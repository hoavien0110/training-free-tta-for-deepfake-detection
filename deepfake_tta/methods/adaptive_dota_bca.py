from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import (
    binary_model_probs,
    class_diag_vars_from_labels,
    class_means_from_labels,
    diag_gaussian_log_prob,
    priors_from_labels,
    weighted_diag_var,
    weighted_mean,
)


@dataclass
class AdaptiveDotaBcaConfig:
    batch_size: int = 512
    num_classes: int = 2
    dota_momentum: float = 0.97
    dota_base_weight: float = 0.75
    dota_confidence_threshold: float = 0.8
    bca_temperature: float = 0.03
    bca_base_weight: float = 0.7
    bca_prior_momentum: float = 0.95
    bca_prototype_momentum: float = 0.98
    min_var: float = 1e-4
    base_floor: float = 0.15
    disagreement_gain: float = 1.5
    confidence_power: float = 1.5


class AdaptiveDotaBca(TTAMethod):
    """Reliability-gated online blend of frozen probe, DOTA, and BCA.

    DOTA and BCA are strong in different settings in the current sweeps. This
    method keeps both online states and routes probability mass toward the
    branch with lower entropy, while falling back to the frozen probe when the
    two adaptation branches disagree.
    """

    def __init__(self, config: AdaptiveDotaBcaConfig | None = None):
        super().__init__(name="adaptive_dota_bca")
        self.config = config or AdaptiveDotaBcaConfig()
        self.dota_means: torch.Tensor | None = None
        self.dota_vars: torch.Tensor | None = None
        self.dota_priors: torch.Tensor | None = None
        self.bca_prototypes: torch.Tensor | None = None
        self.bca_priors: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        means = class_means_from_labels(train_feats, train_labels, num_classes=self.config.num_classes)
        self.dota_means = means
        self.dota_vars = class_diag_vars_from_labels(
            train_feats,
            train_labels,
            means,
            num_classes=self.config.num_classes,
            min_var=self.config.min_var,
        )
        self.dota_priors = priors_from_labels(train_labels, num_classes=self.config.num_classes)
        self.bca_prototypes = means.clone()
        self.bca_priors = self.dota_priors.clone()

    def _confidence(self, probs: torch.Tensor) -> torch.Tensor:
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)
        max_entropy = torch.log(torch.tensor(float(probs.shape[1]), device=probs.device))
        confidence = 1.0 - entropy / max_entropy.clamp_min(1e-8)
        return confidence.clamp(0.0, 1.0).pow(self.config.confidence_power)

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        required = (
            self.dota_means,
            self.dota_vars,
            self.dota_priors,
            self.bca_prototypes,
            self.bca_priors,
        )
        if any(value is None for value in required):
            raise RuntimeError("AdaptiveDotaBca.fit() must be called before predict_proba().")

        model.eval()
        dota_means = self.dota_means.to(device)
        dota_vars = self.dota_vars.to(device)
        dota_priors = self.dota_priors.to(device)
        bca_prototypes = self.bca_prototypes.to(device)
        bca_priors = self.bca_priors.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size)):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)

            dota_logits = diag_gaussian_log_prob(feats, dota_means, dota_vars) + dota_priors.clamp_min(1e-8).log()
            dota_distribution = torch.softmax(dota_logits, dim=1)
            dota_probs = (
                self.config.dota_base_weight * base_probs
                + (1.0 - self.config.dota_base_weight) * dota_distribution
            )

            bca_logits = feats @ bca_prototypes.T / self.config.bca_temperature
            bca_posterior = torch.softmax(bca_logits + bca_priors.clamp_min(1e-8).log(), dim=1)
            bca_probs = self.config.bca_base_weight * base_probs + (1.0 - self.config.bca_base_weight) * bca_posterior

            dota_conf = self._confidence(dota_probs)
            bca_conf = self._confidence(bca_probs)
            base_conf = self._confidence(base_probs)
            disagreement = (dota_probs[:, 1] - bca_probs[:, 1]).abs().clamp(0.0, 1.0)

            base_score = self.config.base_floor + base_conf * (1.0 + self.config.disagreement_gain * disagreement)
            dota_score = dota_conf * (1.0 - disagreement).clamp_min(0.05)
            bca_score = bca_conf * (1.0 - disagreement).clamp_min(0.05)
            scores = torch.stack([base_score, dota_score, bca_score], dim=1).clamp_min(1e-6)
            weights = scores / scores.sum(dim=1, keepdim=True)

            out = (
                weights[:, 0:1] * base_probs
                + weights[:, 1:2] * dota_probs
                + weights[:, 2:3] * bca_probs
            ).clamp_min(1e-8)
            out = out / out.sum(dim=1, keepdim=True)
            probs.append(out.cpu())

            update_weights = out
            update_feats = feats
            if self.config.dota_confidence_threshold > 0:
                confident = out.max(dim=1).values >= self.config.dota_confidence_threshold
                if confident.any():
                    update_weights = out[confident]
                    update_feats = feats[confident]
                else:
                    continue

            batch_means = F.normalize(weighted_mean(update_feats, update_weights), dim=-1)
            batch_vars = weighted_diag_var(
                update_feats,
                update_weights,
                batch_means,
                min_var=self.config.min_var,
            )
            batch_priors = update_weights.mean(dim=0)

            dota_means = F.normalize(
                self.config.dota_momentum * dota_means + (1.0 - self.config.dota_momentum) * batch_means,
                dim=-1,
            )
            dota_vars = (
                self.config.dota_momentum * dota_vars + (1.0 - self.config.dota_momentum) * batch_vars
            ).clamp_min(self.config.min_var)
            dota_priors = self.config.dota_momentum * dota_priors + (1.0 - self.config.dota_momentum) * batch_priors
            dota_priors = dota_priors / dota_priors.sum()

            bca_priors = (
                self.config.bca_prior_momentum * bca_priors
                + (1.0 - self.config.bca_prior_momentum) * batch_priors
            )
            bca_priors = bca_priors / bca_priors.sum()
            bca_prototypes = F.normalize(
                self.config.bca_prototype_momentum * bca_prototypes
                + (1.0 - self.config.bca_prototype_momentum) * batch_means,
                dim=-1,
            )

        return torch.cat(probs, dim=0)
