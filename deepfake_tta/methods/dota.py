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
class DOTAConfig:
    batch_size: int = 512
    num_classes: int = 2
    momentum: float = 0.97
    base_weight: float = 0.35
    min_var: float = 1e-4
    confidence_threshold: float = 0.0


class DOTA(TTAMethod):
    """Distributional TTA with online diagonal Gaussian estimates."""

    def __init__(self, config: DOTAConfig | None = None):
        super().__init__(name="dota")
        self.config = config or DOTAConfig()
        self.means: torch.Tensor | None = None
        self.vars: torch.Tensor | None = None
        self.priors: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        means = class_means_from_labels(train_feats, train_labels, num_classes=self.config.num_classes)
        self.means = means
        self.vars = class_diag_vars_from_labels(
            train_feats,
            train_labels,
            means,
            num_classes=self.config.num_classes,
            min_var=self.config.min_var,
        )
        self.priors = priors_from_labels(train_labels, num_classes=self.config.num_classes)

    @torch.no_grad()
    def predict_proba(
        self,
        model: Any,
        test_feats: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        if self.means is None or self.vars is None or self.priors is None:
            raise RuntimeError("DOTA.fit() must be called before predict_proba().")

        model.eval()
        means = self.means.to(device)
        vars_ = self.vars.to(device)
        priors = self.priors.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size)):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            distribution_logits = diag_gaussian_log_prob(feats, means, vars_) + priors.clamp_min(1e-8).log()
            distribution_probs = torch.softmax(distribution_logits, dim=1)
            out = self.config.base_weight * base_probs + (1 - self.config.base_weight) * distribution_probs
            probs.append(out.cpu())

            update_weights = out
            update_feats = feats
            if self.config.confidence_threshold > 0:
                confident = out.max(dim=1).values >= self.config.confidence_threshold
                if not confident.any():
                    continue
                update_weights = out[confident]
                update_feats = feats[confident]

            batch_means = F.normalize(weighted_mean(update_feats, update_weights), dim=-1)
            batch_vars = weighted_diag_var(
                update_feats,
                update_weights,
                batch_means,
                min_var=self.config.min_var,
            )
            batch_priors = update_weights.mean(dim=0)
            means = F.normalize(self.config.momentum * means + (1 - self.config.momentum) * batch_means, dim=-1)
            vars_ = (self.config.momentum * vars_ + (1 - self.config.momentum) * batch_vars).clamp_min(
                self.config.min_var
            )
            priors = self.config.momentum * priors + (1 - self.config.momentum) * batch_priors
            priors = priors / priors.sum()

        return torch.cat(probs, dim=0)
