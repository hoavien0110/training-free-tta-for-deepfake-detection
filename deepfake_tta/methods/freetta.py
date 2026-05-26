from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import (
    binary_model_probs,
    diag_gaussian_log_prob,
    weighted_diag_var,
    weighted_mean,
)


@dataclass
class FreeTTAConfig:
    batch_size: int = 512
    num_classes: int = 2
    momentum: float = 0.95
    prior_power: float = 1.0
    base_weight: float = 0.4
    min_var: float = 1e-4
    warmup_batches: int = 1


class FreeTTA(TTAMethod):
    """Training-free online EM adaptation for feature probabilities."""

    def __init__(self, config: FreeTTAConfig | None = None):
        super().__init__(name="freetta")
        self.config = config or FreeTTAConfig()
        self.means: torch.Tensor | None = None
        self.vars: torch.Tensor | None = None
        self.priors: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        # FreeTTA is source-free at adaptation time; keep fit as a no-op so the
        # method can share the same runner interface as source-aware methods.
        self.means = None
        self.vars = None
        self.priors = None

    @torch.no_grad()
    def predict_proba(
        self,
        model: Any,
        test_feats: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        model.eval()
        probs = []
        means = self.means.to(device) if self.means is not None else None
        vars_ = self.vars.to(device) if self.vars is not None else None
        priors = self.priors.to(device) if self.priors is not None else None

        for batch_idx, start in enumerate(tqdm(range(0, len(test_feats), self.config.batch_size))):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)

            if means is None or vars_ is None or priors is None:
                posterior = base_probs
            else:
                prior_term = base_probs.clamp_min(1e-8).log() * self.config.prior_power
                distribution_logits = diag_gaussian_log_prob(feats, means, vars_) + priors.clamp_min(1e-8).log()
                posterior = torch.softmax(distribution_logits + prior_term, dim=1)

            out = self.config.base_weight * base_probs + (1 - self.config.base_weight) * posterior
            probs.append(out.cpu())

            if batch_idx + 1 < self.config.warmup_batches:
                continue

            batch_priors = posterior.mean(dim=0)
            batch_means = F.normalize(weighted_mean(feats, posterior), dim=-1)
            batch_vars = weighted_diag_var(
                feats,
                posterior,
                batch_means,
                min_var=self.config.min_var,
            )

            if means is None or vars_ is None or priors is None:
                means = batch_means
                vars_ = batch_vars
                priors = batch_priors / batch_priors.sum()
            else:
                means = F.normalize(self.config.momentum * means + (1 - self.config.momentum) * batch_means, dim=-1)
                vars_ = (self.config.momentum * vars_ + (1 - self.config.momentum) * batch_vars).clamp_min(
                    self.config.min_var
                )
                priors = self.config.momentum * priors + (1 - self.config.momentum) * batch_priors
                priors = priors / priors.sum()

        return torch.cat(probs, dim=0)
        
