from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import (
    binary_model_probs,
    blend_probs,
    class_diag_vars_from_labels,
    class_means_from_labels,
    diag_gaussian_log_prob,
    one_hot,
    topk_cache_probs,
    weighted_diag_var,
    weighted_mean,
)


@dataclass
class CRGConfig:
    batch_size: int = 512
    num_classes: int = 2
    beta: float = 5.5
    top_k: int = 64
    residual_momentum: float = 0.97
    gaussian_momentum: float = 0.97
    cache_weight: float = 0.35
    gaussian_weight: float = 0.35
    min_var: float = 1e-4


class CRG(TTAMethod):
    """Cache + residual prototype alignment + Gaussian distribution modeling."""

    def __init__(self, config: CRGConfig | None = None):
        super().__init__(name="crg")
        self.config = config or CRGConfig()
        self.cache_keys: torch.Tensor | None = None
        self.cache_values: torch.Tensor | None = None
        self.means: torch.Tensor | None = None
        self.vars: torch.Tensor | None = None
        self.residual: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.cache_keys = F.normalize(train_feats.float(), dim=-1)
        self.cache_values = one_hot(train_labels, self.config.num_classes)
        means = class_means_from_labels(train_feats, train_labels, self.config.num_classes)
        self.means = means
        self.vars = class_diag_vars_from_labels(
            train_feats,
            train_labels,
            means,
            self.config.num_classes,
            min_var=self.config.min_var,
        )
        self.residual = torch.zeros_like(means)

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        if any(x is None for x in (self.cache_keys, self.cache_values, self.means, self.vars, self.residual)):
            raise RuntimeError("CRG.fit() must be called before predict_proba().")

        model.eval()
        cache_keys = self.cache_keys.to(device)
        cache_values = self.cache_values.to(device)
        means = self.means.to(device)
        vars_ = self.vars.to(device)
        residual = self.residual.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size)):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            aligned_means = F.normalize(means + residual, dim=-1)
            cache_out = topk_cache_probs(feats, cache_keys, cache_values, beta=self.config.beta, top_k=self.config.top_k)
            gaussian_out = torch.softmax(diag_gaussian_log_prob(feats, aligned_means, vars_), dim=1)
            out = blend_probs((1.0, base_probs), (self.config.cache_weight, cache_out), (self.config.gaussian_weight, gaussian_out))
            probs.append(out.cpu())

            batch_means = F.normalize(weighted_mean(feats, out), dim=-1)
            batch_vars = weighted_diag_var(feats, out, batch_means, min_var=self.config.min_var)
            residual = self.config.residual_momentum * residual + (1 - self.config.residual_momentum) * (
                batch_means - means
            )
            means = F.normalize(
                self.config.gaussian_momentum * means + (1 - self.config.gaussian_momentum) * batch_means,
                dim=-1,
            )
            vars_ = (self.config.gaussian_momentum * vars_ + (1 - self.config.gaussian_momentum) * batch_vars).clamp_min(
                self.config.min_var
            )

        return torch.cat(probs, dim=0)
