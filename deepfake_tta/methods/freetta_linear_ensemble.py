from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs
from deepfake_tta.methods.freetta import FreeTTA, FreeTTAConfig


@dataclass
class FreeTTALinearEnsembleConfig:
    batch_size: int = 512
    linear_weight: float = 0.3
    freetta_weight: float = 0.7
    freetta_momentum: float = 0.95
    freetta_prior_power: float = 1.0
    freetta_base_weight: float = 0.4
    freetta_min_var: float = 1e-4
    freetta_warmup_batches: int = 1


class FreeTTALinearEnsemble(TTAMethod):
    """Lightweight ensemble of the source linear probe and FreeTTA.

    This keeps FreeTTA's source-free memory profile: only online Gaussian
    statistics are maintained during prediction, with no train feature cache.
    """

    def __init__(self, config: FreeTTALinearEnsembleConfig | None = None):
        super().__init__(name="freetta_linear_ensemble")
        self.config = config or FreeTTALinearEnsembleConfig()
        self.freetta = FreeTTA(
            FreeTTAConfig(
                batch_size=self.config.batch_size,
                momentum=self.config.freetta_momentum,
                prior_power=self.config.freetta_prior_power,
                base_weight=self.config.freetta_base_weight,
                min_var=self.config.freetta_min_var,
                warmup_batches=self.config.freetta_warmup_batches,
            )
        )

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.freetta.fit(train_feats, train_labels)

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        model.eval()
        freetta_probs = self.freetta.predict_proba(model, test_feats, device)

        linear_probs = []
        for start in range(0, len(test_feats), self.config.batch_size):
            feats = test_feats[start : start + self.config.batch_size].float().to(device)
            linear_probs.append(binary_model_probs(model, feats).cpu())
        linear_probs = torch.cat(linear_probs, dim=0)

        total_weight = self.config.linear_weight + self.config.freetta_weight
        probs = (
            self.config.linear_weight * linear_probs
            + self.config.freetta_weight * freetta_probs
        ) / total_weight
        return probs.clamp_min(1e-8)
