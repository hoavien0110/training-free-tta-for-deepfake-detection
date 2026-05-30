from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs


@dataclass
class PriorBalancedConfig:
    batch_size: int = 512
    target_prior: tuple[float, float] = (0.5, 0.5)
    strength: float = 0.7


class PriorBalancedWrapper(TTAMethod):
    """Apply batch-level prior correction to an online TTA method.

    This keeps the wrapped method's memory behavior. It is useful when test
    batches are expected to be balanced, but the model probabilities are biased
    toward one class.
    """

    def __init__(
        self,
        inner: TTAMethod,
        name: str,
        config: PriorBalancedConfig | None = None,
    ):
        super().__init__(name=name)
        self.inner = inner
        self.config = config or PriorBalancedConfig()

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.inner.fit(train_feats, train_labels)

    def _balance_probs(self, probs: torch.Tensor) -> torch.Tensor:
        target = torch.tensor(self.config.target_prior, device=probs.device, dtype=probs.dtype)
        target = target / target.sum()
        adjusted = []
        for start in range(0, len(probs), self.config.batch_size):
            batch = probs[start : start + self.config.batch_size]
            batch_prior = batch.mean(dim=0).clamp_min(1e-8)
            factors = (target / batch_prior).pow(self.config.strength)
            corrected = batch * factors[None, :]
            corrected = corrected / corrected.sum(dim=1, keepdim=True).clamp_min(1e-8)
            adjusted.append(corrected)
        return torch.cat(adjusted, dim=0)

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        probs = self.inner.predict_proba(model, test_feats, device)
        return self._balance_probs(probs).cpu()


@dataclass
class LinearEnsembleConfig:
    batch_size: int = 512
    linear_weight: float = 0.3
    inner_weight: float = 0.7


class LinearEnsembleWrapper(TTAMethod):
    """Blend a lightweight online TTA method with the frozen linear probe."""

    def __init__(
        self,
        inner: TTAMethod,
        name: str,
        config: LinearEnsembleConfig | None = None,
    ):
        super().__init__(name=name)
        self.inner = inner
        self.config = config or LinearEnsembleConfig()

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.inner.fit(train_feats, train_labels)

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        inner_probs = self.inner.predict_proba(model, test_feats, device)
        model.eval()
        linear_probs = []
        for start in range(0, len(test_feats), self.config.batch_size):
            feats = test_feats[start : start + self.config.batch_size].float().to(device)
            linear_probs.append(binary_model_probs(model, feats).cpu())
        linear_probs = torch.cat(linear_probs, dim=0)

        total_weight = self.config.linear_weight + self.config.inner_weight
        probs = (
            self.config.linear_weight * linear_probs
            + self.config.inner_weight * inner_probs
        ) / total_weight
        return probs.clamp_min(1e-8)
