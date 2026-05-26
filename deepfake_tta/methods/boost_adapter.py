from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs, blend_probs, one_hot, topk_cache_probs


@dataclass
class BoostAdapterConfig:
    batch_size: int = 512
    cache_batch_size: int = 8192
    num_classes: int = 2
    beta: float = 5.5
    alpha: float = 0.5
    dynamic_weight: float = 0.25
    top_k: int = 64
    confidence_threshold: float = 0.75
    max_dynamic_items: int = 4096


class BoostAdapter(TTAMethod):
    """Feature-space BoostAdapter: source cache + confident online memory."""

    def __init__(self, config: BoostAdapterConfig | None = None):
        super().__init__(name="boost_adapter")
        self.config = config or BoostAdapterConfig()
        self.cache_keys: torch.Tensor | None = None
        self.cache_values: torch.Tensor | None = None
        self.dynamic_keys: list[torch.Tensor] = []
        self.dynamic_values: list[torch.Tensor] = []

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.cache_keys = F.normalize(train_feats.float(), dim=-1)
        self.cache_values = one_hot(train_labels, self.config.num_classes)

    def _dynamic_cache(self, device: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.dynamic_keys:
            return None
        return torch.cat(self.dynamic_keys).to(device), torch.cat(self.dynamic_values).to(device)

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        if self.cache_keys is None or self.cache_values is None:
            raise RuntimeError("BoostAdapter.fit() must be called before predict_proba().")

        model.eval()
        cache_keys = self.cache_keys.to(device)
        cache_values = self.cache_values.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size)):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            source_probs = topk_cache_probs(feats, cache_keys, cache_values, beta=self.config.beta, top_k=self.config.top_k)
            parts = [(1 - self.config.alpha, base_probs), (self.config.alpha, source_probs)]

            dynamic_cache = self._dynamic_cache(device)
            if dynamic_cache is not None:
                dyn_keys, dyn_values = dynamic_cache
                dyn_probs = topk_cache_probs(feats, dyn_keys, dyn_values, beta=self.config.beta, top_k=self.config.top_k)
                parts.append((self.config.dynamic_weight, dyn_probs))

            out = blend_probs(*parts)
            probs.append(out.cpu())

            confident = out.max(dim=1).values >= self.config.confidence_threshold
            if confident.any():
                pseudo = out[confident].argmax(dim=1)
                self.dynamic_keys.append(feats[confident].cpu())
                self.dynamic_values.append(one_hot(pseudo.cpu(), self.config.num_classes))
                while sum(len(x) for x in self.dynamic_keys) > self.config.max_dynamic_items:
                    self.dynamic_keys.pop(0)
                    self.dynamic_values.pop(0)

        return torch.cat(probs, dim=0)
