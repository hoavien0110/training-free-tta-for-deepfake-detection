from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod


@dataclass
class TipAdapterConfig:
    alpha: float = 0.5
    beta: float = 5.5
    test_batch_size: int = 512
    cache_batch_size: int = 8192
    num_classes: int = 2


class TipAdapter(TTAMethod):
    def __init__(self, config: TipAdapterConfig | None = None):
        super().__init__(name="tip_adapter")
        self.config = config or TipAdapterConfig()
        self.cache_keys: torch.Tensor | None = None
        self.cache_values: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.cache_keys = F.normalize(train_feats.float(), dim=-1)
        self.cache_values = F.one_hot(
            train_labels.long(),
            num_classes=self.config.num_classes,
        ).float()

    @torch.no_grad()
    def predict_proba(
        self,
        model: Any,
        test_feats: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        if self.cache_keys is None or self.cache_values is None:
            raise RuntimeError("TipAdapter.fit() must be called before predict_proba().")

        model.eval()
        cache_keys = self.cache_keys.to(device)
        cache_values = self.cache_values.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.test_batch_size)):
            feats = test_feats[start : start + self.config.test_batch_size].float().to(device)
            feats = F.normalize(feats, dim=-1)
            p_fake = torch.sigmoid(model(feats))
            p_base = torch.stack([1 - p_fake, p_fake], dim=1)

            cache_logits = torch.zeros((len(feats), cache_values.shape[1]), device=device)
            cache_norm = torch.zeros((len(feats), 1), device=device)
            for cache_start in range(0, len(cache_keys), self.config.cache_batch_size):
                keys = cache_keys[cache_start : cache_start + self.config.cache_batch_size]
                values = cache_values[cache_start : cache_start + self.config.cache_batch_size]
                affinity = torch.exp(-self.config.beta * (1 - feats @ keys.T))
                cache_logits += affinity @ values
                cache_norm += affinity.sum(dim=1, keepdim=True)

            p_cache = cache_logits / (cache_norm + 1e-8)
            probs.append((self.config.alpha * p_cache + (1 - self.config.alpha) * p_base).cpu())

        return torch.cat(probs, dim=0)


def build_tip_cache(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_keys = F.normalize(train_feats.float(), dim=-1)
    cache_values = F.one_hot(train_labels.long(), num_classes=num_classes).float()
    return cache_keys, cache_values
