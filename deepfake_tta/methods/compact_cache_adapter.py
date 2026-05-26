from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs, one_hot, topk_cache_probs


@dataclass
class CompactCacheAdapterConfig:
    batch_size: int = 512
    num_classes: int = 2
    cache_ratio: float = 0.5
    beta: float = 5.5
    alpha: float = 0.5
    top_k: int = 64
    balance_cache: bool = False
    prior_correction: bool = True


class CompactCacheAdapter(TTAMethod):
    """Cache adapter with deterministic compact source memory.

    The cache is selected per class by keeping samples closest to the class
    centroid. This avoids random subsampling and keeps representative points.
    """

    def __init__(self, config: CompactCacheAdapterConfig | None = None):
        super().__init__(name="compact_cache_adapter")
        self.config = config or CompactCacheAdapterConfig()
        self.cache_keys: torch.Tensor | None = None
        self.cache_values: torch.Tensor | None = None
        self.cache_priors: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        feats = F.normalize(train_feats.float(), dim=-1)
        labels = train_labels.long()
        selected_indices = []
        counts = torch.bincount(labels, minlength=self.config.num_classes)
        balanced_cap = int((counts.min() * self.config.cache_ratio).item()) if self.config.balance_cache else None

        for cls_idx in range(self.config.num_classes):
            cls_idx_tensor = torch.where(labels == cls_idx)[0]
            cls_feats = feats[cls_idx_tensor]
            if len(cls_feats) == 0:
                continue

            if balanced_cap is None:
                keep = max(1, int(len(cls_feats) * self.config.cache_ratio))
            else:
                keep = max(1, min(balanced_cap, len(cls_feats)))

            centroid = F.normalize(cls_feats.mean(dim=0, keepdim=True), dim=-1)
            similarity = (cls_feats @ centroid.T).squeeze(1)
            keep_local = similarity.topk(keep, largest=True).indices
            selected_indices.append(cls_idx_tensor[keep_local])
            print(f"compact cache class {cls_idx}: {keep}/{len(cls_feats)}")

        selected = torch.cat(selected_indices)
        self.cache_keys = feats[selected]
        self.cache_values = one_hot(labels[selected], self.config.num_classes)
        self.cache_priors = self.cache_values.mean(dim=0).clamp_min(1e-8)
        self.cache_priors = self.cache_priors / self.cache_priors.sum()
        print("compact cache memory:", tuple(self.cache_keys.shape))
        print("compact cache priors:", self.cache_priors.tolist())

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        if self.cache_keys is None or self.cache_values is None or self.cache_priors is None:
            raise RuntimeError("CompactCacheAdapter.fit() must be called before predict_proba().")

        model.eval()
        cache_keys = self.cache_keys.to(device)
        cache_values = self.cache_values.to(device)
        cache_priors = self.cache_priors.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size)):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            cache_probs = topk_cache_probs(
                feats,
                cache_keys,
                cache_values,
                beta=self.config.beta,
                top_k=self.config.top_k,
            )

            if self.config.prior_correction:
                cache_probs = cache_probs / cache_priors[None, :]
                cache_probs = cache_probs / cache_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

            out = self.config.alpha * cache_probs + (1 - self.config.alpha) * base_probs
            probs.append(out.cpu())

        return torch.cat(probs, dim=0)
