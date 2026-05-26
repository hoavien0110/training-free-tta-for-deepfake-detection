from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs, blend_probs, one_hot, topk_cache_probs


@dataclass
class DMNConfig:
    batch_size: int = 512
    num_classes: int = 2
    beta: float = 5.5
    top_k: int = 64
    static_weight: float = 0.45
    dynamic_weight: float = 0.35
    confidence_threshold: float = 0.7
    max_dynamic_items: int = 8192


class DMN(TTAMethod):
    """Dual memory network: labeled source memory + confident target memory."""

    def __init__(self, config: DMNConfig | None = None):
        super().__init__(name="dmn")
        self.config = config or DMNConfig()
        self.static_keys: torch.Tensor | None = None
        self.static_values: torch.Tensor | None = None
        self.dynamic_keys: list[torch.Tensor] = []
        self.dynamic_values: list[torch.Tensor] = []

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.static_keys = F.normalize(train_feats.float(), dim=-1)
        self.static_values = one_hot(train_labels, self.config.num_classes)

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        if self.static_keys is None or self.static_values is None:
            raise RuntimeError("DMN.fit() must be called before predict_proba().")

        model.eval()
        static_keys = self.static_keys.to(device)
        static_values = self.static_values.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size)):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            static_probs = topk_cache_probs(feats, static_keys, static_values, beta=self.config.beta, top_k=self.config.top_k)
            parts = [(1.0, base_probs), (self.config.static_weight, static_probs)]

            if self.dynamic_keys:
                dyn_keys = torch.cat(self.dynamic_keys).to(device)
                dyn_values = torch.cat(self.dynamic_values).to(device)
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
