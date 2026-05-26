from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs, class_means_from_labels, weighted_mean


@dataclass
class DPEConfig:
    batch_size: int = 512
    num_classes: int = 2
    temperature: float = 0.07
    visual_momentum: float = 0.95
    text_momentum: float = 0.995
    base_weight: float = 0.4


class DPE(TTAMethod):
    """Dual prototype evolving in feature space."""

    def __init__(self, config: DPEConfig | None = None):
        super().__init__(name="dpe")
        self.config = config or DPEConfig()
        self.text_prototypes: torch.Tensor | None = None
        self.visual_prototypes: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        prototypes = class_means_from_labels(train_feats, train_labels, self.config.num_classes)
        self.text_prototypes = prototypes.clone()
        self.visual_prototypes = prototypes.clone()

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        if self.text_prototypes is None or self.visual_prototypes is None:
            raise RuntimeError("DPE.fit() must be called before predict_proba().")

        model.eval()
        text_proto = self.text_prototypes.to(device)
        visual_proto = self.visual_prototypes.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size)):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            proto = F.normalize((text_proto + visual_proto) / 2, dim=-1)
            proto_probs = torch.softmax(feats @ proto.T / self.config.temperature, dim=1)
            out = self.config.base_weight * base_probs + (1 - self.config.base_weight) * proto_probs
            probs.append(out.cpu())

            batch_visual = F.normalize(weighted_mean(feats, out), dim=-1)
            visual_proto = F.normalize(
                self.config.visual_momentum * visual_proto + (1 - self.config.visual_momentum) * batch_visual,
                dim=-1,
            )
            text_proto = F.normalize(
                self.config.text_momentum * text_proto + (1 - self.config.text_momentum) * visual_proto,
                dim=-1,
            )

        return torch.cat(probs, dim=0)
