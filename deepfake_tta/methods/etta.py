from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs, class_means_from_labels


@dataclass
class ETTAConfig:
    batch_size: int = 512
    num_classes: int = 2
    alpha: float = 0.45
    beta: float = 12.0
    momentum: float = 0.97
    confidence_threshold: float = 0.85
    entropy_power: float = 1.0
    base_weight: float = 1.0


class ETTA(TTAMethod):
    """Efficient recursive embedding update adapter.

    The original ETTA updates contextual text/class embeddings online. In this
    embedding-only pipeline, source class prototypes play that role and are
    recursively updated with confident test samples.
    """

    def __init__(self, config: ETTAConfig | None = None):
        super().__init__(name="etta")
        self.config = config or ETTAConfig()
        self.prototypes: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.prototypes = class_means_from_labels(
            train_feats,
            train_labels,
            num_classes=self.config.num_classes,
        )

    def _normalized_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
        max_entropy = torch.log(torch.tensor(float(probs.shape[1]), device=probs.device))
        return entropy / max_entropy.clamp_min(1e-8)

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        if self.prototypes is None:
            raise RuntimeError("ETTA.fit() must be called before predict_proba().")

        model.eval()
        prototypes = self.prototypes.to(device)
        probs = []
        for start in tqdm(range(0, len(test_feats), self.config.batch_size), desc="etta eval"):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            proto_logits = self.config.beta * feats @ prototypes.T
            proto_probs = torch.softmax(proto_logits, dim=1)

            entropy = self._normalized_entropy(base_probs)
            adaptive_alpha = self.config.alpha * (1 - entropy).clamp(0, 1).pow(self.config.entropy_power)
            out = (
                self.config.base_weight * base_probs
                + adaptive_alpha[:, None] * proto_probs
            ) / (self.config.base_weight + adaptive_alpha[:, None]).clamp_min(1e-8)
            probs.append(out.cpu())

            conf, pred = out.max(dim=1)
            for cls_idx in range(self.config.num_classes):
                mask = (pred == cls_idx) & (conf >= self.config.confidence_threshold)
                if not mask.any():
                    continue
                batch_proto = F.normalize(feats[mask].mean(dim=0), dim=0)
                prototypes[cls_idx] = F.normalize(
                    self.config.momentum * prototypes[cls_idx] + (1 - self.config.momentum) * batch_proto,
                    dim=0,
                )

        self.prototypes = prototypes.detach().cpu()
        return torch.cat(probs, dim=0)
