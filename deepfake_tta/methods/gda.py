from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs, priors_from_labels


@dataclass
class GDAConfig:
    batch_size: int = 512
    num_classes: int = 2
    alpha: float = 1.0
    base_weight: float = 1.0
    temperature: float = 1.0
    shrinkage: float = 0.1
    min_var: float = 1e-4


class GDA(TTAMethod):
    """Gaussian discriminant classifier blended with the frozen probe.

    This is a feature-level version of training-free CLIP GDA: estimate class
    means and a shared diagonal covariance from source embeddings, then ensemble
    the resulting discriminant logits with the existing probe logits.
    """

    def __init__(self, config: GDAConfig | None = None):
        super().__init__(name="gda")
        self.config = config or GDAConfig()
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        feats = F.normalize(train_feats.float(), dim=-1)
        labels = train_labels.long()
        means = []
        global_mean = feats.mean(dim=0)
        for cls_idx in range(self.config.num_classes):
            mask = labels == cls_idx
            means.append(feats[mask].mean(dim=0) if mask.any() else global_mean)
        means = torch.stack(means, dim=0)

        centered = feats - means[labels]
        shared_var = centered.pow(2).mean(dim=0).clamp_min(self.config.min_var)
        if self.config.shrinkage > 0:
            avg_var = shared_var.mean()
            shared_var = (1 - self.config.shrinkage) * shared_var + self.config.shrinkage * avg_var
            shared_var = shared_var.clamp_min(self.config.min_var)

        inv_var = 1.0 / shared_var
        priors = priors_from_labels(labels, num_classes=self.config.num_classes)
        self.weight = means * inv_var[None, :]
        self.bias = -0.5 * (means.pow(2) * inv_var[None, :]).sum(dim=1) + priors.clamp_min(1e-8).log()

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        if self.weight is None or self.bias is None:
            raise RuntimeError("GDA.fit() must be called before predict_proba().")

        model.eval()
        weight = self.weight.to(device)
        bias = self.bias.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size), desc="gda eval"):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            base_logits = base_probs.clamp_min(1e-8).log()
            gda_logits = (feats @ weight.T + bias[None, :]) / max(self.config.temperature, 1e-8)
            logits = self.config.base_weight * base_logits + self.config.alpha * gda_logits
            probs.append(torch.softmax(logits, dim=1).cpu())

        return torch.cat(probs, dim=0)
