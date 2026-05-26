from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import (
    binary_model_probs,
    class_means_from_labels,
    priors_from_labels,
    weighted_mean,
)


@dataclass
class BCAConfig:
    batch_size: int = 512
    num_classes: int = 2
    temperature: float = 0.07
    prior_momentum: float = 0.95
    prototype_momentum: float = 0.98
    base_weight: float = 0.5
    confidence_threshold: float = 0.0


class BCA(TTAMethod):
    """Feature-space Bayesian Class Adaptation.

    The original BCA is designed around VLM class embeddings. In this project we
    adapt source feature prototypes and class priors for the binary probe setup.
    """

    def __init__(self, config: BCAConfig | None = None):
        super().__init__(name="bca")
        self.config = config or BCAConfig()
        self.class_prototypes: torch.Tensor | None = None
        self.class_priors: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.class_prototypes = class_means_from_labels(
            train_feats,
            train_labels,
            num_classes=self.config.num_classes,
        )
        self.class_priors = priors_from_labels(train_labels, num_classes=self.config.num_classes)

    @torch.no_grad()
    def predict_proba(
        self,
        model: Any,
        test_feats: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        if self.class_prototypes is None or self.class_priors is None:
            raise RuntimeError("BCA.fit() must be called before predict_proba().")

        model.eval()
        prototypes = self.class_prototypes.to(device)
        priors = self.class_priors.to(device)
        probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size)):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            likelihood_logits = feats @ prototypes.T / self.config.temperature
            posterior = torch.softmax(likelihood_logits + priors.clamp_min(1e-8).log(), dim=1)
            out = self.config.base_weight * base_probs + (1 - self.config.base_weight) * posterior
            probs.append(out.cpu())

            update_weights = out
            if self.config.confidence_threshold > 0:
                confident = out.max(dim=1).values >= self.config.confidence_threshold
                if confident.any():
                    update_weights = out[confident]
                    update_feats = feats[confident]
                else:
                    continue
            else:
                update_feats = feats

            batch_prior = update_weights.mean(dim=0)
            batch_means = F.normalize(weighted_mean(update_feats, update_weights), dim=-1)
            priors = self.config.prior_momentum * priors + (1 - self.config.prior_momentum) * batch_prior
            priors = priors / priors.sum()
            prototypes = F.normalize(
                self.config.prototype_momentum * prototypes
                + (1 - self.config.prototype_momentum) * batch_means,
                dim=-1,
            )

        return torch.cat(probs, dim=0)
