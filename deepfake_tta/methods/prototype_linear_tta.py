from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs
from deepfake_tta.modeling import evaluate_scores


@dataclass
class PrototypeLinearTTAConfig:
    batch_size: int = 512
    num_classes: int = 2
    prototypes_per_class: int = 32
    prototypes_by_class: dict[int, int] = field(default_factory=lambda: {0: 32, 1: 128})
    kmeans_iters: int = 20
    temperature: float = 0.07
    lr: float = 1e-5
    steps_per_batch: int = 1
    anchor_weight: float = 0.3
    entropy_weight: float = 0.0
    balance_weight: float = 0.0
    reg_weight: float = 0.1
    target_prior: tuple[float, float] | None = None


class PrototypeLinearTTA(TTAMethod):
    def __init__(self, config: PrototypeLinearTTAConfig | None = None):
        super().__init__(name="prototype_linear_tta")
        self.config = config or PrototypeLinearTTAConfig()
        self.prototype_keys: torch.Tensor | None = None
        self.prototype_values: torch.Tensor | None = None

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        feats = F.normalize(train_feats.float(), dim=-1)
        labels = train_labels.long()
        keys = []
        values = []

        for cls_idx in range(self.config.num_classes):
            cls_feats = feats[labels == cls_idx]
            if len(cls_feats) == 0:
                continue

            requested_k = self.config.prototypes_by_class.get(
                cls_idx,
                self.config.prototypes_per_class,
            )
            k = min(requested_k, len(cls_feats))
            centroids = self._kmeans(cls_feats, k)
            keys.append(centroids)
            values.append(
                F.one_hot(
                    torch.full((k,), cls_idx, dtype=torch.long),
                    num_classes=self.config.num_classes,
                ).float()
            )

        self.prototype_keys = F.normalize(torch.cat(keys, dim=0), dim=-1)
        self.prototype_values = torch.cat(values, dim=0)
        print("prototype counts:", self.config.prototypes_by_class)
        print("prototype memory:", tuple(self.prototype_keys.shape))

    def _kmeans(self, feats: torch.Tensor, k: int) -> torch.Tensor:
        idx = torch.randperm(len(feats))[:k]
        centroids = feats[idx].clone()

        for _ in range(self.config.kmeans_iters):
            sim = feats @ centroids.T
            assign = sim.argmax(dim=1)
            new_centroids = []
            for cluster_idx in range(k):
                mask = assign == cluster_idx
                if mask.any():
                    new_centroids.append(feats[mask].mean(dim=0))
                else:
                    new_centroids.append(centroids[cluster_idx])
            centroids = F.normalize(torch.stack(new_centroids), dim=-1)

        return centroids

    def _anchor_probs(self, feats: torch.Tensor) -> torch.Tensor:
        if self.prototype_keys is None or self.prototype_values is None:
            raise RuntimeError("PrototypeLinearTTA.fit() must be called before predict_proba().")

        keys = self.prototype_keys.to(feats.device)
        values = self.prototype_values.to(feats.device)
        logits = feats @ keys.T / self.config.temperature
        proto_weights = torch.softmax(logits, dim=1)
        return proto_weights @ values

    def _soft_ce(self, probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        return -(target_probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        return -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()

    def _balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        mean_probs = probs.mean(dim=0).clamp_min(1e-8)
        if self.config.target_prior is not None:
            prior = torch.tensor(self.config.target_prior, device=probs.device).float()
            prior = prior / prior.sum()
            return (mean_probs * (mean_probs.log() - prior.clamp_min(1e-8).log())).sum()
        return (mean_probs * mean_probs.log()).sum()

    def evaluate(
        self,
        model: Any,
        test_feats: torch.Tensor,
        test_labels: torch.Tensor,
        device: str,
        name: str = "test",
        show_report: bool = True,
    ) -> dict[str, float]:
        probs = self.predict_proba(model, test_feats, device)
        return evaluate_scores(
            y_true=test_labels.numpy(),
            y_score=probs[:, 1].numpy(),
            y_pred=probs.argmax(dim=1).numpy(),
            name=name,
            show_report=show_report,
        )

    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        if self.prototype_keys is None or self.prototype_values is None:
            raise RuntimeError("PrototypeLinearTTA.fit() must be called before predict_proba().")

        adapted_model = copy.deepcopy(model).to(device)
        adapted_model.train()
        init_weight = adapted_model.fc.weight.detach().clone()
        init_bias = adapted_model.fc.bias.detach().clone()
        optimizer = torch.optim.AdamW(adapted_model.parameters(), lr=self.config.lr)
        all_probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size)):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            anchor_probs = self._anchor_probs(feats).detach()

            for _ in range(self.config.steps_per_batch):
                with torch.enable_grad():
                    pred_probs = binary_model_probs(adapted_model, feats)
                    loss_anchor = self._soft_ce(pred_probs, anchor_probs)
                    loss_entropy = self._entropy(pred_probs)
                    loss_balance = self._balance_loss(pred_probs)
                    loss_reg = (
                        (adapted_model.fc.weight - init_weight).pow(2).mean()
                        + (adapted_model.fc.bias - init_bias).pow(2).mean()
                    )
                    loss = (
                        self.config.anchor_weight * loss_anchor
                        + self.config.entropy_weight * loss_entropy
                        + self.config.balance_weight * loss_balance
                        + self.config.reg_weight * loss_reg
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            adapted_model.eval()
            with torch.no_grad():
                all_probs.append(binary_model_probs(adapted_model, feats).cpu())
            adapted_model.train()

        return torch.cat(all_probs, dim=0)
