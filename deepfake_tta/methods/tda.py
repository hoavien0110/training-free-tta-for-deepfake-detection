from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs, one_hot, topk_cache_probs


@dataclass
class TDAConfig:
    batch_size: int = 512
    num_classes: int = 2
    positive_alpha: float = 0.35
    positive_beta: float = 5.5
    positive_shot_capacity: int = 64
    positive_entropy_threshold: float = 0.35
    negative_alpha: float = 0.15
    negative_beta: float = 5.5
    negative_shot_capacity: int = 64
    negative_entropy_lower: float = 0.35
    negative_entropy_upper: float = 0.8
    negative_mask_lower: float = 0.2
    negative_mask_upper: float = 0.8
    top_k: int = 64


class TDA(TTAMethod):
    """Training-free dynamic adapter with positive and negative caches."""

    def __init__(self, config: TDAConfig | None = None):
        super().__init__(name="tda")
        self.config = config or TDAConfig()
        self.pos_keys: list[list[torch.Tensor]] = []
        self.pos_scores: list[list[torch.Tensor]] = []
        self.neg_keys: list[list[torch.Tensor]] = []
        self.neg_values: list[list[torch.Tensor]] = []
        self.neg_scores: list[list[torch.Tensor]] = []

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.pos_keys = [[] for _ in range(self.config.num_classes)]
        self.pos_scores = [[] for _ in range(self.config.num_classes)]
        self.neg_keys = [[] for _ in range(self.config.num_classes)]
        self.neg_values = [[] for _ in range(self.config.num_classes)]
        self.neg_scores = [[] for _ in range(self.config.num_classes)]

    def _normalized_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
        max_entropy = torch.log(torch.tensor(float(probs.shape[1]), device=probs.device))
        return entropy / max_entropy.clamp_min(1e-8)

    def _append_positive(self, feats: torch.Tensor, labels: torch.Tensor, entropy: torch.Tensor) -> None:
        for cls_idx in range(self.config.num_classes):
            mask = labels == cls_idx
            if not mask.any():
                continue
            keys = torch.cat(self.pos_keys[cls_idx] + [feats[mask].cpu()], dim=0)
            scores = torch.cat(self.pos_scores[cls_idx] + [entropy[mask].cpu()], dim=0)
            order = scores.argsort()
            keep = order[: self.config.positive_shot_capacity]
            self.pos_keys[cls_idx] = [keys[keep]]
            self.pos_scores[cls_idx] = [scores[keep]]

    def _append_negative(self, feats: torch.Tensor, pred: torch.Tensor, prob_map: torch.Tensor, entropy: torch.Tensor) -> None:
        neg_values = ((prob_map > self.config.negative_mask_lower) & (prob_map < self.config.negative_mask_upper)).float()
        if not neg_values.any():
            return
        for cls_idx in range(self.config.num_classes):
            mask = pred == cls_idx
            if not mask.any():
                continue
            keys = torch.cat(self.neg_keys[cls_idx] + [feats[mask].cpu()], dim=0)
            values = torch.cat(self.neg_values[cls_idx] + [neg_values[mask].cpu()], dim=0)
            scores = torch.cat(self.neg_scores[cls_idx] + [entropy[mask].cpu()], dim=0)
            order = scores.argsort()
            keep = order[: self.config.negative_shot_capacity]
            self.neg_keys[cls_idx] = [keys[keep]]
            self.neg_values[cls_idx] = [values[keep]]
            self.neg_scores[cls_idx] = [scores[keep]]

    def _cache_tensors(self, cache: list[list[torch.Tensor]], values: list[list[torch.Tensor]] | None, device: str):
        keys = []
        vals = []
        for cls_idx, cls_keys in enumerate(cache):
            if not cls_keys:
                continue
            cls_key_tensor = torch.cat(cls_keys, dim=0)
            keys.append(cls_key_tensor)
            if values is None:
                vals.append(one_hot(torch.full((len(cls_key_tensor),), cls_idx), self.config.num_classes))
            else:
                vals.append(torch.cat(values[cls_idx], dim=0))
        if not keys:
            return None, None
        return torch.cat(keys, dim=0).to(device), torch.cat(vals, dim=0).to(device)

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        model.eval()
        all_probs = []
        for start in tqdm(range(0, len(test_feats), self.config.batch_size), desc="tda eval"):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            logits = base_probs.clamp_min(1e-8).log()

            pos_keys, pos_values = self._cache_tensors(self.pos_keys, None, device)
            if pos_keys is not None and pos_values is not None:
                logits = logits + self.config.positive_alpha * topk_cache_probs(
                    feats, pos_keys, pos_values, beta=self.config.positive_beta, top_k=self.config.top_k
                )

            neg_keys, neg_values = self._cache_tensors(self.neg_keys, self.neg_values, device)
            if neg_keys is not None and neg_values is not None:
                logits = logits - self.config.negative_alpha * topk_cache_probs(
                    feats, neg_keys, neg_values, beta=self.config.negative_beta, top_k=self.config.top_k
                )

            out = torch.softmax(logits, dim=1)
            all_probs.append(out.cpu())

            entropy = self._normalized_entropy(base_probs)
            pred = base_probs.argmax(dim=1)
            pos_mask = entropy <= self.config.positive_entropy_threshold
            if pos_mask.any():
                self._append_positive(feats[pos_mask].detach(), pred[pos_mask].detach(), entropy[pos_mask].detach())
            neg_mask = (entropy >= self.config.negative_entropy_lower) & (entropy <= self.config.negative_entropy_upper)
            if neg_mask.any():
                self._append_negative(
                    feats[neg_mask].detach(),
                    pred[neg_mask].detach(),
                    base_probs[neg_mask].detach(),
                    entropy[neg_mask].detach(),
                )

        return torch.cat(all_probs, dim=0)
