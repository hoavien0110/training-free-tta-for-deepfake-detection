from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from deepfake_tta.methods.base import TTAMethod
from deepfake_tta.methods.common import binary_model_probs, one_hot, topk_cache_probs


@dataclass
class OnlineConfidentCacheAdapterConfig:
    batch_size: int = 512
    num_classes: int = 2
    source_cache_ratio: float = 0.1
    beta: float = 5.5
    alpha_source: float = 0.2
    alpha_dynamic: float = 0.25
    top_k: int = 64
    add_threshold: float = 0.9
    max_entropy: float = 0.35
    entropy_score_weight: float = 0.25
    use_threshold: float = 0.85
    agreement_only: bool = True
    max_dynamic_per_class: int = 2048
    cache_path: str | None = None
    save_cache: bool = True


class OnlineConfidentCacheAdapter(TTAMethod):
    """Online cache adapter with confident source and target memory.

    Source cache is selected by linear-probe confidence and low entropy. During
    test-time, confident target samples are added to a bounded dynamic cache for
    later batches.
    """

    def __init__(self, config: OnlineConfidentCacheAdapterConfig | None = None):
        super().__init__(name="online_confident_cache_adapter")
        self.config = config or OnlineConfidentCacheAdapterConfig()
        self.source_keys: torch.Tensor | None = None
        self.source_values: torch.Tensor | None = None
        self.pending_train_feats: torch.Tensor | None = None
        self.pending_train_labels: torch.Tensor | None = None
        self.dynamic_keys: list[list[torch.Tensor]] = []
        self.dynamic_values: list[list[torch.Tensor]] = []

    def fit(self, train_feats: torch.Tensor, train_labels: torch.Tensor) -> None:
        if self.config.cache_path and Path(self.config.cache_path).exists():
            payload = torch.load(self.config.cache_path, map_location="cpu")
            self.source_keys = payload["source_keys"].float()
            self.source_values = payload["source_values"].float()
            print("loaded source cache from:", self.config.cache_path)
            print("source cache memory:", tuple(self.source_keys.shape))
        else:
            self.pending_train_feats = F.normalize(train_feats.float(), dim=-1).cpu()
            self.pending_train_labels = train_labels.long().cpu()

        self.dynamic_keys = [[] for _ in range(self.config.num_classes)]
        self.dynamic_values = [[] for _ in range(self.config.num_classes)]

    def _normalized_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
        max_entropy = torch.log(torch.tensor(float(probs.shape[1]), device=probs.device))
        return entropy / max_entropy.clamp_min(1e-8)

    @torch.no_grad()
    def _build_source_cache_by_confidence(self, model: Any, device: str) -> None:
        if self.source_keys is not None and self.source_values is not None:
            return
        if self.pending_train_feats is None or self.pending_train_labels is None:
            raise RuntimeError("No train features available to build source cache.")

        model.eval()
        feats = self.pending_train_feats
        labels = self.pending_train_labels
        all_conf = []
        all_pred = []
        all_entropy = []

        for start in tqdm(range(0, len(feats), self.config.batch_size), desc="scoring train confidence"):
            xb = feats[start : start + self.config.batch_size].to(device)
            probs = binary_model_probs(model, xb)
            conf, pred = probs.max(dim=1)
            all_conf.append(conf.cpu())
            all_pred.append(pred.cpu())
            all_entropy.append(self._normalized_entropy(probs).cpu())

        conf = torch.cat(all_conf)
        pred = torch.cat(all_pred)
        entropy = torch.cat(all_entropy)
        selected = []

        for cls_idx in range(self.config.num_classes):
            cls_indices = torch.where(labels == cls_idx)[0]
            cls_conf = conf[cls_indices]
            cls_pred = pred[cls_indices]
            cls_entropy = entropy[cls_indices]
            cls_score = cls_conf - self.config.entropy_score_weight * cls_entropy
            keep = max(1, int(len(cls_indices) * self.config.source_cache_ratio))

            correct_mask = cls_pred == cls_idx
            correct_indices = cls_indices[correct_mask]
            correct_score = cls_score[correct_mask]

            if len(correct_indices) >= keep:
                chosen = correct_indices[correct_score.topk(keep, largest=True).indices]
            else:
                remaining = keep - len(correct_indices)
                incorrect_indices = cls_indices[~correct_mask]
                incorrect_score = cls_score[~correct_mask]
                if len(incorrect_indices) > 0 and remaining > 0:
                    fill = incorrect_score.topk(min(remaining, len(incorrect_indices)), largest=True).indices
                    chosen = torch.cat([correct_indices, incorrect_indices[fill]])
                else:
                    chosen = correct_indices

            selected.append(chosen)
            print(
                f"source cache class {cls_idx}: {len(chosen)}/{len(cls_indices)} "
                f"(correct candidates={int(correct_mask.sum())})"
            )

        selected_indices = torch.cat(selected)
        self.source_keys = feats[selected_indices].cpu()
        self.source_values = one_hot(labels[selected_indices], self.config.num_classes).cpu()
        print("source cache memory:", tuple(self.source_keys.shape))

        if self.config.cache_path and self.config.save_cache:
            cache_path = Path(self.config.cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "source_keys": self.source_keys,
                    "source_values": self.source_values,
                    "source_cache_ratio": self.config.source_cache_ratio,
                    "selection": "linear_probe_confidence_entropy_correct_first",
                    "num_classes": self.config.num_classes,
                },
                cache_path,
            )
            print("saved source cache to:", cache_path)

        self.pending_train_feats = None
        self.pending_train_labels = None

    def _get_dynamic_cache(self, device: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        keys = []
        values = []
        for cls_idx in range(self.config.num_classes):
            if self.dynamic_keys[cls_idx]:
                keys.append(torch.cat(self.dynamic_keys[cls_idx], dim=0))
                values.append(torch.cat(self.dynamic_values[cls_idx], dim=0))
        if not keys:
            return None, None
        return torch.cat(keys, dim=0).to(device), torch.cat(values, dim=0).to(device)

    def _append_dynamic(self, feats_cpu: torch.Tensor, labels_cpu: torch.Tensor) -> None:
        for cls_idx in range(self.config.num_classes):
            mask = labels_cpu == cls_idx
            if not mask.any():
                continue

            self.dynamic_keys[cls_idx].append(feats_cpu[mask])
            self.dynamic_values[cls_idx].append(one_hot(labels_cpu[mask], self.config.num_classes))
            keys = torch.cat(self.dynamic_keys[cls_idx], dim=0)
            values = torch.cat(self.dynamic_values[cls_idx], dim=0)

            if len(keys) > self.config.max_dynamic_per_class:
                keys = keys[-self.config.max_dynamic_per_class :]
                values = values[-self.config.max_dynamic_per_class :]

            self.dynamic_keys[cls_idx] = [keys]
            self.dynamic_values[cls_idx] = [values]

    @torch.no_grad()
    def predict_proba(self, model: Any, test_feats: torch.Tensor, device: str) -> torch.Tensor:
        self._build_source_cache_by_confidence(model, device)
        if self.source_keys is None or self.source_values is None:
            raise RuntimeError("Source cache was not built.")

        model.eval()
        source_keys = self.source_keys.to(device)
        source_values = self.source_values.to(device)
        all_probs = []

        for start in tqdm(range(0, len(test_feats), self.config.batch_size), desc="online cache eval"):
            feats = F.normalize(test_feats[start : start + self.config.batch_size].float().to(device), dim=-1)
            base_probs = binary_model_probs(model, feats)
            source_probs = topk_cache_probs(
                feats,
                source_keys,
                source_values,
                beta=self.config.beta,
                top_k=self.config.top_k,
            )
            out = (1 - self.config.alpha_source) * base_probs + self.config.alpha_source * source_probs

            dyn_keys, dyn_values = self._get_dynamic_cache(device)
            if dyn_keys is not None and dyn_values is not None:
                dynamic_probs = topk_cache_probs(
                    feats,
                    dyn_keys,
                    dyn_values,
                    beta=self.config.beta,
                    top_k=self.config.top_k,
                )
                dynamic_conf, dynamic_pred = dynamic_probs.max(dim=1)
                use_dynamic = dynamic_conf >= self.config.use_threshold
                if self.config.agreement_only:
                    use_dynamic = use_dynamic & (dynamic_pred == out.argmax(dim=1))
                dynamic_blend = (1 - self.config.alpha_dynamic) * out + self.config.alpha_dynamic * dynamic_probs
                out = torch.where(use_dynamic[:, None], dynamic_blend, out)

            all_probs.append(out.cpu())

            out_conf, out_pred = out.max(dim=1)
            out_entropy = self._normalized_entropy(out)
            base_pred = base_probs.argmax(dim=1)
            source_pred = source_probs.argmax(dim=1)
            add_mask = (out_conf >= self.config.add_threshold) & (out_entropy <= self.config.max_entropy)
            if self.config.agreement_only:
                add_mask = add_mask & (out_pred == base_pred) & (out_pred == source_pred)
            if add_mask.any():
                self._append_dynamic(feats[add_mask].detach().cpu(), out_pred[add_mask].detach().cpu())

        dyn_counts = [sum(len(x) for x in self.dynamic_keys[c]) for c in range(self.config.num_classes)]
        print("dynamic cache counts:", dyn_counts)
        return torch.cat(all_probs, dim=0)
