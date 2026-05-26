from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_model_probs(model, feats: torch.Tensor) -> torch.Tensor:
    p_fake = torch.sigmoid(model(feats))
    return torch.stack([1 - p_fake, p_fake], dim=1).clamp_min(1e-8)


def cache_probs(
    feats: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    beta: float = 5.5,
    chunk_size: int = 8192,
) -> torch.Tensor:
    cache_logits = torch.zeros((len(feats), cache_values.shape[1]), device=feats.device)
    cache_norm = torch.zeros((len(feats), 1), device=feats.device)
    for start in range(0, len(cache_keys), chunk_size):
        keys = cache_keys[start : start + chunk_size]
        values = cache_values[start : start + chunk_size]
        affinity = torch.exp(-beta * (1 - feats @ keys.T))
        cache_logits += affinity @ values
        cache_norm += affinity.sum(dim=1, keepdim=True)
    return cache_logits / (cache_norm + 1e-8)


def topk_cache_probs(
    feats: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    beta: float = 5.5,
    top_k: int = 64,
) -> torch.Tensor:
    top_k = min(top_k, len(cache_keys))
    similarity = feats @ cache_keys.T
    top_values, top_idx = similarity.topk(top_k, dim=1)
    affinity = torch.exp(-beta * (1 - top_values))
    values = cache_values[top_idx]
    logits = (affinity[:, :, None] * values).sum(dim=1)
    return logits / (affinity.sum(dim=1, keepdim=True) + 1e-8)


def one_hot(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).float()


def blend_probs(*weighted_probs: tuple[float, torch.Tensor]) -> torch.Tensor:
    total = None
    total_weight = 0.0
    for weight, probs in weighted_probs:
        total = probs * weight if total is None else total + probs * weight
        total_weight += weight
    if total is None or total_weight <= 0:
        raise ValueError("At least one positive-weight probability tensor is required.")
    return (total / total_weight).clamp_min(1e-8)


def class_means_from_labels(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    feats = F.normalize(feats.float(), dim=-1)
    means = []
    global_mean = feats.mean(dim=0)
    for cls_idx in range(num_classes):
        mask = labels.long() == cls_idx
        if mask.any():
            means.append(feats[mask].mean(dim=0))
        else:
            means.append(global_mean)
    return F.normalize(torch.stack(means, dim=0), dim=-1)


def class_diag_vars_from_labels(
    feats: torch.Tensor,
    labels: torch.Tensor,
    means: torch.Tensor,
    num_classes: int = 2,
    min_var: float = 1e-4,
) -> torch.Tensor:
    feats = F.normalize(feats.float(), dim=-1)
    vars_ = []
    global_var = feats.var(dim=0, unbiased=False).clamp_min(min_var)
    for cls_idx in range(num_classes):
        mask = labels.long() == cls_idx
        if mask.sum() > 1:
            vars_.append((feats[mask] - means[cls_idx]).pow(2).mean(dim=0).clamp_min(min_var))
        else:
            vars_.append(global_var)
    return torch.stack(vars_, dim=0)


def priors_from_labels(labels: torch.Tensor, num_classes: int = 2, smoothing: float = 1.0) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    priors = counts + smoothing
    return priors / priors.sum()


def weighted_mean(feats: torch.Tensor, probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    weights = probs.sum(dim=0).clamp_min(eps)
    return probs.T @ feats / weights[:, None]


def weighted_diag_var(
    feats: torch.Tensor,
    probs: torch.Tensor,
    means: torch.Tensor,
    min_var: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    vars_ = []
    for cls_idx in range(probs.shape[1]):
        weights = probs[:, cls_idx]
        denom = weights.sum().clamp_min(eps)
        var = (weights[:, None] * (feats - means[cls_idx]).pow(2)).sum(dim=0) / denom
        vars_.append(var.clamp_min(min_var))
    return torch.stack(vars_, dim=0)


def diag_gaussian_log_prob(
    feats: torch.Tensor,
    means: torch.Tensor,
    vars_: torch.Tensor,
) -> torch.Tensor:
    feats = feats[:, None, :]
    means = means[None, :, :]
    vars_ = vars_[None, :, :]
    return -0.5 * (((feats - means).pow(2) / vars_) + vars_.log()).sum(dim=-1)
