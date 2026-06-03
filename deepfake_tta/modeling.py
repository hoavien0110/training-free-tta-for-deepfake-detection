from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


class LinearProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(1)


class OSDLinearProbe(nn.Module):
    """Feature-space Orthogonal Subspace Decomposition linear probe.

    This is a precomputed-feature adaptation of OSD/Effort: estimate a frozen
    principal semantic subspace with SVD/PCA, then train the detector on the
    orthogonal residual subspace where forgery-specific cues should live.
    """

    def __init__(self, dim: int, center: torch.Tensor, basis: torch.Tensor):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.register_buffer("center", center.float())
        self.register_buffer("basis", basis.float())

    def residualize(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.center.to(x.device)
        basis = self.basis.to(x.device)
        semantic = (centered @ basis) @ basis.T
        return centered - semantic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.residualize(x)).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_file(path: str, verbose: bool = True) -> tuple[torch.Tensor, torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu")
    features = payload["features"].float()
    labels = payload["labels"].long()
    if verbose:
        print(path)
        print("features:", tuple(features.shape))
        print("labels:", tuple(labels.shape))
        print("label counts [REAL, FAKE]:", torch.bincount(labels, minlength=2).tolist())
        print("dataset:", payload.get("dataset_name", payload.get("target_dataset", "unknown")))
        print("transform:", payload.get("transform_name", "none"))
        print("clip:", payload.get("clip_model", "unknown"))
        print()
    return features, labels, payload


def train_linear_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    device: str,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> LinearProbe:
    model = LinearProbe(train_feats.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    x_train = train_feats.to(device)
    y_train = train_labels.float().to(device)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_train), device=device)
        total_loss = 0.0
        total_correct = 0
        total = 0

        for start in range(0, len(x_train), batch_size):
            idx = permutation[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_batch_size = len(xb)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_loss += loss.item() * current_batch_size
            total_correct += (preds == yb.long()).sum().item()
            total += current_batch_size

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={total_loss / total:.4f} | train_acc={total_correct / total:.4f}"
        )

    return model


def build_osd_subspace(
    train_feats: torch.Tensor,
    rank: int = 128,
    max_samples: int = 100_000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    feats = train_feats.float()
    if len(feats) > max_samples:
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(len(feats), generator=generator)[:max_samples]
        feats = feats[indices]

    center = feats.mean(dim=0)
    centered = feats - center
    rank = min(rank, centered.shape[1], centered.shape[0] - 1)
    if rank <= 0:
        raise ValueError(f"Invalid OSD rank={rank} for centered shape={tuple(centered.shape)}")

    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:rank].T.contiguous()
    return center.cpu(), basis.cpu()


def train_osd_linear_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    device: str,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    osd_rank: int = 128,
    osd_max_samples: int = 100_000,
    seed: int = 42,
) -> OSDLinearProbe:
    center, basis = build_osd_subspace(
        train_feats,
        rank=osd_rank,
        max_samples=osd_max_samples,
        seed=seed,
    )
    model = OSDLinearProbe(train_feats.shape[1], center=center, basis=basis).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    x_train = train_feats.to(device)
    y_train = train_labels.float().to(device)

    print("OSD rank:", basis.shape[1])
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_train), device=device)
        total_loss = 0.0
        total_correct = 0
        total = 0

        for start in range(0, len(x_train), batch_size):
            idx = permutation[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_batch_size = len(xb)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_loss += loss.item() * current_batch_size
            total_correct += (preds == yb.long()).sum().item()
            total += current_batch_size

        print(
            f"OSD Epoch {epoch + 1}/{epochs} | "
            f"loss={total_loss / total:.4f} | train_acc={total_correct / total:.4f}"
        )

    return model


def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2), float(thresholds[eer_idx])


def evaluate_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    name: str = "test",
    show_report: bool = True,
) -> dict[str, float]:
    eer, eer_threshold = calculate_eer(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "auc": roc_auc_score(y_true, y_score),
        "ap": average_precision_score(y_true, y_score),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    if show_report:
        print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))
    return metrics


@torch.no_grad()
def evaluate_probe(
    model: LinearProbe,
    feats: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    name: str = "test",
    batch_size: int = 4096,
    show_report: bool = True,
) -> dict[str, float]:
    model.eval()
    scores = []
    for start in range(0, len(feats), batch_size):
        xb = feats[start : start + batch_size].to(device)
        scores.append(torch.sigmoid(model(xb)).cpu())
    y_score = torch.cat(scores).numpy()
    y_true = labels.numpy()
    y_pred = (y_score >= 0.5).astype(int)
    return evaluate_scores(y_true, y_score, y_pred, name=name, show_report=show_report)


def build_tip_cache(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    from deepfake_tta.methods.tip_adapter import build_tip_cache as _build_tip_cache

    return _build_tip_cache(train_feats, train_labels, num_classes=num_classes)


@torch.no_grad()
def tip_adapter_predict(
    model: LinearProbe,
    test_feats: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    device: str,
    alpha: float = 0.5,
    beta: float = 5.5,
    test_batch_size: int = 512,
    cache_batch_size: int = 8192,
) -> torch.Tensor:
    from deepfake_tta.methods.tip_adapter import TipAdapter, TipAdapterConfig

    adapter = TipAdapter(
        TipAdapterConfig(
            alpha=alpha,
            beta=beta,
            test_batch_size=test_batch_size,
            cache_batch_size=cache_batch_size,
            num_classes=cache_values.shape[1],
        )
    )
    adapter.cache_keys = cache_keys
    adapter.cache_values = cache_values
    return adapter.predict_proba(model, test_feats, device)


@torch.no_grad()
def evaluate_tip_adapter(
    model: LinearProbe,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    device: str,
    name: str = "test",
    alpha: float = 0.5,
    beta: float = 5.5,
    test_batch_size: int = 512,
    cache_batch_size: int = 8192,
    show_report: bool = True,
) -> dict[str, float]:
    from deepfake_tta.methods.tip_adapter import TipAdapter, TipAdapterConfig

    adapter = TipAdapter(
        TipAdapterConfig(
            alpha=alpha,
            beta=beta,
            test_batch_size=test_batch_size,
            cache_batch_size=cache_batch_size,
            num_classes=cache_values.shape[1],
        )
    )
    adapter.cache_keys = cache_keys
    adapter.cache_values = cache_values
    print(f"alpha: {alpha} | beta: {beta}")
    return adapter.evaluate(model, test_feats, test_labels, device, name=name, show_report=show_report)
