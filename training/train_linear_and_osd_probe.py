from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from deepfake_tta.modeling import (
    evaluate_probe,
    load_feature_file,
    seed_everything,
    train_linear_probe,
    train_osd_linear_probe,
)


def select_balanced_subset(
    feats: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.long()
    counts = torch.bincount(labels, minlength=2)
    keep_per_class = int(counts.min().item())
    if keep_per_class <= 0:
        raise ValueError(f"Cannot balance {split_name} with label counts: {counts.tolist()}")

    generator = torch.Generator()
    generator.manual_seed(seed)
    selected = []
    for cls_idx in range(len(counts)):
        cls_indices = torch.where(labels == cls_idx)[0]
        perm = torch.randperm(len(cls_indices), generator=generator)
        selected.append(cls_indices[perm[:keep_per_class]])

    indices = torch.cat(selected)
    indices = indices[torch.randperm(len(indices), generator=generator)]
    balanced_feats = feats[indices].contiguous()
    balanced_labels = labels[indices].contiguous()
    print(f"balanced {split_name}:")
    print("original label counts [REAL, FAKE]:", counts.tolist())
    print("balanced label counts [REAL, FAKE]:", torch.bincount(balanced_labels.long(), minlength=2).tolist())
    return balanced_feats, balanced_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="/kaggle/working/ffpp_split_features")
    parser.add_argument("--train-features")
    parser.add_argument("--val-features")
    parser.add_argument("--test-features")
    parser.add_argument("--output-dir", default="/kaggle/working/ffpp_split_models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--osd-rank", type=int, default=128)
    parser.add_argument("--osd-max-samples", type=int, default=100_000)
    parser.add_argument(
        "--balance-splits",
        action="store_true",
        help="Undersample train/val/test to equal REAL/FAKE counts before training and evaluation.",
    )
    parser.add_argument(
        "--select-by-val-f1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the validation split to select the best epoch and decision threshold by macro F1.",
    )
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--skip-osd", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    seed_everything(args.seed)

    feature_dir = Path(args.feature_dir)
    train_path = Path(args.train_features) if args.train_features else feature_dir / "ffpp_train_features.pt"
    val_path = Path(args.val_features) if args.val_features else feature_dir / "ffpp_val_features.pt"
    test_path = Path(args.test_features) if args.test_features else feature_dir / "ffpp_test_features.pt"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    linear_output = output_dir / "ffpp_linear_probe_split.pt"
    osd_output = output_dir / "ffpp_osd_linear_probe_split.pt"
    results_output = output_dir / "ffpp_split_train_results.csv"
    threshold_output = output_dir / "ffpp_probe_thresholds.csv"

    train_feats, train_labels, _ = load_feature_file(str(train_path))
    val_feats, val_labels, _ = load_feature_file(str(val_path))
    test_feats, test_labels, _ = load_feature_file(str(test_path))

    if args.balance_splits:
        train_feats, train_labels = select_balanced_subset(train_feats, train_labels, seed=args.seed + 11, split_name="train")
        val_feats, val_labels = select_balanced_subset(val_feats, val_labels, seed=args.seed + 17, split_name="val")
        test_feats, test_labels = select_balanced_subset(test_feats, test_labels, seed=args.seed + 23, split_name="test")

    models = []
    if not args.skip_linear:
        linear_model = train_linear_probe(
            train_feats,
            train_labels,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_feats=val_feats if args.select_by_val_f1 else None,
            val_labels=val_labels if args.select_by_val_f1 else None,
            eval_batch_size=args.eval_batch_size,
        )
        torch.save(linear_model.state_dict(), linear_output)
        print("saved:", linear_output)
        models.append(("linear_probe", linear_model, linear_output))

    if not args.skip_osd:
        osd_model = train_osd_linear_probe(
            train_feats,
            train_labels,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            osd_rank=args.osd_rank,
            osd_max_samples=args.osd_max_samples,
            seed=args.seed,
            val_feats=val_feats if args.select_by_val_f1 else None,
            val_labels=val_labels if args.select_by_val_f1 else None,
            eval_batch_size=args.eval_batch_size,
        )
        torch.save(osd_model.state_dict(), osd_output)
        print("saved:", osd_output)
        models.append(("osd_linear_probe", osd_model, osd_output))

    rows = []
    threshold_rows = []
    splits = [
        ("train", train_feats, train_labels),
        ("val", val_feats, val_labels),
        ("test", test_feats, test_labels),
    ]
    for model_name, model, model_path in models:
        threshold = float(getattr(model, "best_threshold_", 0.5) or 0.5)
        val_f1 = getattr(model, "best_val_f1_", None)
        threshold_rows.append(
            {
                "model": model_name,
                "model_path": str(model_path),
                "threshold": threshold,
                "best_val_f1": val_f1,
                "selected_by_val_f1": bool(args.select_by_val_f1),
            }
        )
        for split_name, feats, labels in splits:
            metrics = evaluate_probe(
                model,
                feats,
                labels,
                device=device,
                name=f"{model_name} | {split_name}",
                batch_size=args.eval_batch_size,
                show_report=False,
                threshold=threshold,
            )
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "balanced_splits": bool(args.balance_splits),
                    "selected_by_val_f1": bool(args.select_by_val_f1),
                    "best_val_f1": val_f1,
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(results_output, index=False)
    print(results)
    print("saved:", results_output)

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(threshold_output, index=False)
    print(thresholds)
    print("saved:", threshold_output)


if __name__ == "__main__":
    main()
