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

    train_feats, train_labels, _ = load_feature_file(str(train_path))
    val_feats, val_labels, _ = load_feature_file(str(val_path))
    test_feats, test_labels, _ = load_feature_file(str(test_path))

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
        )
        torch.save(linear_model.state_dict(), linear_output)
        print("saved:", linear_output)
        models.append(("linear_probe", linear_model))

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
        )
        torch.save(osd_model.state_dict(), osd_output)
        print("saved:", osd_output)
        models.append(("osd_linear_probe", osd_model))

    rows = []
    splits = [
        ("train", train_feats, train_labels),
        ("val", val_feats, val_labels),
        ("test", test_feats, test_labels),
    ]
    for model_name, model in models:
        for split_name, feats, labels in splits:
            metrics = evaluate_probe(
                model,
                feats,
                labels,
                device=device,
                name=f"{model_name} | {split_name}",
                batch_size=args.eval_batch_size,
                show_report=False,
            )
            rows.append({"model": model_name, "split": split_name, **metrics})

    results = pd.DataFrame(rows)
    results.to_csv(results_output, index=False)
    print(results)
    print("saved:", results_output)


if __name__ == "__main__":
    main()
