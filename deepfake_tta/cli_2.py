from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch

from deepfake_tta.methods import AVAILABLE_TTA_METHODS, create_tta_method
from deepfake_tta.modeling import (
    LinearProbe,
    evaluate_probe,
    load_feature_file,
    seed_everything,
    train_linear_probe,
)


DEFAULT_CORRUPTIONS = ("color_contrast", "color_saturation", "gaussian_blur", "resize")


def device_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def load_or_train_linear_probe(
    args: argparse.Namespace,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    device: str,
) -> LinearProbe:
    if args.load_model:
        model = LinearProbe(train_feats.shape[1]).to(device)
        state = torch.load(args.load_model, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print("loaded model from:", args.load_model)
        return model

    model = train_linear_probe(
        train_feats,
        train_labels,
        device=device,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.model_output:
        Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.model_output)
        print("saved model to:", args.model_output)
    return model


def select_balanced_train_subset(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = train_labels.long()
    counts = torch.bincount(labels, minlength=2)
    keep_per_class = int(counts.min().item())
    if keep_per_class <= 0:
        raise ValueError(f"Cannot balance labels with counts: {counts.tolist()}")

    generator = torch.Generator()
    generator.manual_seed(seed)
    selected = []
    for cls_idx in range(len(counts)):
        cls_indices = torch.where(labels == cls_idx)[0]
        perm = torch.randperm(len(cls_indices), generator=generator)
        selected.append(cls_indices[perm[:keep_per_class]])

    selected_indices = torch.cat(selected)
    selected_indices = selected_indices[torch.randperm(len(selected_indices), generator=generator)]
    balanced_feats = train_feats[selected_indices].contiguous()
    balanced_labels = train_labels[selected_indices].contiguous()

    print("balanced train subset:")
    print("original label counts [REAL, FAKE]:", counts.tolist())
    print("balanced label counts [REAL, FAKE]:", torch.bincount(balanced_labels.long(), minlength=2).tolist())
    return balanced_feats, balanced_labels


def build_tta_methods(args: argparse.Namespace, train_feats: torch.Tensor, train_labels: torch.Tensor):
    methods = []
    for method_name in args.tta_methods:
        method = create_tta_method(
            method_name,
            alpha=args.alpha,
            beta=args.beta,
            test_batch_size=args.test_batch_size,
            cache_batch_size=args.cache_batch_size,
            method_cache_dir=args.method_cache_dir,
        )
        method.fit(train_feats, train_labels)
        methods.append(method)
    return methods


def find_feature_file(feature_root: Path, level: int, corruption: str, suffix: str) -> Path:
    exact_name = f"celebdfv1_level{level}_{corruption}_{suffix}_features.pt"
    exact_matches = sorted(feature_root.glob(f"**/level_{level}/{exact_name}"))
    if exact_matches:
        return exact_matches[0]

    loose_matches = sorted(feature_root.glob(f"**/level_{level}/celebdfv1_level{level}_{corruption}*features.pt"))
    if loose_matches:
        return loose_matches[0]

    raise FileNotFoundError(
        f"Cannot find feature file for level={level}, corruption={corruption}, suffix={suffix} under {feature_root}"
    )


def infer_meta_from_path(path: Path) -> dict[str, str | int]:
    match = re.search(r"celebdfv1_level(\d+)_(.+?)_features\.pt$", path.name)
    if not match:
        return {"level": "", "corruption": "", "order_setting": ""}

    level = int(match.group(1))
    stem = match.group(2)
    suffixes = ("balanced_16_16", "sequential", "shuffle")
    for suffix in suffixes:
        marker = f"_{suffix}"
        if stem.endswith(marker):
            return {
                "level": level,
                "corruption": stem[: -len(marker)],
                "order_setting": suffix,
            }
    return {"level": level, "corruption": stem, "order_setting": ""}


def evaluate_feature_file(
    args: argparse.Namespace,
    feature_path: Path,
    model: LinearProbe,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    device: str,
) -> list[dict]:
    feats, labels, payload = load_feature_file(str(feature_path))
    meta = infer_meta_from_path(feature_path)
    level = payload.get("transform_level", meta["level"])
    corruption = payload.get("transform_name", meta["corruption"])
    order_setting = payload.get("order_setting", meta["order_setting"])

    common = {
        "level": level,
        "corruption": corruption,
        "order_setting": order_setting,
        "feature_path": str(feature_path),
    }

    rows = []
    probe_metrics = evaluate_probe(
        model,
        feats,
        labels,
        device,
        name=f"CelebDFv1 level {level} {corruption} {order_setting} | linear_probe",
        batch_size=args.eval_batch_size,
        show_report=args.show_report,
    )
    rows.append({**common, "method": "linear_probe", **probe_metrics})

    for method in build_tta_methods(args, train_feats, train_labels):
        try:
            method_metrics = method.evaluate(
                model,
                feats,
                labels,
                device,
                name=f"CelebDFv1 level {level} {corruption} {order_setting} | {method.name}",
                show_report=args.show_report,
            )
            rows.append({**common, "method": method.name, **method_metrics})
        except Exception as exc:
            if not args.continue_on_error:
                raise
            rows.append({**common, "method": method.name, "error": repr(exc)})
    return rows


def cmd_eval_template(args: argparse.Namespace) -> None:
    device = device_arg(args.device)
    seed_everything(args.seed)

    train_feats, train_labels, _ = load_feature_file(args.train_features)
    model_train_feats = train_feats
    model_train_labels = train_labels
    if args.balance_train_labels and not args.load_model:
        model_train_feats, model_train_labels = select_balanced_train_subset(train_feats, train_labels, args.seed)
    elif args.balance_train_labels and args.load_model:
        print("--balance-train-labels ignored because --load-model was provided.")

    method_train_feats = train_feats
    method_train_labels = train_labels
    if args.balance_method_fit:
        method_train_feats, method_train_labels = select_balanced_train_subset(train_feats, train_labels, args.seed)

    model = load_or_train_linear_probe(args, model_train_feats, model_train_labels, device)

    feature_root = Path(args.feature_root)
    feature_paths = [
        find_feature_file(feature_root, level, corruption, args.filename_suffix)
        for level in args.levels
        for corruption in args.corruptions
    ]

    print("matched feature files:")
    for path in feature_paths:
        print(" -", path)

    rows = []
    if args.include_train_eval:
        train_metrics = evaluate_probe(
            model,
            train_feats,
            train_labels,
            device,
            name="Train FF++ | linear_probe",
            batch_size=args.eval_batch_size,
            show_report=False,
        )
        rows.append(
            {
                "level": "train",
                "corruption": "train_ffpp",
                "order_setting": "train",
                "feature_path": args.train_features,
                "method": "linear_probe",
                **train_metrics,
            }
        )

    for feature_path in feature_paths:
        rows.extend(evaluate_feature_file(args, feature_path, model, method_train_feats, method_train_labels, device))

    results = pd.DataFrame(rows)
    Path(args.results_output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_output, index=False)
    print("saved results to:", args.results_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepfake-tta-cli-2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser(
        "eval-template",
        help="Evaluate nested CelebDFv1 feature folders like level_1/celebdfv1_level1_*_balanced_16_16_features.pt.",
    )
    p.add_argument("--feature-root", required=True)
    p.add_argument("--train-features", required=True)
    p.add_argument("--load-model")
    p.add_argument("--model-output", default="/kaggle/working/ufd_linear_probe_cli2.pt")
    p.add_argument("--results-output", required=True)
    p.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--corruptions", nargs="+", default=list(DEFAULT_CORRUPTIONS))
    p.add_argument("--filename-suffix", default="balanced_16_16")
    p.add_argument("--tta-methods", nargs="+", choices=AVAILABLE_TTA_METHODS, default=["tip_adapter"])
    p.add_argument("--method-cache-dir")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=5.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--test-batch-size", type=int, default=512)
    p.add_argument("--cache-batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--show-report", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--include-train-eval", action="store_true")
    p.add_argument(
        "--balance-train-labels",
        action="store_true",
        help="Before training a new linear probe, undersample train features to equal REAL/FAKE counts.",
    )
    p.add_argument(
        "--balance-method-fit",
        action="store_true",
        help="Also fit TTA methods on an equal-label train subset instead of the full imbalanced train set.",
    )
    p.set_defaults(func=cmd_eval_template)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
