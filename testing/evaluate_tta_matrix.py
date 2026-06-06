from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from deepfake_tta.methods import AVAILABLE_TTA_METHODS, create_tta_method
from deepfake_tta.modeling import (
    LinearProbe,
    OSDLinearProbe,
    evaluate_probe,
    load_feature_file,
    seed_everything,
)
from testing.evaluate_ffpp_corruption_embeddings import (
    apply_aligned_ids,
    build_aligned_balanced_ids,
)


def device_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def parse_spec(value: str, kind: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"{kind} spec must be NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"{kind} spec name is empty: {value}")
    return name, Path(path)


def find_feature_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    preferred_patterns = (
        "**/ffpp_test_features.pt",
        "**/ffpp_test_level*_features.pt",
        "**/celebdfv1_level*_features.pt",
        "**/*features.pt",
    )
    seen = set()
    files = []
    for pattern in preferred_patterns:
        for feature_path in sorted(path.glob(pattern)):
            if feature_path not in seen:
                files.append(feature_path)
                seen.add(feature_path)
        if files:
            break
    if not files:
        raise FileNotFoundError(f"No feature files found under {path}")
    return files


def infer_feature_meta(dataset_name: str, path: Path, payload: dict) -> dict[str, object]:
    level = payload.get("transform_level")
    corruption = payload.get("transform_name", "none")
    split = payload.get("split_name", "test")

    patterns = (
        r"ffpp_test_level(\d+)_(.+?)_features\.pt$",
        r"celebdfv1_level(\d+)_(.+?)_features\.pt$",
    )
    for pattern in patterns:
        match = re.search(pattern, path.name)
        if match:
            level = int(match.group(1)) if level is None else level
            corruption = match.group(2) if not corruption or corruption == "none" else corruption
            break

    if path.name == "ffpp_test_features.pt":
        split = "test"
        corruption = "none"

    return {
        "dataset": dataset_name,
        "payload_dataset": payload.get("dataset_name", ""),
        "split": split,
        "level": level,
        "corruption": corruption,
        "feature_path": str(path),
    }


def select_balanced_subset(
    feats: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.long()
    counts = torch.bincount(labels, minlength=2)
    keep_per_class = int(counts.min().item())
    if keep_per_class <= 0:
        raise ValueError(f"Cannot balance {name} with label counts: {counts.tolist()}")

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
    print(f"balanced {name}:")
    print("original label counts [REAL, FAKE]:", counts.tolist())
    print("balanced label counts [REAL, FAKE]:", torch.bincount(balanced_labels, minlength=2).tolist())
    return balanced_feats, balanced_labels


def load_probe_model(model_path: Path, dim: int, device: str):
    state = torch.load(model_path, map_location="cpu")
    if "center" in state and "basis" in state:
        model = OSDLinearProbe(dim, center=state["center"], basis=state["basis"])
        inferred_type = "osd_linear_probe"
    else:
        model = LinearProbe(dim)
        inferred_type = "linear_probe"
    model.load_state_dict(state)
    model.to(device).eval()
    return inferred_type, model


def read_thresholds(paths: list[str] | None) -> dict[str, float]:
    thresholds = {}
    for raw_path in paths or []:
        prefix = None
        if "=" in raw_path:
            prefix, raw_path = raw_path.split("=", 1)
            prefix = prefix.strip()
        path = Path(raw_path)
        if not path.exists():
            print("threshold CSV not found, using default 0.5:", path)
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            threshold = float(row.get("threshold", 0.5))
            model = row.get("model")
            if isinstance(model, str) and model:
                thresholds[model] = threshold
                if prefix:
                    thresholds[f"{prefix}:{model}"] = threshold
            model_path = row.get("model_path")
            if isinstance(model_path, str) and model_path:
                thresholds[Path(model_path).name] = threshold
                if prefix:
                    thresholds[f"{prefix}:{Path(model_path).name}"] = threshold
    return thresholds


def build_tta_method(method_name: str, args: argparse.Namespace, train_feats: torch.Tensor, train_labels: torch.Tensor):
    method = create_tta_method(
        method_name,
        alpha=args.alpha,
        beta=args.beta,
        test_batch_size=args.test_batch_size,
        cache_batch_size=args.cache_batch_size,
        method_cache_dir=args.method_cache_dir,
    )
    method.fit(train_feats, train_labels)
    return method


def evaluate_feature_with_model(
    *,
    args: argparse.Namespace,
    rows: list[dict],
    dataset_name: str,
    feature_path: Path,
    feats: torch.Tensor,
    labels: torch.Tensor,
    payload: dict,
    model_name: str,
    model_type: str,
    model_path: Path,
    model,
    tta_train_feats: torch.Tensor,
    tta_train_labels: torch.Tensor,
    thresholds: dict[str, float],
    device: str,
) -> None:
    meta = infer_feature_meta(dataset_name, feature_path, payload)
    threshold = thresholds.get(
        model_name,
        thresholds.get(
            f"{model_name}:{model_type}",
            thresholds.get(
                f"{model_name}:{model_path.name}",
                thresholds.get(model_path.name, thresholds.get(model_type, 0.5)),
            ),
        ),
    )

    for method_name in args.tta_methods:
        common = {
            **meta,
            "model": model_name,
            "model_type": model_type,
            "model_path": str(model_path),
            "method": method_name,
        }
        try:
            if method_name == "none":
                metrics = evaluate_probe(
                    model,
                    feats,
                    labels,
                    device=device,
                    name=f"{dataset_name} | {feature_path.name} | {model_name} | none",
                    batch_size=args.eval_batch_size,
                    show_report=args.show_report,
                    threshold=threshold,
                )
            else:
                method = build_tta_method(method_name, args, tta_train_feats, tta_train_labels)
                metrics = method.evaluate(
                    model,
                    feats,
                    labels,
                    device=device,
                    name=f"{dataset_name} | {feature_path.name} | {model_name} | {method.name}",
                    show_report=args.show_report,
                )
            rows.append({**common, **metrics})
        except Exception as exc:
            if not args.continue_on_error:
                raise
            rows.append({**common, "error": repr(exc)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--dataset", action="append", required=True, help="Repeatable NAME=PATH feature file/folder.")
    parser.add_argument("--model", action="append", required=True, help="Repeatable NAME=PATH model .pt.")
    parser.add_argument("--thresholds-csv", action="append")
    parser.add_argument("--results-output", default="/kaggle/working/tta_testing_matrix_results.csv")
    parser.add_argument("--balanced-dataset", action="append", default=[])
    parser.add_argument("--balanced-aligned-dataset", action="append", default=[])
    parser.add_argument("--balance-method-fit", action="store_true")
    parser.add_argument("--tta-methods", nargs="+", default=["none", *AVAILABLE_TTA_METHODS])
    parser.add_argument("--method-cache-dir")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=5.5)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--cache-batch-size", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-report", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    device = device_arg(args.device)
    seed_everything(args.seed)

    train_feats, train_labels, _ = load_feature_file(args.train_features)
    tta_train_feats, tta_train_labels = train_feats, train_labels
    if args.balance_method_fit:
        tta_train_feats, tta_train_labels = select_balanced_subset(
            train_feats,
            train_labels,
            seed=args.seed + 101,
            name="method fit",
        )

    dataset_specs = [parse_spec(value, "dataset") for value in args.dataset]
    model_specs = [parse_spec(value, "model") for value in args.model]
    thresholds = read_thresholds(args.thresholds_csv)

    rows = []
    loaded_models = {}
    for dataset_name, dataset_path in dataset_specs:
        feature_paths = find_feature_files(dataset_path)
        print(f"\nDataset {dataset_name}:")
        for path in feature_paths:
            print(" -", path)

        aligned_ids = None
        if dataset_name in set(args.balanced_aligned_dataset):
            aligned_ids = build_aligned_balanced_ids(feature_paths, args.block_size)

        for feature_path in feature_paths:
            feats, labels, payload = load_feature_file(str(feature_path))
            if aligned_ids is not None:
                feats, labels = apply_aligned_ids(feats, labels, payload, aligned_ids)
            elif dataset_name in set(args.balanced_dataset):
                feats, labels = select_balanced_subset(
                    feats,
                    labels,
                    seed=args.seed + sum(ord(ch) for ch in dataset_name + feature_path.name),
                    name=f"{dataset_name}/{feature_path.name}",
                )

            dim = feats.shape[1]
            for model_name, model_path in model_specs:
                if model_name not in loaded_models:
                    model_type, model = load_probe_model(model_path, dim, device)
                    loaded_models[model_name] = (model_type, model)
                    print("loaded model:", model_name, model_type, model_path)
                model_type, model = loaded_models[model_name]
                evaluate_feature_with_model(
                    args=args,
                    rows=rows,
                    dataset_name=dataset_name,
                    feature_path=feature_path,
                    feats=feats,
                    labels=labels,
                    payload=payload,
                    model_name=model_name,
                    model_type=model_type,
                    model_path=model_path,
                    model=model,
                    tta_train_feats=tta_train_feats,
                    tta_train_labels=tta_train_labels,
                    thresholds=thresholds,
                    device=device,
                )

    results = pd.DataFrame(rows)
    Path(args.results_output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_output, index=False)
    print(results)
    print("saved:", args.results_output)


if __name__ == "__main__":
    main()
