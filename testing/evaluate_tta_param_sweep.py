from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from deepfake_tta.methods.bca import BCA, BCAConfig
from deepfake_tta.methods.freetta import FreeTTA, FreeTTAConfig
from deepfake_tta.methods.tip_adapter import TipAdapter, TipAdapterConfig
from deepfake_tta.modeling import evaluate_probe, load_feature_file, seed_everything
from testing.evaluate_tta_matrix import (
    apply_aligned_ids,
    build_aligned_balanced_ids,
    find_feature_files,
    infer_feature_meta,
    load_probe_model,
    parse_spec,
    read_thresholds,
    select_balanced_subset,
    select_shots_per_class,
    shuffle_test_features,
)


def device_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def parse_float_list(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return default
    return [float(value) for value in values]


def parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    return [int(value) for value in values]


def compact_grid(args: argparse.Namespace) -> list[dict]:
    rows = [{"method": "none", "param_id": "none"}]

    for shots, alpha, beta in product(args.tip_shots, args.tip_alpha, args.tip_beta):
        rows.append(
            {
                "method": "tip_adapter",
                "param_id": f"tip_s{shots}_a{alpha:g}_b{beta:g}",
                "shots_per_class": shots,
                "alpha": alpha,
                "beta": beta,
            }
        )

    for base_weight, momentum, prior_power in product(args.freetta_base_weight, args.freetta_momentum, args.freetta_prior_power):
        rows.append(
            {
                "method": "freetta",
                "param_id": f"free_bw{base_weight:g}_m{momentum:g}_p{prior_power:g}",
                "base_weight": base_weight,
                "momentum": momentum,
                "prior_power": prior_power,
            }
        )

    for temperature, base_weight, prior_momentum, prototype_momentum in product(
        args.bca_temperature,
        args.bca_base_weight,
        args.bca_prior_momentum,
        args.bca_prototype_momentum,
    ):
        rows.append(
            {
                "method": "bca",
                "param_id": (
                    f"bca_t{temperature:g}_bw{base_weight:g}_"
                    f"pm{prior_momentum:g}_xm{prototype_momentum:g}"
                ),
                "temperature": temperature,
                "base_weight": base_weight,
                "prior_momentum": prior_momentum,
                "prototype_momentum": prototype_momentum,
            }
        )
    return rows


def create_method(config: dict, args: argparse.Namespace, train_feats: torch.Tensor, train_labels: torch.Tensor):
    method_name = config["method"]
    if method_name == "tip_adapter":
        shots = int(config["shots_per_class"])
        fit_feats, fit_labels = select_shots_per_class(
            train_feats,
            train_labels,
            shots_per_class=shots,
            seed=args.seed + 701 + shots,
            name=f"tip_adapter cache {config['param_id']}",
        )
        method = TipAdapter(
            TipAdapterConfig(
                alpha=float(config["alpha"]),
                beta=float(config["beta"]),
                test_batch_size=args.test_batch_size,
                cache_batch_size=args.cache_batch_size,
            )
        )
    elif method_name == "freetta":
        fit_feats, fit_labels = train_feats, train_labels
        method = FreeTTA(
            FreeTTAConfig(
                batch_size=args.test_batch_size,
                momentum=float(config["momentum"]),
                prior_power=float(config["prior_power"]),
                base_weight=float(config["base_weight"]),
                min_var=args.freetta_min_var,
                warmup_batches=args.freetta_warmup_batches,
            )
        )
    elif method_name == "bca":
        fit_feats, fit_labels = train_feats, train_labels
        method = BCA(
            BCAConfig(
                batch_size=args.test_batch_size,
                temperature=float(config["temperature"]),
                prior_momentum=float(config["prior_momentum"]),
                prototype_momentum=float(config["prototype_momentum"]),
                base_weight=float(config["base_weight"]),
                confidence_threshold=args.bca_confidence_threshold,
            )
        )
    else:
        raise ValueError(f"Cannot create TTA method for config: {config}")

    method.fit(fit_feats, fit_labels)
    fit_counts = torch.bincount(fit_labels.long(), minlength=2).tolist()
    return method, int(len(fit_labels)), fit_counts


def evaluate_configs(
    *,
    args: argparse.Namespace,
    rows: list[dict],
    configs: list[dict],
    dataset_name: str,
    feature_path: Path,
    feats: torch.Tensor,
    labels: torch.Tensor,
    payload: dict,
    model_name: str,
    model_type: str,
    model_path: Path,
    model,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
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

    for config in configs:
        common = {
            **meta,
            "model": model_name,
            "model_type": model_type,
            "model_path": str(model_path),
            **config,
        }
        try:
            if config["method"] == "none":
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
                rows.append({**common, "method_fit_samples": None, "method_fit_counts": None, **metrics})
                continue

            method, fit_samples, fit_counts = create_method(config, args, train_feats, train_labels)
            metrics = method.evaluate(
                model,
                feats,
                labels,
                device=device,
                name=f"{dataset_name} | {feature_path.name} | {model_name} | {config['param_id']}",
                show_report=args.show_report,
            )
            rows.append({**common, "method_fit_samples": fit_samples, "method_fit_counts": fit_counts, **metrics})
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
    parser.add_argument("--results-output", default="/kaggle/working/tta_param_sweep_results.csv")
    parser.add_argument("--balanced-dataset", action="append", default=[])
    parser.add_argument("--balanced-aligned-dataset", action="append", default=[])
    parser.add_argument("--shuffle-all-tests", action="store_true")
    parser.add_argument("--shuffle-dataset", action="append", default=[])
    parser.add_argument("--balance-method-fit", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["none", "tip_adapter", "freetta", "bca"])
    parser.add_argument("--tip-shots", nargs="+")
    parser.add_argument("--tip-alpha", nargs="+")
    parser.add_argument("--tip-beta", nargs="+")
    parser.add_argument("--freetta-base-weight", nargs="+")
    parser.add_argument("--freetta-momentum", nargs="+")
    parser.add_argument("--freetta-prior-power", nargs="+")
    parser.add_argument("--freetta-min-var", type=float, default=1e-4)
    parser.add_argument("--freetta-warmup-batches", type=int, default=1)
    parser.add_argument("--bca-temperature", nargs="+")
    parser.add_argument("--bca-base-weight", nargs="+")
    parser.add_argument("--bca-prior-momentum", nargs="+")
    parser.add_argument("--bca-prototype-momentum", nargs="+")
    parser.add_argument("--bca-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--cache-batch-size", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-report", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    args.tip_shots = parse_int_list(args.tip_shots, [8, 16, 32])
    args.tip_alpha = parse_float_list(args.tip_alpha, [0.2, 0.5, 0.8])
    args.tip_beta = parse_float_list(args.tip_beta, [3.5, 5.5, 7.5])
    args.freetta_base_weight = parse_float_list(args.freetta_base_weight, [0.2, 0.4, 0.6])
    args.freetta_momentum = parse_float_list(args.freetta_momentum, [0.9, 0.95, 0.98])
    args.freetta_prior_power = parse_float_list(args.freetta_prior_power, [1.0])
    args.bca_temperature = parse_float_list(args.bca_temperature, [0.03, 0.07, 0.12])
    args.bca_base_weight = parse_float_list(args.bca_base_weight, [0.3, 0.5, 0.7])
    args.bca_prior_momentum = parse_float_list(args.bca_prior_momentum, [0.95])
    args.bca_prototype_momentum = parse_float_list(args.bca_prototype_momentum, [0.98])

    device = device_arg(args.device)
    seed_everything(args.seed)

    train_feats, train_labels, _ = load_feature_file(args.train_features)
    if args.balance_method_fit:
        train_feats, train_labels = select_balanced_subset(
            train_feats,
            train_labels,
            seed=args.seed + 101,
            name="method fit",
        )

    all_configs = compact_grid(args)
    methods = set(args.methods)
    configs = [config for config in all_configs if config["method"] in methods]
    print("sweep configs:", len(configs), flush=True)
    for config in configs:
        print(" -", config, flush=True)

    dataset_specs = [parse_spec(value, "dataset") for value in args.dataset]
    model_specs = [parse_spec(value, "model") for value in args.model]
    thresholds = read_thresholds(args.thresholds_csv)
    balanced_datasets = set(args.balanced_dataset)
    balanced_aligned_datasets = set(args.balanced_aligned_dataset)
    shuffle_datasets = set(args.shuffle_dataset)

    rows = []
    loaded_models = {}
    for dataset_name, dataset_path in dataset_specs:
        feature_paths = find_feature_files(dataset_path)
        print(f"\nDataset {dataset_name}:")
        for path in feature_paths:
            print(" -", path)

        aligned_ids = None
        if dataset_name in balanced_aligned_datasets:
            aligned_ids = build_aligned_balanced_ids(feature_paths, args.block_size)

        for feature_path in feature_paths:
            feats, labels, payload = load_feature_file(str(feature_path))
            if aligned_ids is not None:
                feats, labels = apply_aligned_ids(feats, labels, payload, aligned_ids)
            elif dataset_name in balanced_datasets:
                feats, labels = select_balanced_subset(
                    feats,
                    labels,
                    seed=args.seed + sum(ord(ch) for ch in dataset_name + feature_path.name),
                    name=f"{dataset_name}/{feature_path.name}",
                )
            elif args.shuffle_all_tests or dataset_name in shuffle_datasets:
                feats, labels = shuffle_test_features(
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
                evaluate_configs(
                    args=args,
                    rows=rows,
                    configs=configs,
                    dataset_name=dataset_name,
                    feature_path=feature_path,
                    feats=feats,
                    labels=labels,
                    payload=payload,
                    model_name=model_name,
                    model_type=model_type,
                    model_path=model_path,
                    model=model,
                    train_feats=train_feats,
                    train_labels=train_labels,
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
