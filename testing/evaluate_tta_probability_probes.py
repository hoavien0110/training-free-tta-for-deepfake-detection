from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from deepfake_tta.modeling import load_feature_file, predict_probe_scores, seed_everything
from testing.evaluate_tta_matrix import (
    apply_aligned_ids,
    build_aligned_balanced_ids,
    build_sample_index,
    find_feature_files,
    infer_feature_meta,
    load_probe_model,
    parse_spec,
    read_thresholds,
)
from testing.evaluate_tta_param_sweep import create_method


DEFAULT_METHOD_CONFIGS: dict[str, dict[str, Any]] = {
    "none": {"method": "none", "param_id": "none"},
    "tip_adapter": {
        "method": "tip_adapter",
        "param_id": "tip_s16_a0.5_b5.5",
        "shots_per_class": 16,
        "alpha": 0.5,
        "beta": 5.5,
    },
    "freetta": {
        "method": "freetta",
        "param_id": "free_bw0.6_m0.9_p1",
        "base_weight": 0.6,
        "momentum": 0.9,
        "prior_power": 1.0,
    },
    "bca": {
        "method": "bca",
        "param_id": "bca_t0.03_bw0.7_pm0.95_xm0.98",
        "temperature": 0.03,
        "base_weight": 0.7,
        "prior_momentum": 0.95,
        "prototype_momentum": 0.98,
    },
    "dota": {
        "method": "dota",
        "param_id": "dota_bw0.55_m0.95_ct0.9",
        "base_weight": 0.55,
        "momentum": 0.95,
        "confidence_threshold": 0.9,
    },
    "tda": {
        "method": "tda",
        "param_id": "tda_pa0.4_pb5.5_pe0.4_na0",
        "positive_alpha": 0.4,
        "positive_beta": 5.5,
        "positive_entropy_threshold": 0.4,
        "negative_alpha": 0.0,
    },
    "gda": {
        "method": "gda",
        "param_id": "gda_a1_bw1_t1_s0.1",
        "alpha": 1.0,
        "base_weight": 1.0,
        "temperature": 1.0,
        "shrinkage": 0.1,
    },
    "etta": {
        "method": "etta",
        "param_id": "etta_a0.45_b12_m0.97_ct0.9",
        "alpha": 0.45,
        "beta": 12.0,
        "momentum": 0.97,
        "confidence_threshold": 0.9,
    },
    "best4_ensemble": {
        "method": "best4_ensemble",
        "param_id": "ens_d0.2_f0.2_b0.4_t0.2",
        "dota_weight": 0.2,
        "free_weight": 0.2,
        "bca_weight": 0.4,
        "tda_weight": 0.2,
    },
}


def device_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def build_tta_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
        test_batch_size=args.test_batch_size,
        cache_batch_size=args.cache_batch_size,
        freetta_min_var=args.freetta_min_var,
        freetta_warmup_batches=args.freetta_warmup_batches,
        bca_confidence_threshold=args.bca_confidence_threshold,
        dota_min_var=args.dota_min_var,
        tda_positive_shot_capacity=args.tda_positive_shot_capacity,
        tda_negative_beta=args.tda_negative_beta,
        tda_negative_shot_capacity=args.tda_negative_shot_capacity,
        tda_negative_entropy_lower=args.tda_negative_entropy_lower,
        tda_negative_entropy_upper=args.tda_negative_entropy_upper,
        tda_negative_mask_lower=args.tda_negative_mask_lower,
        tda_negative_mask_upper=args.tda_negative_mask_upper,
        tda_top_k=args.tda_top_k,
        gda_min_var=args.gda_min_var,
        etta_entropy_power=args.etta_entropy_power,
        etta_base_weight=args.etta_base_weight,
    )


def parse_method_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = []
    preset_methods = args.methods
    if preset_methods is None:
        preset_methods = [] if args.method_config_json else list(DEFAULT_METHOD_CONFIGS)

    for method in preset_methods:
        if method not in DEFAULT_METHOD_CONFIGS:
            raise ValueError(f"Unknown preset method {method!r}. Choices: {sorted(DEFAULT_METHOD_CONFIGS)}")
        configs.append(dict(DEFAULT_METHOD_CONFIGS[method]))

    for raw_config in args.method_config_json or []:
        config = json.loads(raw_config)
        if "method" not in config:
            raise ValueError(f"Custom method config must contain 'method': {raw_config}")
        config.setdefault("param_id", config["method"])
        configs.append(config)

    return configs


def sample_ids_from_payload(payload: dict, n: int) -> list[str]:
    paths = payload.get("paths")
    if paths is None:
        return [f"idx_{idx:06d}" for idx in range(n)]

    sample_index = build_sample_index(payload)
    by_position = {idx: sample_id for sample_id, idx in sample_index.items()}
    return [by_position.get(idx, str(paths[idx])) for idx in range(n)]


def select_balanced_subset_with_ids(
    feats: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    *,
    seed: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
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
    ids = [sample_ids[int(idx)] for idx in indices]
    print(f"balanced {name}:")
    print("original label counts [REAL, FAKE]:", counts.tolist())
    print("balanced label counts [REAL, FAKE]:", torch.bincount(labels[indices], minlength=2).tolist())
    return feats[indices].contiguous(), labels[indices].contiguous(), ids


def shuffle_with_ids(
    feats: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(labels), generator=generator)
    ids = [sample_ids[int(idx)] for idx in indices]
    return feats[indices].contiguous(), labels[indices].contiguous(), ids


def metric_or_nan(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return float("nan")


def summarize_scores(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": metric_or_nan(roc_auc_score, y_true, y_score),
        "ap": metric_or_nan(average_precision_score, y_true, y_score),
    }


def threshold_for_model(
    thresholds: dict[str, float],
    model_name: str,
    model_type: str,
    model_path: Path,
) -> float:
    return thresholds.get(
        model_name,
        thresholds.get(
            f"{model_name}:{model_type}",
            thresholds.get(f"{model_name}:{model_path.name}", thresholds.get(model_path.name, thresholds.get(model_type, 0.5))),
        ),
    )


def probability_rows(
    common: dict[str, Any],
    sample_ids: list[str],
    labels: torch.Tensor,
    probs: torch.Tensor,
    y_pred: np.ndarray,
    threshold: float,
    fit_samples: int | None,
    fit_counts: list[int] | None,
) -> list[dict[str, Any]]:
    y_true = labels.detach().cpu().numpy().astype(int)
    probs_np = probs.detach().cpu().numpy()
    rows = []
    for idx, sample_id in enumerate(sample_ids):
        prob_real = float(probs_np[idx, 0])
        prob_fake = float(probs_np[idx, 1])
        rows.append(
            {
                **common,
                "sample_index": idx,
                "sample_id": sample_id,
                "y_true": int(y_true[idx]),
                "prob_real": prob_real,
                "prob_fake": prob_fake,
                "score": prob_fake,
                "y_pred": int(y_pred[idx]),
                "correct": bool(int(y_pred[idx]) == int(y_true[idx])),
                "confidence": float(max(prob_real, prob_fake)),
                "margin": float(abs(prob_fake - prob_real)),
                "threshold": float(threshold),
                "method_fit_samples": fit_samples,
                "method_fit_counts": fit_counts,
            }
        )
    return rows


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--dataset", action="append", required=True, help="Repeatable NAME=PATH feature file/folder.")
    parser.add_argument("--model", action="append", required=True, help="Repeatable NAME=PATH model .pt.")
    parser.add_argument("--thresholds-csv", action="append")
    parser.add_argument("--probes-output", default="/kaggle/working/tta_probability_probes.csv")
    parser.add_argument("--summary-output", default="/kaggle/working/tta_probability_probe_summary.csv")
    parser.add_argument("--balanced-dataset", action="append", default=[])
    parser.add_argument("--balanced-aligned-dataset", action="append", default=[])
    parser.add_argument("--shuffle-all-tests", action="store_true")
    parser.add_argument("--shuffle-dataset", action="append", default=[])
    parser.add_argument("--balance-method-fit", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=f"Preset methods. Choices: {', '.join(DEFAULT_METHOD_CONFIGS)}",
    )
    parser.add_argument(
        "--method-config-json",
        action="append",
        help="Repeatable JSON dict for custom configs, e.g. '{\"method\":\"bca\",\"param_id\":\"...\"}'.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--cache-batch-size", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--freetta-min-var", type=float, default=1e-4)
    parser.add_argument("--freetta-warmup-batches", type=int, default=1)
    parser.add_argument("--bca-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--dota-min-var", type=float, default=1e-4)
    parser.add_argument("--tda-positive-shot-capacity", type=int, default=64)
    parser.add_argument("--tda-negative-beta", type=float, default=5.5)
    parser.add_argument("--tda-negative-shot-capacity", type=int, default=64)
    parser.add_argument("--tda-negative-entropy-lower", type=float, default=0.35)
    parser.add_argument("--tda-negative-entropy-upper", type=float, default=0.8)
    parser.add_argument("--tda-negative-mask-lower", type=float, default=0.2)
    parser.add_argument("--tda-negative-mask-upper", type=float, default=0.8)
    parser.add_argument("--tda-top-k", type=int, default=64)
    parser.add_argument("--gda-min-var", type=float, default=1e-4)
    parser.add_argument("--etta-entropy-power", type=float, default=1.0)
    parser.add_argument("--etta-base-weight", type=float, default=1.0)
    args = parser.parse_args()

    device = device_arg(args.device)
    seed_everything(args.seed)

    train_feats, train_labels, _ = load_feature_file(args.train_features)
    if args.balance_method_fit:
        train_ids = [f"train_{idx:06d}" for idx in range(len(train_labels))]
        train_feats, train_labels, _ = select_balanced_subset_with_ids(
            train_feats,
            train_labels,
            train_ids,
            seed=args.seed + 101,
            name="method fit",
        )

    method_configs = parse_method_configs(args)
    tta_args = build_tta_args(args)
    dataset_specs = [parse_spec(value, "dataset") for value in args.dataset]
    model_specs = [parse_spec(value, "model") for value in args.model]
    thresholds = read_thresholds(args.thresholds_csv)
    balanced_datasets = set(args.balanced_dataset)
    balanced_aligned_datasets = set(args.balanced_aligned_dataset)
    shuffle_datasets = set(args.shuffle_dataset)

    probes_output = Path(args.probes_output)
    summary_output = Path(args.summary_output)
    probes_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    probes_output.unlink(missing_ok=True)
    summary_output.unlink(missing_ok=True)

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
            sample_ids = sample_ids_from_payload(payload, len(labels))

            if aligned_ids is not None:
                feats, labels = apply_aligned_ids(feats, labels, payload, aligned_ids)
                sample_ids = list(aligned_ids)
            elif dataset_name in balanced_datasets:
                feats, labels, sample_ids = select_balanced_subset_with_ids(
                    feats,
                    labels,
                    sample_ids,
                    seed=args.seed + sum(ord(ch) for ch in dataset_name + feature_path.name),
                    name=f"{dataset_name}/{feature_path.name}",
                )
            elif args.shuffle_all_tests or dataset_name in shuffle_datasets:
                feats, labels, sample_ids = shuffle_with_ids(
                    feats,
                    labels,
                    sample_ids,
                    seed=args.seed + sum(ord(ch) for ch in dataset_name + feature_path.name),
                )

            meta = infer_feature_meta(dataset_name, feature_path, payload)
            dim = feats.shape[1]
            y_true = labels.detach().cpu().numpy().astype(int)

            for model_name, model_path in model_specs:
                if model_name not in loaded_models:
                    model_type, model = load_probe_model(model_path, dim, device)
                    loaded_models[model_name] = (model_type, model)
                    print("loaded model:", model_name, model_type, model_path)
                model_type, model = loaded_models[model_name]
                threshold = threshold_for_model(thresholds, model_name, model_type, model_path)

                for config in method_configs:
                    common = {
                        **meta,
                        "model": model_name,
                        "model_type": model_type,
                        "model_path": str(model_path),
                        **config,
                    }
                    try:
                        print("probing:", dataset_name, feature_path.name, model_name, config["param_id"], flush=True)
                        if config["method"] == "none":
                            scores = predict_probe_scores(model, feats, device, batch_size=args.eval_batch_size)
                            probs = torch.tensor(np.stack([1.0 - scores, scores], axis=1), dtype=torch.float32)
                            y_pred = (scores >= threshold).astype(int)
                            fit_samples = None
                            fit_counts = None
                        else:
                            method, fit_samples, fit_counts = create_method(config, tta_args, train_feats, train_labels)
                            probs = method.predict_proba(model, feats, device).detach().cpu().float()
                            scores = probs[:, 1].numpy()
                            y_pred = probs.argmax(dim=1).numpy().astype(int)

                        current_probe_rows = probability_rows(
                            common,
                            sample_ids,
                            labels,
                            probs,
                            y_pred,
                            threshold,
                            fit_samples,
                            fit_counts,
                        )
                        current_summary_rows = [
                            {
                                **common,
                                "n_samples": int(len(labels)),
                                "method_fit_samples": fit_samples,
                                "method_fit_counts": fit_counts,
                                **summarize_scores(y_true, scores, y_pred),
                            }
                        ]
                        append_rows(probes_output, current_probe_rows)
                        append_rows(summary_output, current_summary_rows)
                        print("appended rows:", len(current_probe_rows), "->", probes_output, flush=True)
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        append_rows(summary_output, [{**common, "error": repr(exc)}])

    if not probes_output.exists():
        append_rows(probes_output, [{"error": "No probe rows were produced. Check summary output for method errors."}])
    if not summary_output.exists():
        append_rows(summary_output, [{"error": "No summary rows were produced."}])

    print(pd.read_csv(probes_output).head())
    print(pd.read_csv(summary_output))
    print("saved probes:", probes_output)
    print("saved summary:", summary_output)


if __name__ == "__main__":
    main()
