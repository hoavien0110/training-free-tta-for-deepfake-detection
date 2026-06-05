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

from deepfake_tta.modeling import LinearProbe, OSDLinearProbe, evaluate_probe, load_feature_file, seed_everything


def device_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def find_feature_files(feature_dir: str | Path) -> list[Path]:
    root = Path(feature_dir)
    files = sorted(root.glob("**/ffpp_test_level*_features.pt"))
    if files:
        return files
    files = sorted(root.glob("**/*features.pt"))
    if files:
        return files
    raise FileNotFoundError(f"No feature files found under {root}")


def find_model_files(model_dir: str | Path) -> list[Path]:
    root = Path(model_dir)
    preferred = [
        root / "ffpp_linear_probe_split.pt",
        root / "ffpp_osd_linear_probe_split.pt",
    ]
    files = [path for path in preferred if path.exists()]
    if files:
        return files
    files = sorted(root.glob("**/*.pt"))
    if files:
        return files
    raise FileNotFoundError(f"No model .pt files found under {root}")


def infer_feature_meta(path: Path, payload: dict) -> dict[str, object]:
    match = re.search(r"ffpp_test_level(\d+)_(.+?)_features\.pt$", path.name)
    level = payload.get("transform_level")
    corruption = payload.get("transform_name")
    if match:
        level = int(match.group(1)) if level is None else level
        corruption = match.group(2) if corruption is None else corruption
    return {
        "dataset": payload.get("dataset_name", "FaceForensics++"),
        "split": payload.get("split_name", "test"),
        "level": level,
        "corruption": corruption,
        "feature_path": str(path),
    }


def sample_id_from_path(path: str) -> str:
    parts = Path(path).parts
    anchors = ("original_sequences", "manipulated_sequences")
    for anchor in anchors:
        if anchor in parts:
            idx = parts.index(anchor)
            return "/".join(parts[idx:])
    return "/".join(parts[-8:])


def build_sample_index(payload: dict) -> dict[str, int]:
    paths = payload.get("paths")
    if paths is None:
        raise ValueError("Feature payload does not contain paths, cannot align samples across corruptions.")
    return {sample_id_from_path(path): idx for idx, path in enumerate(paths)}


def interleave_by_blocks(real_ids: list[str], fake_ids: list[str], block_size: int) -> list[str]:
    ordered = []
    usable = min(len(real_ids), len(fake_ids))
    usable = usable - (usable % block_size)
    real_ids = real_ids[:usable]
    fake_ids = fake_ids[:usable]
    for start in range(0, usable, block_size):
        ordered.extend(real_ids[start : start + block_size])
        ordered.extend(fake_ids[start : start + block_size])
    return ordered


def build_aligned_balanced_ids(feature_files: list[Path], block_size: int) -> list[str]:
    common_ids: set[str] | None = None
    label_by_id = {}

    for feature_path in feature_files:
        payload = torch.load(feature_path, map_location="cpu")
        sample_index = build_sample_index(payload)
        ids = set(sample_index)
        common_ids = ids if common_ids is None else common_ids & ids

        labels = payload["labels"].long()
        for sample_id, idx in sample_index.items():
            label = int(labels[idx].item())
            previous = label_by_id.get(sample_id)
            if previous is not None and previous != label:
                raise ValueError(f"Inconsistent label for sample {sample_id}: {previous} vs {label}")
            label_by_id[sample_id] = label

    if not common_ids:
        raise ValueError("No common sample IDs found across feature files.")

    real_ids = sorted(sample_id for sample_id in common_ids if label_by_id[sample_id] == 0)
    fake_ids = sorted(sample_id for sample_id in common_ids if label_by_id[sample_id] == 1)
    ordered_ids = interleave_by_blocks(real_ids, fake_ids, block_size)
    if not ordered_ids:
        raise ValueError(
            f"Cannot build balanced aligned test set with block_size={block_size}. "
            f"Common REAL={len(real_ids)}, FAKE={len(fake_ids)}"
        )

    print("aligned balanced test:")
    print("common ids:", len(common_ids))
    print("common label counts [REAL, FAKE]:", [len(real_ids), len(fake_ids)])
    print("selected label counts [REAL, FAKE]:", [len(ordered_ids) // 2, len(ordered_ids) // 2])
    print("block pattern:", block_size, "REAL then", block_size, "FAKE")
    return ordered_ids


def apply_aligned_ids(feats: torch.Tensor, labels: torch.Tensor, payload: dict, ordered_ids: list[str]):
    sample_index = build_sample_index(payload)
    indices = torch.tensor([sample_index[sample_id] for sample_id in ordered_ids], dtype=torch.long)
    feats = feats[indices].contiguous()
    labels = labels[indices].contiguous()
    return feats, labels


def load_probe_model(model_path: Path, dim: int, device: str):
    state = torch.load(model_path, map_location="cpu")
    if "center" in state and "basis" in state:
        model = OSDLinearProbe(dim, center=state["center"], basis=state["basis"])
        model_name = "osd_linear_probe"
    else:
        model = LinearProbe(dim)
        model_name = "linear_probe"
    model.load_state_dict(state)
    model.to(device).eval()
    return model_name, model


def read_thresholds(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    threshold_path = Path(path)
    if not threshold_path.exists():
        print("threshold CSV not found, using default 0.5:", threshold_path)
        return {}
    df = pd.read_csv(threshold_path)
    out = {}
    for _, row in df.iterrows():
        model = str(row.get("model", ""))
        threshold = float(row.get("threshold", 0.5))
        if model:
            out[model] = threshold
        model_path = row.get("model_path")
        if isinstance(model_path, str) and model_path:
            out[Path(model_path).name] = threshold
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--thresholds-csv")
    parser.add_argument("--results-output", default="/kaggle/working/ffpp_corruption_model_results.csv")
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-report", action="store_true")
    parser.add_argument(
        "--balanced-aligned-test",
        action="store_true",
        help="Use the same balanced sample IDs for every corruption file.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=10,
        help="When --balanced-aligned-test is set, order samples as N REAL then N FAKE repeatedly.",
    )
    args = parser.parse_args()

    device = device_arg(args.device)
    seed_everything(args.seed)

    feature_files = find_feature_files(args.feature_dir)
    model_files = find_model_files(args.model_dir)
    thresholds = read_thresholds(args.thresholds_csv)

    print("feature files:")
    for path in feature_files:
        print(" -", path)
    print("model files:")
    for path in model_files:
        print(" -", path)

    aligned_ids = None
    if args.balanced_aligned_test:
        aligned_ids = build_aligned_balanced_ids(feature_files, args.block_size)

    rows = []
    loaded_models = {}
    for feature_path in feature_files:
        feats, labels, payload = load_feature_file(str(feature_path))
        meta = infer_feature_meta(feature_path, payload)
        if aligned_ids is not None:
            feats, labels = apply_aligned_ids(feats, labels, payload, aligned_ids)
            print("after aligned balance:", feature_path.name)
            print("features:", tuple(feats.shape))
            print("label counts [REAL, FAKE]:", torch.bincount(labels.long(), minlength=2).tolist())
        dim = feats.shape[1]

        for model_path in model_files:
            if model_path not in loaded_models:
                loaded_models[model_path] = load_probe_model(model_path, dim, device)
            model_name, model = loaded_models[model_path]
            threshold = thresholds.get(model_path.name, thresholds.get(model_name, 0.5))
            metrics = evaluate_probe(
                model,
                feats,
                labels,
                device=device,
                name=f"{meta['dataset']} {meta['split']} level {meta['level']} {meta['corruption']} | {model_name}",
                batch_size=args.eval_batch_size,
                show_report=args.show_report,
                threshold=threshold,
            )
            rows.append(
                {
                    **meta,
                    "model": model_name,
                    "model_path": str(model_path),
                    "balanced_aligned_test": bool(args.balanced_aligned_test),
                    "block_size": int(args.block_size) if args.balanced_aligned_test else None,
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    Path(args.results_output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_output, index=False)
    print(results)
    print("saved:", args.results_output)


if __name__ == "__main__":
    main()
