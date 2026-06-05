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

    rows = []
    loaded_models = {}
    for feature_path in feature_files:
        feats, labels, payload = load_feature_file(str(feature_path))
        meta = infer_feature_meta(feature_path, payload)
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
