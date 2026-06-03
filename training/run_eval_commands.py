from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ALL_TTA_METHODS = [
    "tip_adapter",
    "boost_adapter",
    "compact_cache_adapter",
    "online_confident_cache_adapter",
    "crg",
    "dmn",
    "dpe",
    "dota",
    "freetta",
    "freetta_linear_ensemble",
    "freetta_balanced",
    "freetta_linear_ensemble_balanced",
    "bca_balanced",
    "bca_linear_ensemble_balanced",
    "dpe_balanced",
    "dpe_linear_ensemble_balanced",
    "dota_balanced",
    "dota_linear_ensemble_balanced",
    "prototype_linear_tta_balanced",
    "online_cache_10_balanced",
    "bca",
    "dynaprompt",
    "prototype_linear_tta",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="/kaggle/working/training-free-tta-for-deepfake-detection")
    parser.add_argument("--feature-dir", default="/kaggle/working/ffpp_split_features")
    parser.add_argument("--model-dir", default="/kaggle/working/ffpp_split_models")
    parser.add_argument("--eval-ffpp", action="store_true")
    parser.add_argument("--eval-celebdf", action="store_true")
    parser.add_argument("--celebdf-feature-root", default="/kaggle/input/celebdfv1-balanced-16-16-features/celebdfv1_balanced_16_16_features")
    parser.add_argument("--method-cache-dir", default="/kaggle/working/tta_method_cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shuffle-test-features", action="store_true")
    parser.add_argument("--results-dir", default="/kaggle/working")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_ffpp:
        run(
            [
                "python",
                "-m",
                "deepfake_tta.cli",
                "train-eval",
                "--train-features",
                str(feature_dir / "ffpp_train_features.pt"),
                "--test-features",
                str(feature_dir / "ffpp_val_features.pt"),
                str(feature_dir / "ffpp_test_features.pt"),
                "--load-model",
                str(model_dir / "ffpp_linear_probe_split.pt"),
                "--tta-methods",
                "freetta",
                "freetta_linear_ensemble",
                "freetta_linear_ensemble_balanced",
                "--continue-on-error",
                "--device",
                args.device,
                "--results-output",
                str(results_dir / "ffpp_val_test_tta_results.csv"),
            ],
            cwd=repo_dir,
        )

    if args.eval_celebdf:
        cmd = [
            "python",
            "-m",
            "deepfake_tta.cli_2",
            "eval-template",
            "--feature-root",
            args.celebdf_feature_root,
            "--train-features",
            str(feature_dir / "ffpp_train_features.pt"),
            "--load-model",
            str(model_dir / "ffpp_linear_probe_split.pt"),
            "--levels",
            "1",
            "2",
            "3",
            "4",
            "5",
            "--corruptions",
            "color_contrast",
            "color_saturation",
            "gaussian_blur",
            "resize",
            "--filename-suffix",
            "balanced_16_16",
            "--tta-methods",
            *ALL_TTA_METHODS,
            "--method-cache-dir",
            args.method_cache_dir,
            "--continue-on-error",
            "--device",
            args.device,
            "--results-output",
            str(results_dir / "celebdfv1_balanced_16_16_24_methods_results.csv"),
        ]
        if args.shuffle_test_features:
            cmd.append("--shuffle-test-features")
        run(cmd, cwd=repo_dir)

    if not args.eval_ffpp and not args.eval_celebdf:
        print("Nothing to run. Pass --eval-ffpp and/or --eval-celebdf.")


if __name__ == "__main__":
    main()
