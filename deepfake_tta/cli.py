from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from deepfake_tta.corruptions import ALL_METHODS, PostprocessType, process_folder
from deepfake_tta.features import (
    create_clip_model,
    extract_features,
    prepare_deepfakebench_dataframe,
    save_feature_file,
)
from deepfake_tta.modeling import (
    evaluate_probe,
    LinearProbe,
    load_feature_file,
    seed_everything,
    train_linear_probe,
)
from deepfake_tta.methods import AVAILABLE_TTA_METHODS, create_tta_method


def device_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--tip-test-batch-size", type=int, default=512)
    parser.add_argument("--tip-cache-batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--load-model",
        help="Load a saved LinearProbe state_dict and skip training.",
    )
    parser.add_argument(
        "--method-cache-dir",
        help="Directory for reusable method caches, e.g. compact source caches.",
    )
    parser.add_argument(
        "--tta-methods",
        nargs="+",
        choices=AVAILABLE_TTA_METHODS,
        default=["tip_adapter"],
        help="TTA methods to run after the linear probe baseline.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record method errors in the results CSV and continue evaluating other methods.",
    )
    parser.add_argument(
        "--shuffle-test-features",
        action="store_true",
        help="Shuffle test features and labels together before evaluation. Useful for order-sensitive online TTA.",
    )


def build_tta_methods(args: argparse.Namespace, train_feats: torch.Tensor, train_labels: torch.Tensor):
    methods = [
        create_tta_method(
            method_name,
            alpha=args.alpha,
            beta=args.beta,
            test_batch_size=args.tip_test_batch_size,
            cache_batch_size=args.tip_cache_batch_size,
            method_cache_dir=args.method_cache_dir,
        )
        for method_name in args.tta_methods
    ]
    for method in methods:
        method.fit(train_feats, train_labels)
    return methods


def maybe_shuffle_test_features(
    args: argparse.Namespace,
    feats: torch.Tensor,
    labels: torch.Tensor,
    *,
    salt: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not getattr(args, "shuffle_test_features", False):
        return feats, labels

    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + int(salt))
    indices = torch.randperm(len(labels), generator=generator)
    shuffled_feats = feats[indices].contiguous()
    shuffled_labels = labels[indices].contiguous()
    print("shuffled test features with seed:", int(args.seed) + int(salt))
    print("label counts [REAL, FAKE]:", torch.bincount(shuffled_labels.long(), minlength=2).tolist())
    return shuffled_feats, shuffled_labels


def load_or_train_linear_probe(
    args: argparse.Namespace,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    device: str,
) -> LinearProbe:
    if getattr(args, "load_model", None):
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
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    print("saved model to:", args.model_output)
    return model


def cmd_generate_corruptions(args: argparse.Namespace) -> None:
    process_folder(
        input_dir=args.input_dir,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        methods=args.methods,
        levels=args.levels,
        output_size=(args.width, args.height),
    )


def cmd_extract_features(args: argparse.Namespace) -> None:
    device = device_arg(args.device)
    df = prepare_deepfakebench_dataframe(
        csv_path=args.csv_path,
        dataset_name=args.dataset_name,
        deepfakebench_root=args.deepfakebench_root,
        replacement_root=args.replacement_root,
        original_root=args.original_root,
    )
    print(df.shape)
    print(df["label_num"].value_counts())

    model, preprocess = create_clip_model(args.clip_model, args.pretrained, device)
    features, labels, paths = extract_features(
        dataframe=df,
        clip_model=model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    metadata = {
        "dataset_name": args.dataset_name,
        "clip_model": f"{args.clip_model}/{args.pretrained}",
    }
    if args.transform_name:
        metadata["transform_name"] = args.transform_name
    if args.transform_level:
        metadata["transform_level"] = args.transform_level
    save_feature_file(args.output_path, features, labels, paths=paths, metadata=metadata)
    print("features:", tuple(features.shape))
    print("labels:", tuple(labels.shape))


def cmd_extract_celebdf_corruptions(args: argparse.Namespace) -> None:
    device = device_arg(args.device)
    model, preprocess = create_clip_model(args.clip_model, args.pretrained, device)
    output_dir = Path(args.output_dir)

    extract_celebdf_corruption_level(args, model, preprocess, device, output_dir, args.level)


def extract_celebdf_corruption_level(
    args: argparse.Namespace,
    model,
    preprocess,
    device: str,
    output_dir: Path,
    level: int,
) -> None:
    for method in args.methods:
        replacement_root = Path(args.processed_root) / method / f"level_{level}" / args.dataset_name
        df = prepare_deepfakebench_dataframe(
            csv_path=args.csv_path,
            dataset_name=args.dataset_name,
            deepfakebench_root=args.deepfakebench_root,
            replacement_root=replacement_root,
            original_root=Path(args.deepfakebench_root) / args.dataset_name,
        )
        output_path = output_dir / f"celebdfv1_level{level}_{method}_features.pt"
        print(f"\nExtracting level {level} | {method}")
        print("root:", replacement_root)
        features, labels, paths = extract_features(
            dataframe=df,
            clip_model=model,
            preprocess=preprocess,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        save_feature_file(
            output_path,
            features,
            labels,
            paths=paths,
            metadata={
                "dataset_name": args.dataset_name,
                "transform_name": method,
                "transform_level": level,
                "clip_model": f"{args.clip_model}/{args.pretrained}",
            },
        )


def cmd_extract_celebdf_corruption_levels(args: argparse.Namespace) -> None:
    device = device_arg(args.device)
    model, preprocess = create_clip_model(args.clip_model, args.pretrained, device)
    output_dir = Path(args.output_dir)

    for level in args.levels:
        extract_celebdf_corruption_level(args, model, preprocess, device, output_dir, level)


def cmd_train_eval(args: argparse.Namespace) -> None:
    device = device_arg(args.device)
    seed_everything(args.seed)
    train_feats, train_labels, _ = load_feature_file(args.train_features)
    model = load_or_train_linear_probe(args, train_feats, train_labels, device)

    rows = []
    rows.append(
        {
            "dataset": "train",
            "method": "linear_probe",
            **evaluate_probe(
                model,
                train_feats,
                train_labels,
                device,
                name="Train",
                batch_size=args.eval_batch_size,
                show_report=False,
            ),
        }
    )

    for feature_path in args.test_features:
        feats, labels, payload = load_feature_file(feature_path)
        feats, labels = maybe_shuffle_test_features(args, feats, labels, salt=len(rows))
        dataset_name = payload.get("dataset_name", Path(feature_path).stem)
        probe_metrics = evaluate_probe(
            model,
            feats,
            labels,
            device,
            name=f"{dataset_name} | Linear Probe",
            batch_size=args.eval_batch_size,
            show_report=args.show_report,
        )
        rows.append({"dataset": dataset_name, "feature_path": feature_path, "method": "linear_probe", **probe_metrics})

        for method in build_tta_methods(args, train_feats, train_labels):
            try:
                method_metrics = method.evaluate(
                    model,
                    feats,
                    labels,
                    device,
                    name=f"{dataset_name} | {method.name}",
                    show_report=args.show_report,
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "feature_path": feature_path,
                        "method": method.name,
                        **method_metrics,
                    }
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append(
                    {
                        "dataset": dataset_name,
                        "feature_path": feature_path,
                        "method": method.name,
                        "error": repr(exc),
                    }
                )

    if args.results_output:
        pd.DataFrame(rows).to_csv(args.results_output, index=False)
        print("saved results to:", args.results_output)


def cmd_eval_corruptions(args: argparse.Namespace) -> None:
    device = device_arg(args.device)
    seed_everything(args.seed)
    train_feats, train_labels, _ = load_feature_file(args.train_features)
    model = load_or_train_linear_probe(args, train_feats, train_labels, device)

    rows = [
        {
            "level": args.level,
            "corruption": "train_ffpp",
            "method": "linear_probe",
            **evaluate_probe(
                model,
                train_feats,
                train_labels,
                device,
                name="Train FF++",
                batch_size=args.eval_batch_size,
                show_report=False,
            ),
        }
    ]

    for corruption in args.corruptions:
        rows.extend(evaluate_corruption_feature(args, model, train_feats, train_labels, device, args.level, corruption))

    pd.DataFrame(rows).to_csv(args.results_output, index=False)
    print("saved results to:", args.results_output)


def evaluate_corruption_feature(
    args: argparse.Namespace,
    model,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    device: str,
    level: int,
    corruption: str,
) -> list[dict]:
    feature_path = Path(args.feature_dir) / f"celebdfv1_level{level}_{corruption}_features.pt"
    feats, labels, _ = load_feature_file(str(feature_path))
    salt = level * 1000 + sum(ord(ch) for ch in corruption)
    feats, labels = maybe_shuffle_test_features(args, feats, labels, salt=salt)
    rows = []
    probe_metrics = evaluate_probe(
        model,
        feats,
        labels,
        device,
        name=f"CelebDFv1 level {level} {corruption} | Linear Probe",
        batch_size=args.eval_batch_size,
        show_report=False,
    )
    rows.append({"level": level, "corruption": corruption, "method": "linear_probe", **probe_metrics})

    for method in build_tta_methods(args, train_feats, train_labels):
        try:
            method_metrics = method.evaluate(
                model,
                feats,
                labels,
                device,
                name=f"CelebDFv1 level {level} {corruption} | {method.name}",
                show_report=False,
            )
            rows.append(
                {
                    "level": level,
                    "corruption": corruption,
                    "method": method.name,
                    **method_metrics,
                }
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            rows.append(
                {
                    "level": level,
                    "corruption": corruption,
                    "method": method.name,
                    "error": repr(exc),
                }
            )
    return rows


def cmd_eval_corruption_levels(args: argparse.Namespace) -> None:
    device = device_arg(args.device)
    seed_everything(args.seed)
    train_feats, train_labels, _ = load_feature_file(args.train_features)
    model = load_or_train_linear_probe(args, train_feats, train_labels, device)

    rows = [
        {
            "level": "train",
            "corruption": "train_ffpp",
            "method": "linear_probe",
            **evaluate_probe(
                model,
                train_feats,
                train_labels,
                device,
                name="Train FF++",
                batch_size=args.eval_batch_size,
                show_report=False,
            ),
        }
    ]

    for level in args.levels:
        for corruption in args.corruptions:
            rows.extend(evaluate_corruption_feature(args, model, train_feats, train_labels, device, level, corruption))

    pd.DataFrame(rows).to_csv(args.results_output, index=False)
    print("saved results to:", args.results_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepfake-tta")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("generate-corruptions")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--dataset-name", default="Celeb-DF-v1")
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    p.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.set_defaults(func=cmd_generate_corruptions)

    p = subparsers.add_parser("extract-features")
    p.add_argument("--csv-path", required=True)
    p.add_argument("--deepfakebench-root", required=True)
    p.add_argument("--dataset-name", default="FaceForensics++")
    p.add_argument("--replacement-root")
    p.add_argument("--original-root")
    p.add_argument("--output-path", required=True)
    p.add_argument("--transform-name")
    p.add_argument("--transform-level", type=int)
    p.add_argument("--clip-model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_extract_features)

    p = subparsers.add_parser("extract-celebdf-corruptions")
    p.add_argument("--csv-path", required=True)
    p.add_argument("--deepfakebench-root", required=True)
    p.add_argument("--processed-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", default="Celeb-DF-v1")
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--clip-model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_extract_celebdf_corruptions)

    p = subparsers.add_parser("extract-celebdf-corruption-levels")
    p.add_argument("--csv-path", required=True)
    p.add_argument("--deepfakebench-root", required=True)
    p.add_argument("--processed-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", default="Celeb-DF-v1")
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    p.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--clip-model", default="ViT-L-14")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_extract_celebdf_corruption_levels)

    p = subparsers.add_parser("train-eval")
    p.add_argument("--train-features", required=True)
    p.add_argument("--test-features", nargs="+", required=True)
    p.add_argument("--model-output", default="/kaggle/working/ufd_linear_probe.pt")
    p.add_argument("--results-output")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=5.5)
    p.add_argument("--show-report", action="store_true")
    add_common_train_args(p)
    p.set_defaults(func=cmd_train_eval)

    p = subparsers.add_parser("eval-corruptions")
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--train-features", required=True)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--corruptions", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    p.add_argument("--model-output", default="/kaggle/working/ufd_linear_probe.pt")
    p.add_argument("--results-output", required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=5.5)
    add_common_train_args(p)
    p.set_defaults(func=cmd_eval_corruptions)

    p = subparsers.add_parser("eval-corruption-levels")
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--train-features", required=True)
    p.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--corruptions", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    p.add_argument("--model-output", default="/kaggle/working/ufd_linear_probe_all_levels.pt")
    p.add_argument("--results-output", required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=5.5)
    add_common_train_args(p)
    p.set_defaults(func=cmd_eval_corruption_levels)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
