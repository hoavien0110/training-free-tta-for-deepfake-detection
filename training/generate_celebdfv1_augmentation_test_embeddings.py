from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.ffpp_split_utils import create_clip
from training.generate_ffpp_augmentation_test_embeddings import (
    build_augmentation_transform,
    extract_augmented_features,
    save_augmented_feature_file,
)
from training.generate_ffpp_corruption_test_embeddings import (
    DEFAULT_CORRUPTIONS,
    parse_root_map,
    resolve_device,
)


CELEBDFV1_DATASET_NAME = "Celeb-DF-v1"
CELEBDFV1_ANCHORS = ("Celeb-real", "YouTube-real", "Celeb-synthesis")


def prepare_celebdfv1_dataframe(
    *,
    csv_path: str | Path,
    deepfakebench_root: str | Path,
    dataset_name: str = CELEBDFV1_DATASET_NAME,
    allow_missing: bool = False,
    trust_paths: bool = False,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    deepfakebench_root = Path(deepfakebench_root)
    print("prepare_celebdfv1_dataframe", flush=True)
    print("csv_path:", csv_path, flush=True)
    print("deepfakebench_root:", deepfakebench_root, flush=True)

    df = pd.read_csv(csv_path)
    print("csv rows:", len(df), flush=True)
    df = df[df["datasetname"].eq(dataset_name)].copy()
    print("dataset rows:", len(df), "| dataset:", dataset_name, flush=True)
    if df.empty:
        raise ValueError(f"No rows found for datasetname={dataset_name!r}")

    df["label_num"] = df["label"].map({"REAL": 0, "FAKE": 1}).astype(int)
    df["imagepath_fixed"] = df["imagepath"].astype(str).str.replace(
        "../input/deepfakebench",
        str(deepfakebench_root),
        regex=False,
    )
    if trust_paths:
        print("trust paths: skipping clean image existence check", flush=True)
    else:
        exists = df["imagepath_fixed"].map(lambda path: Path(path).exists())
        missing_count = int((~exists).sum())
        print("missing clean files:", missing_count, flush=True)
        if missing_count:
            print(df.loc[~exists, ["imagepath", "imagepath_fixed"]].head(10).to_string(index=False))
            if not allow_missing:
                raise FileNotFoundError(
                    f"{missing_count}/{len(df)} CelebDFv1 images are missing. "
                    "Use --allow-missing to drop missing rows or --trust-paths to skip this check."
                )
            df = df[exists].reset_index(drop=True)
            print("after dropping missing:", len(df), flush=True)

    print(df["label_num"].value_counts().rename(index={0: "REAL", 1: "FAKE"}), flush=True)
    return df.reset_index(drop=True)


def resolve_celebdfv1_corruption_root(
    *,
    corruption: str,
    level: int,
    default_root: str | Path,
    root_map: dict[str, Path],
) -> Path:
    root = root_map.get(corruption, Path(default_root))
    candidates = [
        root / corruption / f"level_{level}",
        root / corruption / CELEBDFV1_DATASET_NAME,
        root / corruption,
        root / f"level_{level}",
        root / CELEBDFV1_DATASET_NAME,
        root,
    ]
    for candidate in candidates:
        if any((candidate / anchor).exists() for anchor in CELEBDFV1_ANCHORS):
            return candidate

    print("could not auto-detect CelebDFv1 corruption root. tried:")
    for candidate in candidates:
        print(" -", candidate)
    return candidates[0]


def remap_celebdfv1_to_corruption_root(
    df: pd.DataFrame,
    *,
    deepfakebench_root: str | Path,
    corruption_level_root: str | Path,
    allow_missing: bool = False,
    trust_paths: bool = False,
) -> pd.DataFrame:
    source_root = Path(deepfakebench_root) / CELEBDFV1_DATASET_NAME
    corruption_level_root = Path(corruption_level_root)
    out = df.copy()
    out["source_imagepath_fixed"] = out["imagepath_fixed"]
    out["imagepath_fixed"] = out["imagepath_fixed"].astype(str).str.replace(
        str(source_root),
        str(corruption_level_root),
        regex=False,
    )

    print("corruption root:", corruption_level_root, flush=True)
    print("mapped rows:", len(out), flush=True)
    if trust_paths:
        print("trust paths: skipping per-file existence check", flush=True)
        return out

    exists = out["imagepath_fixed"].map(lambda path: Path(path).exists())
    missing_count = int((~exists).sum())
    print("missing files:", missing_count, flush=True)
    if missing_count:
        print("missing examples:")
        print(out.loc[~exists, ["source_imagepath_fixed", "imagepath_fixed"]].head(10).to_string(index=False))
        if not allow_missing:
            raise FileNotFoundError(
                f"{missing_count}/{len(out)} mapped files are missing under {corruption_level_root}. "
                "Use --allow-missing to skip missing files."
            )
        out = out[exists].reset_index(drop=True)
        print("after dropping missing:", len(out), flush=True)
    return out


def output_name(*, level: int | None, corruption: str | None, policy: str, views: int) -> str:
    if corruption:
        return f"celebdfv1_level{level}_{corruption}_aug_{policy}_{views}views_features.pt"
    return f"celebdfv1_aug_{policy}_{views}views_features.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="/kaggle/input/datasets/jamestashvik/deepfakebench/deepfakebench_dataset.csv")
    parser.add_argument("--deepfakebench-root", default="/kaggle/input/datasets/jamestashvik/deepfakebench/DeepFakeBench")
    parser.add_argument("--corruption-root", default="/kaggle/input/celebdfv1-corruption-level-2")
    parser.add_argument("--corruption-root-map", nargs="*")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--corruptions", nargs="+", default=list(DEFAULT_CORRUPTIONS))
    parser.add_argument("--include-clean", action="store_true")
    parser.add_argument("--output-dir", default="/kaggle/working/celebdfv1_augmented_test_features")
    parser.add_argument("--augmentation-policy", choices=["weak", "crop", "color", "preprocess"], default="weak")
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--store-views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--trust-paths", action="store_true", help="Skip slow per-file existence checks after path remapping.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.views <= 0:
        raise ValueError("--views must be positive")

    root_map = parse_root_map(args.corruption_root_map)
    device = resolve_device(args.device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = prepare_celebdfv1_dataframe(
        csv_path=args.csv_path,
        deepfakebench_root=args.deepfakebench_root,
        allow_missing=args.allow_missing,
        trust_paths=args.trust_paths,
    )

    print("creating CLIP:", args.clip_model, args.pretrained, flush=True)
    clip_model, preprocess = create_clip(args.clip_model, args.pretrained, device)
    transform = build_augmentation_transform(
        preprocess,
        policy=args.augmentation_policy,
        image_size=args.image_size,
    )
    print("CLIP and augmentation ready", flush=True)

    tasks = []
    if args.include_clean:
        tasks.append((None, None, base_df))
    for corruption in args.corruptions:
        corruption_level_root = resolve_celebdfv1_corruption_root(
            corruption=corruption,
            level=args.level,
            default_root=args.corruption_root,
            root_map=root_map,
        )
        df = remap_celebdfv1_to_corruption_root(
            base_df,
            deepfakebench_root=args.deepfakebench_root,
            corruption_level_root=corruption_level_root,
            allow_missing=args.allow_missing,
            trust_paths=args.trust_paths,
        )
        tasks.append((corruption, corruption_level_root, df))

    saved = []
    for task_idx, (corruption, root, df) in enumerate(tasks):
        out_path = output_dir / output_name(
            level=args.level if corruption else None,
            corruption=corruption,
            policy=args.augmentation_policy,
            views=args.views,
        )
        if args.skip_existing and out_path.exists():
            print("skip existing:", out_path)
            saved.append(out_path)
            continue

        label = "clean" if corruption is None else f"level {args.level} | {corruption}"
        print(f"\nExtracting augmented CelebDFv1 | {label}", flush=True)
        features, features_views, labels, paths = extract_augmented_features(
            df,
            clip_model=clip_model,
            transform=transform,
            views=args.views,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=args.use_amp,
            seed=args.seed + task_idx,
        )
        save_augmented_feature_file(
            out_path,
            features=features,
            features_views=features_views,
            labels=labels,
            paths=paths,
            store_views=args.store_views,
            metadata={
                "dataset_name": CELEBDFV1_DATASET_NAME,
                "split_name": "test",
                "transform_name": corruption or "none",
                "transform_level": args.level if corruption else None,
                "clip_model": f"{args.clip_model}/{args.pretrained}",
                "augmentation_policy": args.augmentation_policy,
                "augmentation_views": args.views,
                "image_size": args.image_size,
                "features_are_augmented_mean": True,
                "features_views_shape": tuple(features_views.shape) if args.store_views else None,
                "corruption_root": str(root) if root else None,
                "seed": args.seed,
            },
        )
        saved.append(out_path)

    print("saved files:")
    for path in saved:
        print(" -", path)


if __name__ == "__main__":
    main()
