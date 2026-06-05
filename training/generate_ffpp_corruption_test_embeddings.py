from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from training.ffpp_split_utils import (
    create_clip,
    extract_features,
    prepare_ffpp_split_dataframe,
    save_feature_file,
)


DEFAULT_CORRUPTIONS = ("color_contrast", "color_saturation", "gaussian_blur", "resize")


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def remap_to_corruption_root(
    df,
    *,
    deepfakebench_root: str | Path,
    corruption_level_root: str | Path,
    allow_missing: bool = False,
):
    source_root = Path(deepfakebench_root) / "FaceForensics++"
    corruption_level_root = Path(corruption_level_root)
    out = df.copy()
    out["source_imagepath_fixed"] = out["imagepath_fixed"]
    out["imagepath_fixed"] = out["imagepath_fixed"].astype(str).str.replace(
        str(source_root),
        str(corruption_level_root),
        regex=False,
    )

    exists = out["imagepath_fixed"].map(lambda path: Path(path).exists())
    missing_count = int((~exists).sum())
    print("corruption root:", corruption_level_root)
    print("mapped rows:", len(out), "| missing files:", missing_count)
    if missing_count:
        print("missing examples:")
        print(out.loc[~exists, ["source_imagepath_fixed", "imagepath_fixed"]].head(10).to_string(index=False))
        if not allow_missing:
            raise FileNotFoundError(
                f"{missing_count}/{len(out)} mapped files are missing under {corruption_level_root}. "
                "Use --allow-missing to skip missing files."
            )
        out = out[exists].reset_index(drop=True)
        print("after dropping missing:", len(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="/kaggle/input/datasets/jamestashvik/deepfakebench/deepfakebench_dataset.csv")
    parser.add_argument("--deepfakebench-root", default="/kaggle/input/datasets/jamestashvik/deepfakebench/DeepFakeBench")
    parser.add_argument("--corruption-root", default="/kaggle/input/ffpp-corruption-level-2")
    parser.add_argument("--split-root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--corruptions", nargs="+", default=list(DEFAULT_CORRUPTIONS))
    parser.add_argument("--output-dir", default="/kaggle/working/ffpp_corruption_test_features")
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("preparing base FF++ split dataframe:", args.split, flush=True)
    base_df = prepare_ffpp_split_dataframe(
        csv_path=args.csv_path,
        deepfakebench_root=args.deepfakebench_root,
        split_name=args.split,
        split_root=args.split_root,
    )

    print("creating CLIP:", args.clip_model, args.pretrained, flush=True)
    clip_model, preprocess = create_clip(args.clip_model, args.pretrained, device)
    print("CLIP ready", flush=True)

    saved = []
    for corruption in args.corruptions:
        output_path = output_dir / f"ffpp_{args.split}_level{args.level}_{corruption}_features.pt"
        if args.skip_existing and output_path.exists():
            print("skip existing:", output_path)
            saved.append(output_path)
            continue

        corruption_level_root = Path(args.corruption_root) / corruption / f"level_{args.level}"
        print(f"\nExtracting FF++ {args.split} | level {args.level} | {corruption}", flush=True)
        df = remap_to_corruption_root(
            base_df,
            deepfakebench_root=args.deepfakebench_root,
            corruption_level_root=corruption_level_root,
            allow_missing=args.allow_missing,
        )
        features, labels, paths = extract_features(
            df,
            clip_model=clip_model,
            preprocess=preprocess,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=args.use_amp,
        )
        save_feature_file(
            output_path,
            features,
            labels,
            paths,
            metadata={
                "dataset_name": "FaceForensics++",
                "split_name": args.split,
                "transform_name": corruption,
                "transform_level": args.level,
                "clip_model": f"{args.clip_model}/{args.pretrained}",
                "corruption_root": str(corruption_level_root),
            },
        )
        saved.append(output_path)

    print("saved files:")
    for path in saved:
        print(" -", path)


if __name__ == "__main__":
    main()
