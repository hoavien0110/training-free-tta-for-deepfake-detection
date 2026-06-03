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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="/kaggle/input/datasets/jamestashvik/deepfakebench/deepfakebench_dataset.csv")
    parser.add_argument("--deepfakebench-root", default="/kaggle/input/datasets/jamestashvik/deepfakebench/DeepFakeBench")
    parser.add_argument("--split-root")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output-dir", default="/kaggle/working/ffpp_split_features")
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_model, preprocess = create_clip(args.clip_model, args.pretrained, device)
    saved = []

    for split_name in args.splits:
        output_path = output_dir / f"ffpp_{split_name}_features.pt"
        if args.skip_existing and output_path.exists():
            print("skip existing:", output_path)
            saved.append(output_path)
            continue

        df = prepare_ffpp_split_dataframe(
            csv_path=args.csv_path,
            deepfakebench_root=args.deepfakebench_root,
            split_name=split_name,
            split_root=args.split_root,
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
                "split_name": split_name,
                "clip_model": f"{args.clip_model}/{args.pretrained}",
            },
        )
        saved.append(output_path)

    print("saved files:")
    for path in saved:
        print(" -", path)


if __name__ == "__main__":
    main()
