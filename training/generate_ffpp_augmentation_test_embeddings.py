from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from training.ffpp_split_utils import (
    create_clip,
    prepare_ffpp_split_dataframe,
    save_feature_file,
)
from training.generate_ffpp_corruption_test_embeddings import (
    DEFAULT_CORRUPTIONS,
    parse_root_map,
    remap_to_corruption_root,
    resolve_corruption_level_root,
    resolve_device,
)


def find_normalize(preprocess):
    for transform in getattr(preprocess, "transforms", []):
        if transform.__class__.__name__ == "Normalize":
            return transform
    return None


def build_augmentation_transform(preprocess, *, policy: str, image_size: int):
    import torchvision.transforms as T

    normalize = find_normalize(preprocess)
    mean = getattr(normalize, "mean", (0.48145466, 0.4578275, 0.40821073))
    std = getattr(normalize, "std", (0.26862954, 0.26130258, 0.27577711))

    if policy == "weak":
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.85, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    if policy == "crop":
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    if policy == "color":
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    if policy == "preprocess":
        return preprocess
    raise ValueError(f"Unknown augmentation policy: {policy}")


class MultiViewImageDataset(Dataset):
    def __init__(self, dataframe, transform, views: int):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.views = views

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["imagepath_fixed"]).convert("RGB")
        views = torch.stack([self.transform(image) for _ in range(self.views)], dim=0)
        return views, int(row["label_num"]), row["imagepath_fixed"]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


@torch.inference_mode()
def extract_augmented_features(
    dataframe,
    clip_model,
    transform,
    *,
    views: int,
    device: str,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    seed: int,
):
    print(
        "building augmentation dataloader:",
        f"samples={len(dataframe)}",
        f"views={views}",
        f"batch_size={batch_size}",
        f"effective_clip_batch={batch_size * views}",
        f"num_workers={num_workers}",
        flush=True,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        MultiViewImageDataset(dataframe, transform, views),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    print("dataloader batches:", len(loader), flush=True)

    features_mean = []
    features_views = []
    labels = []
    paths = []
    print("starting augmented feature extraction loop", flush=True)
    for batch_idx, (images, batch_labels, batch_paths) in enumerate(tqdm(loader)):
        if batch_idx == 0:
            print("first batch loaded:", tuple(images.shape), flush=True)
        batch_size_current = images.shape[0]
        flat_images = images.flatten(0, 1).to(device, non_blocking=True)
        if batch_idx == 0:
            print("first batch flattened for CLIP:", tuple(flat_images.shape), flush=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device == "cuda")):
            flat_features = clip_model.encode_image(flat_images)
        if batch_idx == 0:
            print("first batch encoded:", tuple(flat_features.shape), flush=True)
        elif batch_idx % 25 == 0:
            print("processed batches:", batch_idx, "/", len(loader), flush=True)
        flat_features = F.normalize(flat_features.float(), dim=-1)
        view_features = flat_features.view(batch_size_current, views, -1)
        mean_features = F.normalize(view_features.mean(dim=1), dim=-1)
        features_mean.append(mean_features.cpu())
        features_views.append(view_features.cpu())
        labels.append(batch_labels.cpu())
        paths.extend(batch_paths)

    return torch.cat(features_mean), torch.cat(features_views), torch.cat(labels), paths


def save_augmented_feature_file(
    output_path: Path,
    *,
    features: torch.Tensor,
    features_views: torch.Tensor,
    labels: torch.Tensor,
    paths: list[str],
    metadata: dict,
    store_views: bool,
) -> None:
    extra = {"features_views": features_views} if store_views else {}
    save_feature_file(
        output_path,
        features,
        labels,
        paths,
        metadata={**metadata, **extra},
    )
    if store_views:
        print("features_views:", tuple(features_views.shape))


def output_name(*, split: str, level: int | None, corruption: str | None, policy: str, views: int) -> str:
    if corruption:
        return f"ffpp_{split}_level{level}_{corruption}_aug_{policy}_{views}views_features.pt"
    return f"ffpp_{split}_aug_{policy}_{views}views_features.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="/kaggle/input/datasets/jamestashvik/deepfakebench/deepfakebench_dataset.csv")
    parser.add_argument("--deepfakebench-root", default="/kaggle/input/datasets/jamestashvik/deepfakebench/DeepFakeBench")
    parser.add_argument("--corruption-root", default="/kaggle/input/ffpp-corruption-level-2")
    parser.add_argument("--corruption-root-map", nargs="*")
    parser.add_argument("--split-root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--corruptions", nargs="+", default=list(DEFAULT_CORRUPTIONS))
    parser.add_argument("--include-clean", action="store_true")
    parser.add_argument("--output-dir", default="/kaggle/working/ffpp_augmented_test_features")
    parser.add_argument("--augmentation-policy", choices=["weak", "crop", "color", "preprocess"], default="weak")
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--store-views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=64)
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

    print("preparing base FF++ split dataframe:", args.split, flush=True)
    base_df = prepare_ffpp_split_dataframe(
        csv_path=args.csv_path,
        deepfakebench_root=args.deepfakebench_root,
        split_name=args.split,
        split_root=args.split_root,
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
        corruption_level_root = resolve_corruption_level_root(
            corruption=corruption,
            level=args.level,
            default_root=args.corruption_root,
            root_map=root_map,
        )
        df = remap_to_corruption_root(
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
            split=args.split,
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
        print(f"\nExtracting augmented FF++ {args.split} | {label}", flush=True)
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
                "dataset_name": "FaceForensics++",
                "split_name": args.split,
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
