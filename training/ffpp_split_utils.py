from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json",
    "val": "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json",
    "test": "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json",
}


def resolve_split_json(split_name: str, dataset_root: str | Path, split_root: str | Path | None = None) -> Path:
    split_name = split_name.lower()
    candidates = []
    if split_root:
        candidates.append(Path(split_root) / f"{split_name}.json")

    dataset_root = Path(dataset_root)
    candidates.extend(
        [
            dataset_root / "splits" / f"{split_name}.json",
            dataset_root / "dataset" / "splits" / f"{split_name}.json",
            dataset_root / "DeepFakeBench" / "FaceForensics++" / "splits" / f"{split_name}.json",
            dataset_root / "FaceForensics++" / "splits" / f"{split_name}.json",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in dataset_root.rglob(f"{split_name}.json"):
        if "split" in str(candidate).lower():
            return candidate

    out_dir = Path("/kaggle/working/ffpp_splits")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}.json"
    if not out_path.exists():
        print(f"split json not found locally, downloading {split_name}.json")
        try:
            urllib.request.urlretrieve(SPLIT_URLS[split_name], out_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Cannot find or download {split_name}.json. "
                "Set SPLIT_ROOT to a folder containing train.json, val.json, test.json."
            ) from exc
    return out_path


def load_split_pairs(split_name: str, dataset_root: str | Path, split_root: str | Path | None = None):
    split_path = resolve_split_json(split_name, dataset_root, split_root)
    with split_path.open("r") as f:
        pairs = json.load(f)
    print("split:", split_name, "| path:", split_path, "| pairs:", len(pairs))
    return [(str(a), str(b)) for a, b in pairs]


def _extract_ffpp_video_key(path: str) -> str:
    parts = Path(path).parts
    if "frames" in parts:
        idx = parts.index("frames")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(path).parent.name


def _is_original_path(path: str) -> bool:
    return "original_sequences" in path


def prepare_ffpp_split_dataframe(
    csv_path: str | Path,
    deepfakebench_root: str | Path,
    split_name: str,
    split_root: str | Path | None = None,
    dataset_name: str = "FaceForensics++",
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    deepfakebench_root = Path(deepfakebench_root)
    dataset_root = csv_path.parent
    pairs = load_split_pairs(split_name, dataset_root=dataset_root, split_root=split_root)
    video_ids = {vid for pair in pairs for vid in pair}
    pair_keys = {f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs}

    df = pd.read_csv(csv_path)
    df = df[df["datasetname"].eq(dataset_name)].copy()
    df["label_num"] = df["label"].map({"REAL": 0, "FAKE": 1}).astype(int)
    df["imagepath_fixed"] = df["imagepath"].astype(str).str.replace(
        "../input/deepfakebench",
        str(deepfakebench_root),
        regex=False,
    )
    df["video_key"] = df["imagepath_fixed"].map(_extract_ffpp_video_key)
    df["is_original"] = df["imagepath_fixed"].map(_is_original_path)

    original_mask = df["is_original"] & df["video_key"].isin(video_ids)
    fake_pair_mask = (~df["is_original"]) & df["video_key"].isin(pair_keys)
    fake_component_mask = (~df["is_original"]) & df["video_key"].str.split("_").map(
        lambda x: len(x) >= 2 and x[0] in video_ids and x[1] in video_ids
    )
    out = df[original_mask | fake_pair_mask | fake_component_mask].reset_index(drop=True)
    print("split dataframe:", split_name, out.shape)
    print(out["label_num"].value_counts().rename(index={0: "REAL", 1: "FAKE"}))
    return out


class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, preprocess):
        self.df = dataframe.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["imagepath_fixed"]).convert("RGB")
        image = self.preprocess(image)
        return image, int(row["label_num"]), row["imagepath_fixed"]


def create_clip(model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, preprocess


@torch.inference_mode()
def extract_features(
    dataframe: pd.DataFrame,
    clip_model,
    preprocess,
    device: str,
    batch_size: int = 128,
    num_workers: int = 4,
    use_amp: bool = False,
):
    loader = DataLoader(
        ImageDataset(dataframe, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    features = []
    labels = []
    paths = []
    for images, batch_labels, batch_paths in tqdm(loader):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device == "cuda")):
            batch_features = clip_model.encode_image(images)
        features.append(F.normalize(batch_features.float(), dim=-1).cpu())
        labels.append(batch_labels.cpu())
        paths.extend(batch_paths)
    return torch.cat(features), torch.cat(labels), paths


def save_feature_file(
    output_path: str | Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    paths: list[str],
    metadata: dict,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"features": features, "labels": labels, "paths": paths, **metadata}
    torch.save(payload, output_path)
    print("saved:", output_path)
    print("features:", tuple(features.shape))
    print("labels:", tuple(labels.shape))
    print("counts [REAL, FAKE]:", torch.bincount(labels.long(), minlength=2).tolist())
