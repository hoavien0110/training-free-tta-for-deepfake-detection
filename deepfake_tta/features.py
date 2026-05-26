from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, preprocess):
        self.df = dataframe.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["imagepath_fixed"]).convert("RGB")
        return self.preprocess(image), int(row["label_num"]), row["imagepath_fixed"]


def prepare_deepfakebench_dataframe(
    csv_path: str | Path,
    dataset_name: str,
    deepfakebench_root: str | Path,
    replacement_root: str | Path | None = None,
    original_root: str | Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["datasetname"] == dataset_name].copy()
    df["label_num"] = df["label"].map({"REAL": 0, "FAKE": 1}).astype(int)
    df["imagepath_fixed"] = df["imagepath"].astype(str).str.replace(
        "../input/deepfakebench",
        str(deepfakebench_root),
        regex=False,
    )

    if replacement_root:
        source_root = str(original_root or deepfakebench_root)
        df["imagepath_fixed"] = df["imagepath_fixed"].str.replace(
            source_root,
            str(replacement_root),
            regex=False,
        )

    return df.reset_index(drop=True)


def create_clip_model(model_name: str, pretrained: str, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    return model.to(device).eval(), preprocess


@torch.no_grad()
def extract_features(
    dataframe: pd.DataFrame,
    clip_model,
    preprocess,
    device: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    loader = DataLoader(
        ImageDataset(dataframe, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    paths: list[str] = []

    for images, batch_labels, batch_paths in tqdm(loader):
        images = images.to(device, non_blocking=True)
        batch_features = clip_model.encode_image(images)
        features.append(F.normalize(batch_features, dim=-1).cpu())
        labels.append(batch_labels.cpu())
        paths.extend(batch_paths)

    return torch.cat(features), torch.cat(labels), paths


def save_feature_file(
    output_path: str | Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    paths: list[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    payload = {"features": features, "labels": labels}
    if paths is not None:
        payload["paths"] = paths
    if metadata:
        payload.update(metadata)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"saved to: {output_path}")
