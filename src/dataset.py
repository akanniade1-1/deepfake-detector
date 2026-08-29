import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST_PATH = DATA_DIR / "subset_manifest.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FaceDataset(Dataset):
    """Reads (path, label, split) rows from the manifest CSV for one split."""

    def __init__(self, split: str, manifest_path: Path = MANIFEST_PATH):
        self.split = split
        self.transform = get_transforms(split)
        self.samples = []
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.samples.append((row["path"], int(row["label"])))
        if not self.samples:
            raise RuntimeError(
                f"No samples found for split={split!r} in {manifest_path}. "
                "Did you run download_data.py?"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
