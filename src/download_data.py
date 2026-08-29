"""Download the 140k Real and Fake Faces dataset from Kaggle and build a
manifest CSV covering the full dataset (labels matching the dataset's own
convention: 1 = real, 0 = fake)."""

import csv
import subprocess
import zipfile
from pathlib import Path

DATASET = "xhlulu/140k-real-and-fake-faces"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_PATH = DATA_DIR / "140k-real-and-fake-faces.zip"


def download_and_extract() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    root = DATA_DIR / "real_vs_fake" / "real-vs-fake"
    if root.exists():
        print(f"Dataset already extracted at {root}")
        return root

    if not ZIP_PATH.exists():
        print(f"Downloading {DATASET} via Kaggle API...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR)],
            check=True,
        )

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

    if not root.exists():
        raise RuntimeError(
            f"Expected extracted folder not found at {root}. "
            "Check the zip contents and adjust the path."
        )
    return root


def build_manifest(root: Path) -> None:
    manifest_path = DATA_DIR / "subset_manifest.csv"

    rows = []
    for split in ("train", "valid", "test"):
        split_dir = root / split
        for label_name, label in (("real", 1), ("fake", 0)):
            class_dir = split_dir / label_name
            paths = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
            for p in paths:
                rows.append((str(p), label, split))
            print(f"{split}/{label_name}: {len(paths)} images")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split"])
        writer.writerows(rows)

    print(f"Wrote manifest with {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    dataset_root = download_and_extract()
    build_manifest(dataset_root)
