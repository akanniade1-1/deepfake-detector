"""Show sample real/fake face images from the test set with the trained
model's predictions, for a visual sanity check (analogous to Fig. 1 in the
original report, but with real data and real model predictions)."""

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import get_transforms
from model import build_model
from train import get_device

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
_FINETUNED = OUTPUT_DIR / "checkpoints" / "best_model_finetuned.pt"
CHECKPOINT_PATH = _FINETUNED if _FINETUNED.exists() else OUTPUT_DIR / "checkpoints" / "best_model.pt"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "real_vs_fake" / "real-vs-fake" / "test"
SAMPLES_PLOT_PATH = OUTPUT_DIR / "plots" / "sample_predictions.png"

N_PER_CLASS = 4
SEED = 7


def pick_samples():
    rng = random.Random(SEED)
    samples = []
    for label_name, label in (("real", 1), ("fake", 0)):
        class_dir = DATA_ROOT / label_name
        paths = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        chosen = rng.sample(paths, N_PER_CLASS)
        samples.extend((p, label) for p in chosen)
    rng.shuffle(samples)
    return samples


def main():
    device = get_device()
    model = build_model(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    transform = get_transforms("test")
    samples = pick_samples()

    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    for ax, (path, true_label) in zip(axes.flat, samples):
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(tensor)
            prob_real = torch.sigmoid(logit).item()

        pred_label = 1 if prob_real > 0.5 else 0
        confidence = prob_real if pred_label == 1 else 1 - prob_real

        true_str = "real" if true_label == 1 else "fake"
        pred_str = "real" if pred_label == 1 else "fake"
        correct = "✓" if pred_label == true_label else "✗"
        color = "green" if pred_label == true_label else "red"

        ax.imshow(image)
        ax.set_title(
            f"true: {true_str} | pred: {pred_str} ({confidence:.0%}) {correct}",
            color=color,
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle("Sample test images with model predictions", fontsize=13)
    fig.tight_layout()
    SAMPLES_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAMPLES_PLOT_PATH)
    print(f"Saved sample prediction grid to {SAMPLES_PLOT_PATH}")


if __name__ == "__main__":
    main()
