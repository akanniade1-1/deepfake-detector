import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import FaceDataset
from model import build_model
from train import BATCH_SIZE, get_device

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(OUTPUT_DIR / "checkpoints" / "best_model.pt"),
        help="Path to a model checkpoint (.pt) to evaluate.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix for output filenames, e.g. 'finetuned' -> "
        "metrics_finetuned.json, confusion_matrix_finetuned.png, etc. "
        "Leave empty to use the default (unsuffixed) filenames.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    metrics_path = OUTPUT_DIR / f"metrics{suffix}.json"
    confusion_plot_path = OUTPUT_DIR / "plots" / f"confusion_matrix{suffix}.png"
    roc_plot_path = OUTPUT_DIR / "plots" / f"roc_curve{suffix}.png"

    device = get_device()
    print(f"Using device: {device}")

    test_ds = FaceDataset("test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"test samples: {len(test_ds)}")

    model = build_model(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(float)

    metrics = {
        "checkpoint": args.checkpoint,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "n_test_samples": len(all_labels),
    }

    print(json.dumps(metrics, indent=2))
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    confusion_plot_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["fake", "real"])
    disp.plot()
    plt.title("Confusion Matrix (test set)")
    plt.savefig(confusion_plot_path)
    plt.close()

    RocCurveDisplay.from_predictions(all_labels, all_probs)
    plt.title("ROC Curve (test set)")
    plt.savefig(roc_plot_path)
    plt.close()

    print(f"Saved plots to {confusion_plot_path} and {roc_plot_path}")


if __name__ == "__main__":
    main()
