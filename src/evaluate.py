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
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints" / "best_model.pt"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CONFUSION_PLOT_PATH = OUTPUT_DIR / "plots" / "confusion_matrix.png"
ROC_PLOT_PATH = OUTPUT_DIR / "plots" / "roc_curve.png"


def main():
    device = get_device()
    print(f"Using device: {device}")

    test_ds = FaceDataset("test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"test samples: {len(test_ds)}")

    model = build_model(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
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
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "n_test_samples": len(all_labels),
    }

    print(json.dumps(metrics, indent=2))
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    CONFUSION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["real", "fake"])
    disp.plot()
    plt.title("Confusion Matrix (test set)")
    plt.savefig(CONFUSION_PLOT_PATH)
    plt.close()

    RocCurveDisplay.from_predictions(all_labels, all_probs)
    plt.title("ROC Curve (test set)")
    plt.savefig(ROC_PLOT_PATH)
    plt.close()

    print(f"Saved plots to {CONFUSION_PLOT_PATH} and {ROC_PLOT_PATH}")


if __name__ == "__main__":
    main()
