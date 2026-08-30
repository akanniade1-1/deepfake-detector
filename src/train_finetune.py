"""Continue training from the frozen-head checkpoint, this time unfreezing
the full ResNet18 backbone for full fine-tuning at a lower learning rate."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FaceDataset
from model import build_model
from train import BATCH_SIZE, get_device, run_epoch

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
BASE_CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints" / "best_model.pt"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints" / "best_model_finetuned.pt"
HISTORY_PATH = OUTPUT_DIR / "history_finetuned.json"
LOSS_PLOT_PATH = OUTPUT_DIR / "plots" / "loss_curve_finetuned.png"

EPOCHS = 4
LR = 1e-4


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_ds = FaceDataset("train")
    valid_ds = FaceDataset("valid")
    print(f"train samples: {len(train_ds)}, valid samples: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model(freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(BASE_CHECKPOINT_PATH, map_location=device))
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, valid_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{EPOCHS} - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> saved new best fine-tuned checkpoint (val_acc={val_acc:.4f})")

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    epochs_range = range(1, EPOCHS + 1)
    plt.figure()
    plt.plot(epochs_range, history["train_loss"], label="train loss")
    plt.plot(epochs_range, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fine-tuning Loss (full backbone unfrozen)")
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH)
    print(f"Saved loss curve to {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
