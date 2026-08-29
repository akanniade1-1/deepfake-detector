import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FaceDataset
from model import build_model

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints" / "best_model.pt"
HISTORY_PATH = OUTPUT_DIR / "history.json"
LOSS_PLOT_PATH = OUTPUT_DIR / "plots" / "loss_curve.png"

BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_ds = FaceDataset("train")
    valid_ds = FaceDataset("valid")
    print(f"train samples: {len(train_ds)}, valid samples: {len(valid_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model(freeze_backbone=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

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
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    epochs_range = range(1, EPOCHS + 1)
    plt.figure()
    plt.plot(epochs_range, history["train_loss"], label="train loss")
    plt.plot(epochs_range, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH)
    print(f"Saved loss curve to {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
