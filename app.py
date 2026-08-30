"""Local Gradio demo for the deepfake face detector.

Run with: python app.py
Then open the printed local URL and upload a face photo.
"""

import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataset import get_transforms  # noqa: E402
from model import build_model  # noqa: E402
from train import get_device  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FINETUNED_CHECKPOINT = OUTPUT_DIR / "checkpoints" / "best_model_finetuned.pt"
BASE_CHECKPOINT = OUTPUT_DIR / "checkpoints" / "best_model.pt"

CHECKPOINT_PATH = FINETUNED_CHECKPOINT if FINETUNED_CHECKPOINT.exists() else BASE_CHECKPOINT

device = get_device()
model = build_model(freeze_backbone=True).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
transform = get_transforms("test")

print(f"Loaded checkpoint: {CHECKPOINT_PATH}")


def predict(image):
    if image is None:
        return {}
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        prob_real = torch.sigmoid(logit).item()
    return {"real": prob_real, "fake": 1 - prob_real}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a face photo"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Deepfake Face Detector",
    description=(
        "ResNet18 fine-tuned on the 140k-real-and-fake-faces dataset "
        "(real photos vs. StyleGAN-generated fakes). "
        f"Using checkpoint: {CHECKPOINT_PATH.name}. "
        "See the repo's README/MODEL_CARD for real evaluation metrics "
        "and known limitations (e.g. not tested on non-StyleGAN fakes "
        "or video deepfakes)."
    ),
)

if __name__ == "__main__":
    demo.launch()
