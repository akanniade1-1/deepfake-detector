# Model Card: Deepfake Face Detector

## Model Description

A binary image classifier that predicts whether a face photo is a real
photograph or a GAN-generated fake. Built by fine-tuning a
ResNet18 (ImageNet-pretrained) with a single-logit binary head.

Two checkpoints are provided, representing two points on a compute/accuracy
tradeoff:

| Variant | Description | Checkpoint |
|---|---|---|
| Frozen-head | Only the final `Linear(512, 1)` layer is trained; the ResNet18 backbone stays at its ImageNet weights. | `outputs/checkpoints/best_model.pt` |
| Fine-tuned | Continued training from the frozen-head checkpoint with the full backbone unfrozen at a lower learning rate. | `outputs/checkpoints/best_model_finetuned.pt` |

## Intended Use

- **Intended**: a demonstration/portfolio project showing a real,
  reproducible deepfake-image detection pipeline — dataset, training,
  evaluation, and an interactive demo, all backed by runnable code.
- **Not intended**: production content moderation, legal/forensic evidence,
  or any deployment where a wrong classification has real consequences.
  See Limitations below for why.

## Training Data

[`xhlulu/140k-real-and-fake-faces`](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
on Kaggle — 140,000 face images:
- **Real**: photographs from [Flickr-Faces-HQ (FFHQ)](https://github.com/NVlabs/ffhq-dataset) (Karras et al., NVIDIA).
- **Fake**: StyleGAN-generated faces, also from the same dataset authors' pipeline.

Split (as shipped by the dataset, used in full — no subsampling):

| Split | Real | Fake | Total |
|---|---|---|---|
| Train | 50,000 | 50,000 | 100,000 |
| Valid | 10,000 | 10,000 | 20,000 |
| Test | 10,000 | 10,000 | 20,000 |

Label convention: **1 = real, 0 = fake** (matches the dataset's own
`label`/`label_str` columns).

## Training Procedure

- **Architecture**: `torchvision.models.resnet18(weights=IMAGENET1K_V1)`,
  final FC layer replaced with `Linear(512, 1)`.
- **Loss**: `BCEWithLogitsLoss`.
- **Preprocessing**: resize to 224x224, ImageNet mean/std normalization,
  random horizontal flip (train only).
- **Frozen-head stage**: Adam, lr=1e-3, batch size 32, 8 epochs, only the
  head's parameters trained.
- **Fine-tuning stage**: continued from the frozen-head best checkpoint,
  full backbone unfrozen, Adam, lr=1e-4, batch size 32, 4 epochs.
- **Hardware**: Apple M3 GPU via PyTorch's MPS backend.
- Best checkpoint in each stage selected by validation accuracy.

## Evaluation Results

Evaluated once on the full 20,000-image held-out test split
(`src/evaluate.py`). "Positive" class = 1 = real.

| Metric | Frozen-head | Fine-tuned |
|---|---|---|
| Accuracy | 0.808 | **0.993** |
| Precision | 0.818 | 0.990 |
| Recall | 0.792 | 0.997 |
| F1 | 0.804 | 0.993 |
| ROC-AUC | 0.889 | 0.9998 |

(See `outputs/metrics.json` and `outputs/metrics_finetuned.json` for the
exact numbers used above, and `outputs/plots/` for confusion matrices,
ROC curves, and sample predictions.)

## Limitations

- **In-distribution result**: fine-tuning the full backbone raised test
  accuracy from 80.8% to 99.3%, but all fakes (train and test) come from
  the same StyleGAN pipeline. This is a well-documented property of this
  particular dataset — a fully fine-tuned CNN can key in on that
  generator's specific artifacts — and is not evidence of general
  deepfake-detection capability. No cross-generator or cross-dataset
  generalization test has been performed.
- **Single generation method**: all fakes in the training/test data are
  StyleGAN-generated still images. The model has not been evaluated on
  other generators (diffusion models, other GAN variants) or on video
  deepfakes (face-swap, lip-sync), and should not be assumed to
  generalize to them.
- **No temporal modeling**: this is an image classifier only. It cannot
  detect deepfakes that rely on temporal/motion artifacts, which would
  require a video-based (e.g. frame-sequence) approach.
- **Single-machine training**: no federated or multi-institution
  training was performed; this is a standard centralized fine-tuning
  setup.
- **Dataset demographic coverage** is whatever FFHQ/StyleGAN happen to
  represent — not independently audited here for bias across age,
  gender, or ethnicity.

## Ethical Considerations

This repository is detection-only — it does not include code for
generating deepfakes. The classifier's purpose is defensive (helping
identify synthetic media), consistent with the broader goal of countering
misuse of deepfake technology in misinformation, fraud, and social
engineering contexts.
