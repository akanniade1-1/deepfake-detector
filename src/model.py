import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(freeze_backbone: bool = True) -> nn.Module:
    """ResNet18 pretrained on ImageNet, with a single-logit binary head.

    Labels are 1=real, 0=fake (matches the dataset's own convention);
    use BCEWithLogitsLoss on the raw output.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)  # fc is always trainable
    return model
