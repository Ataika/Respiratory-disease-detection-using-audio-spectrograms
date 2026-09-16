"""CNN baselines for respiratory sound classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNetBaseline(nn.Module):
    """
    ResNet50 baseline for 4-class respiratory sound classification.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetBaseline(nn.Module):
    """
    EfficientNet-B3 baseline for 4-class respiratory sound classification.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_cnn_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Build a CNN baseline by name.
    """
    model_name = model_name.lower()

    if model_name == "resnet50":
        return ResNetBaseline(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if model_name == "efficientnet_b3":
        return EfficientNetBaseline(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    raise ValueError(f"Unsupported CNN model: {model_name}")
