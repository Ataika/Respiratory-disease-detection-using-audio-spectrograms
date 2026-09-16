"""Audio self-supervised backbones for respiratory sound classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


AUDIO_MAE_HF_ID = "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m"


class AudioMAEBaseline(nn.Module):
    """
    AudioMAE backbone with a classification head for respiratory sound classes.

    This wrapper uses a timm-compatible port of the AudioMAE ViT-B/16 encoder
    pretrained on AudioSet-2M with masked autoencoding.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            AUDIO_MAE_HF_ID,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """
        Freeze all pretrained backbone parameters and keep only the classifier head trainable.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.head.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze the full AudioMAE model.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True


def build_audio_ssl_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build an audio self-supervised model by name.
    """
    model_name = model_name.lower()

    if model_name == "audiomae":
        return AudioMAEBaseline(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(f"Unsupported audio SSL model: {model_name}")
