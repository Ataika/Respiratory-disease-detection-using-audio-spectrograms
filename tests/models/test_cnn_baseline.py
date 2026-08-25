import torch
import torch.nn as nn

from src.models.cnn_baseline import (
    EfficientNetBaseline,
    ResNetBaseline,
    build_cnn_model,
)


def test_resnet_baseline_forward_pass():
    model = ResNetBaseline(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 128, 216)

    y = model(x)

    assert isinstance(model, nn.Module)
    assert y.shape == (2, 4)


def test_efficientnet_baseline_forward_pass():
    model = EfficientNetBaseline(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 128, 216)

    y = model(x)

    assert isinstance(model, nn.Module)
    assert y.shape == (2, 4)


def test_build_cnn_model_returns_resnet():
    model = build_cnn_model("resnet50", pretrained=False)

    assert isinstance(model, ResNetBaseline)


def test_build_cnn_model_returns_efficientnet():
    model = build_cnn_model("efficientnet_b3", pretrained=False)

    assert isinstance(model, EfficientNetBaseline)
