"""
Grad-CAM for the tuned ResNet50 baseline on ICBHI.

Produces side-by-side figures (mel spectrogram | Grad-CAM overlay) for
representative correct and incorrect validation predictions, including a
`both`-class example. This is the single explainability method required by the
frozen experimental design.

Usage:
    MPLCONFIGDIR=results/.mplconfig venv/bin/python scripts/gradcam_resnet50.py

Light on memory: audio is loaded per sample on demand, runs on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_splits
from src.data.parsers.icbhi import parse_icbhi
from src.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_and_preprocess,
)
from src.models.cnn_baseline import ResNetBaseline

CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
DATASET_PATH = (
    "./data/raw/datasets/archive/Respiratory_Sound_Database/"
    "Respiratory_Sound_Database"
)
CHECKPOINT = "results/checkpoints/final_resnet50_tuned_best.pth"
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "gradcam"
SAMPLE_RATE = 22050
DURATION = 5.0
SCAN_LIMIT = 250  # cap how many val samples we probe when selecting examples


def load_model() -> ResNetBaseline:
    model = ResNetBaseline(num_classes=4, pretrained=False)
    state = torch.load(PROJECT_ROOT / CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def input_tensor(sample: dict) -> torch.Tensor:
    """Build the CNN input (3, 128, T) for one sample."""
    offset = float(sample.get("start", 0.0))
    x = load_and_preprocess(
        file_path=sample["wav_path"],
        offset=offset,
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        branch="cnn",
    )
    return x.unsqueeze(0)  # (1, 3, 128, T)


def display_mel(x: torch.Tensor) -> np.ndarray:
    """Recover the [0,1] mel image from channel 0 (undo ImageNet norm)."""
    ch0 = x[0, 0].cpu().numpy()
    return ch0 * IMAGENET_STD[0] + IMAGENET_MEAN[0]


def grad_cam(model: ResNetBaseline, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Compute a Grad-CAM heatmap (H, W in mel resolution) for class_idx."""
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    target_layer = model.backbone.layer4[-1]

    def fwd_hook(_m, _i, output):
        activations["v"] = output

    def bwd_hook(_m, _gi, grad_output):
        gradients["v"] = grad_output[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        logits = model(x)
        model.zero_grad()
        logits[0, class_idx].backward()

        acts = activations["v"][0]        # (C, h, w)
        grads = gradients["v"][0]         # (C, h, w)
        weights = grads.mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * acts).sum(dim=0)  # (h, w)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
        cam = F.interpolate(
            cam, size=(x.shape[2], x.shape[3]), mode="bilinear",
            align_corners=False,
        )
        return cam[0, 0].detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def save_example(model, sample, true_idx, pred_idx, x, tag) -> Path:
    mel = display_mel(x)
    cam = grad_cam(model, x, pred_idx)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Mel spectrogram")
    axes[0].set_xlabel("Time frames")
    axes[0].set_ylabel("Mel bins")

    axes[1].imshow(mel, origin="lower", aspect="auto", cmap="gray")
    axes[1].imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.5)
    axes[1].set_title("Grad-CAM overlay (predicted class)")
    axes[1].set_xlabel("Time frames")
    axes[1].set_ylabel("Mel bins")

    correct = "correct" if true_idx == pred_idx else "INCORRECT"
    fig.suptitle(
        f"[{tag}] true: {CLASS_NAMES[true_idx]}  |  "
        f"pred: {CLASS_NAMES[pred_idx]}  ({correct})",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = OUT_DIR / f"gradcam_{tag}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Parsing ICBHI...", flush=True)
    samples = parse_icbhi(DATASET_PATH)
    _, val_samples = create_splits(samples, val_ratio=0.15, seed=42)
    print(f"Validation samples: {len(val_samples)}", flush=True)

    model = load_model()
    print("Model loaded.", flush=True)

    # Predict over a bounded scan and remember each sample's prediction.
    scanned: list[tuple[int, int, int]] = []  # (val_index, true, pred)
    for i, sample in enumerate(val_samples[:SCAN_LIMIT]):
        x = input_tensor(sample)
        with torch.no_grad():
            pred = int(model(x).argmax(dim=1).item())
        scanned.append((i, int(sample["label"]), pred))
    print(f"Scanned {len(scanned)} samples for example selection.", flush=True)

    chosen: dict[str, int] = {}

    def pick(tag, predicate):
        for idx, true, pred in scanned:
            if idx in chosen.values():
                continue
            if predicate(true, pred):
                chosen[tag] = idx
                return

    # correct normal, correct abnormal, a clear misclassification, both class
    pick("correct_normal", lambda t, p: t == 0 and p == 0)
    pick("correct_abnormal", lambda t, p: t in (1, 2) and t == p)
    pick("incorrect", lambda t, p: t != p and t != 0)
    pick("both_correct", lambda t, p: t == 3 and p == 3)
    if "both_correct" not in chosen:
        pick("both_any", lambda t, p: t == 3)

    saved = []
    for tag, idx in chosen.items():
        sample = val_samples[idx]
        true_idx = int(sample["label"])
        x = input_tensor(sample)
        with torch.no_grad():
            pred_idx = int(model(x).argmax(dim=1).item())
        path = save_example(model, sample, true_idx, pred_idx, x, tag)
        saved.append(path)
        print(
            f"  {tag}: true={CLASS_NAMES[true_idx]} pred={CLASS_NAMES[pred_idx]} "
            f"-> {path.name}",
            flush=True,
        )

    print(f"\nSaved {len(saved)} Grad-CAM figures to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
