"""
Export per-sample predictions + probabilities on the ICBHI validation set
for a trained model, so downstream scripts (statistical significance
tests, ICBHI Score, AUC-ROC, ECE) can work from a single cached array
instead of re-running inference each time.

The validation split is always create_splits(seed=42) on parse_icbhi(...)
— the same underlying breathing-cycle samples regardless of which model's
branch/sample-rate is used to preprocess them, which is what makes
per-sample paired comparisons (McNemar, DeLong) between models valid.

Usage:
    MPLCONFIGDIR=results/.mplconfig venv/bin/python scripts/export_val_predictions.py \
        --model-name resnet50 \
        --checkpoint results/checkpoints/final_resnet50_tuned_best.pth \
        --out results/predictions/resnet50_tuned_val.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_splits
from src.data.parsers.icbhi import parse_icbhi
from src.data.preprocessing import load_and_preprocess
from src.models.cnn_baseline import build_cnn_model
from src.models.audio_ssl import build_audio_ssl_model

DATASET_PATH = (
    "./data/raw/datasets/archive/Respiratory_Sound_Database/"
    "Respiratory_Sound_Database"
)
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export val-set predictions/probabilities for a checkpoint."
    )
    parser.add_argument("--model-name", required=True,
                        choices=["resnet50", "efficientnet_b3", "audiomae"])
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    return parser.parse_args()


def build_model(model_name: str, checkpoint: str) -> torch.nn.Module:
    if model_name in ("resnet50", "efficientnet_b3"):
        model = build_cnn_model(model_name=model_name, num_classes=4, pretrained=False)
    else:
        model = build_audio_ssl_model(model_name=model_name, num_classes=4, pretrained=False)
    state = torch.load(PROJECT_ROOT / checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model


def branch_config(model_name: str) -> tuple[str, int, float]:
    if model_name == "audiomae":
        return "audiomae", 16000, 10.0
    return "cnn", 22050, 5.0


def main() -> None:
    args = parse_args()
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Parsing ICBHI...", flush=True)
    samples = parse_icbhi(DATASET_PATH)
    _, val_samples = create_splits(samples, val_ratio=0.15, seed=42)
    print(f"Validation samples: {len(val_samples)}", flush=True)

    branch, sample_rate, duration = branch_config(args.model_name)
    model = build_model(args.model_name, args.checkpoint)
    print(f"Model loaded on {DEVICE}. branch={branch} sr={sample_rate} dur={duration}",
          flush=True)

    targets: list[int] = []
    probs_all: list[np.ndarray] = []

    with torch.no_grad():
        for i, sample in enumerate(val_samples):
            offset = float(sample.get("start", 0.0))
            x = load_and_preprocess(
                file_path=sample["wav_path"],
                offset=offset,
                duration=duration,
                sample_rate=sample_rate,
                branch=branch,
            ).unsqueeze(0).to(DEVICE)

            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            probs_all.append(probs)
            targets.append(int(sample["label"]))

            if (i + 1) % 200 == 0:
                print(f"  processed {i + 1}/{len(val_samples)}", flush=True)

    targets_arr = np.array(targets, dtype=np.int64)
    probs_arr = np.stack(probs_all, axis=0)  # (N, 4)
    preds_arr = probs_arr.argmax(axis=1)

    np.savez(
        out_path,
        targets=targets_arr,
        preds=preds_arr,
        probs=probs_arr,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
    )

    acc = float((preds_arr == targets_arr).mean())
    print(f"\nAccuracy: {acc:.4f}", flush=True)
    print(f"Saved predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
