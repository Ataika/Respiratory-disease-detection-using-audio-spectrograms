"""
External cross-domain validation of the tuned ResNet50 on SPRSound.

Unlike BRACETS (binary), SPRSound already uses the same unified 4-class
label scheme (normal/crackle/wheeze/both) as ICBHI, so no label remapping
is needed here — this is a direct multiclass transfer test: does a model
trained only on ICBHI generalize to a completely different pediatric
respiratory-sound dataset (SPRSound, SJTU) with no retraining?

Usage:
    MPLCONFIGDIR=results/.mplconfig venv/bin/python scripts/eval_spr_sound_external.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.parsers.spr_sound import parse_spr_sound
from src.data.preprocessing import load_and_preprocess
from src.models.cnn_baseline import ResNetBaseline

SPR_SOUND_PATH = "data/raw/datasets/SPRSound-main/Classification"
CHECKPOINT = "results/checkpoints/final_resnet50_tuned_best.pth"
OUT_DIR = PROJECT_ROOT / "results" / "external"
SAMPLE_RATE = 22050
DURATION = 5.0
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


def load_model() -> ResNetBaseline:
    model = ResNetBaseline(num_classes=4, pretrained=False)
    state = torch.load(PROJECT_ROOT / CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Parsing SPRSound...", flush=True)
    records = parse_spr_sound(SPR_SOUND_PATH)
    print(f"SPRSound recordings: {len(records)}", flush=True)

    model = load_model()
    print("Model loaded. Running inference (first 5s window per recording)...",
          flush=True)

    y_true: list[int] = []
    y_pred: list[int] = []

    for i, rec in enumerate(records):
        try:
            x = load_and_preprocess(
                file_path=rec["wav_path"],
                offset=0.0,
                duration=DURATION,
                sample_rate=SAMPLE_RATE,
                branch="cnn",
            ).unsqueeze(0).to(DEVICE)
        except Exception as exc:  # skip unreadable files, keep eval robust
            print(f"  skip {rec['wav_path']}: {exc}", flush=True)
            continue

        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0].cpu()
        pred = int(probs.argmax().item())

        y_true.append(int(rec["label"]))
        y_pred.append(pred)

        if (i + 1) % 300 == 0:
            print(f"  processed {i + 1}/{len(records)}", flush=True)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()
    report_text = classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3], target_names=CLASS_NAMES,
        digits=4, zero_division=0,
    )

    report = {
        "checkpoint": CHECKPOINT,
        "n_recordings_evaluated": len(y_true),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix_rows_true_cols_pred": cm,
        "class_names": CLASS_NAMES,
        "note": "ICBHI-trained ResNet50, no retraining, no label remapping "
                "(SPRSound already uses the unified 4-class scheme); "
                "first 5s window per recording.",
    }
    (OUT_DIR / "sprsound_resnet50_tuned.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "SPRSound external validation (ICBHI-trained ResNet50, no retraining)",
        f"Checkpoint: {CHECKPOINT}",
        f"Recordings evaluated: {len(y_true)}",
        "",
        f"Accuracy:  {acc:.4f}",
        f"Macro F1:  {macro_f1:.4f}",
        "",
        "Confusion matrix (rows=true, cols=pred) "
        "[normal, crackle, wheeze, both]:",
    ]
    for row in cm:
        lines.append(f"  {row}")
    lines += ["", report_text]

    (OUT_DIR / "sprsound_resnet50_tuned.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("SPRSound external (4-class)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    for r in range(4):
        for c in range(4):
            ax.text(c, r, str(cm[r][c]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sprsound_resnet50_tuned_confusion.png", dpi=180)
    plt.close(fig)

    print("\n".join(lines), flush=True)
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
