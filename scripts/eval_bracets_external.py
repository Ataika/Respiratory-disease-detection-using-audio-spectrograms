"""
External cross-domain validation of the tuned ResNet50 on BRACETS.

The model is trained on ICBHI four-class acoustic patterns and evaluated, with
NO retraining, on the unseen BRACETS dataset as a binary screening task:

    model prediction `normal` (0)        -> healthy (0)
    model prediction crackle/wheeze/both -> pathological (1)

This measures domain shift: did the model learn transferable respiratory
patterns, or ICBHI-specific artifacts? Inference only, CPU, light on memory.

Usage:
    MPLCONFIGDIR=results/.mplconfig venv/bin/python scripts/eval_bracets_external.py
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
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.parsers.bracets import parse_bracets
from src.data.preprocessing import load_and_preprocess
from src.models.cnn_baseline import ResNetBaseline

BRACETS_PATH = "data/raw/datasets/BRACETS"
CHECKPOINT = "results/checkpoints/final_resnet50_tuned_best.pth"
OUT_DIR = PROJECT_ROOT / "results" / "external"
SAMPLE_RATE = 22050
DURATION = 5.0
BINARY_NAMES = ["healthy", "pathological"]


def load_model() -> ResNetBaseline:
    model = ResNetBaseline(num_classes=4, pretrained=False)
    state = torch.load(PROJECT_ROOT / CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Parsing BRACETS...", flush=True)
    records = parse_bracets(BRACETS_PATH)
    print(f"BRACETS recordings: {len(records)}", flush=True)

    model = load_model()
    print("Model loaded. Running inference (first 5s window per recording)...",
          flush=True)

    y_true: list[int] = []
    y_pred: list[int] = []
    abnormal_prob: list[float] = []

    for i, rec in enumerate(records):
        try:
            x = load_and_preprocess(
                file_path=rec["wav_path"],
                offset=0.0,
                duration=DURATION,
                sample_rate=SAMPLE_RATE,
                branch="cnn",
            ).unsqueeze(0)
        except Exception as exc:  # skip unreadable files, keep eval robust
            print(f"  skip {rec['wav_path']}: {exc}", flush=True)
            continue

        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
        pred4 = int(probs.argmax().item())
        pred_bin = 0 if pred4 == 0 else 1
        p_abnormal = float(1.0 - probs[0].item())

        y_true.append(int(rec["label"]))
        y_pred.append(pred_bin)
        abnormal_prob.append(p_abnormal)

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(records)}", flush=True)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    try:
        auroc = roc_auc_score(y_true, abnormal_prob)
    except ValueError:
        auroc = float("nan")

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    report = {
        "checkpoint": CHECKPOINT,
        "n_recordings_evaluated": len(y_true),
        "n_healthy": n_neg,
        "n_pathological": n_pos,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "auroc_abnormal": round(auroc, 4),
        "confusion_matrix_rows_true_cols_pred": cm,
        "binary_label_names": BINARY_NAMES,
        "note": "ICBHI-trained ResNet50, no retraining; first 5s window per "
                "recording; normal->healthy, crackle/wheeze/both->pathological.",
    }
    (OUT_DIR / "bracets_resnet50_tuned.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "BRACETS external validation (ICBHI-trained ResNet50, no retraining)",
        f"Checkpoint: {CHECKPOINT}",
        f"Recordings evaluated: {len(y_true)} "
        f"(healthy={n_neg}, pathological={n_pos})",
        "",
        f"Accuracy:     {acc:.4f}",
        f"Macro F1:     {macro_f1:.4f}",
        f"AUROC:        {auroc:.4f}",
        "",
        "Confusion matrix (rows=true, cols=pred) [healthy, pathological]:",
        f"  {cm[0]}",
        f"  {cm[1]}",
    ]
    (OUT_DIR / "bracets_resnet50_tuned.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("BRACETS external (binary)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(BINARY_NAMES); ax.set_yticklabels(BINARY_NAMES)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r][c]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bracets_resnet50_tuned_confusion.png", dpi=180)
    plt.close(fig)

    print("\n".join(lines), flush=True)
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
