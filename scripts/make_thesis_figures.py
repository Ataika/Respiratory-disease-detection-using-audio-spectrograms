"""
Generate the summary figures the thesis still needs:

  1. class_distribution.png       - ICBHI cycle counts per class (shows imbalance)
  2. model_comparison_macro_f1.png - best-epoch macro F1 per final model
  3. per_class_f1_comparison.png   - grouped per-class F1 across final models
  4. class_examples_spectrograms.png - one mel spectrogram example per class

Reads existing artifacts (per_class/*.json) and the ICBHI dataset. Light on
memory, but do NOT run while AudioMAE is training (8 GB machine). Run it once
after AudioMAE finishes:

    MPLCONFIGDIR=results/.mplconfig venv/bin/python scripts/make_thesis_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.parsers.icbhi import parse_icbhi
from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, load_and_preprocess

OUT_DIR = PROJECT_ROOT / "results" / "figures" / "thesis"
PER_CLASS = PROJECT_ROOT / "results" / "per_class"
CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
DATASET_PATH = (
    "./data/raw/datasets/archive/Respiratory_Sound_Database/"
    "Respiratory_Sound_Database"
)

# Final models to compare: label -> per_class json stem
MODELS = {
    "ResNet50\ntuned": "final_resnet50_tuned_best",
    "EfficientNet-B3\ntuned": "final_efficientnet_b3_tuned_best",
    "ResNet50\n+SpecAug": "final_resnet50_specaugment_best",
    "ResNet50\n+class-weighted": "final_resnet50_class_weighted_no_sampler_best",
    "AudioMAE\nno-sampler": "final_audiomae_no_sampler_best",
}


def load_report(stem: str) -> dict | None:
    path = PER_CLASS / f"{stem}_per_class.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fig_class_distribution(samples: list[dict]) -> None:
    counts = [sum(1 for s in samples if int(s["label"]) == i) for i in range(4)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(CLASS_NAMES, counts, color=["#4aa3ff", "#5be08a", "#ffb454", "#ff6b6b"])
    ax.set_title("ICBHI 2017 — class distribution (annotated segments)")
    ax.set_ylabel("Number of segments")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)
    print("  class_distribution.png", counts)


def fig_macro_f1() -> None:
    labels, scores = [], []
    for label, stem in MODELS.items():
        rep = load_report(stem)
        if rep is None:
            continue
        f1 = rep.get("best_val_macro_f1")
        if f1 is None:
            f1 = rep["classification_report"]["macro avg"]["f1-score"]
        labels.append(label)
        scores.append(float(f1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, scores, color="#4aa3ff")
    ax.set_title("Model comparison — best-epoch validation macro F1 (ICBHI 4-class)")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, max(scores) * 1.25 if scores else 1)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s, f"{s:.3f}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison_macro_f1.png", dpi=200)
    plt.close(fig)
    print("  model_comparison_macro_f1.png", [round(s, 3) for s in scores])


def fig_per_class_f1() -> None:
    model_labels, matrix = [], []
    for label, stem in MODELS.items():
        rep = load_report(stem)
        if rep is None:
            continue
        cr = rep["classification_report"]
        matrix.append([cr[c]["f1-score"] for c in CLASS_NAMES])
        model_labels.append(label.replace("\n", " "))
    if not matrix:
        return
    matrix = np.array(matrix)
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(model_labels)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, ml in enumerate(model_labels):
        ax.bar(x + i * width, matrix[i], width, label=ml)
    ax.set_title("Per-class F1 across final models (the `both` class is the hardest)")
    ax.set_ylabel("F1-score")
    ax.set_xticks(x + width * (len(model_labels) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_class_f1_comparison.png", dpi=200)
    plt.close(fig)
    print("  per_class_f1_comparison.png")


def fig_class_examples(samples: list[dict]) -> None:
    chosen: dict[int, dict] = {}
    for s in samples:
        lbl = int(s["label"])
        if lbl not in chosen:
            chosen[lbl] = s
        if len(chosen) == 4:
            break
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for lbl in range(4):
        ax = axes[lbl // 2][lbl % 2]
        s = chosen.get(lbl)
        if s is None:
            ax.set_visible(False)
            continue
        x = load_and_preprocess(
            file_path=s["wav_path"], offset=float(s.get("start", 0.0)),
            duration=5.0, sample_rate=22050, branch="cnn",
        )
        mel = x[0].numpy() * IMAGENET_STD[0] + IMAGENET_MEAN[0]
        ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"{CLASS_NAMES[lbl]}")
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
    fig.suptitle("Example mel spectrograms per class (ICBHI)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_examples_spectrograms.png", dpi=200)
    plt.close(fig)
    print("  class_examples_spectrograms.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Parsing ICBHI...", flush=True)
    samples = parse_icbhi(DATASET_PATH)
    print(f"Segments: {len(samples)}", flush=True)
    print("Generating figures:", flush=True)
    fig_class_distribution(samples)
    fig_macro_f1()
    fig_per_class_f1()
    fig_class_examples(samples)
    print(f"\nDone. Figures in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
