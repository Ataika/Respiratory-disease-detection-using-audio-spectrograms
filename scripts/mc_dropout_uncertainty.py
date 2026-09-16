"""
MC Dropout uncertainty estimation for the tuned ResNet50 baseline on ICBHI.

At inference time we keep BatchNorm frozen (eval mode) but force the
classifier-head Dropout layer back into train mode, so every forward pass
samples a different dropout mask. Running N stochastic forward passes per
validation sample gives:

  - a Monte-Carlo mean prediction (averaged softmax probabilities)
  - predictive entropy of that mean distribution (higher = more uncertain)
  - whether high uncertainty actually correlates with wrong predictions,
    which is the whole point of MC Dropout for a medical screening tool
    ("I'm not sure -> see a doctor" only works if uncertainty is informative)

Usage:
    MPLCONFIGDIR=results/.mplconfig venv/bin/python scripts/mc_dropout_uncertainty.py
"""
from __future__ import annotations

import json
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
from src.data.preprocessing import load_and_preprocess
from src.models.cnn_baseline import ResNetBaseline

CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
DATASET_PATH = (
    "./data/raw/datasets/archive/Respiratory_Sound_Database/"
    "Respiratory_Sound_Database"
)
CHECKPOINT = "results/checkpoints/final_resnet50_tuned_best.pth"
OUT_DIR = PROJECT_ROOT / "results" / "uncertainty"
SAMPLE_RATE = 22050
DURATION = 5.0
N_PASSES = 30
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def load_model() -> ResNetBaseline:
    model = ResNetBaseline(num_classes=4, pretrained=False)
    state = torch.load(PROJECT_ROOT / CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Keep the model in eval mode but force Dropout layers back to train
    mode, so BatchNorm statistics stay frozen while dropout stays stochastic.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def input_tensor(sample: dict) -> torch.Tensor:
    offset = float(sample.get("start", 0.0))
    x = load_and_preprocess(
        file_path=sample["wav_path"],
        offset=offset,
        duration=DURATION,
        sample_rate=SAMPLE_RATE,
        branch="cnn",
    )
    return x.unsqueeze(0)


def predictive_entropy(mean_probs: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.sum(mean_probs * np.log(mean_probs + eps)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Parsing ICBHI...", flush=True)
    samples = parse_icbhi(DATASET_PATH)
    _, val_samples = create_splits(samples, val_ratio=0.15, seed=42)
    print(f"Validation samples: {len(val_samples)}", flush=True)

    model = load_model()
    enable_mc_dropout(model)
    print(f"Model loaded. Running {N_PASSES} MC-Dropout passes per sample.",
          flush=True)

    records: list[dict] = []
    with torch.no_grad():
        for i, sample in enumerate(val_samples):
            x = input_tensor(sample).to(DEVICE)
            # Repeat the same input N_PASSES times along the batch dim so a
            # single forward call gives N_PASSES independent dropout masks
            # (each row samples its own mask) instead of N_PASSES sequential
            # calls — mathematically identical, far fewer MPS dispatches.
            x_repeated = x.repeat(N_PASSES, 1, 1, 1)
            logits = model(x_repeated)
            probs_stack = F.softmax(logits, dim=1).cpu().numpy()  # (N, 4)

            mean_probs = probs_stack.mean(axis=0)
            std_probs = probs_stack.std(axis=0)
            pred = int(mean_probs.argmax())
            true = int(sample["label"])
            entropy = predictive_entropy(mean_probs)

            records.append({
                "true": true,
                "pred": pred,
                "correct": pred == true,
                "entropy": entropy,
                "mean_confidence": float(mean_probs[pred]),
                "std_top_class": float(std_probs[pred]),
            })

            if (i + 1) % 200 == 0:
                print(f"  processed {i + 1}/{len(val_samples)}", flush=True)

    entropies = np.array([r["entropy"] for r in records])
    correct_mask = np.array([r["correct"] for r in records])

    entropy_correct = entropies[correct_mask]
    entropy_wrong = entropies[~correct_mask]

    summary = {
        "checkpoint": CHECKPOINT,
        "n_passes": N_PASSES,
        "n_val_samples": len(records),
        "accuracy": round(float(correct_mask.mean()), 4),
        "mean_entropy_correct": round(float(entropy_correct.mean()), 4),
        "mean_entropy_wrong": round(float(entropy_wrong.mean()), 4)
            if len(entropy_wrong) else None,
        "n_wrong": int((~correct_mask).sum()),
    }

    per_class = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_records = [r for r in records if r["true"] == class_idx]
        if not class_records:
            continue
        class_entropies = np.array([r["entropy"] for r in class_records])
        per_class[class_name] = {
            "n": len(class_records),
            "mean_entropy": round(float(class_entropies.mean()), 4),
        }
    summary["per_class_entropy"] = per_class

    (OUT_DIR / "mc_dropout_resnet50_tuned.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "MC Dropout uncertainty — tuned ResNet50 on ICBHI validation set",
        f"Checkpoint: {CHECKPOINT}",
        f"Passes per sample: {N_PASSES}",
        f"Validation samples: {len(records)}",
        "",
        f"Accuracy: {summary['accuracy']:.4f}",
        f"Mean predictive entropy | correct predictions:   "
        f"{summary['mean_entropy_correct']:.4f}",
        f"Mean predictive entropy | incorrect predictions: "
        f"{summary['mean_entropy_wrong']:.4f}"
        if summary["mean_entropy_wrong"] is not None else
        "Mean predictive entropy | incorrect predictions: n/a",
        "",
        "Per-class mean entropy:",
    ]
    for name, stats in per_class.items():
        lines.append(f"  {name}: {stats['mean_entropy']:.4f} (n={stats['n']})")

    is_informative = (
        summary["mean_entropy_wrong"] is not None
        and summary["mean_entropy_wrong"] > summary["mean_entropy_correct"]
    )
    lines += [
        "",
        "Uncertainty is " + ("informative" if is_informative else "NOT clearly informative")
        + " for this model: incorrect predictions have "
        + ("higher" if is_informative else "similar or lower")
        + " predictive entropy than correct ones.",
    ]
    (OUT_DIR / "mc_dropout_resnet50_tuned.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, entropies.max() + 1e-6, 30)
    ax.hist(entropy_correct, bins=bins, alpha=0.6, label="correct", color="#2f7d4f")
    ax.hist(entropy_wrong, bins=bins, alpha=0.6, label="incorrect", color="#a83f34")
    ax.set_xlabel("Predictive entropy (MC Dropout, N=%d)" % N_PASSES)
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty: correct vs incorrect predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mc_dropout_entropy_hist.png", dpi=180)
    plt.close(fig)

    print("\n".join(lines), flush=True)
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
