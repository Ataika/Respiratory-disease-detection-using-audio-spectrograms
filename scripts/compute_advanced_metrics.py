"""
Compute the metrics/tests required by the design spec but missing from the
core training pipeline: the official ICBHI Score, in-domain multiclass
AUC-ROC, ECE calibration (+ reliability diagram), bootstrap confidence
intervals for macro F1, and pairwise McNemar's / DeLong's tests between
whichever models have exported prediction caches (see
scripts/export_val_predictions.py).

Works with 1..N prediction caches — metrics that need only one model
(ICBHI Score, AUC-ROC, ECE, bootstrap CI) are computed for every cache
found; the pairwise tests (McNemar, DeLong) run over every pair.

Usage:
    MPLCONFIGDIR=results/.mplconfig venv/bin/python scripts/compute_advanced_metrics.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRED_DIR = PROJECT_ROOT / "results" / "predictions"
OUT_DIR = PROJECT_ROOT / "results" / "advanced_metrics"
CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
ABNORMAL_LABELS = {1, 2, 3}
N_BOOTSTRAP = 2000
N_ECE_BINS = 10
RNG_SEED = 42


# ---------------------------------------------------------------------------
# ICBHI Score
# ---------------------------------------------------------------------------

def icbhi_score(targets: np.ndarray, preds: np.ndarray) -> dict:
    """Official ICBHI 2017 challenge metric: Se = recall on abnormal cycles
    (crackle/wheeze/both pooled), Sp = recall on normal cycles,
    Score = (Se + Sp) / 2.
    """
    normal_mask = targets == 0
    abnormal_mask = ~normal_mask

    sp = float((preds[normal_mask] == targets[normal_mask]).mean()) if normal_mask.any() else float("nan")
    se = float((preds[abnormal_mask] == targets[abnormal_mask]).mean()) if abnormal_mask.any() else float("nan")
    score = (se + sp) / 2

    return {"sensitivity_se": se, "specificity_sp": sp, "icbhi_score": score}


# ---------------------------------------------------------------------------
# AUC-ROC (macro, one-vs-rest, in-domain multiclass)
# ---------------------------------------------------------------------------

def multiclass_auc(targets: np.ndarray, probs: np.ndarray) -> dict:
    present = sorted(set(targets.tolist()))
    try:
        macro_auc = float(
            roc_auc_score(targets, probs, multi_class="ovr", average="macro",
                          labels=list(range(4)))
        )
    except ValueError as exc:
        macro_auc = float("nan")
        print(f"  [warn] macro AUC failed: {exc}")

    per_class = {}
    for c in range(4):
        y_true_bin = (targets == c).astype(int)
        if y_true_bin.sum() == 0 or y_true_bin.sum() == len(targets):
            per_class[CLASS_NAMES[c]] = None
            continue
        per_class[CLASS_NAMES[c]] = float(roc_auc_score(y_true_bin, probs[:, c]))

    return {"macro_auc_ovr": macro_auc, "per_class_auc_ovr": per_class}


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error) + reliability diagram
# ---------------------------------------------------------------------------

def expected_calibration_error(
    targets: np.ndarray, preds: np.ndarray, probs: np.ndarray, n_bins: int = N_ECE_BINS
) -> dict:
    confidences = probs[np.arange(len(preds)), preds]
    correct = (preds == targets).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_report = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi) if lo > 0 else (
            (confidences >= lo) & (confidences <= hi)
        )
        count = int(in_bin.sum())
        if count == 0:
            bins_report.append({"range": [round(float(lo), 2), round(float(hi), 2)],
                                "count": 0, "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = float(confidences[in_bin].mean())
        acc = float(correct[in_bin].mean())
        ece += (count / len(preds)) * abs(avg_conf - acc)
        bins_report.append({
            "range": [round(float(lo), 2), round(float(hi), 2)],
            "count": count, "avg_confidence": round(avg_conf, 4), "accuracy": round(acc, 4),
        })

    return {"ece": float(ece), "n_bins": n_bins, "bins": bins_report}


def plot_reliability_diagram(ece_report: dict, model_name: str, out_path: Path) -> None:
    bins = [b for b in ece_report["bins"] if b["count"] > 0]
    mids = [(b["range"][0] + b["range"][1]) / 2 for b in bins]
    accs = [b["accuracy"] for b in bins]
    confs = [b["avg_confidence"] for b in bins]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.bar(mids, accs, width=1 / ece_report["n_bins"], edgecolor="black",
          alpha=0.7, label="accuracy per bin")
    ax.scatter(mids, confs, color="red", zorder=5, label="avg confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram — {model_name} (ECE={ece_report['ece']:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bootstrap CI for macro F1
# ---------------------------------------------------------------------------

def bootstrap_macro_f1_ci(
    targets: np.ndarray, preds: np.ndarray, n_bootstrap: int = N_BOOTSTRAP
) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    n = len(targets)
    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(targets[idx], preds[idx], average="macro", zero_division=0)

    point = float(f1_score(targets, preds, average="macro", zero_division=0))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {
        "point_estimate": point,
        "ci_95_low": float(lo),
        "ci_95_high": float(hi),
        "n_bootstrap": n_bootstrap,
        "bootstrap_std": float(scores.std()),
    }


# ---------------------------------------------------------------------------
# McNemar's test (paired, on per-sample correctness)
# ---------------------------------------------------------------------------

def mcnemar_test(targets: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    correct_a = preds_a == targets
    correct_b = preds_b == targets

    # b = A right, B wrong ; c = A wrong, B right (classic McNemar notation)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    n_discordant = b + c
    if n_discordant == 0:
        return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c,
                "statistic": 0.0, "p_value": 1.0, "note": "no discordant pairs"}

    # Exact binomial test is preferred for small discordant counts;
    # continuity-corrected chi-square otherwise (standard McNemar practice).
    if n_discordant < 25:
        from math import comb
        k = min(b, c)
        p_value = 0.0
        for i in range(0, k + 1):
            p_value += comb(n_discordant, i) * (0.5 ** n_discordant)
        p_value = min(1.0, 2 * p_value)
        statistic = None
    else:
        statistic = float((abs(b - c) - 1) ** 2 / (b + c))
        from math import erf, sqrt
        # chi-square(1) survival function via erf (chi2_1 = Z^2)
        z = statistic ** 0.5
        p_value = 1 - erf(z / sqrt(2))

    return {
        "b_a_right_b_wrong": b, "c_a_wrong_b_right": c,
        "statistic": statistic, "p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# DeLong's test (paired AUC comparison, binary normal-vs-abnormal reduction)
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance computation (Sun & Xu, 2014)."""
    n = preds_sorted.shape[1] - m
    positive = preds_sorted[:, :m]
    negative = preds_sorted[:, m:]
    k = preds_sorted.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(preds_sorted[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(
    targets_binary: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> dict:
    """Two-sided DeLong test that two correlated ROC-AUCs are equal."""
    order = np.argsort(-targets_binary, kind="mergesort")
    y = targets_binary[order]
    m = int(y.sum())  # positives
    preds_sorted = np.vstack([scores_a[order], scores_b[order]])

    aucs, cov = _fast_delong(preds_sorted, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
                "z": None, "p_value": None, "note": "non-positive variance"}

    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    from math import erf, sqrt
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return {
        "auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
        "z": float(z), "p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "targets": data["targets"],
        "preds": data["preds"],
        "probs": data["probs"],
        "model_name": str(data["model_name"]),
        "checkpoint": str(data["checkpoint"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    caches = sorted(PRED_DIR.glob("*.npz"))
    if not caches:
        print(f"No prediction caches found in {PRED_DIR}. "
              f"Run scripts/export_val_predictions.py first.")
        return

    print(f"Found {len(caches)} prediction cache(s): "
          f"{[c.name for c in caches]}\n")

    per_model_report: dict[str, dict] = {}
    for cache_path in caches:
        c = load_cache(cache_path)
        name = c["model_name"]
        targets, preds, probs = c["targets"], c["preds"], c["probs"]
        print(f"=== {name} ({cache_path.name}) ===")

        icbhi = icbhi_score(targets, preds)
        print(f"  ICBHI Score: Se={icbhi['sensitivity_se']:.4f} "
              f"Sp={icbhi['specificity_sp']:.4f} "
              f"Score={icbhi['icbhi_score']:.4f}")

        auc = multiclass_auc(targets, probs)
        print(f"  Macro AUC-ROC (OvR): {auc['macro_auc_ovr']:.4f}")

        ece = expected_calibration_error(targets, preds, probs)
        print(f"  ECE: {ece['ece']:.4f}")
        plot_reliability_diagram(ece, name, OUT_DIR / f"{name}_reliability_diagram.png")

        boot = bootstrap_macro_f1_ci(targets, preds)
        print(f"  Macro F1: {boot['point_estimate']:.4f} "
              f"95% CI [{boot['ci_95_low']:.4f}, {boot['ci_95_high']:.4f}]")

        per_model_report[name] = {
            "checkpoint": c["checkpoint"],
            "n_samples": int(len(targets)),
            "icbhi_score": icbhi,
            "auc_roc": auc,
            "ece": ece,
            "bootstrap_macro_f1": boot,
        }
        print()

    pairwise_report: dict[str, dict] = {}
    model_caches = {load_cache(p)["model_name"]: load_cache(p) for p in caches}
    for name_a, name_b in combinations(model_caches.keys(), 2):
        ca, cb = model_caches[name_a], model_caches[name_b]
        if not np.array_equal(ca["targets"], cb["targets"]):
            print(f"  [warn] {name_a} vs {name_b}: target arrays differ in length/order, "
                  f"skipping paired tests")
            continue

        key = f"{name_a}_vs_{name_b}"
        print(f"=== {key} ===")
        mcnemar = mcnemar_test(ca["targets"], ca["preds"], cb["preds"])
        print(f"  McNemar: b={mcnemar['b_a_right_b_wrong']} "
              f"c={mcnemar['c_a_wrong_b_right']} p={mcnemar['p_value']:.4f}")

        targets_bin = (ca["targets"] != 0).astype(int)  # abnormal=1, normal=0
        scores_a = 1.0 - ca["probs"][:, 0]
        scores_b = 1.0 - cb["probs"][:, 0]
        delong = delong_test(targets_bin, scores_a, scores_b)
        if delong.get("p_value") is not None:
            print(f"  DeLong (normal-vs-abnormal AUC): "
                  f"AUC_a={delong['auc_a']:.4f} AUC_b={delong['auc_b']:.4f} "
                  f"z={delong['z']:.4f} p={delong['p_value']:.4f}")
        else:
            print(f"  DeLong: {delong.get('note')}")

        pairwise_report[key] = {"mcnemar": mcnemar, "delong_normal_vs_abnormal": delong}
        print()

    full_report = {"per_model": per_model_report, "pairwise": pairwise_report}
    out_json = OUT_DIR / "advanced_metrics_report.json"
    out_json.write_text(json.dumps(full_report, indent=2, default=str), encoding="utf-8")
    print(f"Full report written to {out_json}")


if __name__ == "__main__":
    main()
