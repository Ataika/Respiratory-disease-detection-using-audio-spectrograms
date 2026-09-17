"""Aggregate multi-seed ICBHI training logs into a robustness summary.

Reads the raw console logs saved under results/.mplconfig/log_seed_*.txt
(five seeds each for ResNet50 and EfficientNet-B3) and produces:
  - results/multiseed_results.csv        per-seed rows
  - results/MULTISEED_SUMMARY_RU.md      mean +/- std table and a short
                                          honest conclusion for Chapter 4.5

No training dependencies required (pure stdlib) so it can run before the
venv/torch install finishes.
"""
from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "results" / ".mplconfig"
OUT_CSV = PROJECT_ROOT / "results" / "multiseed_results.csv"
OUT_MD = PROJECT_ROOT / "results" / "MULTISEED_SUMMARY_RU.md"

# (model_name, seed, log filename)
RUNS = [
    ("resnet50", 42, "log_seed_resnet50_42.txt"),
    ("resnet50", 0, "log_seed_resnet50_0.txt"),
    ("resnet50", 1, "log_seed_resnet50_1.txt"),
    ("resnet50", 2, "log_seed_resnet50_2.txt"),
    ("resnet50", 3, "log_seed_resnet50_3.txt"),
    ("efficientnet_b3", 42, "log_seed_eff_42.txt"),
    ("efficientnet_b3", 0, "log_seed_eff_0.txt"),
    ("efficientnet_b3", 1, "log_seed_eff_1.txt"),
    ("efficientnet_b3", 2, "log_seed_efficientnet_b3_2.txt"),
    ("efficientnet_b3", 3, "log_seed_efficientnet_b3_3.txt"),
    ("audiomae", 42, "log_audiomae_overnight_main.txt"),
    ("audiomae", 0, "log_seed_audiomae_0.txt"),
    ("audiomae", 1, "log_seed_audiomae_1.txt"),
    ("audiomae", 2, "log_seed_audiomae_2.txt"),
    ("audiomae", 3, "log_seed_audiomae_3.txt"),
]

FIELDS = {
    "best_epoch": r"Best epoch:\s*(\d+)",
    "best_val_loss": r"Best val loss:\s*([\d.]+)",
    "best_val_acc": r"Best val acc:\s*([\d.]+)",
    "best_val_f1": r"Best val macro F1:\s*([\d.]+)",
    "final_val_acc": r"Final val acc:\s*([\d.]+)",
    "final_val_f1": r"Final val macro F1:\s*([\d.]+)",
}


def parse_log(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "Best val macro F1" not in text:
        return None  # failed/aborted run, e.g. a CLI-argument typo
    row: dict = {}
    for key, pattern in FIELDS.items():
        match = re.search(pattern, text)
        row[key] = float(match.group(1)) if match else None
    return row


def main() -> None:
    rows: list[dict] = []
    skipped: list[str] = []

    for model_name, seed, filename in RUNS:
        path = LOG_DIR / filename
        if not path.exists():
            skipped.append(f"{filename} (not found)")
            continue
        parsed = parse_log(path)
        if parsed is None:
            skipped.append(f"{filename} (no results — aborted run)")
            continue
        parsed["model_name"] = model_name
        parsed["seed"] = seed
        parsed["source_log"] = filename
        rows.append(parsed)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name", "seed", "best_epoch", "best_val_loss",
                "best_val_acc", "best_val_f1", "final_val_acc",
                "final_val_f1", "source_log",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model_name"], []).append(row)

    lines = [
        "# Multi-Seed Robustness Summary: ICBHI CNN Baselines",
        "",
        "## Назначение",
        "",
        "Этот файл агрегирует 5 независимых прогонов (seed = 42, 0, 1, 2, 3) "
        "каждой из двух CNN-моделей на ICBHI. Источник — сырые консольные "
        "логи в `results/.mplconfig/log_seed_*.txt`, до этого нигде не сведённые.",
        "",
        "## Сводная таблица (mean ± std по 5 seed)",
        "",
        "| Model | n | val acc (mean ± std) | val macro F1 (mean ± std) | "
        "min F1 | max F1 |",
        "|---|---|---|---|---|---|",
    ]

    conclusions = []
    for model_name, model_rows in by_model.items():
        accs = [r["best_val_acc"] for r in model_rows if r["best_val_acc"] is not None]
        f1s = [r["best_val_f1"] for r in model_rows if r["best_val_f1"] is not None]
        acc_mean, acc_std = statistics.mean(accs), statistics.pstdev(accs)
        f1_mean, f1_std = statistics.mean(f1s), statistics.pstdev(f1s)
        lines.append(
            f"| `{model_name}` | {len(model_rows)} | "
            f"{acc_mean:.4f} ± {acc_std:.4f} | "
            f"{f1_mean:.4f} ± {f1_std:.4f} | "
            f"{min(f1s):.4f} | {max(f1s):.4f} |"
        )
        conclusions.append((model_name, f1_mean, f1_std, min(f1s), max(f1s)))

    lines += [
        "",
        "## По каждому seed",
        "",
        "| Model | seed | best epoch | val acc | val macro F1 |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["model_name"], r["seed"])):
        lines.append(
            f"| `{row['model_name']}` | {row['seed']} | {row['best_epoch']} | "
            f"{row['best_val_acc']:.4f} | {row['best_val_f1']:.4f} |"
        )

    lines += ["", "## Честный вывод"]
    if len(conclusions) == 2:
        (m1, f1_a, std_a, min_a, max_a), (m2, f1_b, std_b, min_b, max_b) = conclusions
        spread_a = max_a - min_a
        spread_b = max_b - min_b
        gap = abs(f1_a - f1_b)
        lines += [
            "",
            f"- Разброс `macro F1` между seed внутри одной модели: "
            f"`{m1}` — {spread_a:.4f}, `{m2}` — {spread_b:.4f}.",
            f"- Разница средних `macro F1` между моделями: {gap:.4f}.",
        ]
        wider = max(spread_a, spread_b)
        if gap < wider:
            lines.append(
                "- **Разброс между seed одной модели больше (или сопоставим), "
                "чем разница средних между моделями.** Это значит, что "
                "заявление вида «модель X лучше модели Y» по одному "
                "прогону не выдерживает проверки на устойчивость — "
                "разница может объясняться случайной инициализацией, "
                "а не архитектурой."
            )
        else:
            lines.append(
                "- Разница между моделями превышает внутримодельный "
                "разброс — сравнение по средним значениям в этом случае "
                "статистически осмысленно."
            )
    if skipped:
        lines += ["", "## Пропущенные/повреждённые логи", ""]
        lines += [f"- `{s}`" for s in skipped]

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Parsed {len(rows)} runs across {len(by_model)} models.")
    if skipped:
        print(f"Skipped {len(skipped)}: {skipped}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
