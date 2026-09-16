from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import TrainConfig, run_icbhi_baseline


CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the ICBHI baseline experiment.
    """
    parser = argparse.ArgumentParser(
        description="Train a baseline model on the ICBHI dataset."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b3", "audiomae"],
        help="Baseline model to train.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data/raw/datasets/archive/Respiratory_Sound_Database/Respiratory_Sound_Database",
        help="Path to the ICBHI dataset root.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio at patient level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the patient-level split.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="How often to print epoch-level training logs.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="results/checkpoints/icbhi_best_model.pth",
        help="Path to save the best checkpoint.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="results/final_experiment_registry.csv",
        help="CSV file where experiment summaries will be appended.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Number of epochs without validation loss improvement"
             "before early stopping.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=1,
        help="Number of epochs without validation loss "
             "improvement before reducing learning rate "
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Factor used to reduce the learning rate when validation "
             "loss plateaus"
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class-weighted cross-entropy loss during training"
    )
    parser.add_argument(
        "--disable-weighted-sampler",
        action="store_true",
        help="Disable weighted sampling and use regular dataloader sampling.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing value for cross-entropy loss.",
    )
    parser.add_argument(
        "--use-spec-augment",
        action="store_true",
        help="Apply light SpecAugment to CNN training spectrograms.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the pretrained backbone and train only the classifier head.",
    )

    return parser.parse_args()


def save_training_plots(
    history,
    model_name: str,
    checkpoint_path: str,
) -> tuple[Path, Path]:
    """
    Save learning curves and the best validation confusion matrix.
    """
    checkpoint = Path(checkpoint_path)
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    stem = checkpoint.stem
    curves_path = figures_dir / f"{stem}_curves.png"
    confusion_path = figures_dir / f"{stem}_confusion_matrix.png"

    epochs = list(range(1, len(history.train_loss) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history.train_loss, marker="o", label="Train loss")
    axes[0].plot(epochs, history.val_loss, marker="o", label="Val loss")
    axes[0].set_title(f"{model_name} Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.train_acc, marker="o", label="Train acc")
    axes[1].plot(epochs, history.val_acc, marker="o", label="Val acc")
    axes[1].plot(epochs, history.train_f1, marker="o", label="Train macro F1")
    axes[1].plot(epochs, history.val_f1, marker="o", label="Val macro F1")
    axes[1].set_title(f"{model_name} Accuracy and Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    class_names = ["normal", "crackle", "wheeze", "both"]
    matrix = history.best_val_confusion_matrix
    if not matrix:
        matrix = history.val_confusion_matrices[-1]

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(f"{model_name} Best Validation Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(confusion_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return curves_path, confusion_path


def save_per_class_report(
    history,
    model_name: str,
    checkpoint_path: str,
) -> tuple[Path, Path]:
    """
    Save best-epoch class-level precision, recall, F1, support, and prediction counts.
    """
    reports_dir = PROJECT_ROOT / "results" / "per_class"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(checkpoint_path).stem
    text_path = reports_dir / f"{stem}_per_class.txt"
    json_path = reports_dir / f"{stem}_per_class.json"

    targets = history.best_val_targets
    predictions = history.best_val_predictions

    if not targets or not predictions:
        targets = history.final_val_targets
        predictions = history.final_val_predictions

    if not targets or not predictions:
        raise ValueError("Validation targets and predictions are required.")

    labels = list(range(len(CLASS_NAMES)))
    text_report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    dict_report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    prediction_distribution = {
        class_name: predictions.count(class_index)
        for class_index, class_name in enumerate(CLASS_NAMES)
    }
    target_distribution = {
        class_name: targets.count(class_index)
        for class_index, class_name in enumerate(CLASS_NAMES)
    }

    with text_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Model: {model_name}\n")
        handle.write(f"Checkpoint: {checkpoint_path}\n\n")
        handle.write(f"Evaluation epoch: {history.best_epoch}\n")
        handle.write(f"Best val loss: {history.best_val_loss:.4f}\n")
        handle.write(f"Best val acc: {history.best_val_acc:.4f}\n")
        handle.write(f"Best val macro F1: {history.best_val_f1:.4f}\n\n")
        handle.write(text_report)
        handle.write("\n\nTarget distribution:\n")
        for class_name, count in target_distribution.items():
            handle.write(f"{class_name}: {count}\n")
        handle.write("\nPrediction distribution:\n")
        for class_name, count in prediction_distribution.items():
            handle.write(f"{class_name}: {count}\n")

    report_payload = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "evaluation_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "best_val_acc": history.best_val_acc,
        "best_val_macro_f1": history.best_val_f1,
        "classification_report": dict_report,
        "target_distribution": target_distribution,
        "prediction_distribution": prediction_distribution,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

    return text_path, json_path


def append_result_summary(
    history,
    args: argparse.Namespace,
    device: str,
) -> Path:
    """
    Append one experiment summary to the clean experiment registry.

    The registry stores both metrics and configuration flags so each run can be
    interpreted later without guessing which training options were active.
    """
    results_path = PROJECT_ROOT / args.results_csv
    results_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    row = {
        "run_id": run_id,
        "model_name": args.model_name,
        "device": device,
        "epochs_requested": args.epochs,
        "epochs_trained": len(history.train_loss),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "use_early_stopping": args.early_stopping_patience > 0,
        "early_stopping_patience": args.early_stopping_patience,
        "use_scheduler": True,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "use_class_weights": args.use_class_weights,
        "use_weighted_sampler": not args.disable_weighted_sampler,
        "label_smoothing": args.label_smoothing,
        "use_spec_augment": args.use_spec_augment,
        "freeze_backbone": args.freeze_backbone,
        "task_mode": "multiclass_4",
        "best_epoch": history.best_epoch,
        "best_val_loss": f"{history.best_val_loss:.4f}",
        "best_val_acc": f"{history.best_val_acc:.4f}",
        "best_val_macro_f1": f"{history.best_val_f1:.4f}",
        "final_train_loss": f"{history.train_loss[-1]:.4f}",
        "final_train_acc": f"{history.train_acc[-1]:.4f}",
        "final_train_macro_f1": f"{history.train_f1[-1]:.4f}",
        "final_val_loss": f"{history.val_loss[-1]:.4f}",
        "final_val_acc": f"{history.val_acc[-1]:.4f}",
        "final_val_macro_f1": f"{history.val_f1[-1]:.4f}",
        "checkpoint_path": args.checkpoint_path,
        "notes": "",
    }

    fieldnames = list(row.keys())
    write_header = not results_path.exists()

    with results_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return results_path


def main() -> None:
    """
    Run the ICBHI baseline experiment.
    """
    args = parse_args()
    # `--seed` previously only drove the patient-level split (via
    # create_splits). Model weight init, WeightedRandomSampler draws, and
    # SpecAugment masks all read torch's global RNG, which was never seeded
    # here — so two runs with the same --seed were not actually
    # reproducible. Seeding it makes --seed control the *entire* run.
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    config = TrainConfig(
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        log_every=args.log_every,
        checkpoint_path=args.checkpoint_path,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        use_class_weights=args.use_class_weights,
        label_smoothing=args.label_smoothing,
        use_spec_augment=args.use_spec_augment,
        freeze_backbone=args.freeze_backbone,
    )

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    model, history = run_icbhi_baseline(
        model_name=args.model_name,
        dataset_path=str(dataset_path),
        config=config,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        use_weighted_sampler = not args.disable_weighted_sampler,
    )

    best_confusion_matrix = history.best_val_confusion_matrix
    if not best_confusion_matrix:
        best_confusion_matrix = history.val_confusion_matrices[-1]
    curves_path, confusion_path = save_training_plots(
        history=history,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
    )
    per_class_text_path, per_class_json_path = save_per_class_report(
        history=history,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
    )
    results_csv_path = append_result_summary(
        history=history,
        args=args,
        device=device,
    )

    print("\nTraining finished.")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Best checkpoint: {args.checkpoint_path}")
    print(f"Curves plot: {curves_path}")
    print(f"Confusion matrix plot: {confusion_path}")
    print(f"Per-class report: {per_class_text_path}")
    print(f"Per-class JSON: {per_class_json_path}")
    print(f"Results CSV: {results_csv_path}")
    print(f"Best epoch: {history.best_epoch}")
    print(f"Best val loss: {history.best_val_loss:.4f}")
    print(f"Best val acc: {history.best_val_acc:.4f}")
    print(f"Best val macro F1: {history.best_val_f1:.4f}")
    print(f"Final train loss: {history.train_loss[-1]:.4f}")
    print(f"Final train acc: {history.train_acc[-1]:.4f}")
    print(f"Final train macro F1: {history.train_f1[-1]:.4f}")
    print(f"Final val loss: {history.val_loss[-1]:.4f}")
    print(f"Final val acc: {history.val_acc[-1]:.4f}")
    print(f"Final val macro F1: {history.val_f1[-1]:.4f}")
    print("Best val confusion matrix:")
    for row in best_confusion_matrix:
        print(row)


if __name__ == "__main__":
    main()

