from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import TrainConfig, run_icbhi_baseline


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the ICBHI baseline experiment.
    """
    parser = argparse.ArgumentParser(
        description="Train a CNN baseline on the ICBHI dataset."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b3"],
        help="CNN baseline model to train.",
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
        default="results/baseline_results.csv",
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

    return parser.parse_args()


def save_training_plots(
    history,
    model_name: str,
    checkpoint_path: str,
) -> tuple[Path, Path]:
    """
    Save learning curves and the final validation confusion matrix.
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
    matrix = history.val_confusion_matrices[-1]

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(f"{model_name} Validation Confusion Matrix")
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


def append_result_summary(
    history,
    args: argparse.Namespace,
    device: str,
) -> Path:
    """
    Append the current experiment summary to a CSV file.
    """
    results_path = PROJECT_ROOT / args.results_csv
    results_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "model_name": args.model_name,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "best_epoch": history.best_epoch,
        "best_val_loss": f"{history.best_val_loss:.4f}",
        "final_train_loss": f"{history.train_loss[-1]:.4f}",
        "final_train_acc": f"{history.train_acc[-1]:.4f}",
        "final_train_macro_f1": f"{history.train_f1[-1]:.4f}",
        "final_val_loss": f"{history.val_loss[-1]:.4f}",
        "final_val_acc": f"{history.val_acc[-1]:.4f}",
        "final_val_macro_f1": f"{history.val_f1[-1]:.4f}",
        "checkpoint_path": args.checkpoint_path,
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
    )

    final_confusion_matrix = history.val_confusion_matrices[-1]
    curves_path, confusion_path = save_training_plots(
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
    print(f"Results CSV: {results_csv_path}")
    print(f"Best epoch: {history.best_epoch}")
    print(f"Best val loss: {history.best_val_loss:.4f}")
    print(f"Final train loss: {history.train_loss[-1]:.4f}")
    print(f"Final train acc: {history.train_acc[-1]:.4f}")
    print(f"Final train macro F1: {history.train_f1[-1]:.4f}")
    print(f"Final val loss: {history.val_loss[-1]:.4f}")
    print(f"Final val acc: {history.val_acc[-1]:.4f}")
    print(f"Final val macro F1: {history.val_f1[-1]:.4f}")
    print("Final val confusion matrix:")
    for row in final_confusion_matrix:
        print(row)


if __name__ == "__main__":
    main()




