from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_splits
from src.data.loaders import create_dataloader
from src.data.parsers.icbhi import parse_icbhi


CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


def parse_args() -> argparse.Namespace:
    """
    Parse arguments for AudioMAE input diagnostics.
    """
    parser = argparse.ArgumentParser(
        description="Inspect ICBHI AudioMAE inputs before training."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data/raw/datasets/archive/Respiratory_Sound_Database/Respiratory_Sound_Database",
        help="Path to the ICBHI dataset root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used for the diagnostic dataloader.",
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
    return parser.parse_args()


def format_distribution(samples: list[dict]) -> dict[str, int]:
    """
    Count labels in a sample list and return readable class names.
    """
    counts = Counter(sample["label"] for sample in samples)
    return {
        class_name: counts.get(class_index, 0)
        for class_index, class_name in enumerate(CLASS_NAMES)
    }


def print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """
    Print basic tensor statistics for model input diagnostics.
    """
    print(f"{name}.shape: {tuple(tensor.shape)}")
    print(f"{name}.mean: {tensor.mean().item():.4f}")
    print(f"{name}.std: {tensor.std().item():.4f}")
    print(f"{name}.min: {tensor.min().item():.4f}")
    print(f"{name}.max: {tensor.max().item():.4f}")


def main() -> None:
    """
    Run AudioMAE input diagnostics on the ICBHI split.
    """
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    samples = parse_icbhi(str(dataset_path))
    train_samples, val_samples = create_splits(
        samples=samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("AudioMAE input diagnostics")
    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Train class distribution: {format_distribution(train_samples)}")
    print(f"Validation class distribution: {format_distribution(val_samples)}")

    dataloader = create_dataloader(
        samples=train_samples,
        branch="audiomae",
        sample_rate=16000,
        duration=10.0,
        batch_size=args.batch_size,
        shuffle=False,
        use_weighted_sampler=False,
    )

    batch_x, batch_y = next(iter(dataloader))
    print_tensor_stats("batch_x", batch_x)
    print(f"batch_y: {batch_y.tolist()}")
    print(f"batch_y class names: {[CLASS_NAMES[label] for label in batch_y.tolist()]}")


if __name__ == "__main__":
    main()
