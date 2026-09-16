"""DataLoader utilities for respiratory sound classification."""
from __future__ import annotations
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.dataset import RespDataset


def make_weighted_sampler(samples: list[dict]) -> WeightedRandomSampler:
    """
    Create a weighted sampler to reduce class imbalance.
    """
    labels = np.array(
        [sample["label"] for sample in samples],
        dtype=np.int64,
    )

    class_counts = np.bincount(labels, minlength=4)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloader(
    samples: list[dict],
    branch: str = "cnn",
    sample_rate: int = 22050,
    duration: float = 5.0,
    batch_size: int = 8,
    shuffle: bool = False,
    use_weighted_sampler: bool = False,
    use_spec_augment: bool = False,
) -> DataLoader:
    """
    Build a dataloader from a list of samples.
    """
    dataset = RespDataset(
        samples=samples,
        branch=branch,
        sample_rate=sample_rate,
        duration=duration,
        use_spec_augment=use_spec_augment,
    )

    sampler = None
    if use_weighted_sampler:
        sampler = make_weighted_sampler(samples)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
    )