"""Dataset utilities for respiratory sound classification."""
from __future__ import annotations

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from src.data.preprocessing import apply_spec_augment, load_and_preprocess

def create_splits(
    samples: list[dict],
    val_ratio: float = 0.15,
    seed: int = 42,
)-> tuple[list[dict], list[dict]]:
    """
    Split samples into train and validation sets at patient level.

    This prevents leakage where recordings from the same patient
    appear in both train and validation.
    """
    random.seed(seed)
    samples_by_patient = defaultdict(list)
    for sample in samples:
        samples_by_patient[sample["patient_id"]].append(sample)

    patient_ids = list(samples_by_patient.keys())
    random.shuffle(patient_ids)

    n_val = max(1, int(len(patient_ids) * val_ratio))
    val_patient_ids = set(patient_ids[:n_val])

    train_samples: list[dict] = []
    val_samples: list[dict] = []

    for patient_id , patient_samples in samples_by_patient.items():
        if patient_id in val_patient_ids:
            val_samples.extend(patient_samples)
        else:
            train_samples.extend(patient_samples)
    return train_samples, val_samples

class RespDataset(Dataset):
    """
    Unified dataset wrapper for respiratory sound samples.
    """

    def __init__(
        self,
        samples: list[dict],
        branch: str = "cnn",
        sample_rate: int = 22050,
        duration: float = 5.0,
        use_spec_augment: bool = False,
    ) -> None:
        self.samples = samples
        self.branch = branch
        self.sample_rate = sample_rate
        self.duration = duration
        self.use_spec_augment = use_spec_augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        # When segment annotations exist, start from the annotated offset.
        offset = float(sample.get("start", 0.0))

        x = load_and_preprocess(
            file_path=sample["wav_path"],
            offset=offset,
            duration=self.duration,
            sample_rate=self.sample_rate,
            branch=self.branch,
        )

        if self.use_spec_augment and self.branch == "cnn":
            x = apply_spec_augment(x)

        y = torch.tensor(sample["label"], dtype=torch.long)

        return x, y
