"""Parser for the BRACETS respiratory sound dataset."""

from __future__ import annotations

import csv
from pathlib import Path


HEALTHY_LABELS = {
    "healthy",
}


def _map_bracets_diagnosis(raw_diagnosis: str) -> int:
    """
    Map BRACETS diagnosis into a binary label.

    0 = healthy
    1 = pathological
    """
    diagnosis = raw_diagnosis.strip().lower()

    if diagnosis in HEALTHY_LABELS:
        return 0

    return 1


def _load_bracets_metadata(metadata_path: Path) -> dict[str, int]:
    """
    Read BRACETS metadata and return patient_id -> binary label.
    """
    patient_labels: dict[str, int] = {}

    with metadata_path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = str(row["SubjectID"]).strip()
            diagnosis = str(row["Diagnosis"]).strip()
            patient_labels[subject_id] = _map_bracets_diagnosis(diagnosis)

    return patient_labels


def parse_bracets(dataset_path: str | Path) -> list[dict]:
    """
    Parse BRACETS into one record per audio file.

    BRACETS is used as an unseen binary evaluation set:
    0 = healthy
    1 = pathological
    """
    dataset_root = Path(dataset_path)
    data_dir = dataset_root / "Data"
    metadata_path = dataset_root / "Metadata" / "Metadata.txt"

    patient_labels = _load_bracets_metadata(metadata_path)
    records: list[dict] = []

    for patient_dir in sorted(data_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        label = patient_labels.get(patient_id)

        if label is None:
            continue

        for wav_path in sorted(patient_dir.rglob("*.wav")):
            records.append(
                {
                    "dataset": "bracets",
                    "split": "unseen",
                    "patient_id": patient_id,
                    "recording_id": wav_path.stem,
                    "wav_path": str(wav_path),
                    "label": label,
                }
            )

    return records
