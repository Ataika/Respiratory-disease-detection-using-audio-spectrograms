"""Parser for the SPRSound respiratory sound dataset."""

from __future__ import annotations

import json
from pathlib import Path


SPR_LABEL_MAP = {
    "normal": 0,
    "crackle": 1,
    "wheeze": 2,
    "wheeze&crackle": 3,
    "wheeze_crackle": 3,
    "crackle&wheeze": 3,
    "das": 1,
    "cas": 2,
    "cas & das": 3,
    "cas&das": 3,
}


def _normalize_spr_label(raw_label: str) -> int | None:
    """
    Normalize SPRSound annotation into the unified label scheme.
    """
    label = raw_label.strip().lower()

    if label == "poor quality":
        return None

    return SPR_LABEL_MAP.get(label, 0)


def parse_spr_sound(dataset_path: str | Path) -> list[dict]:
    """
    Parse SPRSound into one record per audio file.

    This dataset is used as an unseen evaluation set.
    """
    dataset_root = Path(dataset_path)
    wav_dir = dataset_root / "train_classification_wav"
    json_dir = dataset_root / "train_classification_json"

    records: list[dict] = []

    for wav_path in sorted(wav_dir.glob("*.wav")):
        recording_id = wav_path.stem
        json_path = json_dir / f"{recording_id}.json"

        if not json_path.exists():
            continue

        with json_path.open("r", encoding="utf-8", errors="ignore") as handle:
            metadata = json.load(handle)

        raw_label = str(metadata.get("record_annotation", "normal")).strip()
        label = _normalize_spr_label(raw_label)

        if label is None:
            continue

        records.append(
            {
                "dataset": "spr_sound",
                "split": "unseen",
                "patient_id": recording_id,
                "recording_id": recording_id,
                "wav_path": str(wav_path),
                "label": label,
                "raw_label": raw_label,
            }
        )

    return records
