"""Parser for the SPRSound respiratory sound dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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

    Returns None (record skipped by the caller) for "poor quality" and
    for any annotation string not in SPR_LABEL_MAP, rather than silently
    defaulting to "normal" — an unrecognized label (typo, casing
    variant, or a category this map hasn't seen yet) is not evidence the
    recording is actually normal, and defaulting it that way would
    quietly corrupt ground truth for cross-domain evaluation metrics.
    """
    label = raw_label.strip().lower()

    if label == "poor quality":
        return None

    if label not in SPR_LABEL_MAP:
        logger.warning(
            "SPRSound: unrecognized record_annotation %r, skipping record "
            "(known labels: %s)", raw_label, sorted(SPR_LABEL_MAP)
        )
        return None

    return SPR_LABEL_MAP[label]


def parse_spr_sound(dataset_path: str | Path) -> list[dict]:
    """
    Parse SPRSound into one record per audio file.

    This dataset is used as an unseen evaluation set. Both the "train" and
    "valid" classification splits published by SPRSound are read here,
    since neither was used for training in this thesis — both are unseen
    data as far as our ICBHI-trained models are concerned.
    """
    dataset_root = Path(dataset_path)
    records: list[dict] = []

    for split_name in ("train_classification", "valid_classification"):
        wav_dir = dataset_root / f"{split_name}_wav"
        json_dir = dataset_root / f"{split_name}_json"

        if not wav_dir.exists():
            continue

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
                    "source_split": split_name,
                    "patient_id": recording_id,
                    "recording_id": recording_id,
                    "wav_path": str(wav_path),
                    "label": label,
                    "raw_label": raw_label,
                }
            )

    return records
