"""Parser for the ICBHI 2017 respiratory sound dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DIAGNOSIS_MAP = {
    "Healthy": "normal",
    "URTI": "urti",
    "COPD": "copd",
    "Bronchiectasis": "bronchiectasis",
    "Pneumonia": "pneumonia",
    "Bronchiolitis": "bronchiolitis",
    "Asthma": "asthma",
    "LRTI": "lrti",
}

_FLAG_TO_LABEL = {
    (0, 0): 0,
    (1, 0): 1,
    (0, 1): 2,
    (1, 1): 3,
}


def _flags_to_label(crackle: int, wheeze: int) -> int:
    """Convert binary crackle/wheeze flags into the unified label scheme."""

    flags = (int(crackle), int(wheeze))
    if flags not in _FLAG_TO_LABEL:
        raise ValueError(
            "ICBHI labels must use binary crackle/wheeze flags, "
            f"got crackle={crackle}, wheeze={wheeze}."
        )
    return _FLAG_TO_LABEL[flags]


def _load_diagnosis_map(dataset_path: Path) -> dict[str, str]:
    diagnosis_csv = dataset_path / "patient_diagnosis.csv"
    diagnosis_df = pd.read_csv(
        diagnosis_csv,
        header=None,
        names=["patient_id", "diagnosis"],
    )

    diagnosis_map: dict[str, str] = {}
    for patient_id, diagnosis in zip(
        diagnosis_df["patient_id"], diagnosis_df["diagnosis"], strict=False
    ):
        diagnosis_map[str(patient_id)] = DIAGNOSIS_MAP.get(
            str(diagnosis), str(diagnosis).strip().lower()
        )

    return diagnosis_map


def parse_icbhi(dataset_path: str | Path) -> list[dict]:
    """
    Parse ICBHI recordings into one record per breathing cycle.

    Expected dataset structure:
    - patient_diagnosis.csv
    - audio_and_txt_files/
        - <recording_id>.wav
        - <recording_id>.txt
    """

    dataset_root = Path(dataset_path)
    audio_dir = dataset_root / "audio_and_txt_files"
    diagnosis_map = _load_diagnosis_map(dataset_root)

    if not audio_dir.exists():
        raise FileNotFoundError(f"ICBHI audio directory not found: {audio_dir}")

    records: list[dict] = []

    for txt_path in sorted(audio_dir.glob("*.txt")):
        wav_path = txt_path.with_suffix(".wav")
        if not wav_path.exists():
            continue

        recording_id = txt_path.stem
        patient_id = recording_id.split("_")[0]
        diagnosis = diagnosis_map.get(patient_id, "unknown")

        with txt_path.open("r", encoding="utf-8") as handle:
            for annotation_index, raw_line in enumerate(handle):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(
                        "Invalid ICBHI annotation line "
                        f"in {txt_path} at line {annotation_index + 1}: {raw_line.rstrip()}"
                    )

                start = float(parts[0])
                end = float(parts[1])
                crackle = int(parts[2])
                wheeze = int(parts[3])
                label = _flags_to_label(crackle=crackle, wheeze=wheeze)

                records.append(
                    {
                        "dataset": "icbhi",
                        "patient_id": patient_id,
                        "recording_id": recording_id,
                        "wav_path": str(wav_path),
                        "annotation_path": str(txt_path),
                        "annotation_index": annotation_index,
                        "start": start,
                        "end": end,
                        "crackle": crackle,
                        "wheeze": wheeze,
                        "label": label,
                        "diagnosis": diagnosis,
                    }
                )

    return records
