"""Parser for the KAUH respiratory sound dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FILENAME_COLUMNS = [
    "filename",
    "file",
    "audio",
    "audio_file",
    "recording",
    "recording_name",
    "wav",
    "name",
]
LABEL_COLUMNS = [
    "label",
    "diagnosis",
    "class",
    "category",
    "annotation",
    "status",
    "condition",
    "pathology",
]


def _find_matching_column(columns: list[str], candidates: list[str]) -> str | None:
    """
    Find the first matching column name in a case-insensitive way.
    """
    normalized = {column.strip().lower(): column for column in columns}

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def _map_kauh_label(raw_label: str) -> int:
    """
    Map a raw KAUH label into the unified thesis label scheme.

    KAUH is less granular than ICBHI, so some labels may only say
    "normal" or "abnormal". In those cases we map conservatively.
    """
    label = raw_label.strip().lower()
    if "abnormal" in label or "pathology" in label:
        return 1

    if "normal" in label or "healthy" in label:
        return 0

    if "crackle" in label and "wheeze" in label:
        return 3

    if "wheeze" in label or "rhonchi" in label:
        return 2

    if "crackle" in label:
        return 1

    return 0


def _map_kauh_sound_type(raw_sound_type: str) -> int | None:
    """
    Map KAUH sound-type notation into the unified label scheme.

    Examples seen in filenames:
    - `N`
    - `E W`
    - `I E W`
    - `C`
    - `I C E W`
    - `Crep`
    - `Bronchial`
    """
    sound_type = raw_sound_type.strip().lower()
    tokens = [token for token in sound_type.replace(",", " ").split() if token]

    crackle = "crep" in sound_type or "crackle" in sound_type or "c" in tokens
    wheeze = (
        "wheeze" in sound_type
        or "rhonchi" in sound_type
        or "stridor" in sound_type
        or "w" in tokens
    )

    if crackle and wheeze:
        return 3
    if crackle:
        return 1
    if wheeze:
        return 2

    if "normal" in sound_type or "n" in tokens:
        return 0

    return None


def _parse_kauh_filename(wav_path: Path) -> dict[str, str]:
    """
    Extract metadata from real KAUH filenames.

    Expected pattern (approximate):
    RECORDINGID_DIAGNOSIS,SOUNDTYPE,LOCATION,AGE,GENDER.wav
    """
    parts = [part.strip() for part in wav_path.stem.split(",")]

    record_and_diag = parts[0] if parts else wav_path.stem
    if "_" in record_and_diag:
        recording_id, diagnosis = record_and_diag.split("_", 1)
    else:
        recording_id, diagnosis = record_and_diag, ""

    return {
        "recording_id": recording_id.strip(),
        "diagnosis": diagnosis.strip(),
        "sound_type": parts[1].strip() if len(parts) > 1 else "",
        "location": parts[2].strip() if len(parts) > 2 else "",
        "age": parts[3].strip() if len(parts) > 3 else "",
        "gender": parts[4].strip() if len(parts) > 4 else "",
    }


def parse_kauh(dataset_path: str | Path) -> list[dict]:
    """
    Parse KAUH into one record per audio file.
    """
    dataset_root = Path(dataset_path)
    audio_dir = dataset_root / "Audio Files"
    annotation_path = dataset_root / "Data annotation.xlsx"

    records: list[dict] = []

    # Path 1: synthetic/unit-test friendly parsing from Excel with filename + label.
    if annotation_path.exists():
        annotation_df = pd.read_excel(annotation_path)

        filename_column = _find_matching_column(
            annotation_df.columns.tolist(),
            FILENAME_COLUMNS,
        )
        label_column = _find_matching_column(
            annotation_df.columns.tolist(),
            LABEL_COLUMNS,
        )

        if filename_column is not None and label_column is not None:
            for annotation_index, row in annotation_df.iterrows():
                raw_filename = str(row[filename_column]).strip()
                raw_label = str(row[label_column]).strip()

                if not raw_filename or raw_filename.lower() == "nan":
                    continue

                wav_path = audio_dir / raw_filename
                if wav_path.suffix.lower() != ".wav":
                    wav_path = wav_path.with_suffix(".wav")

                if not wav_path.exists():
                    continue

                records.append(
                    {
                        "dataset": "kauh",
                        "patient_id": wav_path.stem,
                        "recording_id": wav_path.stem,
                        "wav_path": str(wav_path),
                        "annotation_index": annotation_index,
                        "label": _map_kauh_label(raw_label),
                        "raw_label": raw_label,
                    }
                )

            return records

    # Path 2: real KAUH parsing from filenames when Excel lacks filename linkage.
    for annotation_index, wav_path in enumerate(sorted(audio_dir.glob("*.wav"))):
        meta = _parse_kauh_filename(wav_path)

        label = _map_kauh_sound_type(meta["sound_type"])
        if label is None:
            label = _map_kauh_label(meta["diagnosis"])

        raw_label = meta["sound_type"] or meta["diagnosis"]

        records.append(
            {
                "dataset": "kauh",
                "patient_id": meta["recording_id"],
                "recording_id": meta["recording_id"],
                "wav_path": str(wav_path),
                "annotation_index": annotation_index,
                "label": label,
                "raw_label": raw_label,
                "diagnosis": meta["diagnosis"],
                "sound_type": meta["sound_type"],
                "location": meta["location"],
                "age": meta["age"],
                "gender": meta["gender"],
            }
        )

    return records



