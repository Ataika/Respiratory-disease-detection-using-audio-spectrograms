"""Parser for the HF Lung V1 respiratory sound dataset"""

from __future__ import annotations
from pathlib import Path
from src.data.parsers.icbhi import _flags_to_label
import re

#Events that indicate wheeze -like pathology in HF Lung labels.

WHEEZE_EVENTS = {
    "wheeze" ,
    "rhonchi",
    "stridor",
}

#Events that indicate crackle-like pathology.
CRACKLE_EVENTS = {
    "crackle",
    "d",
}

DISPLAY_EVENT_NAMES = {
    "d": "D",
    "crackle": "Crackle",
    "wheeze": "Wheeze",
    "rhonchi": "Rhonchi",
    "stridor": "Stridor",
}

#Pure respiratory phase markers , not pathology labels
IGNORE_EVENTS = {
    "i","e"
}

def _normalize_event(event: str) -> str:
    """
    Normalize raw event labels so matching in case - insensitive and robust to extra spaces.
    """
    return event.strip().lower()


#Human-readable names for repsiratory phase markers.

PHASE_MAP = {
    "i": "inhalation",
    "e": "exhalation",
}

# Regex helpers for extracting a stable patient/session identifier
# from common HF Lung filename patterns.

_STETH_DATE_RE = re.compile(r"^steth_(\d{8})_")
_TRUNC_DATE_RE = re.compile(r"^trunc_(\d{4}-\d{2}-\d{2})-")


def _parse_hf_timestamp(value: str) -> float:
    """
    Convert timestamps like 00:01:02.500 into seconds.
    """
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_label_line(line: str) -> tuple[str, float, float]:
    """
    Parse one HF Lung label line.
    Expected format:
    Event HH:MM:SS.mmm HH:MM:SS.mmm
    """
    parts = line.split()
    if len(parts) != 3:
     raise ValueError(f"Invalid HF Lung label line: {line!r}")

    event = _normalize_event(parts[0])
    start = _parse_hf_timestamp(parts[1])
    end = _parse_hf_timestamp(parts[2])
    return event, start, end


def _extract_patient_id(recording_id: str) -> str:
    """
    Extract a stable patient-like identifier from the recording name.
    """
    steth_match = _STETH_DATE_RE.match(recording_id)
    if steth_match:
        return steth_match.group(1)

    trunc_match = _TRUNC_DATE_RE.match(recording_id)
    if trunc_match:
        return trunc_match.group(1).replace("-", "")

    return recording_id


def _overlaps(start_a: float, end_a: float, start_b: float, end_b: float)-> bool:
      """
      Check whether two time intervals overlap.
      """
      return max(start_a, start_b) < min(end_a, end_b)


def parse_hf_lung(dataset_path: str | Path) -> list[dict]:
    """
    Parse HF Lung V1 into one record per respiratory phase segment.

    The raw labels contain:
    - respiratory phases: I, E
    - pathological sound events: wheeze/rhonchi/stridor/crackle/D

    We convert overlapping sound events into the same unified label
    space used in the thesis:
    0 = normal
    1 = crackle
    2 = wheeze
    3 = both
    """
    dataset_root = Path(dataset_path)
    records: list[dict] = []

    for split in ("train", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue

        for wav_path in sorted(split_dir.glob("*.wav")):
            recording_id = wav_path.stem
            label_path = wav_path.with_name(f"{recording_id}_label.txt")

            if not label_path.exists():
                continue

            phase_events: list[tuple[str, float, float]] = []
            sound_events: list[tuple[str, float, float]] = []

            lines = label_path.read_text(encoding="utf-8",
                                         errors="ignore").splitlines()
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue

                event, start, end = _parse_label_line(line)

                if event in IGNORE_EVENTS:
                    phase_events.append((event, start, end))
                elif event in WHEEZE_EVENTS or event in CRACKLE_EVENTS:
                    sound_events.append((event, start, end))

            patient_id = _extract_patient_id(recording_id)
            for annotation_index, (phase_event, start, end) in enumerate(phase_events):
                crackle = 0
                wheeze = 0
                source_events: set[str] = set()

                for sound_event, sound_start, sound_end in sound_events:
                    if not _overlaps(start, end, sound_start, sound_end):
                        continue

                    if sound_event in CRACKLE_EVENTS:
                        crackle = 1
                    if sound_event in WHEEZE_EVENTS:
                        wheeze = 1

                    source_events.add(DISPLAY_EVENT_NAMES.get(sound_event, sound_event))

                records.append(
                    {
                        "dataset": "hf_lung",
                        "split": split,
                        "patient_id": patient_id,
                        "recording_id": recording_id,
                        "wav_path": str(wav_path),
                        "annotation_path": str(label_path),
                        "annotation_index": annotation_index,
                        "phase": PHASE_MAP[phase_event],
                        "start": start,
                        "end": end,
                        "crackle": crackle,
                        "wheeze": wheeze,
                        "label": _flags_to_label(crackle=crackle,
                                                 wheeze=wheeze),
                        "source_events": sorted(source_events),
                    }
                )

    return records
