from pathlib import Path

import pandas as pd

from src.data.parsers.kauh import _map_kauh_label, parse_kauh


def _build_kauh_fixture(root: Path) -> Path:
    dataset_path = root / "jwyy9np4gv-3"
    audio_dir = dataset_path / "Audio Files"
    audio_dir.mkdir(parents=True)

    # creating fake wav files
    (audio_dir / "patient_001.wav").write_bytes(b"fake-wav")
    (audio_dir / "patient_002.wav").write_bytes(b"fake-wav")
    (audio_dir / "patient_003.wav").write_bytes(b"fake-wav")

    # creating Excel annotations
    annotation_df = pd.DataFrame(
        {
            "filename": ["patient_001.wav", "patient_002.wav",
                         "patient_003.wav"],
            "label": ["normal", "crackle", "wheeze"],
        }
    )
    annotation_df.to_excel(dataset_path / "Data annotation.xlsx",
                           index=False)

    return dataset_path


def test_parse_kauh_returns_one_record_per_file(tmp_path):
    dataset_path = _build_kauh_fixture(tmp_path)

    records = parse_kauh(dataset_path)

    assert len(records) == 3
    assert [record["label"] for record in records] == [0, 1, 2]
    assert all(record["dataset"] == "kauh" for record in records)
    assert records[0]["recording_id"] == "patient_001"
    assert records[1]["recording_id"] == "patient_002"
    assert records[2]["recording_id"] == "patient_003"


def test_map_kauh_label_matches_unified_scheme():
    assert _map_kauh_label("normal") == 0
    assert _map_kauh_label("healthy") == 0
    assert _map_kauh_label("crackle") == 1
    assert _map_kauh_label("wheeze") == 2
    assert _map_kauh_label("crackle wheeze") == 3
    assert _map_kauh_label("abnormal") == 1

