from pathlib import Path

from src.data.parsers.hf_lung import _extract_patient_id, _parse_hf_timestamp, parse_hf_lung


def _build_hf_lung_fixture(root: Path) -> Path:
    dataset_path = root / "HF_LUNG_V1"
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"
    duplicate_dir = dataset_path / "train 2"

    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    duplicate_dir.mkdir(parents=True)

    train_recording = "steth_20190821_12_00_25"
    (train_dir / f"{train_recording}.wav").write_bytes(b"fake-wav")
    (train_dir / f"{train_recording}_label.txt").write_text(
        "I 00:00:00.000 00:00:02.000\n"
        "D 00:00:00.500 00:00:01.000\n"
        "E 00:00:02.000 00:00:04.000\n"
        "Wheeze 00:00:02.500 00:00:03.500\n"
        "I 00:00:04.000 00:00:06.000\n"
        "D 00:00:04.200 00:00:04.600\n"
        "Rhonchi 00:00:05.000 00:00:05.500\n",
        encoding="utf-8",
    )

    test_recording = "trunc_2019-08-21-10-35-06-L1_1"
    (test_dir / f"{test_recording}.wav").write_bytes(b"fake-wav")
    (test_dir / f"{test_recording}_label.txt").write_text(
        "I 00:00:01.000 00:00:02.500\n"
        "Stridor 00:00:01.500 00:00:02.000\n",
        encoding="utf-8",
    )

    duplicate_recording = "steth_20190822_08_00_00"
    (duplicate_dir / f"{duplicate_recording}.wav").write_bytes(b"fake-wav")
    (duplicate_dir / f"{duplicate_recording}_label.txt").write_text(
        "I 00:00:00.000 00:00:01.000\n"
        "D 00:00:00.100 00:00:00.500\n",
        encoding="utf-8",
    )

    return dataset_path


def test_parse_hf_lung_builds_phase_level_records(tmp_path):
    dataset_path = _build_hf_lung_fixture(tmp_path)

    records = parse_hf_lung(dataset_path)

    assert len(records) == 4
    assert [record["label"] for record in records] == [1, 2, 3, 2]
    assert [record["phase"] for record in records] == [
        "inhalation",
        "exhalation",
        "inhalation",
        "inhalation",
    ]
    assert records[0]["patient_id"] == "20190821"
    assert records[-1]["patient_id"] == "20190821"
    assert records[0]["source_events"] == ["D"]
    assert records[1]["source_events"] == ["Wheeze"]
    assert records[2]["source_events"] == ["D", "Rhonchi"]
    assert records[3]["source_events"] == ["Stridor"]


def test_parse_hf_lung_ignores_duplicate_export_folders(tmp_path):
    dataset_path = _build_hf_lung_fixture(tmp_path)

    records = parse_hf_lung(dataset_path)

    recording_ids = {record["recording_id"] for record in records}
    assert "steth_20190822_08_00_00" not in recording_ids


def test_parse_hf_timestamp_converts_hms_to_seconds():
    assert _parse_hf_timestamp("00:00:03.638") == 3.638
    assert _parse_hf_timestamp("00:01:10.500") == 70.5


def test_extract_patient_id_uses_recording_date():
    assert _extract_patient_id("steth_20190821_12_00_25") == "20190821"
    assert _extract_patient_id("trunc_2019-08-21-10-35-06-L1_1") == "20190821"
