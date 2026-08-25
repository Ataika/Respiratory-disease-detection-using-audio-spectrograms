from pathlib import Path

import pytest

from src.data.parsers.icbhi import _flags_to_label, parse_icbhi


def _build_icbhi_fixture(root: Path) -> Path:
    dataset_path = root / "icbhi"
    audio_dir = dataset_path / "audio_and_txt_files"
    audio_dir.mkdir(parents=True)

    (dataset_path / "patient_diagnosis.csv").write_text(
        "101,Healthy\n102,COPD\n",
        encoding="utf-8",
    )

    recording_id = "101_1b1_Al_sc_Meditron"
    (audio_dir / f"{recording_id}.wav").write_bytes(b"fake-wav")
    (audio_dir / f"{recording_id}.txt").write_text(
        "0.0 1.0 0 0\n"
        "1.0 2.0 1 0\n"
        "2.0 3.0 0 1\n"
        "3.0 4.0 1 1\n",
        encoding="utf-8",
    )

    missing_recording_id = "102_1b1_Pr_sc_Litt3200"
    (audio_dir / f"{missing_recording_id}.txt").write_text(
        "0.0 1.5 0 0\n",
        encoding="utf-8",
    )

    return dataset_path


def test_parse_icbhi_returns_one_record_per_breath_cycle(tmp_path):
    dataset_path = _build_icbhi_fixture(tmp_path)

    records = parse_icbhi(dataset_path)

    assert len(records) == 4
    assert [record["label"] for record in records] == [0, 1, 2, 3]
    assert all(record["dataset"] == "icbhi" for record in records)
    assert all(record["patient_id"] == "101" for record in records)
    assert all(record["recording_id"] == "101_1b1_Al_sc_Meditron" for record in records)
    assert all(record["diagnosis"] == "normal" for record in records)
    assert records[0]["wav_path"].endswith("101_1b1_Al_sc_Meditron.wav")
    assert records[0]["start"] == pytest.approx(0.0)
    assert records[-1]["end"] == pytest.approx(4.0)


def test_parse_icbhi_skips_annotations_without_audio_file(tmp_path):
    dataset_path = _build_icbhi_fixture(tmp_path)

    records = parse_icbhi(dataset_path)

    recording_ids = {record["recording_id"] for record in records}
    assert "102_1b1_Pr_sc_Litt3200" not in recording_ids


@pytest.mark.parametrize(
    ("crackle", "wheeze", "expected_label"),
    [
        (0, 0, 0),
        (1, 0, 1),
        (0, 1, 2),
        (1, 1, 3),
    ],
)
def test_flags_to_label_matches_unified_scheme(crackle, wheeze, expected_label):
    assert _flags_to_label(crackle, wheeze) == expected_label


def test_flags_to_label_rejects_non_binary_values():
    with pytest.raises(ValueError, match="binary crackle/wheeze flags"):
        _flags_to_label(2, 0)
