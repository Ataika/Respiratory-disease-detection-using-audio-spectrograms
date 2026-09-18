from pathlib import Path
import json

from src.data.parsers.spr_sound import _normalize_spr_label,parse_spr_sound


def _build_spr_sound_fixture(root: Path) -> Path:
    dataset_path = root / "SPRSound-main" / "Classification"
    wav_dir = dataset_path / "train_classification_wav"
    json_dir = dataset_path / "train_classification_json"

    wav_dir.mkdir(parents=True)
    json_dir.mkdir(parents=True)

    (wav_dir / "sample_001.wav").write_bytes(b"fake-wav")
    (wav_dir / "sample_002.wav").write_bytes(b"fake-wav")
    (wav_dir / "sample_003.wav").write_bytes(b"fake-wav")

    (json_dir / "sample_001.json").write_text(
        json.dumps({"record_annotation": "Normal"}),
        encoding="utf-8",
    )
    (json_dir / "sample_002.json").write_text(
        json.dumps({"record_annotation": "Crackle"}),
        encoding="utf-8",
    )
    (json_dir / "sample_003.json").write_text(
        json.dumps({"record_annotation": "Wheeze&Crackle"}),
        encoding="utf-8",
    )

    return dataset_path


def test_parse_spr_sound_returns_unseen_records(tmp_path):
    dataset_path = _build_spr_sound_fixture(tmp_path)

    records = parse_spr_sound(dataset_path)

    assert len(records) == 3
    assert [record["label"] for record in records] == [0, 1, 3]
    assert all(record["dataset"] == "spr_sound" for record in records)
    assert all(record["split"] == "unseen" for record in records)
    assert records[0]["recording_id"] == "sample_001"
    assert records[1]["recording_id"] == "sample_002"
    assert records[2]["recording_id"] == "sample_003"


def test_normalize_spr_label_matches_unified_scheme():
    assert _normalize_spr_label("Normal") == 0
    assert _normalize_spr_label("Crackle") == 1
    assert _normalize_spr_label("Wheeze") == 2
    assert _normalize_spr_label("Wheeze&Crackle") == 3


def test_normalize_spr_label_skips_unrecognized_annotation_instead_of_defaulting_to_normal():
    """An unrecognized annotation string must not silently become
    ground-truth `normal` (label 0) — that would quietly corrupt
    cross-domain evaluation metrics for a category this map hasn't seen.
    """
    assert _normalize_spr_label("Rhonchi") is None
    assert _normalize_spr_label("Poor Quality") is None


def test_parse_spr_sound_skips_records_with_unrecognized_annotation(tmp_path):
    dataset_path = _build_spr_sound_fixture(tmp_path)
    json_dir = dataset_path / "train_classification_json"
    (dataset_path / "train_classification_wav" / "sample_004.wav").write_bytes(b"fake-wav")
    (json_dir / "sample_004.json").write_text(
        json.dumps({"record_annotation": "Rhonchi"}),
        encoding="utf-8",
    )

    records = parse_spr_sound(dataset_path)

    assert len(records) == 3  # sample_004 excluded, not mislabeled as normal
    assert "sample_004" not in {r["recording_id"] for r in records}

