from pathlib import Path
from src.data.parsers.bracets import _map_bracets_diagnosis, parse_bracets


def _build_bracets_fixture(root: Path) -> Path:
    dataset_path = root / "BRACETS"
    data_dir = dataset_path / "Data"
    metadata_dir = dataset_path / "Metadata"

    metadata_dir.mkdir(parents=True)
    (data_dir / "01" / "Sound" / "Lung").mkdir(parents=True)
    (data_dir / "02" / "Sound" / "Lung").mkdir(parents=True)

    (data_dir / "01" / "Sound" / "Lung" /
     "rec_01.wav").write_bytes(b"fake-wav")
    (data_dir / "02" / "Sound" / "Lung" /
     "rec_02.wav").write_bytes(b"fake-wav")

    (metadata_dir / "Metadata.txt").write_text(
        "SubjectID,Diagnosis\n"
        "01,Healthy\n"
        "02,COPD\n",
        encoding="utf-8",
    )

    return dataset_path


def test_parse_bracets_returns_unseen_binary_records(tmp_path):
    dataset_path = _build_bracets_fixture(tmp_path)

    records = parse_bracets(dataset_path)

    assert len(records) == 2
    assert [record["label"] for record in records] == [0, 1]
    assert all(record["dataset"] == "bracets" for record in records)
    assert all(record["split"] == "unseen" for record in records)
    assert records[0]["patient_id"] == "01"
    assert records[1]["patient_id"] == "02"


def test_map_bracets_diagnosis_binary_mapping():
    assert _map_bracets_diagnosis("Healthy") == 0
    assert _map_bracets_diagnosis("COPD") == 1
    assert _map_bracets_diagnosis("Asthma") == 1


