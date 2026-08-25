import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "datasets"


def data_available() -> bool:
    """Check whether the local raw dataset directory is available."""

    icbhi_audio_dir = (
        DATA_ROOT
        / "archive"
        / "Respiratory_Sound_Database"
        / "Respiratory_Sound_Database"
        / "audio_and_txt_files"
    )
    return icbhi_audio_dir.is_dir()


@pytest.fixture(scope="session")
def data_root():
    if not data_available():
        pytest.skip("data not available")
    return DATA_ROOT


@pytest.fixture(scope="session")
def icbhi_path(data_root):
    return data_root / "archive" / "Respiratory_Sound_Database" / "Respiratory_Sound_Database"


@pytest.fixture(scope="session")
def hf_path(data_root):
    return data_root / "HF_LUNG_V1"


@pytest.fixture(scope="session")
def kauh_path(data_root):
    return data_root / "jwyy9np4gv-3"


@pytest.fixture(scope="session")
def bracets_path(data_root):
    return data_root / "BRACETS"


@pytest.fixture(scope="session")
def spr_path(data_root):
    return data_root / "SPRSound-main" / "Classification"


@pytest.fixture(scope="session")
def sample_wav(icbhi_path):
    """Return the path to the first ICBHI wav file for quick tests."""

    wavs = sorted((icbhi_path / "audio_and_txt_files").glob("*.wav"))
    if not wavs:
        pytest.skip("wav not found")
    return str(wavs[0])
