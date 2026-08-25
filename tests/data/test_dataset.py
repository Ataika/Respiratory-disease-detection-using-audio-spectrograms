import wave
from pathlib import Path
import numpy as np
import torch
from src.data.dataset import RespDataset, create_splits


def _write_test_wav(path:Path, sample_rate:int, duration_sec : float)->None:
    """
    Create a simple mono WAV file for dataset tests.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec),
                    endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t)
    audio_int16 = (audio * 32768).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio_int16.tobytes())


def _build_samples(tmp_path: Path) -> list[dict]:
    wav_a = tmp_path / "patient_a.wav"
    wav_b = tmp_path / "patient_b.wav"
    wav_c = tmp_path / "patient_c.wav"

    _write_test_wav(wav_a, sample_rate=22050, duration_sec=5.0)
    _write_test_wav(wav_b, sample_rate=22050, duration_sec=5.0)
    _write_test_wav(wav_c, sample_rate=22050, duration_sec=5.0)

    return [
        {
            "patient_id":"patient_1",
            "wav_path": str(wav_a),
            "label": 0,
        },
        {
            "patient_id": "patient_1",
            "wav_path": str(wav_b),
            "label": 1,
        },
        {
            "patient_id": "patient_2",
            "wav_path": str(wav_c),
            "label": 2,
        },
    ]

def test_create_splits_separates_patients(tmp_path):
    samples = _build_samples(tmp_path)

    train_samples , val_samples = create_splits(samples, val_ratio= 0.5, seed=42)
    train_patients = {sample["patient_id"] for sample in train_samples}
    val_patients = {sample["patient_id"] for sample in val_samples}

    assert train_patients.isdisjoint(val_patients)

def test_resp_dataset_returns_tensor_and_lavel(tmp_path):
    samples = _build_samples(tmp_path)

    dataset = RespDataset(
        samples=samples,
        branch="cnn",
        sample_rate=22050,
        duration=5.0,
    )
    x,y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.ndim == 3
    assert x.shape[0] == 3
    assert y.dtype == torch.long





