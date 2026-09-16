import wave
from pathlib import Path
import numpy as np
import torch
from torch.distributed.elastic.utils.store import barrier

from src.data.loaders import create_dataloader, make_weighted_sampler


def _write_test_wav(path: Path, sample_rate: int, duration_sec: float) ->None:
    """
    Create a simple mono WAV file for loader tests.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec),
                    endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 200 * t)

    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio_int16.tobytes())

def _build_samples(tmp_path: Path) -> list[dict]:
    wav_a = tmp_path / "sample_a.wav"
    wav_b = tmp_path / "sample_b.wav"
    wav_c = tmp_path / "sample_c.wav"

    _write_test_wav(wav_a, sample_rate=22050, duration_sec=5.0)
    _write_test_wav(wav_b, sample_rate=22050, duration_sec=5.0)
    _write_test_wav(wav_c, sample_rate=22050, duration_sec=5.0)

    return [
        {"patient_id": "p1", "wav_path": str(wav_a), "label": 0},
        {"patient_id": "p2", "wav_path": str(wav_b), "label": 1},
        {"patient_id": "p3", "wav_path": str(wav_c), "label": 1},
    ]

def test_make_weighted_sampler_returns_sampler(tmp_path):
    samples = _build_samples(tmp_path)

    sampler = make_weighted_sampler(samples)

    assert sampler is not None
    assert len(list(sampler))== len(samples)

def test_create_dataloader_returns_batches(tmp_path):
    samples = _build_samples(tmp_path)

    dataloader = create_dataloader(
        samples = samples,
        branch = "cnn",
        sample_rate = 22050,
        duration = 5.0,
        batch_size = 2,
        use_weighted_sampler=False,
    )

    batch_x , batch_y = next(iter(dataloader))

    assert isinstance(batch_x, torch.Tensor)
    assert isinstance(batch_y, torch.Tensor)
    assert batch_x.ndim == 4
    assert batch_x.shape[1] == 3
    assert batch_y.ndim == 1




