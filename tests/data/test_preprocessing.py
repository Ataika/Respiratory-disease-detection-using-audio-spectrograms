import numpy as np
import torch
import wave
from pathlib import Path

from src.data.preprocessing import (
    build_ast_features,
    build_cnn_mel_spectrogram,
    instance_normalize,
    load_and_preprocess,
    load_audio_window,
)


def test_instance_normalize_zero_mean():
    audio = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    normalized = instance_normalize(audio)

    assert abs(normalized.mean()) < 1e-6

def test_instance_normalize_handles_constant_SIGNAL():
    audio = np.ones(8, dtype=np.float32)
    normalized = instance_normalize(audio)
    assert normalized.shape == audio.shape
    assert not np.isnan(normalized).any()

def test_build_cnn_mel_spectrogram_returns_3d_tensor():
    audio = np.random.randn(22050 * 5).astype(np.float32)

    tensor = build_cnn_mel_spectrogram(audio = audio, sample_rate = 22050)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 3
    assert tensor.shape[0] == 3
    assert tensor.shape[1] == 128

def test_build_ast_features_returns_2d_tensor():
    audio = np.random.randn(16000 * 5).astype(np.float32)

    tensor = build_ast_features(audio = audio, sample_rate = 16000)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 2
    assert tensor.shape[0] == 128


def _write_test_wav(path: Path, sample_rate: int, duration_sec: float)-> None:
    """
    Create a simple mono WAV file for preprocessing tests.
    """
    t = np.linspace(0, duration_sec , int(sample_rate * duration_sec), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 220 * t)
    audio_int16 =(audio * 32767).astype(np.int16)

    with wave.open(str(path),"wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio_int16.tobytes())

def test_load_audio_window_pads_cnn_short_signal(tmp_path):
    wav_path = tmp_path / "cnn.wav"
    _write_test_wav(wav_path, sample_rate=22050, duration_sec=1.0)
    audio = load_audio_window(
        file_path = str(wav_path),
        offset = 0.0,
        duration = 5.0,
        sample_rate = 22050,
    )

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == 22050 * 5


def test_load_and_preprocess_audio_returns_3d_tensor(tmp_path):
    wav_path = tmp_path / "cnn.wav"
    _write_test_wav(wav_path, sample_rate=22050, duration_sec=5.0)

    tensor = load_and_preprocess(
        file_path = str(wav_path),
        offset = 0.0,
        duration = 5.0,
        sample_rate = 22050,
        branch="cnn"
    )

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 3
    assert tensor.shape[0] == 3
    assert tensor.shape[1] == 128

def test_load_and_preprocess_ast_returns_2d_tensor(tmp_path):
    wav_path = tmp_path / "ast.wav"
    _write_test_wav(wav_path, sample_rate=16000, duration_sec=5.0)

    tensor = load_and_preprocess(
        file_path = str(wav_path),
        offset = 0.0,
        duration = 5.0,
        sample_rate = 16000,
        branch="ast",
    )

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 2
    assert tensor.shape[0] == 128




