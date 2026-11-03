"""Tests for core data models, 24-bit HD WAV I/O, and synthetic concert benchmark generation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from concert_studio.io import read_wav, write_wav_hd
from concert_studio.models import AudioTrack
from concert_studio.synthetic import generate_synthetic_concert


def test_audio_track_conversions() -> None:
    mono = np.sin(np.linspace(0.0, 20.0, 4800))
    trk = AudioTrack(name="test_mono", samples=mono, sample_rate=48000)
    assert trk.num_channels == 1
    assert pytest.approx(trk.duration_sec, rel=1e-4) == 0.1
    stereo = trk.to_stereo()
    assert stereo.shape == (4800, 2)
    assert np.allclose(trk.to_mono(), mono)


def test_wav_24bit_and_16bit_roundtrip(tmp_path: Path) -> None:
    sr = 48000
    t = np.arange(sr // 5, dtype=np.float64) / sr
    stereo = np.column_stack([0.6 * np.sin(2 * np.pi * 440 * t), -0.4 * np.cos(2 * np.pi * 880 * t)])
    track = AudioTrack(name="roundtrip_hd", samples=stereo, sample_rate=sr)

    wav24 = write_wav_hd(tmp_path / "hd24.wav", track, bit_depth=24)
    loaded24 = read_wav(wav24)
    assert loaded24.sample_rate == sr
    assert loaded24.samples.shape == stereo.shape
    assert np.max(np.abs(loaded24.samples - stereo)) < 1e-5

    wav16 = write_wav_hd(tmp_path / "std16.wav", track, bit_depth=16)
    loaded16 = read_wav(wav16)
    assert np.max(np.abs(loaded16.samples - stereo)) < 1e-3


def test_generate_synthetic_concert_properties() -> None:
    scenario = generate_synthetic_concert(duration_sec=2.5, sr=48000, num_povs=5, seed=7)
    assert len(scenario.pov_tracks) == 5
    assert scenario.studio_reference.samples.shape == (int(2.5 * 48000), 2)
    for stem_name, arr in scenario.clean_stems.as_dict().items():
        if stem_name != "crowd_removed":
            assert float(np.sqrt(np.mean(arr**2))) > 0.02
