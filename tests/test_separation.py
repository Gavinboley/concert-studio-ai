"""Tests for Stage 4: Multi-stem neural and spectral source separation."""

from __future__ import annotations

import numpy as np
from scipy import signal

from concert_studio.separation import separate_concert_stems
from concert_studio.synthetic import generate_synthetic_concert


def test_separate_concert_stems_produces_five_valid_stems() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.5, sr=sr, num_povs=2, seed=19)
    bundle = separate_concert_stems(scenario.studio_reference, backend="auto")

    for name, arr in bundle.as_dict().items():
        assert arr.shape == scenario.studio_reference.samples.shape, f"Shape mismatch on {name}"
        assert float(np.sqrt(np.mean(arr**2))) > 1e-4, f"Empty stem: {name}"

    # Verify bass stem concentrates energy below 250 Hz compared to vocals
    sos_low = signal.butter(3, 220.0, btype="lowpass", fs=sr, output="sos")
    bass_low_ratio = float(
        np.mean(signal.sosfilt(sos_low, bundle.bass, axis=0) ** 2) / (np.mean(bundle.bass**2) + 1e-12)
    )
    vocal_low_ratio = float(
        np.mean(signal.sosfilt(sos_low, bundle.vocals, axis=0) ** 2) / (np.mean(bundle.vocals**2) + 1e-12)
    )
    assert bass_low_ratio > vocal_low_ratio * 2.5
