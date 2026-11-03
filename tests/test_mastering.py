"""Tests for Stage 6: Sidechain stem mixing, Reference Matchering & ITU-R BS.1770-4 HD Mastering."""

from __future__ import annotations

import numpy as np

from concert_studio.mastering import (
    measure_lufs,
    measure_true_peak_dbtp,
    mix_and_master_stems,
)
from concert_studio.synthetic import generate_synthetic_concert


def test_mix_and_master_stems_hits_target_lufs_and_true_peak_ceiling() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=3.0, sr=sr, num_povs=2, seed=88)

    target_lufs = -14.0
    ceiling_dbtp = -1.0
    mastered, stem_lufs = mix_and_master_stems(
        scenario.clean_stems,
        reference_audio=scenario.studio_reference.samples,
        target_lufs=target_lufs,
        true_peak_limit_db=ceiling_dbtp,
    )

    measured_lufs = measure_lufs(mastered, sr=sr)
    measured_tp = measure_true_peak_dbtp(mastered)

    # Check integrated LUFS is near -14.0 LUFS and True-Peak strictly respects -1.0 dBTP ceiling
    assert abs(measured_lufs - target_lufs) < 1.2
    assert measured_tp <= ceiling_dbtp + 0.05
    assert set(stem_lufs.keys()) == {"vocals", "drums", "bass", "other"}
