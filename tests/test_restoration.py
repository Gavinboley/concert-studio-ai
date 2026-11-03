"""Tests for Stage 3: Cubic spline waveform de-clipping & stadium WPE dereverberation."""

from __future__ import annotations

import numpy as np

from concert_studio.restoration import declip_waveform, dereverberate_stadium
from concert_studio.synthetic import _apply_stadium_reverb


def test_declip_waveform_repairs_saturated_peaks() -> None:
    sr = 48000
    t = np.arange(sr // 4, dtype=np.float64) / sr
    clean = 0.92 * np.sin(2.0 * np.pi * 220.0 * t)
    clipped = np.clip(clean * 1.35, -0.75, 0.75)

    repaired, n_repaired = declip_waveform(clipped, clip_threshold=0.74)
    assert n_repaired > 50
    # Correlation with unclipped sine wave should improve over flat-topped clipped wave
    corr_clipped = float(np.corrcoef(clean, clipped)[0, 1])
    corr_repaired = float(np.corrcoef(clean, repaired)[0, 1])
    assert corr_repaired > corr_clipped


def test_dereverberate_stadium_reduces_late_reverb_tail() -> None:
    sr = 48000
    n = int(2.0 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    # Create dry percussive bursts followed by silence gaps
    dry = np.zeros((n, 2), dtype=np.float64)
    for pulse_s in (0.2, 0.7, 1.2):
        i0 = int(pulse_s * sr)
        plen = int(0.08 * sr)
        burst = np.sin(2 * np.pi * 440.0 * t[:plen]) * np.exp(-t[:plen] / 0.02)
        dry[i0 : i0 + plen, 0] = burst
        dry[i0 : i0 + plen, 1] = burst

    reverberant = _apply_stadium_reverb(dry, sr=sr, rt60=0.6, wet=0.45)
    cleaned, reduction_db = dereverberate_stadium(reverberant, sr=sr, alpha_subtract=0.48)

    assert reduction_db > 0.4
    # Tail energy in silent gap (e.g. 0.32s to 0.55s) should be lower after dereverberation
    g0, g1 = int(0.32 * sr), int(0.55 * sr)
    tail_before = float(np.mean(reverberant[g0:g1] ** 2))
    tail_after = float(np.mean(cleaned[g0:g1] ** 2))
    assert tail_after < tail_before * 0.75
