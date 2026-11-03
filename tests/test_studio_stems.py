"""Tests for Stage 5: Stem-specific studio resynthesis & high-frequency super-resolution."""

from __future__ import annotations

import numpy as np
from scipy import signal

from concert_studio.metrics import crest_factor_db
from concert_studio.studio_stems import (
    enhance_bass_studio,
    enhance_drums_studio,
    enhance_instruments_studio,
    enhance_vocals_studio,
    extend_high_frequency_bandwidth,
    process_all_studio_stems,
)
from concert_studio.synthetic import generate_synthetic_concert


def test_enhance_drums_and_bass_and_instruments() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.5, sr=sr, num_povs=2, seed=55)
    stems = scenario.clean_stems

    # Drums transient shaper + sub-bass boost
    drums_out = enhance_drums_studio(stems.drums, sr=sr)
    assert drums_out.shape == stems.drums.shape
    assert float(np.sqrt(np.mean(drums_out**2))) > float(np.sqrt(np.mean(stems.drums**2)))

    # Bass sub-120Hz mono phase locking
    bass_stereo_unbalanced = stems.bass.copy()
    bass_stereo_unbalanced[:, 1] *= -0.5
    bass_out = enhance_bass_studio(bass_stereo_unbalanced, sr=sr)
    sos_sub = signal.butter(3, 100.0, btype="lowpass", fs=sr, output="sos")
    sub_out = signal.sosfilt(sos_sub, bass_out, axis=0)
    assert np.allclose(sub_out[:, 0], sub_out[:, 1], atol=1e-2)

    # Instruments Mid/Side stereo widening increases side-to-mid ratio
    inst_out = enhance_instruments_studio(stems.other, sr=sr)
    side_in = float(np.std(0.5 * (stems.other[:, 0] - stems.other[:, 1])))
    side_out = float(np.std(0.5 * (inst_out[:, 0] - inst_out[:, 1])))
    assert side_out > side_in * 1.15


def test_extend_high_frequency_bandwidth_restores_air_band() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.0, sr=sr, num_povs=2, seed=66)
    # Brickwall low-pass at 11.5 kHz to simulate AAC phone cutoff
    sos_lp = signal.butter(6, 11500.0, btype="lowpass", fs=sr, output="sos")
    bandlimited = signal.sosfilt(sos_lp, scenario.studio_reference.samples, axis=0)

    extended, hf_gain_db = extend_high_frequency_bandwidth(bandlimited, sr=sr, cutoff_hz=12000.0)
    assert extended.shape == bandlimited.shape
    assert hf_gain_db > 3.0


def test_process_all_studio_stems_bundle() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.0, sr=sr, num_povs=2, seed=77)
    enhanced_bundle, avg_hf_db = process_all_studio_stems(scenario.clean_stems)
    assert avg_hf_db >= 0.0
    assert enhanced_bundle.vocals.shape == scenario.clean_stems.vocals.shape
