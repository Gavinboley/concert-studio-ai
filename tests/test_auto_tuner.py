"""Tests for Stage 0 Dynamic Acoustic Analyzer & Auto-Setting Optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from concert_studio.auto_tuner import AutoSettingOptimizer, AcousticProfile
from concert_studio.models import AudioTrack
from concert_studio.pipeline import PipelineConfig


def test_auto_tuner_synthetic_tracks():
    sr = 48000
    dur = 4.0
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)

    # Track 1: Heavy harsh treble and high noise floor
    sig1 = (
        0.2 * np.sin(2 * np.pi * 440 * t)
        + 0.6 * np.sin(2 * np.pi * 5000 * t)  # Harsh 5kHz horn
        + 0.05 * np.random.randn(len(t))
    )
    t1 = AudioTrack(name="pov_noisy.wav", sample_rate=sr, samples=sig1)

    # Track 2: Clean bass, dual mono
    sig2 = 0.5 * np.sin(2 * np.pi * 80 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    stereo2 = np.column_stack([sig2, sig2])
    t2 = AudioTrack(name="pov_bass.wav", sample_rate=sr, samples=stereo2)

    result = AutoSettingOptimizer.analyze_and_optimize([t1, t2], sample_rate=sr)

    assert result.acoustic_profile.num_tracks_analyzed == 2
    assert result.optimized_config.quality_tier == "archival_max"
    assert 0.15 <= result.optimized_config.dereverb_alpha <= 0.55
    assert 1.20 <= result.optimized_config.mad_outlier_threshold <= 2.0
    assert len(result.diagnostic_findings) >= 4
    assert result.expected_quality_improvement_db > 0.0

    d = result.to_dict()
    assert "acoustic_profile" in d
    assert "optimized_config" in d
    assert d["optimized_config"]["quality_tier"] == "archival_max"
