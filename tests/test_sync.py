"""Tests for Stage 1: Sub-sample GCC-PHAT alignment and clock drift (SRO) compensation."""

from __future__ import annotations

import numpy as np

from concert_studio.models import AudioTrack
from concert_studio.sync import (
    align_and_undrift_tracks,
    estimate_alignment,
    select_reference_pov,
)
from concert_studio.synthetic import _apply_delay_and_drift, generate_synthetic_concert


def test_estimate_alignment_recovers_sub_millisecond_delay_and_drift() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=4.0, sr=sr, num_povs=2, seed=11)
    ref = scenario.studio_reference.to_mono()

    true_offset_sec = 0.0175  # 17.5 ms
    true_drift_ppm = 25.0     # 25 ppm clock skew
    delayed = _apply_delay_and_drift(ref, sr=sr, offset_sec=true_offset_sec, drift_ppm=true_drift_ppm)

    params = estimate_alignment(ref, delayed, sr=sr, track_name="drifted_pov")
    # Sub-millisecond accuracy on initial offset (< 0.5 ms)
    assert abs(params.initial_offset_sec - true_offset_sec) < 0.0005
    # Clock drift estimation accuracy
    assert abs(params.clock_drift_ppm - true_drift_ppm) < 8.0
    assert params.peak_coherence > 0.25


def test_align_and_undrift_tracks_locks_multi_pov_phase() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=3.0, sr=sr, num_povs=4, seed=21)
    aligned, alignments, ref_idx = align_and_undrift_tracks(scenario.pov_tracks, target_sr=sr)

    assert len(aligned) == 4
    assert len(alignments) == 4
    assert ref_idx == select_reference_pov(scenario.pov_tracks)

    # Check residual lag after alignment is near zero (< 0.5 ms)
    ref_mono = aligned[ref_idx].to_mono()
    for idx, trk in enumerate(aligned):
        if idx == ref_idx:
            continue
        post = estimate_alignment(ref_mono, trk.to_mono(), sr=sr)
        assert abs(post.initial_offset_sec) < 0.0006
