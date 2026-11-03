"""Tests for Stage 2: LTAS EQ matching and Time-Frequency Median MAD Outlier Consensus."""

from __future__ import annotations

import numpy as np
from scipy import signal

from concert_studio.consensus import fuse_multiview_consensus, match_ltas_eq
from concert_studio.metrics import si_sdr_db
from concert_studio.models import AudioTrack
from concert_studio.sync import align_and_undrift_tracks
from concert_studio.synthetic import generate_synthetic_concert


def test_match_ltas_eq_restores_muffled_pocket_pov() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.5, sr=sr, num_povs=2, seed=99)
    ref = scenario.studio_reference.samples

    # Simulate muffled pocket recording (low-pass at 3500 Hz)
    sos_lp = signal.butter(4, 3500.0, btype="lowpass", fs=sr, output="sos")
    muffled = signal.sosfilt(sos_lp, ref, axis=0)

    eq_restored = match_ltas_eq(muffled, ref)
    sos_hf = signal.butter(6, 5500.0, btype="highpass", fs=sr, output="sos")
    hf_muffled = float(np.mean(signal.sosfilt(sos_hf, muffled, axis=0) ** 2))
    hf_restored = float(np.mean(signal.sosfilt(sos_hf, eq_restored, axis=0) ** 2))
    assert hf_restored > hf_muffled * 3.0


def test_fuse_multiview_consensus_suppresses_localized_crowd_screams() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=3.0, sr=sr, num_povs=5, seed=42)
    aligned, _, ref_idx = align_and_undrift_tracks(scenario.pov_tracks, target_sr=sr)

    consensus, outliers = fuse_multiview_consensus(aligned, ref_idx=ref_idx, mad_threshold=1.55)

    clean = scenario.studio_reference.samples
    individual_sdrs = [si_sdr_db(clean, trk.samples) for trk in aligned]
    consensus_sdr = si_sdr_db(clean, consensus.samples)

    # Consensus must beat the average individual POV SI-SDR and isolate non-zero crowd outliers
    assert consensus_sdr > float(np.mean(individual_sdrs)) + 1.5
    assert float(np.sqrt(np.mean(outliers.samples**2))) > 0.01
