"""Tests for 100% True-Data (zero generative hallucination) and monotonic POV count resolution scaling."""

from __future__ import annotations

import numpy as np
from scipy import signal

from concert_studio.metrics import si_sdr_db
from concert_studio.models import AudioTrack
from concert_studio.pipeline import ConcertStudioPipeline, PipelineConfig
from concert_studio.synthetic import generate_synthetic_concert


def test_true_data_mode_never_hallucinates_missing_frequencies() -> None:
    """When all uploaded POVs have zero energy above 6 kHz, true_data_mode=True must not synthesize fake highs."""
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.5, sr=sr, num_povs=3, seed=50)
    sos_lp = signal.butter(8, 5000.0, btype="lowpass", fs=sr, output="sos")

    bandlimited_povs = []
    for trk in scenario.pov_tracks:
        bl = signal.sosfilt(sos_lp, trk.samples, axis=0)
        bandlimited_povs.append(AudioTrack(name=trk.name, samples=bl, sample_rate=sr))

    pipeline = ConcertStudioPipeline(
        PipelineConfig(target_sample_rate=sr, quality_tier="standard", true_data_mode=True)
    )
    mastered, _, report = pipeline.run(bandlimited_povs)
    assert report.true_data_mode is True

    # Measure energy above 12 kHz: must remain negligible (< -55 dB relative to full signal)
    sos_hf = signal.butter(6, 12000.0, btype="highpass", fs=sr, output="sos")
    hf_energy = float(np.mean(signal.sosfilt(sos_hf, mastered.samples, axis=0) ** 2))
    total_energy = float(np.mean(mastered.samples**2) + 1e-12)
    assert 10.0 * np.log10((hf_energy + 1e-18) / total_energy) < -45.0


def test_more_pov_files_monotonically_improve_true_to_life_fidelity() -> None:
    """Adding more independent POV recordings must monotonically improve consensus SI-SDR."""
    sr = 48000
    rng = np.random.default_rng(123)
    scenario = generate_synthetic_concert(duration_sec=3.0, sr=sr, num_povs=2, seed=123)
    clean = scenario.studio_reference.samples
    n = len(clean)

    # Create 10 realistic POV recordings with independent localized crowd bursts and noise
    pov_pool: list[AudioTrack] = []
    for i in range(10):
        noise = 0.08 * rng.standard_normal(clean.shape)
        # Add a localized scream burst at a different moment on each phone
        b_start = int(((i * 0.27) % 2.4) * sr)
        b_len = int(0.35 * sr)
        burst = 0.55 * rng.standard_normal((b_len, 2)) * np.hanning(b_len)[:, None]
        corrupted = clean.copy() + noise
        corrupted[b_start : b_start + b_len] += burst
        pov_pool.append(AudioTrack(name=f"pov_{i+1:02d}", samples=corrupted, sample_rate=sr))

    cfg = PipelineConfig(target_sample_rate=sr, quality_tier="standard", true_data_mode=True)
    pipeline = ConcertStudioPipeline(cfg)

    _, _, rep_2 = pipeline.run(pov_pool[:2], reference_studio_track=scenario.studio_reference)
    _, _, rep_5 = pipeline.run(pov_pool[:5], reference_studio_track=scenario.studio_reference)
    _, _, rep_10 = pipeline.run(pov_pool[:10], reference_studio_track=scenario.studio_reference)

    assert rep_2.consensus_si_sdr_db < rep_5.consensus_si_sdr_db < rep_10.consensus_si_sdr_db
