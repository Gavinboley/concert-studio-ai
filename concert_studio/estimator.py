"""Quality-tier scaling configuration and analytical processing time & resolution estimator."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from concert_studio.models import ProcessingEstimate


@dataclass(frozen=True)
class QualityTierSpec:
    """Internal parameters controlled by the user-selected Quality Tier."""

    name: str
    stft_resolutions: list[int]
    sync_subwindows: int
    wpe_taps: int
    enable_echo_cancel: bool
    enable_nonrigid_motion: bool
    enable_agc_inversion: bool
    cost_multiplier: float
    tier_resolution_bonus: float


QUALITY_TIERS: dict[str, QualityTierSpec] = {
    "draft": QualityTierSpec(
        name="draft",
        stft_resolutions=[1024],
        sync_subwindows=4,
        wpe_taps=3,
        enable_echo_cancel=False,
        enable_nonrigid_motion=False,
        enable_agc_inversion=False,
        cost_multiplier=0.45,
        tier_resolution_bonus=0.0,
    ),
    "standard": QualityTierSpec(
        name="standard",
        stft_resolutions=[2048],
        sync_subwindows=8,
        wpe_taps=5,
        enable_echo_cancel=True,
        enable_nonrigid_motion=True,
        enable_agc_inversion=True,
        cost_multiplier=1.0,
        tier_resolution_bonus=8.0,
    ),
    "high": QualityTierSpec(
        name="high",
        stft_resolutions=[1024, 2048],
        sync_subwindows=14,
        wpe_taps=6,
        enable_echo_cancel=True,
        enable_nonrigid_motion=True,
        enable_agc_inversion=True,
        cost_multiplier=1.85,
        tier_resolution_bonus=15.0,
    ),
    "ultra": QualityTierSpec(
        name="ultra",
        stft_resolutions=[1024, 2048, 4096],
        sync_subwindows=22,
        wpe_taps=8,
        enable_echo_cancel=True,
        enable_nonrigid_motion=True,
        enable_agc_inversion=True,
        cost_multiplier=3.1,
        tier_resolution_bonus=22.0,
    ),
    "archival_max": QualityTierSpec(
        name="archival_max",
        stft_resolutions=[512, 1024, 2048, 4096],
        sync_subwindows=32,
        wpe_taps=10,
        enable_echo_cancel=True,
        enable_nonrigid_motion=True,
        enable_agc_inversion=True,
        cost_multiplier=4.8,
        tier_resolution_bonus=28.0,
    ),
}


def get_quality_tier_spec(quality_tier: str) -> QualityTierSpec:
    key = quality_tier.lower().strip()
    if key not in QUALITY_TIERS:
        raise ValueError(f"Unknown quality_tier '{quality_tier}'. Valid tiers: {list(QUALITY_TIERS)}")
    return QUALITY_TIERS[key]


def estimate_processing_plan(
    num_files: int,
    durations_sec: list[float] | None = None,
    avg_duration_sec: float = 180.0,
    sample_rate: int = 48000,
    quality_tier: str = "ultra",
) -> ProcessingEstimate:
    """Estimate processing wall-clock time, SNR gain, and resolution score from file count & tier.

    More POV files directly increase spatial outlier rejection, unoccluded coverage,
    and coherent wavefront SNR (+10 * log10(N_eff) dB).
    """
    if num_files < 1:
        raise ValueError("num_files must be >= 1")

    spec = get_quality_tier_spec(quality_tier)
    if durations_sec and len(durations_sec) > 0:
        durs = [max(float(d), 0.5) for d in durations_sec]
        total_in_sec = float(np.sum(durs))
        est_master_sec = float(np.max(durs) * min(1.0 + 0.08 * np.log1p(len(durs)), 1.35))
        n = len(durs)
    else:
        n = int(num_files)
        total_in_sec = float(n * max(avg_duration_sec, 0.5))
        est_master_sec = float(max(avg_duration_sec, 0.5))

    sr_factor = float(sample_rate) / 48000.0
    # Pairwise timeline graph alignment scales as O(N^2) on compact 250Hz envelopes
    pairwise_sec = 0.0035 * (n**2) * (est_master_sec / 60.0)
    # Per-track anomaly + multi-res STFT consensus scales linearly with total input duration & tier
    per_sec_cost = 0.042 * spec.cost_multiplier * sr_factor
    master_cost = 0.085 * est_master_sec * spec.cost_multiplier * sr_factor

    est_proc_sec = float(max(0.25, pairwise_sec + total_in_sec * per_sec_cost + master_cost))

    # Theoretical coherent wavefront SNR improvement scales as 10 * log10(N_effective)
    coverage_ratio = min(total_in_sec / max(est_master_sec, 1e-3), float(n))
    snr_gain_db = float(
        10.0 * np.log10(max(coverage_ratio, 1.0))
        + 0.18 * spec.tier_resolution_bonus
    )

    # Resolution score [0..100]: scales with POV count (more angles = truer wavefield) + tier depth
    pov_score = 62.0 * (1.0 - np.exp(-float(n) / 6.5))
    res_score = float(np.clip(10.0 + pov_score + spec.tier_resolution_bonus, 15.0, 99.8))

    passes = [
        "Partial-Clip Start/Stop Timeline Graph Auto-Detection",
        "Sub-Sample GCC-PHAT & Quartz Clock-Drift (SRO) Lock",
        "Waveform Cubic Peak De-Clipping",
        "Mic Bump / Shake & Finger Occlusion TF Hand-Off",
        "Multi-Zone Speaker Tuning Composite Equalization",
        f"Multi-Resolution Coherent Wavefront Fusion ({'/'.join(map(str, spec.stft_resolutions))} FFT)",
        f"Stadium Late-Reverb WPE Suppression ({spec.wpe_taps} taps)",
        "True-to-Life Live Capture Loudness Mastering",
    ]
    if spec.enable_nonrigid_motion:
        passes.insert(2, "Non-Rigid Moving-Phone Doppler & Delay Trajectory Tracking")
    if spec.enable_echo_cancel:
        passes.insert(4, "Discrete Delay-Tower & Slapback Cepstral Echo Cancellation")
    if spec.enable_agc_inversion:
        passes.insert(5, "Smartphone AGC Pumping Dynamic Restoration")

    return ProcessingEstimate(
        quality_tier=spec.name,
        num_files=n,
        total_input_duration_sec=round(total_in_sec, 2),
        estimated_master_duration_sec=round(est_master_sec, 2),
        estimated_processing_sec=round(est_proc_sec, 2),
        expected_snr_gain_db=round(snr_gain_db, 2),
        expected_resolution_score=round(res_score, 1),
        stft_resolutions=list(spec.stft_resolutions),
        active_passes=passes,
    )
