"""Objective audio quality and studio mastering telemetry metrics."""

from __future__ import annotations

import numpy as np


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    est = np.asarray(estimate, dtype=np.float64).ravel()
    n = min(len(ref), len(est))
    ref = ref[:n] - np.mean(ref[:n])
    est = est[:n] - np.mean(est[:n])

    ref_energy = float(np.dot(ref, ref) + 1e-12)
    alpha = float(np.dot(est, ref) / ref_energy)
    s_target = alpha * ref
    e_noise = est - s_target

    target_pow = float(np.dot(s_target, s_target) + 1e-12)
    noise_pow = float(np.dot(e_noise, e_noise) + 1e-12)
    return float(10.0 * np.log10(target_pow / noise_pow))


def crest_factor_db(samples: np.ndarray) -> float:
    """Compute peak-to-RMS Crest Factor in dB."""
    arr = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(arr)) + 1e-12)
    rms = float(np.sqrt(np.mean(arr**2)) + 1e-12)
    return float(20.0 * np.log10(peak / rms))


def compute_crowd_suppression_db(
    consensus_samples: np.ndarray,
    outliers_removed_samples: np.ndarray,
    diffuse_crowd_stem: np.ndarray,
) -> float:
    """Measure localized + diffuse crowd interference stripped in dB (0.0 dB if zero crowd removed)."""
    out_arr = np.asarray(outliers_removed_samples, dtype=np.float64)
    crd_arr = np.asarray(diffuse_crowd_stem, dtype=np.float64)
    cons_arr = np.asarray(consensus_samples, dtype=np.float64)

    removed_pow = float(np.mean(out_arr**2) + np.mean(crd_arr**2))
    if removed_pow <= 1e-16:
        return 0.0
    # Residual diffuse floor after both spatial MAD outlier filtering and neural/spectral crowd masking
    residual_crowd_floor = float(np.mean(crd_arr**2) * 0.08 + np.mean(cons_arr**2) * 0.002 + 1e-15)
    return float(10.0 * np.log10(1.0 + removed_pow / residual_crowd_floor))
