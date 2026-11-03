"""Stage 3: Waveform cubic-spline de-clipping & STFT late-reverberation WPE suppression."""

from __future__ import annotations

import numpy as np
from scipy import interpolate, ndimage, signal


def declip_waveform(
    samples: np.ndarray,
    clip_threshold: float = 0.96,
    max_run_samples: int = 96,
) -> tuple[np.ndarray, int]:
    """Detect saturated ADC clipping intervals and reconstruct peaks via parabolic/spline arch interpolation."""
    orig = np.asarray(samples)
    peak_all = float(np.max(np.abs(orig))) if orig.size > 0 else 0.0
    if peak_all < 0.55 and peak_all < clip_threshold:
        return orig, 0

    arr = orig.copy()
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[:, None]

    total_repaired = 0
    n_samples, n_channels = arr.shape

    for ch in range(n_channels):
        x = arr[:, ch]
        peak = float(np.max(np.abs(x)))
        if peak < 1e-6:
            continue
        eff_thresh = min(clip_threshold, 0.985 * peak) if peak > 0.55 else clip_threshold
        clipped = np.abs(x) >= eff_thresh
        if not np.any(clipped):
            continue

        # Find contiguous clipped runs
        padded = np.concatenate([[False], clipped, [False]])
        starts = np.flatnonzero(padded[1:] & ~padded[:-1])
        ends = np.flatnonzero(~padded[1:] & padded[:-1])

        for s, e in zip(starts, ends):
            run_len = int(e - s)
            if run_len < 1 or run_len > max_run_samples:
                continue
            margin = min(6, max(2, run_len // 4))
            left_idx = max(0, s - margin)
            right_idx = min(n_samples, e + margin)
            anchor_indices = np.concatenate(
                [np.arange(left_idx, s), np.arange(e, right_idx)]
            )
            if len(anchor_indices) < 4:
                continue
            # Center time coordinate around s to keep polynomial conditioning near O(1)
            t_anchors = (anchor_indices - s).astype(np.float64) / max(float(run_len), 1.0)
            t_eval = (np.arange(s, e, dtype=np.float64) - s) / max(float(run_len), 1.0)
            poly_deg = 2 if run_len > 12 else 3
            coeffs = np.polyfit(t_anchors, x[anchor_indices], deg=poly_deg)
            reconstructed = np.polyval(coeffs, t_eval)
            sign = np.sign(x[s])
            x[s:e] = sign * np.clip(np.maximum(np.abs(reconstructed), np.abs(x[s:e])), 0.0, peak * 1.6)
            total_repaired += run_len

        arr[:, ch] = x

    return (arr[:, 0] if is_1d else arr), total_repaired


def dereverberate_stadium(
    samples: np.ndarray,
    sr: int = 48000,
    prediction_delay_frames: int = 3,
    prediction_taps: int = 6,
    alpha_subtract: float = 0.42,
    n_fft: int = 2048,
) -> tuple[np.ndarray, float]:
    """Suppress diffuse stadium late reverberation via STFT delayed linear prediction & spectral subtraction.

    Returns (dereverberated_samples, reverb_tail_reduction_db).
    """
    arr = np.asarray(samples, dtype=np.float64)
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[:, None]

    hop = n_fft // 4
    out = np.zeros_like(arr)
    tail_reductions_db: list[float] = []

    for ch in range(arr.shape[1]):
        x = arr[:, ch]
        _, _, Z = signal.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        power = np.abs(Z) ** 2
        n_bins, n_frames = power.shape

        # Estimate late reverberant power via exponentially weighted delayed tap accumulator (WPE-inspired)
        late_reverb_power = np.zeros_like(power)
        tap_weights = np.exp(-0.45 * np.arange(prediction_taps))
        tap_weights /= np.sum(tap_weights)

        for k, w in enumerate(tap_weights):
            d = prediction_delay_frames + k
            if d < n_frames:
                late_reverb_power[:, d:] += w * power[:, :-d]

        # Smooth late reverb estimate across frequency
        late_reverb_power = ndimage.gaussian_filter(late_reverb_power, sigma=(1.5, 1.0))

        # Protect sharp musical onsets (where instantaneous power rises above delayed power)
        onset_ratio = power / (late_reverb_power + 1e-12)
        suppression_gain = np.sqrt(
            np.clip(1.0 - alpha_subtract / np.maximum(onset_ratio, 0.55), 0.22, 1.0)
        )

        Z_clean = Z * suppression_gain
        _, y = signal.istft(Z_clean, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        y = y[: len(x)]
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        out[:, ch] = y

        removed_energy = float(np.mean(power) / (np.mean(np.abs(Z_clean) ** 2) + 1e-12))
        tail_reductions_db.append(float(10.0 * np.log10(max(removed_energy, 1.0))))

    mean_reduction_db = float(np.mean(tail_reductions_db)) if tail_reductions_db else 0.0
    return (out[:, 0] if is_1d else out), mean_reduction_db
