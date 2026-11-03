"""Stage 1: Sub-sample temporal alignment & Sampling Rate Offset (SRO) clock-drift correction."""

from __future__ import annotations

import numpy as np
from scipy import signal

from concert_studio.models import AlignmentParameters, AudioTrack


def _estimate_snr_and_clipping(samples: np.ndarray, clip_limit: float = 0.96) -> tuple[float, float]:
    """Estimate broadband signal-to-noise ratio (dB) and fraction of clipped samples."""
    mono = samples if samples.ndim == 1 else np.mean(samples, axis=1)
    rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
    # Estimate noise floor via high-frequency median absolute difference normalized by signal RMS
    hf_noise = float(np.median(np.abs(np.diff(mono, n=2))) / (2.0 * 0.6745) + 1e-12)
    snr_db = float(20.0 * np.log10(max(rms / hf_noise, 1e-3)))
    peak = float(np.max(np.abs(mono)) + 1e-12)
    # Detect both near-full-scale clipping and flat-top consecutive plateau clipping
    near_fs = np.abs(mono) >= min(clip_limit, 0.985 * peak)
    flat_top = (np.abs(mono[:-1]) > 0.45 * peak) & (np.abs(np.diff(mono)) < 1e-6)
    clip_ratio = float(np.mean(near_fs) + np.mean(flat_top))
    return snr_db, clip_ratio


def select_reference_pov(tracks: list[AudioTrack]) -> int:
    """Select the highest-quality POV track as the temporal and spectral anchor."""
    if not tracks:
        raise ValueError("tracks list cannot be empty")

    best_idx = 0
    best_score = -1e9
    for idx, trk in enumerate(tracks):
        mono = trk.to_mono()
        snr_db, clip_ratio = _estimate_snr_and_clipping(mono)
        # Prefer wide high-frequency bandwidth (non-pocket POV) and minimal clipping
        freqs, psd = signal.welch(mono, fs=trk.sample_rate, nperseg=min(2048, len(mono)))
        hf_mask = freqs >= 6000.0
        hf_ratio_db = 10.0 * np.log10(
            (np.mean(psd[hf_mask]) + 1e-15) / (np.mean(psd) + 1e-15)
        )
        score = snr_db + 0.5 * hf_ratio_db - 120.0 * clip_ratio
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _gcc_phat_lag(
    ref: np.ndarray,
    target: np.ndarray,
    sr: int,
    max_tau_sec: float = 0.5,
    beta: float = 0.75,
) -> tuple[float, float]:
    """Compute sub-sample lag (t_target - t_ref) in seconds via regularized GCC-PHAT."""
    n = len(ref) + len(target)
    n_fft = 1 << int(np.ceil(np.log2(max(n, 2048))))
    R = np.fft.rfft(ref, n=n_fft)
    T = np.fft.rfft(target, n=n_fft)
    # Cross-power spectrum with partial phase transform (beta=0.75 suppresses local scream spikes)
    G = T * np.conj(R)
    denom = (np.abs(G) ** beta) + 1e-10
    cc = np.fft.irfft(G / denom, n=n_fft)

    max_shift = min(int(max_tau_sec * sr), n_fft // 2 - 2)
    # Rearrange [-max_shift, +max_shift]
    cc_window = np.concatenate([cc[-max_shift:], cc[: max_shift + 1]])
    lags = np.arange(-max_shift, max_shift + 1, dtype=np.float64)

    peak_idx = int(np.argmax(cc_window))
    peak_val = float(cc_window[peak_idx]) / (float(np.std(cc_window)) * 10.0 + 1e-9)
    coherence = float(np.clip(peak_val, 0.0, 1.0))

    # Parabolic sub-sample refinement
    delta = 0.0
    if 0 < peak_idx < len(cc_window) - 1:
        y0, y1, y2 = cc_window[peak_idx - 1], cc_window[peak_idx], cc_window[peak_idx + 1]
        denom_p = y0 - 2.0 * y1 + y2
        if abs(denom_p) > 1e-12:
            delta = float(np.clip(0.5 * (y0 - y2) / denom_p, -0.5, 0.5))

    lag_samples = lags[peak_idx] + delta
    return float(lag_samples / sr), coherence


def estimate_alignment(
    ref: np.ndarray,
    target: np.ndarray,
    sr: int,
    track_name: str = "target",
    max_tau_sec: float = 0.5,
) -> AlignmentParameters:
    """Estimate initial Time-of-Arrival offset (sec) and Sampling Rate Offset clock drift (ppm)."""
    ref_mono = ref if ref.ndim == 1 else np.mean(ref, axis=1)
    tgt_mono = target if target.ndim == 1 else np.mean(target, axis=1)

    snr_db, clip_ratio = _estimate_snr_and_clipping(tgt_mono)

    # Global coarse delay via GCC-PHAT
    coarse_offset_sec, coherence = _gcc_phat_lag(
        ref_mono, tgt_mono, sr=sr, max_tau_sec=max_tau_sec
    )

    # Pre-shift target by integer coarse delay to estimate fine windowed drift (SRO)
    coarse_samples = int(round(coarse_offset_sec * sr))
    applied_coarse_sec = float(coarse_samples) / float(sr)
    n = min(len(ref_mono), len(tgt_mono))
    shifted_tgt = np.zeros(n, dtype=np.float64)
    if coarse_samples >= 0:
        valid = n - coarse_samples
        if valid > 0:
            shifted_tgt[:valid] = tgt_mono[coarse_samples : coarse_samples + valid]
    else:
        s = -coarse_samples
        valid = n - s
        if valid > 0:
            shifted_tgt[s : s + valid] = tgt_mono[:valid]

    # Multi-window local lag tracking for SRO clock-drift estimation
    win_len = min(int(1.2 * sr), n // 3)
    if win_len >= int(0.4 * sr):
        hop = max(win_len // 2, 1)
        centers_sec: list[float] = []
        local_lags_sec: list[float] = []
        for start in range(0, n - win_len + 1, hop):
            end = start + win_len
            w_ref = ref_mono[start:end] * np.hanning(win_len)
            w_tgt = shifted_tgt[start:end] * np.hanning(win_len)
            res_lag, w_coh = _gcc_phat_lag(w_ref, w_tgt, sr=sr, max_tau_sec=0.015, beta=0.7)
            if w_coh > 0.15:
                centers_sec.append((start + 0.5 * win_len) / sr)
                local_lags_sec.append(applied_coarse_sec + res_lag)

        if len(centers_sec) >= 3:
            # Robust Theil-Sen median-slope regression: lag(t) = tau_0 + (drift_ppm * 1e-6) * t
            slopes: list[float] = []
            for i in range(len(centers_sec)):
                for j in range(i + 1, len(centers_sec)):
                    dt = centers_sec[j] - centers_sec[i]
                    if dt > 0.2:
                        slopes.append((local_lags_sec[j] - local_lags_sec[i]) / dt)
            slope = float(np.median(slopes)) if slopes else 0.0
            drift_ppm = float(np.clip(slope * 1e6, -500.0, 500.0))
            intercepts = [
                lag_i - slope * t_i for t_i, lag_i in zip(centers_sec, local_lags_sec)
            ]
            initial_offset_sec = float(np.median(intercepts))
        else:
            initial_offset_sec = coarse_offset_sec
            drift_ppm = 0.0
    else:
        initial_offset_sec = coarse_offset_sec
        drift_ppm = 0.0

    return AlignmentParameters(
        track_name=track_name,
        initial_offset_sec=initial_offset_sec,
        clock_drift_ppm=drift_ppm,
        peak_coherence=coherence,
        snr_db=snr_db,
        clipping_ratio=clip_ratio,
    )


def _undrift_and_shift(
    samples: np.ndarray,
    sr: int,
    offset_sec: float,
    drift_ppm: float,
    target_length: int,
) -> np.ndarray:
    """Invert clock drift (SRO) and initial Time-of-Arrival delay to lock phase to reference."""
    t_ref = np.arange(target_length, dtype=np.float64) / sr
    # Forward corruption was: t_corrupt_lookup = t * (1 - drift) - offset
    # Inverse warps lookup to sample the delayed signal at t_sample = (t_ref + offset) / (1 - drift)
    scale = 1.0 / max(1.0 - drift_ppm * 1e-6, 0.99)
    t_lookup = (t_ref + offset_sec) * scale
    t_src = np.arange(samples.shape[0], dtype=np.float64) / sr

    if samples.ndim == 1:
        return np.interp(t_lookup, t_src, samples, left=0.0, right=0.0)

    out = np.zeros((target_length, samples.shape[1]), dtype=np.float64)
    for ch in range(samples.shape[1]):
        out[:, ch] = np.interp(t_lookup, t_src, samples[:, ch], left=0.0, right=0.0)
    return out


def align_and_undrift_tracks(
    tracks: list[AudioTrack],
    target_sr: int = 48000,
) -> tuple[list[AudioTrack], list[AlignmentParameters], int]:
    """Synchronize and clock-lock all POV tracks to the automatically selected reference POV."""
    if not tracks:
        raise ValueError("No tracks provided for alignment")

    # Resample any mismatched nominal sample rates first
    standardized: list[AudioTrack] = []
    for trk in tracks:
        if trk.sample_rate != target_sr:
            gcd = int(np.gcd(target_sr, trk.sample_rate))
            resampled = signal.resample_poly(
                trk.samples, target_sr // gcd, trk.sample_rate // gcd, axis=0
            )
            standardized.append(AudioTrack(name=trk.name, samples=resampled, sample_rate=target_sr))
        else:
            standardized.append(trk)

    ref_idx = select_reference_pov(standardized)
    ref_track = standardized[ref_idx]
    ref_mono = ref_track.to_mono()
    target_len = ref_track.samples.shape[0]

    aligned_tracks: list[AudioTrack] = []
    alignments: list[AlignmentParameters] = []

    for idx, trk in enumerate(standardized):
        if idx == ref_idx:
            snr_db, clip_ratio = _estimate_snr_and_clipping(trk.samples)
            params = AlignmentParameters(
                track_name=trk.name,
                initial_offset_sec=0.0,
                clock_drift_ppm=0.0,
                peak_coherence=1.0,
                snr_db=snr_db,
                clipping_ratio=clip_ratio,
            )
            aligned_tracks.append(
                AudioTrack(name=trk.name, samples=trk.to_stereo(), sample_rate=target_sr)
            )
            alignments.append(params)
        else:
            params = estimate_alignment(
                ref=ref_mono,
                target=trk.to_mono(),
                sr=target_sr,
                track_name=trk.name,
            )
            warped = _undrift_and_shift(
                trk.to_stereo(),
                sr=target_sr,
                offset_sec=params.initial_offset_sec,
                drift_ppm=params.clock_drift_ppm,
                target_length=target_len,
            )
            aligned_tracks.append(AudioTrack(name=trk.name, samples=warped, sample_rate=target_sr))
            alignments.append(params)

    return aligned_tracks, alignments, ref_idx
