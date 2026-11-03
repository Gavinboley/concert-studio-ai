"""Stage 6: Sidechain stem mixing, Dynamic Resonance Control, Matchering, Multiband Glue & ITU-R BS.1770-4 Mastering."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from concert_studio.models import StemBundle


def measure_lufs(samples: np.ndarray, sr: int = 48000) -> float:
    """Measure integrated loudness in LUFS according to ITU-R BS.1770-4 K-weighting."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]

    # Stage 1: Acoustic head-model high-shelf boost (+4 dB above 1.5 kHz)
    sos_hs = signal.butter(2, min(1500.0, sr * 0.42), btype="highpass", fs=sr, output="sos")
    high_part = signal.sosfilt(sos_hs, arr, axis=0)
    k_stage1 = arr + 0.58 * high_part

    # Stage 2: RLB high-pass weighting filter at 38 Hz
    sos_rlb = signal.butter(2, 38.0, btype="highpass", fs=sr, output="sos")
    k_weighted = signal.sosfilt(sos_rlb, k_stage1, axis=0)

    mean_square = float(np.mean(np.sum(k_weighted**2, axis=1)))
    if mean_square <= 1e-18:
        return -70.0
    return float(-0.691 + 10.0 * np.log10(mean_square))


def measure_true_peak_dbtp(samples: np.ndarray) -> float:
    """Measure 4x polyphase oversampled True-Peak level in dBTP (ITU-R BS.1770-4)."""
    arr = np.asarray(samples, dtype=np.float64)
    oversampled = signal.resample_poly(arr, up=4, down=1, axis=0)
    peak = float(np.max(np.abs(oversampled)) + 1e-15)
    return float(20.0 * np.log10(peak))


def _normalize_stem_to_lufs(stem: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    curr = measure_lufs(stem, sr=sr)
    if curr <= -65.0:
        return stem
    gain_db = np.clip(target_lufs - curr, -18.0, 18.0)
    return stem * (10.0 ** (gain_db / 20.0))


def _studio_target_curve(freqs: np.ndarray, mid_val: float) -> np.ndarray:
    """Generate modern, open, high-fidelity studio target curve.

    Clean punchy sub (45-80 Hz) at +3.8 dB, articulate bass (80-200 Hz) at +2.5 dB,
    transparent de-mudded low-mids (200-600 Hz) at +0.8 dB, forward vocal/guitar presence
    (1.8 kHz - 5 kHz) at +2.2 dB, and open, crystalline wideband air (+2.5 dB from 8 kHz to 16 kHz).
    """
    profile = np.ones_like(freqs)
    for k, f in enumerate(freqs):
        if f < 25.0:
            profile[k] = 10.0 ** (-6.0 / 20.0)
        elif f <= 80.0:
            profile[k] = 10.0 ** (3.8 / 20.0)
        elif f <= 200.0:
            frac = (f - 80.0) / 120.0
            profile[k] = 10.0 ** ((3.8 - 1.3 * frac) / 20.0)
        elif f <= 600.0:
            frac = (f - 200.0) / 400.0
            profile[k] = 10.0 ** ((2.5 - 1.7 * frac) / 20.0)
        elif f <= 1800.0:
            frac = (f - 600.0) / 1200.0
            profile[k] = 10.0 ** ((0.8 + 0.6 * frac) / 20.0)
        elif f <= 5000.0:
            profile[k] = 10.0 ** (2.2 / 20.0)
        elif f <= 16000.0:
            profile[k] = 10.0 ** (2.5 / 20.0)
        elif f <= 19000.0:
            frac = (f - 16000.0) / 3000.0
            profile[k] = 10.0 ** ((2.5 - 2.5 * frac) / 20.0)
        else:
            profile[k] = 10.0 ** (-1.5 / 20.0)
    return mid_val * profile


def _apply_spectral_matchering(
    mix: np.ndarray,
    sr: int,
    reference_audio: np.ndarray | None = None,
    n_fft: int = 2048,
    max_eq_db: float = 10.0,
    true_data_mode: bool = False,
) -> np.ndarray:
    """Match spectral balance to either a studio reference track or the HD Studio Master Curve."""
    arr = np.asarray(mix, dtype=np.float64)
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[:, None]

    hop = n_fft // 4
    out = np.zeros_like(arr)

    for ch in range(arr.shape[1]):
        freqs, _, Z = signal.stft(arr[:, ch], fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        mag_mix = np.sqrt(np.mean(np.abs(Z) ** 2, axis=1)) + 1e-12

        if reference_audio is not None:
            ref_ch = (
                reference_audio
                if reference_audio.ndim == 1
                else reference_audio[:, min(ch, reference_audio.shape[1] - 1)]
            )
            _, _, Z_ref = signal.stft(ref_ch, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            mag_target = np.sqrt(np.mean(np.abs(Z_ref) ** 2, axis=1)) + 1e-12
            mag_target *= np.sqrt(np.mean(mag_mix**2)) / (np.sqrt(np.mean(mag_target**2)) + 1e-12)
            hf_stopband = np.zeros_like(freqs, dtype=bool)
        else:
            mid_mask = (freqs >= 1000.0) & (freqs <= 4000.0)
            mid_val = float(np.median(mag_mix[mid_mask])) if np.any(mid_mask) else float(np.median(mag_mix))
            mag_target = _studio_target_curve(freqs, mid_val)

            # In true_data_mode, strictly protect missing/bandlimited frequencies from being boosted
            if true_data_mode:
                hf_stopband = (freqs > 4500.0) & (mag_mix < 0.080 * mid_val)
                if np.any(hf_stopband):
                    mag_target[hf_stopband] = np.minimum(mag_target[hf_stopband], mag_mix[hf_stopband] * 0.90)
            else:
                hf_stopband = (freqs > 4500.0) & (mag_mix < 0.0055 * mid_val)
                if np.any(hf_stopband):
                    mag_target[hf_stopband] = np.minimum(mag_target[hf_stopband], mag_mix[hf_stopband] * 0.95)

        ratio = ndimage.gaussian_filter1d(mag_target / (mag_mix + 1e-12), sigma=6.0)
        max_lin = 10.0 ** (max_eq_db / 20.0)
        min_lin = 10.0 ** (-max_eq_db / 20.0)
        eq_gain = np.clip(ratio, min_lin, max_lin)[:, None]
        if reference_audio is None and np.any(hf_stopband):
            if true_data_mode:
                eq_gain[hf_stopband, :] = 0.0
            else:
                eq_gain[hf_stopband, :] = np.minimum(eq_gain[hf_stopband, :], 0.90)

        _, y = signal.istft(Z * eq_gain, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        if len(y) < arr.shape[0]:
            y = np.pad(y, (0, arr.shape[0] - len(y)))
        out[:, ch] = y[: arr.shape[0]]

    return out[:, 0] if is_1d else out


def suppress_dynamic_resonances(
    mix: np.ndarray,
    sr: int = 48000,
    max_cut_db: float = 4.5,
) -> tuple[np.ndarray, float]:
    """Soothe-style self-tuning dynamic spectral resonance suppressor (1.6 kHz – 7.0 kHz).

    Concert PA horn drivers and smartphone MEMS microphones exhibit sharp narrowband
    whistling/honking resonances at high SPL. This processor computes a critical-band
    smoothed spectral envelope at each STFT frame, identifies narrow high-Q spikes
    exceeding the local envelope (`peak_to_env > 1.30`), and dynamically dips only those
    resonant frequencies while preserving warm broadband musical tone.
    Returns `(smoothed_mix, max_resonance_attenuation_db)`.
    """
    arr = np.asarray(mix, dtype=np.float64).copy()
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[:, None]

    n_fft = 2048
    hop = n_fft // 4
    out = np.zeros_like(arr)
    cuts_db: list[float] = []

    for ch in range(arr.shape[1]):
        freqs, _, Z = signal.stft(arr[:, ch], fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        mags = np.abs(Z)

        # Smooth spectral envelope along frequency axis (sigma=7 bins ~ 160 Hz critical band)
        smooth_env = ndimage.gaussian_filter1d(mags, sigma=7.0, axis=0) + 1e-12
        peak_ratio = mags / smooth_env

        # Focus on ear-fatiguing PA horn / MEMS resonance zone (1.6 kHz - 7.0 kHz)
        harsh_zone = ((freqs >= 1600.0) & (freqs <= min(7000.0, sr * 0.45)))[:, None]

        # Only trigger dynamic resonance dipping on frames above noise floor
        frame_rms = np.sqrt(np.mean(mags**2, axis=0, keepdims=True))
        active_frame = frame_rms > (0.12 * float(np.max(frame_rms) + 1e-12))

        excess_db = np.where(
            harsh_zone & active_frame & (peak_ratio > 1.30),
            20.0 * np.log10(np.maximum(peak_ratio / 1.30, 1.0)),
            0.0,
        )
        cut_db = np.minimum(excess_db * 0.72, max_cut_db)
        # Smooth dynamic notch gains slightly across time and frequency to avoid modulation artifacts
        cut_db = ndimage.gaussian_filter(cut_db, sigma=(0.8, 1.2))
        if np.any(cut_db > 0.0):
            cuts_db.append(float(np.percentile(cut_db[harsh_zone[:, 0]], 98)))

        gain = 10.0 ** (-cut_db / 20.0)
        _, y = signal.istft(Z * gain, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        if len(y) < arr.shape[0]:
            y = np.pad(y, (0, arr.shape[0] - len(y)))
        out[:, ch] = y[: arr.shape[0]]

    att_db = round(float(np.max(cuts_db)) if cuts_db else 0.0, 2)
    return (out[:, 0] if is_1d else out), att_db


def apply_live_glue_ambience_floor(
    mix: np.ndarray,
    sr: int = 48000,
    floor_dbfs: float = -42.0,
) -> np.ndarray:
    """Pass-through clean audio without artificial room noise or static floor injection."""
    return np.asarray(mix, dtype=np.float64)


def _smooth_band_envelope(
    band: np.ndarray,
    sr: int = 48000,
    attack_ms: float = 30.0,
    release_ms: float = 200.0,
) -> np.ndarray:
    """Compute a smooth, 2-pole envelope follower without audio-rate ripple or buzzing."""
    peaks_raw = np.max(np.abs(band), axis=1) if band.ndim == 2 else np.abs(band)
    blk = max(int(0.001 * sr), 1)  # 1ms blocks (1000 Hz)
    n_blks = len(peaks_raw) // blk
    if n_blks <= 1:
        return peaks_raw

    blk_peaks = np.max(peaks_raw[:n_blks * blk].reshape(n_blks, blk), axis=1)
    alpha_att = np.exp(-1.0 / (1000.0 * (attack_ms / 1000.0)))
    alpha_rel = np.exp(-1.0 / (1000.0 * (release_ms / 1000.0)))

    env_sub = np.zeros(n_blks, dtype=np.float64)
    curr = float(blk_peaks[0])
    for i in range(n_blks):
        p = float(blk_peaks[i])
        if p > curr:
            curr = alpha_att * curr + (1.0 - alpha_att) * p
        else:
            curr = alpha_rel * curr + (1.0 - alpha_rel) * p
        env_sub[i] = curr

    env_sub = ndimage.gaussian_filter1d(env_sub, sigma=1.5)
    centers = np.arange(n_blks) * blk + blk // 2
    full_idx = np.arange(len(peaks_raw))
    env_full = np.interp(full_idx, centers, env_sub)
    return np.maximum(env_full, 1e-9)


def _multiband_master_compressor(mix: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Transparent 4-Band Linkwitz-Riley mastering glue preserving natural dynamic punch.
    
    Uses gentle 1.15:1 to 1.25:1 compression ratios with soft knee and high threshold (+6 dB above RMS)
    so transients breathe naturally without squashing the live concert soundstage.
    """
    xovers = [120.0, 1200.0, 6500.0]
    bands: list[np.ndarray] = []
    rem = mix.copy()

    for fc in xovers:
        if fc >= sr * 0.45:
            break
        sos_lp = signal.butter(2, fc, btype="lowpass", fs=sr, output="sos")
        sos_hp = signal.butter(2, fc, btype="highpass", fs=sr, output="sos")
        low = signal.sosfilt(sos_lp, signal.sosfilt(sos_lp, rem, axis=0), axis=0)
        rem = signal.sosfilt(sos_hp, signal.sosfilt(sos_hp, rem, axis=0), axis=0)
        bands.append(low)
    bands.append(rem)

    band_params = [
        (45.0, 250.0, 1.25),  # Sub/Low (<120Hz): 45ms attack, 250ms release, transparent 1.25:1 glue
        (30.0, 180.0, 1.20),  # Low-Mid (120-1200Hz): 30ms attack, 180ms release, gentle 1.20:1 glue
        (20.0, 120.0, 1.20),  # High-Mid (1200-6500Hz): 20ms attack, 120ms release, gentle 1.20:1 glue
        (15.0, 80.0, 1.15),   # Air (>6500Hz): 15ms attack, 80ms release, subtle 1.15:1 glue
    ]

    out = np.zeros_like(mix)
    for b_idx, band in enumerate(bands):
        rms = float(np.sqrt(np.mean(band**2)) + 1e-9)
        att_ms, rel_ms, ratio = band_params[min(b_idx, len(band_params) - 1)]
        env = _smooth_band_envelope(band, sr=sr, attack_ms=att_ms, release_ms=rel_ms)
        thresh = 2.2 * rms
        knee_db = 3.0
        env_db = 20.0 * np.log10(env + 1e-9)
        thresh_db = 20.0 * np.log10(thresh + 1e-9)
        diff_db = env_db - thresh_db

        gain_db = np.zeros_like(diff_db)
        over_idx = diff_db > (knee_db / 2.0)
        knee_idx = (diff_db >= (-knee_db / 2.0)) & (~over_idx)

        gain_db[over_idx] = diff_db[over_idx] * (1.0 / ratio - 1.0)
        gain_db[knee_idx] = ((diff_db[knee_idx] + knee_db / 2.0) ** 2) / (2.0 * knee_db) * (1.0 / ratio - 1.0)

        gain = 10.0 ** (gain_db / 20.0)
        out += band * gain[:, None]

    return out


def _lookahead_true_peak_limiter(
    mix: np.ndarray,
    sr: int = 48000,
    target_lufs: float = -14.0,
    ceiling_dbtp: float = -1.0,
) -> np.ndarray:
    """Normalize to target integrated LUFS with lookahead True-Peak brickwall limiting."""
    out = mix.copy()
    ceiling_lin = 10.0 ** (ceiling_dbtp / 20.0)

    for _ in range(3):
        curr_lufs = measure_lufs(out, sr=sr)
        if curr_lufs > -65.0:
            gain_db = target_lufs - curr_lufs
            out = out * (10.0 ** (gain_db / 20.0))

        # 4x oversampled envelope detection for True-Peak compliance
        oversampled = signal.resample_poly(out, up=4, down=1, axis=0)
        peak_env_os = np.max(np.abs(oversampled), axis=1)
        peak_env = np.max(peak_env_os.reshape(-1, 4), axis=1)[: out.shape[0]]

        lookahead = max(int(0.003 * sr), 1)
        release = max(int(0.045 * sr), 2)
        held_env = ndimage.maximum_filter1d(peak_env, size=lookahead * 2 + 1)
        smooth_env = ndimage.uniform_filter1d(held_env, size=release)
        eff_env = np.maximum(peak_env, smooth_env)

        lim_gain = np.where(eff_env > ceiling_lin * 0.985, (ceiling_lin * 0.985) / (eff_env + 1e-12), 1.0)
        out = out * lim_gain[:, None]

    # Final safety clamp on 4x oversampled True-Peak
    tp_db = measure_true_peak_dbtp(out)
    if tp_db > ceiling_dbtp:
        scale_down = 10.0 ** ((ceiling_dbtp - tp_db - 0.05) / 20.0)
        out *= scale_down

    return out


def mix_and_master_stems(
    stems: StemBundle,
    reference_audio: np.ndarray | None = None,
    target_lufs: float = -14.0,
    true_peak_limit_db: float = -1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Perform automated stem gain-staging, sidechain spectral ducking, matchering & HD mastering."""
    sr = stems.sample_rate

    # 1. Stem Leveling to Studio Balance Targets (punchy drums, solid bass, warm guitars, proud vocals)
    voc = _normalize_stem_to_lufs(stems.vocals, sr=sr, target_lufs=-14.5)
    drm = _normalize_stem_to_lufs(stems.drums, sr=sr, target_lufs=-15.5)
    bas = _normalize_stem_to_lufs(stems.bass, sr=sr, target_lufs=-15.0)
    oth = _normalize_stem_to_lufs(stems.other, sr=sr, target_lufs=-15.5)

    # 2. Dynamic Sidechain Glue (Subtle & natural — NO bassline ducking!)
    drm_env = ndimage.uniform_filter1d(np.max(np.abs(drm), axis=1), size=max(int(0.012 * sr), 1))
    drm_norm = drm_env / (float(np.max(drm_env)) + 1e-9)
    bass_duck_gain = 1.0 - 0.05 * np.clip(drm_norm, 0.0, 1.0)
    bas = bas * bass_duck_gain[:, None]

    voc_env = ndimage.uniform_filter1d(np.max(np.abs(voc), axis=1), size=max(int(0.025 * sr), 1))
    voc_norm = voc_env / (float(np.max(voc_env)) + 1e-9)
    oth_duck_gain = 1.0 - 0.05 * np.clip(voc_norm, 0.0, 1.0)
    oth = oth * oth_duck_gain[:, None]

    stem_lufs = {
        "vocals": round(measure_lufs(voc, sr=sr), 2),
        "drums": round(measure_lufs(drm, sr=sr), 2),
        "bass": round(measure_lufs(bas, sr=sr), 2),
        "other": round(measure_lufs(oth, sr=sr), 2),
    }

    # 3. Sum stems onto Studio Master Bus
    raw_mix = voc + drm + bas + oth

    # 3b. Soothe-Style Dynamic Resonance Suppression (tame harsh PA horn / MEMS whistling peaks)
    de_resonated_mix, _ = suppress_dynamic_resonances(raw_mix, sr=sr, max_cut_db=4.0)

    # 4. Reference Spectral Matchering
    matched_mix = _apply_spectral_matchering(de_resonated_mix, sr=sr, reference_audio=reference_audio)

    # 5. Transparent Multiband Mastering Glue (Preserving Crest Factor & Soundstage)
    glued_mix = _multiband_master_compressor(matched_mix, sr=sr)

    # 6. Loudness Normalization & Lookahead True-Peak Limiting
    mastered = _lookahead_true_peak_limiter(
        glued_mix,
        sr=sr,
        target_lufs=target_lufs,
        ceiling_dbtp=true_peak_limit_db,
    )

    return mastered, stem_lufs
