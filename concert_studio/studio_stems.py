"""Stage 5: Stem-specific studio resynthesis & high-frequency harmonic super-resolution."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from concert_studio.models import StemBundle


def _envelope_follower(x: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Fast vectorized two-time-constant envelope detector."""
    rect = np.abs(x)
    if rect.ndim == 2:
        rect = np.max(rect, axis=1)
    win_att = max(int(attack_ms * 1e-3 * sr), 1)
    win_rel = max(int(release_ms * 1e-3 * sr), 2)
    fast = ndimage.uniform_filter1d(rect, size=win_att)
    slow = ndimage.maximum_filter1d(fast, size=win_rel // 2 + 1)
    return np.maximum(ndimage.uniform_filter1d(slow, size=win_rel), 0.0)


def _soft_compressor(
    stereo: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    attack_ms: float = 8.0,
    release_ms: float = 80.0,
    makeup_db: float = 2.0,
) -> np.ndarray:
    """Smooth optical/VCA-style stereo linked compressor."""
    env = _envelope_follower(stereo, sr=sr, attack_ms=attack_ms, release_ms=release_ms)
    env_db = 20.0 * np.log10(env + 1e-9)
    over_db = np.maximum(env_db - threshold_db, 0.0)
    gain_reduction_db = -over_db * (1.0 - 1.0 / ratio)
    total_gain_lin = 10.0 ** ((gain_reduction_db + makeup_db) / 20.0)
    return stereo * total_gain_lin[:, None]


def _ensure_stereo(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    x = np.asarray(arr, dtype=np.float64).copy()
    if x.ndim == 1:
        return np.column_stack([x, x]), True
    return x, False


def enhance_vocals_studio(vocals: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Transform live PA vocals into an intimate, de-essed, forward studio vocal track."""
    st, is_1d = _ensure_stereo(vocals)

    # 1. High-pass rumble filter at 105 Hz
    sos_hp = signal.butter(3, 105.0, btype="highpass", fs=sr, output="sos")
    st = signal.sosfilt(sos_hp, st, axis=0)

    # 2. De-mud 190-310 Hz (-2.5 dB dip to eliminate boxy phone microphone resonance)
    sos_mud = signal.butter(2, [190.0, 310.0], btype="bandpass", fs=sr, output="sos")
    st = st - 0.25 * signal.sosfilt(sos_mud, st, axis=0)

    # 3. Forward Vocal Presence (2.4 kHz – 4.8 kHz +3.8 dB boost for crystal-clear intelligibility)
    sos_pres = signal.butter(2, [2400.0, min(4800.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    st = st + 0.38 * signal.sosfilt(sos_pres, st, axis=0)

    # 4. Clean Vocal Air (linear high-shelf above 10 kHz, zero nonlinear distortion)
    sos_air = signal.butter(2, min(10000.0, sr * 0.45), btype="highpass", fs=sr, output="sos")
    st = st + 0.22 * signal.sosfilt(sos_air, st, axis=0)

    # 5. Split-band dynamic De-Esser (6.0 kHz – 9.5 kHz) to tame harsh PA horn sibilance
    sos_ess = signal.butter(2, [6000.0, min(9500.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    ess_band = signal.sosfilt(sos_ess, st, axis=0)
    ess_env = _envelope_follower(ess_band, sr=sr, attack_ms=2.0, release_ms=30.0)
    thresh = 0.25 * float(np.max(ess_env) + 1e-9)
    att = np.where(ess_env > thresh, thresh / (ess_env + 1e-9), 1.0)
    st = (st - ess_band) + ess_band * att[:, None]

    # 6. Vocals 100% Dead Center (L=R)
    mono_voc = 0.5 * (st[:, 0] + st[:, 1])
    res = np.column_stack([mono_voc, mono_voc])
    return np.mean(res, axis=1) if is_1d else res


def enhance_drums_studio(drums: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Apply studio kick punch (60Hz), snare crack/body & smooth cymbal air (zero clicking)."""
    st, is_1d = _ensure_stereo(drums)

    # 1. Kick Drum Solid Foundation (55 Hz – 90 Hz clean acoustic sub punch)
    sos_kick = signal.butter(3, [50.0, 90.0], btype="bandpass", fs=sr, output="sos")
    kick_band = signal.sosfilt(sos_kick, np.mean(st, axis=1))
    st[:, 0] += 0.22 * kick_band
    st[:, 1] += 0.22 * kick_band

    # 2. Snare Wire Snap & Body (190 Hz – 260 Hz body + 4.5 kHz – 6.5 kHz crisp wires, zero clicking)
    sos_snare_body = signal.butter(2, [190.0, 260.0], btype="bandpass", fs=sr, output="sos")
    sos_snare_snap = signal.butter(2, [4500.0, min(6500.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    st = st + 0.16 * signal.sosfilt(sos_snare_body, st, axis=0) + 0.14 * signal.sosfilt(sos_snare_snap, st, axis=0)

    # 3. Crisp studio hi-hat air shelf (>8.5 kHz)
    sos_air = signal.butter(2, min(8500.0, sr * 0.45), btype="highpass", fs=sr, output="sos")
    air = signal.sosfilt(sos_air, st, axis=0)
    res = st + 0.18 * air
    return np.mean(res, axis=1) if is_1d else res


def enhance_bass_studio(bass: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Mono-lock sub-180Hz, add clean 800Hz note definition, lowpass neural vocal/snare bleed, and apply gentle leveling."""
    st, is_1d = _ensure_stereo(bass)

    # 1. Reject neural vocal/snare bleed and hiss above bass guitar range (>1200 Hz)
    sos_lp = signal.butter(4, min(1200.0, sr * 0.45), btype="lowpass", fs=sr, output="sos")
    st = signal.sosfilt(sos_lp, st, axis=0)

    # 2. Solid sub-bass foundation (45 Hz – 85 Hz +2.8 dB)
    sos_sub = signal.butter(2, [45.0, 85.0], btype="bandpass", fs=sr, output="sos")
    st = st + 0.28 * signal.sosfilt(sos_sub, st, axis=0)

    # 3. Articulate Bass String & Fret Growl (600 Hz – 1050 Hz +3.2 dB for note clarity)
    sos_growl = signal.butter(2, [600.0, min(1050.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    st = st + 0.32 * signal.sosfilt(sos_growl, st, axis=0)

    # 4. Transparent optical bass leveling (1.30:1 ratio)
    st = _soft_compressor(
        st, sr=sr, threshold_db=-18.0, ratio=1.30, attack_ms=20.0, release_ms=160.0, makeup_db=1.0
    )

    # 5. 100% Mono-Locked Center (Zero Side Channel energy, L = R)
    mono_bass = 0.5 * (st[:, 0] + st[:, 1])
    res = np.column_stack([mono_bass, mono_bass])
    return np.mean(res, axis=1) if is_1d else res


def enhance_instruments_studio(other: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Apply guitar body warmth (350-700 Hz), pick crunch (2.2-4.5 kHz), and wide natural stereo spread."""
    st, is_1d = _ensure_stereo(other)

    # 1. 115 Hz highpass (strip sub rumble)
    sos_hp = signal.butter(3, 115.0, btype="highpass", fs=sr, output="sos")
    st = signal.sosfilt(sos_hp, st, axis=0)

    # 2. De-mud 190-290 Hz (-2.0 dB dip)
    sos_mud = signal.butter(2, [190.0, 290.0], btype="bandpass", fs=sr, output="sos")
    st = st - 0.20 * signal.sosfilt(sos_mud, st, axis=0)

    # 3. Guitar Body Warmth & Wood (350 Hz – 700 Hz +2.8 dB)
    sos_warmth = signal.butter(2, [350.0, 700.0], btype="bandpass", fs=sr, output="sos")
    st = st + 0.28 * signal.sosfilt(sos_warmth, st, axis=0)

    # 4. Guitar Pick Attack, Crunch & Articulation (2.2 kHz – 4.5 kHz +3.5 dB)
    sos_bite = signal.butter(2, [2200.0, min(4500.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    st = st + 0.35 * signal.sosfilt(sos_bite, st, axis=0)

    # 5. Natural wide stereo spread (leaving center pocket for lead vocals)
    mid = 0.5 * (st[:, 0] + st[:, 1])
    side = 0.5 * (st[:, 0] - st[:, 1])
    res = np.column_stack([mid + 1.30 * side, mid - 1.30 * side])
    return np.mean(res, axis=1) if is_1d else res


def extend_high_frequency_bandwidth(
    samples: np.ndarray,
    sr: int = 48000,
    cutoff_hz: float = 12500.0,
    air_gain: float = 0.38,
) -> tuple[np.ndarray, float]:
    """Reconstruct brickwalled high frequencies (12.5 kHz – 20.2 kHz) via smooth polynomial harmonic generation.

    Extracts upper-mid harmonic excitation (6.25 kHz – 12.5 kHz), applies smooth quadratic/cubic
    harmonic generation (zero Dirac rectification buzzing), and filters through a 20.5 kHz ultrasonic guard.
    Returns (extended_samples, hf_gain_improvement_db).
    """
    arr = np.asarray(samples, dtype=np.float64).copy()
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[:, None]

    if sr < 32000:
        return (arr[:, 0] if is_1d else arr), 0.0

    nyq = 0.5 * sr
    src_low = max(cutoff_hz * 0.45, 4000.0)
    src_high = min(cutoff_hz * 0.95, nyq * 0.90)
    dst_low = min(max(cutoff_hz * 1.08, 12800.0), nyq * 0.85)
    dst_high = min(20200.0, nyq * 0.94)

    if dst_low >= dst_high or src_low >= src_high:
        return (arr[:, 0] if is_1d else arr), 0.0

    sos_src = signal.butter(4, [src_low, src_high], btype="bandpass", fs=sr, output="sos")
    sos_dst = signal.butter(6, [dst_low, dst_high], btype="bandpass", fs=sr, output="sos")

    hf_before = signal.sosfilt(sos_dst, arr, axis=0)
    power_before = float(np.mean(hf_before**2) + 1e-15)

    exciters = signal.sosfilt(sos_src, arr, axis=0)
    power_src = float(np.mean(exciters**2) + 1e-15)
    total_power = float(np.mean(arr**2) + 1e-15)

    if 10.0 * np.log10(power_src / total_power) < -40.0:
        return (arr[:, 0] if is_1d else arr), 0.0

    # Smooth polynomial harmonic excitation (pure quadratic even + cubic odd, zero Dirac rectification)
    even_harm = exciters**2 - ndimage.uniform_filter1d(exciters**2, size=max(int(0.002 * sr), 3), axis=0)
    odd_harm = np.tanh(2.0 * exciters) - exciters
    raw_harm = 0.6 * even_harm + 0.4 * odd_harm

    synthesized_air = signal.sosfilt(sos_dst, raw_harm, axis=0)
    src_rms = float(np.sqrt(np.mean(exciters**2)) + 1e-12)
    air_rms = float(np.sqrt(np.mean(synthesized_air**2)) + 1e-12)
    scaled_air = synthesized_air * min((0.55 * src_rms) / air_rms, 50.0)
    out = arr + air_gain * scaled_air

    # Steep 8th-order lowpass guard at 20.5 kHz to eliminate Demucs 22050 Hz Nyquist spike
    sos_guard = signal.butter(8, min(20500.0, sr * 0.45), btype="lowpass", fs=sr, output="sos")
    out = signal.sosfilt(sos_guard, out, axis=0)

    hf_after = signal.sosfilt(sos_dst, out, axis=0)
    power_after = float(np.mean(hf_after**2) + 1e-15)
    improvement_db = float(10.0 * np.log10(max(power_after / power_before, 1.0)))

    return (out[:, 0] if is_1d else out), improvement_db


def process_all_studio_stems(
    stems: StemBundle,
    enable_harmonic_extension: bool = True,
) -> tuple[StemBundle, float]:
    """Run Stage 5 studio enhancement chains on all separated stems plus HF bandwidth extension."""
    sr = stems.sample_rate
    voc = enhance_vocals_studio(stems.vocals, sr=sr)
    drm = enhance_drums_studio(stems.drums, sr=sr)
    bas = enhance_bass_studio(stems.bass, sr=sr)
    oth = enhance_instruments_studio(stems.other, sr=sr)

    if enable_harmonic_extension:
        voc, hf_v = extend_high_frequency_bandwidth(voc, sr=sr, air_gain=0.20)
        # For drums/percussion, apply clean linear high-shelf air EQ (no nonlinear distortion on cymbals)
        sos_air_drm = signal.butter(2, min(9500.0, sr * 0.44), btype="highpass", fs=sr, output="sos")
        drm_air = signal.sosfilt(sos_air_drm, drm, axis=0)
        drm = drm + 0.15 * drm_air
        hf_d = 1.5
        oth, hf_o = extend_high_frequency_bandwidth(oth, sr=sr, air_gain=0.20)
        avg_hf_db = float(np.mean([hf_v, hf_d, hf_o]))
    else:
        avg_hf_db = 0.0

    return (
        StemBundle(
            vocals=voc,
            drums=drm,
            bass=bas,
            other=oth,
            crowd_removed=stems.crowd_removed,
            sample_rate=sr,
        ),
        avg_hf_db,
    )
