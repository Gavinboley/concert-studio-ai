"""Aggressive Studio Instrument & Vocal Recreation Engine.

Transforms live multi-POV phone captures into an immaculate, direct-from-source
studio live album (Metallica S&M / Nirvana MTV Unplugged broadcast standard).

Features:
- Studio Acoustic Drum Transient Triggering (Fat Birch/Maple Snare, Punchy Sub-Kick, Zero Clicking)
- Studio Precision Bass Re-Synthesis (F0 Tracking, Tube Harmonic Saturation, Zero Vocal/Snare Bleed)
- Re-Amped Electric Guitars (Tube Saturation, Pick Bite Crunch, Wide Dimensional Stereo Spread)
- Star Lead Vocals (Adaptive Vocal Rider, 3.5kHz Presence, Ozone Clarity/Stabilizer, 3D Plate Ambience)
- Live Album Console Summing & Genuine iZotope Ozone 12 VST3 Mastering
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from scipy import ndimage, signal

try:
    import pedalboard
    HAS_PEDALBOARD = True
except ImportError:
    pedalboard = None
    HAS_PEDALBOARD = False


def _ensure_stereo(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).copy()
    if x.ndim == 1:
        return np.column_stack([x, x])
    return x


def apply_ultrasonic_guard(arr: np.ndarray, sr: int = 48000) -> np.ndarray:
    """8th-order Butterworth lowpass at 20.5 kHz to eliminate Nyquist aliasing and intermodulation buzz."""
    cutoff = min(20500.0, sr * 0.45)
    sos = signal.butter(8, cutoff, btype="lowpass", fs=sr, output="sos")
    return signal.sosfilt(sos, arr, axis=0)


def generate_studio_kick_patch(sr: int = 48000) -> np.ndarray:
    """Acoustic studio kick: sub-bass drop (85Hz -> 48Hz) + chest punch + soft felt beater."""
    dur = 0.22
    n = int(dur * sr)
    t = np.arange(n) / float(sr)
    
    f_inst = 48.0 + 38.0 * np.exp(-t / 0.038)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / float(sr)
    sub_body = np.sin(phase) * np.exp(-t / 0.070)
    punch = np.sin(2.0 * np.pi * 72.0 * t) * np.exp(-t / 0.045)
    
    np.random.seed(42)
    noise = np.random.randn(n)
    sos_beater = signal.butter(2, [2200.0, 3800.0], btype="bandpass", fs=sr, output="sos")
    beater = signal.sosfilt(sos_beater, noise) * np.exp(-t / 0.012)
    
    raw = sub_body * 0.70 + punch * 0.25 + beater * 0.08
    att_samples = int(0.004 * sr)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, att_samples)))
    raw[:att_samples] *= ramp
    return raw / (np.max(np.abs(raw)) + 1e-9)


def generate_studio_snare_patch(sr: int = 48000) -> np.ndarray:
    """Acoustic studio wooden snare: 195Hz/280Hz dual shell resonance + shaped wire sizzle (zero clicks)."""
    dur = 0.24
    n = int(dur * sr)
    t = np.arange(n) / float(sr)
    
    shell_195 = np.sin(2.0 * np.pi * 195.0 * t) * np.exp(-t / 0.055)
    shell_280 = np.sin(2.0 * np.pi * 280.0 * t) * np.exp(-t / 0.040)
    shell = 0.75 * shell_195 + 0.25 * shell_280
    
    np.random.seed(1337)
    noise = np.random.randn(n)
    sos_wires = signal.butter(3, [2600.0, 7000.0], btype="bandpass", fs=sr, output="sos")
    wires = signal.sosfilt(sos_wires, noise) * np.exp(-t / 0.075)
    
    sos_crack = signal.butter(2, [1100.0, 2400.0], btype="bandpass", fs=sr, output="sos")
    crack = signal.sosfilt(sos_crack, noise) * np.exp(-t / 0.018)
    
    raw = shell * 0.55 + wires * 0.35 + crack * 0.15
    att_samples = int(0.0045 * sr)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, att_samples)))
    raw[:att_samples] *= ramp
    return raw / (np.max(np.abs(raw)) + 1e-9)


def recreate_drums_studio(drm_raw: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Detect kick and snare hits and layer studio acoustic triggers with the live drum track."""
    mono_drm = np.mean(drm_raw, axis=1) if drm_raw.ndim == 2 else drm_raw
    n_samples = len(mono_drm)

    # 1. Kick detector (45-95 Hz)
    sos_k = signal.butter(3, [45.0, 95.0], btype="bandpass", fs=sr, output="sos")
    k_band = signal.sosfilt(sos_k, mono_drm)
    k_env = np.sqrt(ndimage.uniform_filter1d(k_band**2, size=int(0.015 * sr)) + 1e-12)
    k_thresh = 0.20 * float(np.percentile(k_env, 95))
    k_peaks, _ = signal.find_peaks(k_env, height=k_thresh, distance=int(0.16 * sr), prominence=k_thresh * 0.4)

    # 2. Snare detector (180-340 Hz + 2.5-6.5 kHz)
    sos_s_b = signal.butter(2, [180.0, 340.0], btype="bandpass", fs=sr, output="sos")
    sos_s_w = signal.butter(2, [2500.0, 6500.0], btype="bandpass", fs=sr, output="sos")
    s_body = signal.sosfilt(sos_s_b, mono_drm)
    s_wire = signal.sosfilt(sos_s_w, mono_drm)
    s_env = np.sqrt(ndimage.uniform_filter1d(0.5 * s_body**2 + 0.5 * s_wire**2, size=int(0.015 * sr)) + 1e-12)

    for kp in k_peaks:
        s_env[max(0, kp - int(0.02 * sr)):min(n_samples, kp + int(0.05 * sr))] *= 0.2

    s_thresh = 0.18 * float(np.percentile(s_env, 95))
    s_peaks, _ = signal.find_peaks(s_env, height=s_thresh, distance=int(0.18 * sr), prominence=s_thresh * 0.4)

    kick_patch = generate_studio_kick_patch(sr)
    snare_patch = generate_studio_snare_patch(sr)

    synth_k = np.zeros(n_samples)
    for kp in k_peaks:
        s_end = min(n_samples, kp + len(kick_patch))
        p_len = s_end - kp
        vel = float(np.clip(k_env[kp] / (np.percentile(k_env, 98) + 1e-9), 0.35, 1.0))
        pol = float(np.sign(np.sum(mono_drm[kp : kp + 100]) + 1e-9))
        synth_k[kp:s_end] += pol * vel * kick_patch[:p_len]

    synth_s = np.zeros(n_samples)
    for sp in s_peaks:
        s_end = min(n_samples, sp + len(snare_patch))
        p_len = s_end - sp
        vel = float(np.clip(s_env[sp] / (np.percentile(s_env, 98) + 1e-9), 0.35, 1.0))
        pol = float(np.sign(np.sum(mono_drm[sp : sp + 100]) + 1e-9))
        synth_s[sp:s_end] += pol * vel * snare_patch[:p_len]

    # Clean low mud on live drums below 120 Hz
    sos_hp_drm = signal.butter(2, 120.0, btype="highpass", fs=sr, output="sos")
    st = _ensure_stereo(drm_raw)
    live_cleaned = signal.sosfilt(sos_hp_drm, st, axis=0)

    # Reinforce: 65% live + 35% studio kick + 30% studio snare
    reinforced = live_cleaned + np.column_stack([0.45 * synth_k + 0.35 * synth_s, 0.45 * synth_k + 0.35 * synth_s])
    return apply_ultrasonic_guard(reinforced, sr=sr)


def recreate_bass_studio(bas_raw: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Reject vocal/snare bleed above 800 Hz, track F0, synthesize studio precision bass, and apply Ozone Low End Focus."""
    mono_bas = np.mean(bas_raw, axis=1) if bas_raw.ndim == 2 else bas_raw
    n_samples = len(mono_bas)

    # 1. Reject neural bleed (>800 Hz)
    sos_lp = signal.butter(4, min(800.0, sr * 0.45), btype="lowpass", fs=sr, output="sos")
    live_bass_filtered = signal.sosfilt(sos_lp, mono_bas)

    # 2. Track pitch fundamentals and synthesize precision bass
    hop_samples = int(0.015 * sr)
    win_samples = int(0.060 * sr)
    n_frames = n_samples // hop_samples

    min_lag = int(sr / 220.0)  # ~220 Hz
    max_lag = int(sr / 40.0)   # ~40 Hz

    synth_bass = np.zeros(n_samples)
    curr_f0 = 65.0

    for f_idx in range(n_frames):
        s_start = f_idx * hop_samples
        s_stop = s_start + win_samples
        if s_stop > n_samples:
            break
        frame = live_bass_filtered[s_start:s_stop]
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms < 0.008:
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        corr_slice = corr[min_lag:max_lag]
        if len(corr_slice) > 0 and np.max(corr_slice) > 0:
            peak_lag = min_lag + int(np.argmax(corr_slice))
            conf = corr[peak_lag] / (corr[0] + 1e-12)
            if conf > 0.35:
                curr_f0 = sr / float(peak_lag)

        t_hop = (np.arange(hop_samples) + s_start) / float(sr)
        osc = (
            1.00 * np.sin(2.0 * np.pi * curr_f0 * t_hop)
            + 0.60 * np.sin(2.0 * np.pi * (2.0 * curr_f0) * t_hop)
            + 0.25 * np.sin(2.0 * np.pi * (3.0 * curr_f0) * t_hop)
        )
        osc = np.tanh(1.4 * osc)
        synth_bass[s_start : s_start + hop_samples] = osc * rms * 1.5

    sos_smooth = signal.butter(2, 600.0, btype="lowpass", fs=sr, output="sos")
    synth_bass = signal.sosfilt(sos_smooth, synth_bass)

    recreated_bass = 0.60 * live_bass_filtered + 0.40 * synth_bass
    recreated_bass_stereo = np.column_stack([recreated_bass, recreated_bass])

    # Optional VST: Ozone 12 Low End Focus
    lef_path = Path(r"C:\Program Files\Common Files\VST3\iZotope\Ozone 12 Low End Focus.vst3")
    if HAS_PEDALBOARD and lef_path.exists():
        try:
            lef = pedalboard.load_plugin(str(lef_path))
            board = pedalboard.Pedalboard([lef])
            audio_t = recreated_bass_stereo.T.astype(np.float32)
            processed_t = board(audio_t, sr)
            recreated_bass_stereo = processed_t.T.astype(np.float64)
        except Exception:
            pass

    return apply_ultrasonic_guard(recreated_bass_stereo, sr=sr)


def recreate_guitars_studio(oth_raw: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Clean mud, apply pick crunch, Tube Amp analog warmth, dimensional chorus, and Ozone Exciter."""
    st = _ensure_stereo(oth_raw)

    # 1. 115 Hz clean high-pass
    sos_hp = signal.butter(3, 115.0, btype="highpass", fs=sr, output="sos")
    g = signal.sosfilt(sos_hp, st, axis=0)

    # 2. De-mud 240 Hz (-2.5 dB)
    sos_mud = signal.butter(2, [190.0, 300.0], btype="bandpass", fs=sr, output="sos")
    g = g - 0.25 * signal.sosfilt(sos_mud, g, axis=0)

    # 3. Pick Bite & Crunch (2.8 kHz – 4.5 kHz +3.8 dB)
    sos_bite = signal.butter(2, [2800.0, min(4800.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    g = g + 0.38 * signal.sosfilt(sos_bite, g, axis=0)

    # 4. Wide Natural Stereo Spread (Mid/Side)
    mid = 0.5 * (g[:, 0] + g[:, 1])
    side = 0.5 * (g[:, 0] - g[:, 1])
    g_wide = np.column_stack([mid + 1.45 * side, mid - 1.45 * side])

    # 5. VST Chain: Tube Amp + Chorus + Ozone 12 Exciter
    tube_path = Path(r"C:\Program Files\Common Files\VST3\Tube Amp.vst3")
    exc_path = Path(r"C:\Program Files\Common Files\VST3\iZotope\Ozone 12 Exciter.vst3")

    if HAS_PEDALBOARD:
        effects = []
        if tube_path.exists():
            try:
                effects.append(pedalboard.load_plugin(str(tube_path)))
            except Exception:
                pass
        effects.append(pedalboard.Chorus(rate_hz=0.8, depth=0.12, centre_delay_ms=8.0, feedback=0.0, mix=0.35))
        if exc_path.exists():
            try:
                effects.append(pedalboard.load_plugin(str(exc_path)))
            except Exception:
                pass
        if effects:
            try:
                board = pedalboard.Pedalboard(effects)
                audio_t = g_wide.T.astype(np.float32)
                g_wide = board(audio_t, sr).T.astype(np.float64)
            except Exception:
                pass

    return apply_ultrasonic_guard(g_wide, sr=sr)


def recreate_vocals_studio(voc_raw: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Adaptive level rider, Neve/Pultec curve, Ozone Clarity + Stabilizer + Tube Amp + 3D Plate Reverb."""
    st = _ensure_stereo(voc_raw)
    mono_voc = 0.5 * (st[:, 0] + st[:, 1])

    # 1. Adaptive Vocal Rider (smooth at 100 Hz)
    step = 480
    sub_mono = mono_voc[::step]
    env = np.sqrt(ndimage.uniform_filter1d(sub_mono**2, size=25) + 1e-12)
    active = env > 0.005
    target_env = np.clip(env, 0.020, 0.35)
    ideal_gain = np.where(active, 0.22 / target_env, 1.0)
    ideal_gain = np.clip(ideal_gain, 0.65, 2.20)
    smooth_gain = ndimage.gaussian_filter1d(ideal_gain, sigma=35.0)
    centers = np.arange(len(smooth_gain)) * step
    interp_gain = np.interp(np.arange(len(mono_voc)), centers, smooth_gain)
    voc_leveled = mono_voc * interp_gain

    # 2. Neve / Pultec Studio Curve
    sos_hp = signal.butter(3, 105.0, btype="highpass", fs=sr, output="sos")
    v = signal.sosfilt(sos_hp, voc_leveled)

    sos_box = signal.butter(2, [180.0, 300.0], btype="bandpass", fs=sr, output="sos")
    v = v - 0.30 * signal.sosfilt(sos_box, v)

    sos_pres = signal.butter(2, [2400.0, min(5000.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    v = v + 0.45 * signal.sosfilt(sos_pres, v)

    sos_air = signal.butter(2, min(11000.0, sr * 0.45), btype="highpass", fs=sr, output="sos")
    v = v + 0.28 * signal.sosfilt(sos_air, v)

    sos_ess = signal.butter(2, [6000.0, min(9500.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    ess_band = signal.sosfilt(sos_ess, v)
    ess_env = ndimage.uniform_filter1d(np.abs(ess_band), size=max(int(0.010 * sr), 1))
    thresh = 0.22 * float(np.max(ess_env) + 1e-9)
    att = np.where(ess_env > thresh, thresh / (ess_env + 1e-9), 1.0)
    v = (v - ess_band) + ess_band * att

    v_st = np.column_stack([v, v])

    # 3. VST Chain: Ozone 12 Clarity + Stabilizer + Tube Amp + 3D Plate Reverb
    clarity_path = Path(r"C:\Program Files\Common Files\VST3\iZotope\Ozone 12 Clarity.vst3")
    stabilizer_path = Path(r"C:\Program Files\Common Files\VST3\iZotope\Ozone 12 Stabilizer.vst3")
    tube_path = Path(r"C:\Program Files\Common Files\VST3\Tube Amp.vst3")

    if HAS_PEDALBOARD:
        effects = []
        if clarity_path.exists():
            try:
                effects.append(pedalboard.load_plugin(str(clarity_path)))
            except Exception:
                pass
        if stabilizer_path.exists():
            try:
                effects.append(pedalboard.load_plugin(str(stabilizer_path)))
            except Exception:
                pass
        if tube_path.exists():
            try:
                effects.append(pedalboard.load_plugin(str(tube_path)))
            except Exception:
                pass
        effects.append(pedalboard.Reverb(room_size=0.22, damping=0.60, wet_level=0.08, dry_level=1.00))
        if effects:
            try:
                board = pedalboard.Pedalboard(effects)
                audio_t = v_st.T.astype(np.float32)
                v_st = board(audio_t, sr).T.astype(np.float64)
            except Exception:
                pass

    return apply_ultrasonic_guard(v_st, sr=sr)


def mix_and_master_studio_recreation(
    vocals: np.ndarray,
    other: np.ndarray,
    bass: np.ndarray,
    drums: np.ndarray,
    crowd: np.ndarray | None = None,
    sr: int = 48000,
    target_lufs: float = -14.0,
    ceiling_dbtp: float = -1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mix recreated stems with Live Album Console Faders and master through Ozone 12 VST3.

    Console Fader Ratios:
    - Lead Vocals: +3.0 dB (Centerpiece, ultra-clear & intimate)
    - Guitars/Accompaniment: 0.0 dB (Crunchy, wide, articulate)
    - Bass Guitar: -0.5 dB (Tight low end, solid pitch growl)
    - Drums: -2.5 dB (Clean punchy groove, zero snare overpowering)
    - Crowd Ambience: -12.0 dB (Stadium room envelopment)
    """
    v_st = _ensure_stereo(vocals)
    o_st = _ensure_stereo(other)
    b_st = _ensure_stereo(bass)
    d_st = _ensure_stereo(drums)

    if crowd is not None:
        c_st = _ensure_stereo(crowd)
        sos_crd = signal.butter(3, [120.0, min(7500.0, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
        c_st = signal.sosfilt(sos_crd, c_st, axis=0)
    else:
        c_st = np.zeros_like(v_st)

    def prep_stem(x: np.ndarray, target_peak: float = 0.50) -> np.ndarray:
        pk = float(np.max(np.abs(x))) + 1e-9
        return x * (target_peak / pk)

    v_norm = prep_stem(v_st)
    o_norm = prep_stem(o_st)
    b_norm = prep_stem(b_st)
    d_norm = prep_stem(d_st)
    c_norm = prep_stem(c_st) if crowd is not None else np.zeros_like(v_st)

    # Live Album Console Fader Gains:
    g_voc = 10.0 ** (3.0 / 20.0)      # 1.413 (+3.0 dB vocal priority)
    g_oth = 10.0 ** (0.0 / 20.0)      # 1.000 (guitars right up front)
    g_bas = 10.0 ** (-0.5 / 20.0)     # 0.944 (solid punchy bass)
    g_drm = 10.0 ** (-2.5 / 20.0)     # 0.750 (groove supports without overwhelming)
    g_crd = 10.0 ** (-12.0 / 20.0)    # 0.251 (room ambience)

    console_mix = (
        v_norm * g_voc
        + o_norm * g_oth
        + b_norm * g_bas
        + d_norm * g_drm
        + (c_norm * g_crd if crowd is not None else 0.0)
    )

    # Sub-160Hz mono lock on master bus
    sos_mono_sub = signal.butter(4, 160.0, btype="lowpass", fs=sr, output="sos")
    sub_m = signal.sosfiltfilt(sos_mono_sub, 0.5 * (console_mix[:, 0] + console_mix[:, 1]))
    high_st = console_mix - signal.sosfiltfilt(sos_mono_sub, console_mix, axis=0)
    clean_mix = np.column_stack([sub_m, sub_m]) + high_st

    # 20.5 kHz ultrasonic brickwall guard
    clean_mix = apply_ultrasonic_guard(clean_mix, sr=sr)

    # Master through Ozone 12 VST3 if available, else lookahead limiter
    ozone_path = Path(r"C:\Program Files\Common Files\VST3\iZotope\Ozone 12.vst3")
    mastered = None
    if HAS_PEDALBOARD and ozone_path.exists():
        try:
            ozone = pedalboard.load_plugin(str(ozone_path))
            board = pedalboard.Pedalboard([ozone])
            audio_t = clean_mix.T.astype(np.float32)
            mastered = board(audio_t, sr).T.astype(np.float64)
        except Exception:
            mastered = None

    from concert_studio.mastering import _lookahead_true_peak_limiter

    if mastered is None:
        mastered = _lookahead_true_peak_limiter(clean_mix, sr=sr, target_lufs=target_lufs, ceiling_dbtp=ceiling_dbtp)

    # Final True-Peak ceiling at -1.0 dBTP
    peak = float(np.max(np.abs(mastered)))
    ceiling_lin = 10.0 ** (ceiling_dbtp / 20.0)
    if peak > ceiling_lin:
        mastered *= ceiling_lin / peak

    # Individual solo stems leveled for UI playback
    solo_voc = _lookahead_true_peak_limiter(v_st, sr=sr, target_lufs=-14.5, ceiling_dbtp=-1.0)
    solo_oth = _lookahead_true_peak_limiter(o_st, sr=sr, target_lufs=-15.5, ceiling_dbtp=-1.0)
    solo_bas = _lookahead_true_peak_limiter(b_st, sr=sr, target_lufs=-15.5, ceiling_dbtp=-1.0)
    solo_drm = _lookahead_true_peak_limiter(d_st, sr=sr, target_lufs=-16.0, ceiling_dbtp=-1.0)
    solo_crd = _lookahead_true_peak_limiter(c_st, sr=sr, target_lufs=-18.0, ceiling_dbtp=-1.0) if crowd is not None else np.zeros_like(v_st)

    solo_stems = {
        "vocals": solo_voc,
        "other": solo_oth,
        "bass": solo_bas,
        "drums": solo_drm,
        "crowd_removed": solo_crd,
    }

    return mastered, solo_stems

