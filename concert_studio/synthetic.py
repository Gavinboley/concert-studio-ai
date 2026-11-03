"""Synthetic multi-stem concert & multi-POV audience degradation generator (v1 & v2 partial/anomaly clips)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal

from concert_studio.models import AudioTrack, IMUTelemetry, StemBundle
from concert_studio.spatial import synthesize_stage_video_frame


@dataclass(frozen=True)
class SyntheticConcertScenario:
    """Ground-truth studio stems, clean studio mix, and corrupted fan POV tracks."""

    clean_stems: StemBundle
    studio_reference: AudioTrack
    pov_tracks: list[AudioTrack]
    ground_truth_offsets_sec: dict[str, float]
    ground_truth_drifts_ppm: dict[str, float]


def _synthesize_clean_stems(
    duration_sec: float = 6.0,
    sr: int = 48000,
    seed: int = 42,
) -> StemBundle:
    """Synthesize a rich 4-stem musical performance (vocals, drums, bass, instruments)."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_sec * sr))
    t = np.arange(n, dtype=np.float64) / sr

    # 1. Drums: 120 BPM (0.5s beat), sub-kick on beats, crisp snare on backbeats, 16th hi-hats
    drums_l = np.zeros(n, dtype=np.float64)
    drums_r = np.zeros(n, dtype=np.float64)
    beat_sec = 0.5
    num_beats = int(duration_sec / beat_sec)
    for b in range(num_beats):
        t0 = b * beat_sec
        idx0 = int(t0 * sr)
        hit_len = min(int(0.22 * sr), n - idx0)
        if hit_len <= 0:
            continue
        th = np.arange(hit_len, dtype=np.float64) / sr
        f_inst = 52.0 + 83.0 * np.exp(-th / 0.025)
        phase = 2.0 * np.pi * np.cumsum(f_inst) / sr
        kick = 0.55 * np.sin(phase) * np.exp(-th / 0.085)
        drums_l[idx0 : idx0 + hit_len] += kick
        drums_r[idx0 : idx0 + hit_len] += kick

        if b % 2 == 1:
            snare_body = 0.25 * np.sin(2.0 * np.pi * 195.0 * th) * np.exp(-th / 0.045)
            snare_wires = 0.30 * rng.standard_normal(hit_len) * np.exp(-th / 0.065)
            sos_bp = signal.butter(2, [900.0, 9500.0], btype="bandpass", fs=sr, output="sos")
            snare_wires = signal.sosfilt(sos_bp, snare_wires)
            drums_l[idx0 : idx0 + hit_len] += snare_body + 1.05 * snare_wires
            drums_r[idx0 : idx0 + hit_len] += snare_body + 0.95 * snare_wires

    sixteenth_sec = beat_sec / 4.0
    for s_idx in range(int(duration_sec / sixteenth_sec)):
        idx0 = int(s_idx * sixteenth_sec * sr)
        hh_len = min(int(0.045 * sr), n - idx0)
        if hh_len <= 0:
            continue
        th = np.arange(hh_len, dtype=np.float64) / sr
        accent = 0.16 if (s_idx % 2 == 0) else 0.09
        raw_hh = accent * rng.standard_normal(hh_len) * np.exp(-th / 0.014)
        sos_hp = signal.butter(3, 7500.0, btype="highpass", fs=sr, output="sos")
        hh = signal.sosfilt(sos_hp, raw_hh)
        drums_l[idx0 : idx0 + hh_len] += 0.9 * hh
        drums_r[idx0 : idx0 + hh_len] += 1.1 * hh

    drums = np.column_stack([drums_l, drums_r])

    # 2. Bass: Warm synth/electric bass progression
    bass_notes = [55.0, 43.65, 65.41, 49.0]
    bass_mono = np.zeros(n, dtype=np.float64)
    bar_sec = 0.85
    for i in range(n):
        bar = int((t[i] / bar_sec)) % len(bass_notes)
        f0 = bass_notes[bar]
        bass_mono[i] = (
            0.36 * np.sin(2.0 * np.pi * f0 * t[i])
            + 0.18 * np.sin(2.0 * np.pi * 2.0 * f0 * t[i] + 0.3)
            + 0.08 * np.sin(2.0 * np.pi * 3.0 * f0 * t[i])
        )
    groove_env = 0.55 + 0.45 * np.abs(np.cos(2.0 * np.pi * (1.5 + 0.4 * np.sin(t)) * t))
    bass_mono *= groove_env
    bass = np.column_stack([bass_mono, bass_mono])

    # 3. Vocals: Expressive lead vocal with non-repeating melodic phrasing (enables unambiguous clip sync)
    melody_notes = [220.0, 261.63, 293.66, 329.63, 392.00, 246.94, 349.23, 277.18]
    note_dur = duration_sec / len(melody_notes)
    vocal_mono = np.zeros(n, dtype=np.float64)
    for m_idx, f_vocal in enumerate(melody_notes):
        i0 = int(m_idx * note_dur * sr)
        i1 = min(n, int((m_idx + 1) * note_dur * sr))
        tn = np.arange(i1 - i0, dtype=np.float64) / sr
        vibrato = 4.5 * np.sin(2.0 * np.pi * 5.6 * tn)
        inst_phase = 2.0 * np.pi * np.cumsum(f_vocal + vibrato) / sr
        sig = np.zeros_like(tn)
        for h in range(1, 14):
            fh = h * f_vocal
            formant_gain = (
                np.exp(-0.5 * ((fh - 800.0) / 350.0) ** 2)
                + 0.75 * np.exp(-0.5 * ((fh - 2650.0) / 500.0) ** 2)
                + 0.25 / h
            )
            sig += (0.20 * formant_gain) * np.sin(h * inst_phase)
        env = np.ones_like(tn)
        fade = min(int(0.04 * sr), len(tn) // 4)
        if fade > 0:
            env[:fade] = np.linspace(0.0, 1.0, fade)
            env[-fade:] = np.linspace(1.0, 0.08, fade)
        # Distinct dynamic accent per phrase so partial clips have unique macro-envelopes
        phrase_accent = 0.65 + 0.35 * ((m_idx * 37 % 11) / 10.0)
        vocal_mono[i0:i1] = sig * env * phrase_accent
    vocals = np.column_stack([vocal_mono, vocal_mono])

    # 4. Other (Stereo panned chords / synths / guitars with upper harmonics up to 18 kHz)
    chords = [
        (440.0, 523.25, 659.25),
        (349.23, 440.0, 523.25),
        (523.25, 659.25, 783.99),
        (392.00, 493.88, 587.33),
    ]
    other_l = np.zeros(n, dtype=np.float64)
    other_r = np.zeros(n, dtype=np.float64)
    for k in (1, 2, 3, 6, 12, 20):
        for c_idx, (f1, f2, f3) in enumerate(chords):
            mask = (np.floor(t / bar_sec).astype(int) % len(chords)) == c_idx
            tm = t[mask]
            c1 = (0.08 / k) * np.sin(2.0 * np.pi * (f1 * k) * tm)
            c2 = (0.08 / k) * np.sin(2.0 * np.pi * (f2 * k) * tm + 0.4)
            c3 = (0.08 / k) * np.sin(2.0 * np.pi * (f3 * k) * tm + 0.9)
            other_l[mask] += 0.85 * c1 + 0.55 * c2 + 0.25 * c3
            other_r[mask] += 0.25 * c1 + 0.55 * c2 + 0.85 * c3
    other = np.column_stack([other_l, other_r])

    return StemBundle(
        vocals=vocals,
        drums=drums,
        bass=bass,
        other=other,
        crowd_removed=np.zeros_like(vocals),
        sample_rate=sr,
    )


def _apply_delay_and_drift(
    audio: np.ndarray,
    sr: int,
    offset_sec: float,
    drift_ppm: float,
    motion_excursion_sec: float = 0.0,
) -> np.ndarray:
    """Apply fractional initial time delay + clock drift (SRO) + optional moving-phone Doppler wave."""
    n = audio.shape[0]
    t_orig = np.arange(n, dtype=np.float64) / sr
    motion_wave = (
        motion_excursion_sec * np.sin(2.0 * np.pi * 0.45 * t_orig)
        if motion_excursion_sec > 0.0
        else 0.0
    )
    t_query = t_orig * (1.0 - drift_ppm * 1e-6) - offset_sec - motion_wave
    if audio.ndim == 1:
        return np.interp(t_query, t_orig, audio, left=0.0, right=0.0)
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        out[:, ch] = np.interp(t_query, t_orig, audio[:, ch], left=0.0, right=0.0)
    return out


def _apply_stadium_reverb(
    audio: np.ndarray, sr: int, rt60: float = 0.55, wet: float = 0.28, ir_seed: int = 101
) -> np.ndarray:
    """Convolve with a position-specific multi-reflection stadium impulse response."""
    rng = np.random.default_rng(ir_seed)
    ir_len = int(rt60 * sr)
    t_ir = np.arange(ir_len, dtype=np.float64) / sr
    decay = np.exp(-6.91 * t_ir / rt60)
    ir = rng.standard_normal(ir_len) * decay
    ir[0] = 1.0
    j1 = float(rng.uniform(-4.0, 6.0))
    j2 = float(rng.uniform(-6.0, 9.0))
    for delay_ms, gain in [(18.0 + j1, 0.35), (37.0 + j2, 0.24), (64.0 - j1, 0.16)]:
        idx = max(1, int(delay_ms * 1e-3 * sr))
        if idx < ir_len:
            ir[idx] += gain
    ir /= np.linalg.norm(ir) + 1e-12
    if audio.ndim == 1:
        rev = signal.fftconvolve(audio, ir, mode="full")[: audio.shape[0]]
        return (1.0 - wet) * audio + wet * rev
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        rev = signal.fftconvolve(audio[:, ch], ir, mode="full")[: audio.shape[0]]
        out[:, ch] = (1.0 - wet) * audio[:, ch] + wet * rev
    return out


def generate_synthetic_concert(
    duration_sec: float = 6.0,
    sr: int = 48000,
    num_povs: int = 5,
    seed: int = 42,
    include_partial_clips: bool = False,
    include_physical_anomalies: bool = False,
) -> SyntheticConcertScenario:
    """Generate a synthetic concert benchmark with ground-truth studio mix & corrupted POVs.

    When `include_partial_clips=True` and `include_physical_anomalies=True`, generates:
    - Partial clips starting and stopping at random points on the timeline
    - Moving-phone dynamic Doppler/ToA excursion
    - Discrete delay-tower / slapback echoes
    - Mechanical mic bumps + IMU accelerometer/gyroscope telemetry spikes
    - Finger occlusion (blocked mic) intervals
    """
    if num_povs < 2:
        raise ValueError("num_povs must be at least 2")
    rng = np.random.default_rng(seed)
    stems = _synthesize_clean_stems(duration_sec=duration_sec, sr=sr, seed=seed)

    studio_mix = stems.vocals + stems.drums + stems.bass + stems.other
    peak = np.max(np.abs(studio_mix)) + 1e-12
    studio_mix = 0.85 * (studio_mix / peak)
    studio_ref = AudioTrack(name="studio_reference", samples=studio_mix, sample_rate=sr)

    pov_specs = [
        ("pov_01_front_center", 0.000, 0.0, 0.12, 16500.0, 0.99),
        ("pov_02_left_screamer", 0.014, 18.5, 0.20, 14200.0, 0.99),
        ("pov_03_right_clipped", 0.009, -15.0, 0.16, 14800.0, 0.62),
        ("pov_04_balcony_echo", 0.028, 28.0, 0.30, 12500.0, 0.99),
        ("pov_05_pocket_muffled", 0.019, -24.0, 0.24, 7200.0, 0.99),
    ]

    pov_tracks: list[AudioTrack] = []
    gt_offsets: dict[str, float] = {}
    gt_drifts: dict[str, float] = {}
    n = studio_mix.shape[0]

    for i in range(num_povs):
        name, offset_s, drift_ppm, wet, lp_cutoff, clip_thresh = pov_specs[i % len(pov_specs)]
        if i >= len(pov_specs):
            name = f"pov_{i+1:02d}_angle"
            offset_s = float(rng.uniform(0.004, 0.030))
            drift_ppm = float(rng.uniform(-25.0, 25.0))
            wet = float(rng.uniform(0.12, 0.28))
            lp_cutoff = float(rng.uniform(9500.0, 16500.0))
            clip_thresh = 0.99

        # 1. Stadium reverberation (distinct acoustic reflection path per physical POV spot)
        corrupted = _apply_stadium_reverb(studio_mix, sr=sr, rt60=0.48, wet=wet, ir_seed=101 + i * 37)

        # 2. Position-dependent speaker tuning & high-frequency air rolloff
        sos_lp = signal.butter(4, min(lp_cutoff, sr * 0.46), btype="lowpass", fs=sr, output="sos")
        corrupted = signal.sosfilt(sos_lp, corrupted, axis=0)

        # Optional proximity subwoofer boom on POV #2
        if include_physical_anomalies and i % 5 == 1:
            sos_sub = signal.butter(2, 95.0, btype="lowpass", fs=sr, output="sos")
            corrupted = corrupted + 1.6 * signal.sosfilt(sos_sub, corrupted, axis=0)

        # Optional discrete delay-tower / slapback echo (52 ms delay, alpha=0.35) on POV #4
        if include_physical_anomalies and i % 5 == 3:
            d_echo = int(0.052 * sr)
            if d_echo < n:
                corrupted[d_echo:] += 0.35 * corrupted[:-d_echo]

        # 3. Localized uncorrelated fan scream / applause burst unique to this POV
        local_crowd = np.zeros_like(corrupted)
        scream_start = int(((0.65 + 0.95 * i) % max(duration_sec - 0.8, 0.5)) * sr)
        scream_len = min(int(0.45 * sr), n - scream_start)
        if scream_len > 0:
            ts = np.arange(scream_len, dtype=np.float64) / sr
            f_chirp = 1100.0 + 1600.0 * np.sin(np.pi * ts / (scream_len / sr))
            scream_sig = 0.65 * np.sin(2.0 * np.pi * np.cumsum(f_chirp) / sr)
            scream_noise = 0.40 * rng.standard_normal(scream_len)
            sos_sc = signal.butter(2, [900.0, 4200.0], btype="bandpass", fs=sr, output="sos")
            burst = (scream_sig + signal.sosfilt(sos_sc, scream_noise)) * np.hanning(scream_len)
            local_crowd[scream_start : scream_start + scream_len, 0] += burst
            local_crowd[scream_start : scream_start + scream_len, 1] += 0.9 * burst

        murmur = (0.018 / np.sqrt(max(1, i // 3 + 1))) * rng.standard_normal(corrupted.shape)
        corrupted = corrupted + local_crowd + murmur

        # 4. Optional Mechanical Mic Bump + IMU Telemetry spike on POV #2
        imu_data: IMUTelemetry | None = None
        if include_physical_anomalies and i % 5 == 1:
            bump_start = int(min(1.1, duration_sec * 0.35) * sr)
            bump_len = min(int(0.16 * sr), n - bump_start)
            if bump_len > 0:
                tb = np.arange(bump_len, dtype=np.float64) / sr
                thump = 0.85 * np.sin(2.0 * np.pi * 38.0 * tb) * np.hanning(bump_len)
                corrupted[bump_start : bump_start + bump_len, 0] += thump
                corrupted[bump_start : bump_start + bump_len, 1] += thump
                imu_t = np.linspace(0.0, duration_sec, 100)
                imu_acc = np.full_like(imu_t, 0.2)
                imu_gyr = np.full_like(imu_t, 0.1)
                b_sec = bump_start / float(sr)
                spike_mask = np.abs(imu_t - b_sec) < 0.12
                imu_acc[spike_mask] = 8.5
                imu_gyr[spike_mask] = 4.2
                imu_data = IMUTelemetry(
                    timestamps_sec=imu_t,
                    accel_magnitude=imu_acc,
                    gyro_magnitude=imu_gyr,
                )

        # 5. Optional Finger Occlusion (mic blocked for 0.6s) on POV #3
        if include_physical_anomalies and i % 5 == 2:
            occ_s = int(min(1.8, duration_sec * 0.45) * sr)
            occ_e = min(n, occ_s + int(0.65 * sr))
            if occ_e > occ_s:
                sos_muf = signal.butter(4, 650.0, btype="lowpass", fs=sr, output="sos")
                corrupted[occ_s:occ_e] = signal.sosfilt(sos_muf, corrupted[occ_s:occ_e], axis=0)

        # 6. Delay, clock drift, and optional moving-phone Doppler wave
        motion_exc = 0.0022 if (include_physical_anomalies and i % 5 == 4) else 0.0
        corrupted = _apply_delay_and_drift(
            corrupted,
            sr=sr,
            offset_sec=offset_s,
            drift_ppm=drift_ppm,
            motion_excursion_sec=motion_exc,
        )

        # 7. Front-row MEMS ADC hard clipping if applicable
        if clip_thresh < 0.95:
            corrupted = np.clip(corrupted * 1.25, -clip_thresh, clip_thresh)

        # 8. Optional partial clip slicing (arbitrary start and stop times)
        if include_partial_clips and i > 0:
            clip_frac = float(rng.uniform(0.55, 0.82))
            clip_len = max(int(clip_frac * n), int(1.2 * sr))
            max_start = max(n - clip_len, 1)
            start_idx = int(rng.integers(0, max_start))
            corrupted = corrupted[start_idx : start_idx + clip_len]
            gt_offsets[name] = round(start_idx / float(sr) + offset_s, 4)
        else:
            gt_offsets[name] = offset_s

        gt_drifts[name] = drift_ppm
        vid_frames = None
        if include_physical_anomalies:
            sim_dist = 6.0 + offset_s * 343.0
            sim_az = [-18.0, -32.0, 22.0, 8.0, -10.0][i % 5]
            sim_elev = "balcony" if sim_dist > 15.0 else "mid_floor"
            vid_frames = synthesize_stage_video_frame(
                distance_m=sim_dist, azimuth_deg=sim_az, elevation_zone=sim_elev
            )

        pov_tracks.append(
            AudioTrack(
                name=name,
                samples=corrupted,
                sample_rate=sr,
                imu_telemetry=imu_data,
                video_frames_rgb=vid_frames,
            )
        )

    return SyntheticConcertScenario(
        clean_stems=stems,
        studio_reference=studio_ref,
        pov_tracks=pov_tracks,
        ground_truth_offsets_sec=gt_offsets,
        ground_truth_drifts_ppm=gt_drifts,
    )
