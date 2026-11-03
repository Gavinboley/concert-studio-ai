"""Stage 1B: Arbitrary start/stop partial-clip timeline auto-detection & moving-phone trajectory sync."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage, signal

from concert_studio.models import AnomalyEvent, AudioTrack
from concert_studio.sync import _estimate_snr_and_clipping, _gcc_phat_lag


@dataclass(frozen=True)
class PlacedClip:
    """A POV clip placed and phase-locked onto the unified master concert timeline."""

    track: AudioTrack                 # Padded to full master timeline length (T_master, 2)
    active_mask: np.ndarray           # 1D float64 (T_master,) in [0.0, 1.0] with cosine entry/exit fades
    timeline_start_sec: float
    timeline_end_sec: float
    clip_duration_sec: float
    initial_toa_ms: float
    clock_drift_ppm: float
    max_motion_drift_ms: float
    coherence_score: float
    snr_db: float
    clipping_ratio: float
    motion_events: list[AnomalyEvent]


def trim_active_bounds(mono: np.ndarray, sr: int) -> tuple[int, int]:
    """Find active musical start and end sample indices (stripping leading/trailing silence & TikTok end-cards)."""
    if len(mono) < int(4.0 * sr):
        return 0, len(mono)
    hop = max(sr // 50, 1)  # 20ms frames
    n_f = len(mono) // hop
    if n_f < 20:
        return 0, len(mono)
    frame_rms = np.sqrt(np.mean(mono[:n_f * hop].reshape(n_f, hop) ** 2, axis=1))
    med_rms = float(np.percentile(frame_rms, 60))
    active = frame_rms > max(med_rms * 0.08, 1e-4)
    idx = np.flatnonzero(active)
    if len(idx) < 10:
        return 0, len(mono)
    # Only trim if there is actual silence (> 250ms) at the boundaries (e.g. TikTok end-cards)
    s0 = int(idx[0] * hop) if idx[0] > 12 else 0
    s1 = int(idx[-1] * hop) if idx[-1] < n_f - 12 else len(mono)
    return s0, s1


def _compute_multiband_envelope(mono: np.ndarray, sr: int, env_sr: int = 250) -> np.ndarray:
    """Compute a 5-band log-spectral onset/energy fingerprint at `env_sr` Hz for partial-clip matching."""
    n_env = max(int(round(len(mono) * env_sr / sr)), 4)
    # Focus bands on the musical range while stripping <80Hz stadium subwoofer boom
    bands = [
        (100.0, 250.0),
        (250.0, 550.0),
        (550.0, 1100.0),
        (1100.0, 2500.0),
        (2500.0, min(8000.0, sr * 0.45)),
    ]
    feats = []
    hop = max(int(round(sr / env_sr)), 1)

    # Frame RMS detection to identify silent leading/trailing frames
    n_frames_mono = len(mono) // hop
    if n_frames_mono >= 10:
        frame_rms_mono = np.sqrt(np.mean(mono[:n_frames_mono * hop].reshape(n_frames_mono, hop) ** 2, axis=1))
        med_rms = float(np.percentile(frame_rms_mono, 60))
        is_silent = frame_rms_mono < max(med_rms * 0.08, 1e-5)
    else:
        is_silent = np.zeros(n_env, dtype=bool)

    for f_lo, f_hi in bands:
        if f_lo >= f_hi:
            continue
        sos = signal.butter(2, [f_lo, f_hi], btype="bandpass", fs=sr, output="sos")
        bp = signal.sosfilt(sos, mono)
        power = bp**2
        sm = ndimage.uniform_filter1d(power, size=hop * 2 + 1)
        idx = np.clip((np.arange(n_env) * sr / env_sr).astype(int), 0, len(sm) - 1)
        pwr_sampled = sm[idx]
        floor_p = max(float(np.percentile(pwr_sampled, 15)), 1e-12)
        log_e = np.log10(np.maximum(pwr_sampled, floor_p * 0.25) + 1e-12)

        # High-pass detrend slow AGC changes (>2.5 sec) when clip duration is > 10s
        if n_env > int(10.0 * env_sr):
            local_mean = ndimage.uniform_filter1d(log_e, size=int(2.5 * env_sr))
            detrended = log_e - local_mean
        else:
            detrended = log_e - np.mean(log_e)

        onset = np.diff(log_e, prepend=log_e[0])
        combined = 0.6 * detrended + 0.4 * (onset / (np.std(onset) + 1e-6))

        # Zero out silent frames so TikTok end-cards/silences produce 0.0 in the normalized envelope
        if len(is_silent) > 0:
            sil_idx = np.clip(np.arange(n_env), 0, len(is_silent) - 1)
            combined = np.where(is_silent[sil_idx], 0.0, combined)

        non_zero = combined[combined != 0.0]
        std_val = float(np.std(non_zero)) if len(non_zero) > 10 else float(np.std(combined))
        norm = combined / (std_val + 1e-6)
        feats.append(norm)
    return np.vstack(feats)  # shape (B, n_env)


def _pairwise_envelope_offset(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    env_sr: int = 250,
    min_overlap_sec: float = 0.6,
) -> tuple[float, float]:
    """Find offset `t_start_b - t_start_a` (in seconds) where clip B best aligns inside/against clip A."""
    n_a = feat_a.shape[1]
    n_b = feat_b.shape[1]
    shorter = min(n_a, n_b)
    if shorter > 8 * env_sr:
        min_ov = max(int(min_overlap_sec * env_sr), int(0.65 * shorter), 4)
        ov_exp = 0.55
    else:
        min_ov = max(int(min_overlap_sec * env_sr), shorter // 3, 4)
        ov_exp = 0.20

    # Full cross-correlation summed across bands
    n_full = n_a + n_b - 1
    n_fft = 1 << int(np.ceil(np.log2(max(n_full, 64))))
    FA = np.fft.rfft(feat_a, n=n_fft, axis=1)
    FB = np.fft.rfft(feat_b[:, ::-1], n=n_fft, axis=1)
    xcorr = np.sum(np.fft.irfft(FA * FB, n=n_fft, axis=1)[:, :n_full], axis=0)

    # Compute local energy norms for normalized cross-correlation over valid overlap lengths
    ones_a = np.ones((1, n_a))
    ones_b = np.ones((1, n_b))
    sq_a = np.sum(feat_a**2, axis=0, keepdims=True)
    sq_b = np.sum(feat_b**2, axis=0, keepdims=True)

    EA = np.fft.rfft(sq_a, n=n_fft, axis=1)
    EB = np.fft.rfft(sq_b[:, ::-1], n=n_fft, axis=1)
    OA = np.fft.rfft(ones_a, n=n_fft, axis=1)
    OB = np.fft.rfft(ones_b, n=n_fft, axis=1)

    energy_a_ov = np.fft.irfft(EA * OB, n=n_fft, axis=1)[0, :n_full]
    energy_b_ov = np.fft.irfft(OA * EB, n=n_fft, axis=1)[0, :n_full]
    ov_counts = np.fft.irfft(OA * OB, n=n_fft, axis=1)[0, :n_full]

    valid_mask = ov_counts >= min_ov
    denom = np.sqrt(np.maximum(energy_a_ov * energy_b_ov, 1e-9))
    # Prior favoring larger overlap when correlation is similar
    overlap_weight = np.clip(ov_counts / float(shorter), 0.25, 1.0) ** ov_exp
    ncc = np.where(valid_mask, (xcorr / denom) * overlap_weight, -1.0)

    best_k = int(np.argmax(ncc))
    best_score = float(np.clip(ncc[best_k], 0.0, 1.0))
    # Index k in conv(a, reverse(b)) corresponds to lag = k - (n_b - 1) samples of B's start in A
    lag_frames = best_k - (n_b - 1)
    return float(lag_frames) / float(env_sr), best_score


def _refine_and_track_nonrigid_motion(
    ref_mono: np.ndarray,
    clip_stereo: np.ndarray,
    sr: int,
    coarse_start_sample: int,
    num_subwindows: int = 12,
    enable_nonrigid: bool = True,
) -> tuple[np.ndarray, int, float, float, float, float, list[AnomalyEvent]]:
    """Refine partial clip start to sub-sample precision and warp out clock drift + moving-phone Doppler."""
    clip_mono = np.mean(clip_stereo, axis=1)
    n_clip = len(clip_mono)
    n_master = len(ref_mono)

    # Extract overlapping slice from reference composite
    s0 = max(0, coarse_start_sample)
    c0 = max(0, -coarse_start_sample)
    ov_len = min(n_master - s0, n_clip - c0)
    if ov_len < int(0.25 * sr):
        return clip_stereo, max(0, coarse_start_sample), 0.0, 0.0, 0.0, 0.2, []

    ref_slice = ref_mono[s0 : s0 + ov_len]
    clip_slice = clip_mono[c0 : c0 + ov_len]

    # Bound initial fine GCC-PHAT window to 20s max so 6+ minute concert tracks align in <20ms
    max_fine_len = int(20.0 * sr)
    if ov_len > max_fine_len:
        mid_off = (ov_len - max_fine_len) // 2
        ref_fine = ref_slice[mid_off : mid_off + max_fine_len]
        clip_fine = clip_slice[mid_off : mid_off + max_fine_len]
    else:
        ref_fine = ref_slice
        clip_fine = clip_slice

    # Fine GCC-PHAT lag over the overlapping region (also test -clip_slice for 180-deg polarity inversion)
    fine_lag_pos, coh_pos = _gcc_phat_lag(
        ref_fine, clip_fine, sr=sr, max_tau_sec=0.12, beta=0.72
    )
    fine_lag_neg, coh_neg = _gcc_phat_lag(
        ref_fine, -clip_fine, sr=sr, max_tau_sec=0.12, beta=0.72
    )

    polarity_events: list[AnomalyEvent] = []
    if coh_neg > coh_pos * 1.15 and coh_neg > 0.25:
        clip_stereo = -clip_stereo
        clip_mono = -clip_mono
        clip_slice = -clip_slice
        fine_lag_sec = fine_lag_neg
        coherence = coh_neg
        polarity_events.append(
            AnomalyEvent(
                kind="polarity_inverted",
                start_sec=0.0,
                end_sec=round(n_clip / float(sr), 2),
                severity=0.85,
                detail=f"Auto-corrected 180-deg inverted waveform polarity (coh={coh_neg:.2f} vs {coh_pos:.2f})",
            )
        )
    else:
        fine_lag_sec = fine_lag_pos
        coherence = coh_pos

    total_shift_sec = fine_lag_sec
    shift_samples = int(round(total_shift_sec * sr))
    frac_shift_sec = total_shift_sec - (shift_samples / float(sr))

    # Pre-align clip by integer shift_samples relative to s0 for windowed trajectory tracking
    aligned_clip_mono = np.zeros(ov_len, dtype=np.float64)
    if shift_samples >= 0:
        v = ov_len - shift_samples
        if v > 0:
            aligned_clip_mono[:v] = clip_slice[shift_samples : shift_samples + v]
    else:
        sh = -shift_samples
        v = ov_len - sh
        if v > 0:
            aligned_clip_mono[sh : sh + v] = clip_slice[:v]

    # Windowed tracking across the clip to measure linear clock drift (ppm) + non-rigid phone motion (ms)
    win_len = min(
        max(int(0.45 * sr), ov_len // max(num_subwindows // 2, 2)),
        ov_len // 2,
        int(3.0 * sr),
    )
    centers_sec: list[float] = []
    lags_sec: list[float] = []

    if win_len >= int(0.25 * sr):
        hop = max(
            (ov_len - win_len) // max(num_subwindows, 4),
            win_len // 2,
            int(0.15 * sr),
        )
        for w_start in range(0, ov_len - win_len + 1, hop):
            w_end = w_start + win_len
            win_r = ref_slice[w_start:w_end] * np.hanning(win_len)
            win_c = aligned_clip_mono[w_start:w_end] * np.hanning(win_len)
            if float(np.std(win_r)) < 1e-5 or float(np.std(win_c)) < 1e-5:
                continue
            loc_lag, loc_coh = _gcc_phat_lag(win_r, win_c, sr=sr, max_tau_sec=0.020, beta=0.70)
            if loc_coh > 0.12:
                centers_sec.append((c0 + w_start + 0.5 * win_len) / float(sr))
                lags_sec.append((shift_samples / float(sr)) + loc_lag)

    drift_ppm = 0.0
    max_motion_ms = 0.0
    motion_events: list[AnomalyEvent] = list(polarity_events)
    t_clip = np.arange(n_clip, dtype=np.float64) / float(sr)

    if len(centers_sec) >= 3:
        c_arr = np.asarray(centers_sec, dtype=np.float64)
        l_arr = np.asarray(lags_sec, dtype=np.float64)
        # Outlier rejection on windowed lags via median absolute deviation
        med_l = float(np.median(l_arr))
        mad_l = float(np.median(np.abs(l_arr - med_l)) + 1e-6)
        inliers = np.abs(l_arr - med_l) <= max(3.5 * 1.4826 * mad_l, 0.008)
        if np.sum(inliers) >= 3:
            c_arr = c_arr[inliers]
            l_arr = l_arr[inliers]

        # Robust Theil-Sen linear fit: lag(t) = tau_0 + slope * t
        slopes = []
        for i in range(len(c_arr)):
            for j in range(i + 1, len(c_arr)):
                dt = c_arr[j] - c_arr[i]
                if dt > 0.15:
                    slopes.append((l_arr[j] - l_arr[i]) / dt)
        slope = float(np.median(slopes)) if slopes else 0.0
        drift_ppm = float(np.clip(slope * 1e6, -500.0, 500.0))
        tau_0 = float(np.median(l_arr - slope * c_arr))

        linear_trend = tau_0 + (drift_ppm * 1e-6) * c_arr
        residual_motion_sec = l_arr - linear_trend
        smooth_res = ndimage.gaussian_filter1d(residual_motion_sec, sigma=1.0)
        max_motion_ms = float((np.max(smooth_res) - np.min(smooth_res)) * 1000.0)

        if enable_nonrigid and max_motion_ms > 0.35:
            # Interpolate smooth non-rigid moving-phone trajectory across full clip
            nonrigid_curve = np.interp(t_clip, c_arr, smooth_res, left=smooth_res[0], right=smooth_res[-1])
            tau_curve = tau_0 + (drift_ppm * 1e-6) * t_clip + nonrigid_curve
            if max_motion_ms > 0.75:
                motion_events.append(
                    AnomalyEvent(
                        kind="moving_phone",
                        start_sec=round(float(c_arr[0]), 2),
                        end_sec=round(float(c_arr[-1]), 2),
                        severity=round(float(np.clip(max_motion_ms / 8.0, 0.15, 1.0)), 2),
                        detail=f"Moving phone Doppler/ToA shift stabilized ({max_motion_ms:.2f} ms excursion)",
                    )
                )
        else:
            tau_curve = tau_0 + (drift_ppm * 1e-6) * t_clip
        initial_toa_ms = float(tau_0 * 1000.0)
    else:
        tau_curve = np.full(n_clip, total_shift_sec, dtype=np.float64)
        initial_toa_ms = float(total_shift_sec * 1000.0)
        frac_shift_sec = total_shift_sec

    # If coarse_start_sample was > 0 (a partial clip starting mid-song), separate timeline start vs acoustic ToA:
    # When coarse_start_sample > 0, `tau_curve` is the fine adjustment relative to `s0`.
    # We keep the clip anchored at `s0` on the master timeline and warp internal samples by `tau_curve`.
    t_lookup = t_clip + tau_curve
    warped = np.zeros_like(clip_stereo)
    for ch in range(clip_stereo.shape[1]):
        warped[:, ch] = np.interp(t_lookup, t_clip, clip_stereo[:, ch], left=0.0, right=0.0)

    true_start_sample = max(0, s0)
    return (
        warped,
        true_start_sample,
        initial_toa_ms,
        drift_ppm,
        max_motion_ms,
        coherence,
        motion_events,
    )


def assemble_concert_timeline(
    tracks: list[AudioTrack],
    target_sr: int = 48000,
    num_subwindows: int = 12,
    enable_nonrigid_motion: bool = True,
) -> tuple[list[PlacedClip], int, np.ndarray]:
    """Auto-detect start/stop positions of arbitrary partial clips and lock them onto a unified timeline.

    Returns (placed_clips, reference_idx, coverage_curve_per_sec).
    """
    if not tracks:
        raise ValueError("tracks list cannot be empty")

    # 1. Standardize sample rate to target_sr
    std_tracks: list[AudioTrack] = []
    for trk in tracks:
        if trk.sample_rate != target_sr:
            gcd = int(np.gcd(target_sr, trk.sample_rate))
            res = signal.resample_poly(trk.samples, target_sr // gcd, trk.sample_rate // gcd, axis=0)
            std_tracks.append(
                AudioTrack(
                    name=trk.name,
                    samples=res,
                    sample_rate=target_sr,
                    imu_telemetry=trk.imu_telemetry,
                    video_frames_rgb=trk.video_frames_rgb,
                )
            )
        else:
            std_tracks.append(trk)

    n_tracks = len(std_tracks)
    durations = [trk.duration_sec for trk in std_tracks]
    max_dur = max(durations)

    # 2. Pick primary anchor track: prefer long duration + wide high-frequency bandwidth + high SNR
    snr_clips = [_estimate_snr_and_clipping(trk.to_mono()) for trk in std_tracks]
    hf_scores: list[float] = []
    for trk in std_tracks:
        m = trk.to_mono()
        if len(m) < target_sr:
            hf_scores.append(-50.0)
            continue
        sig_len = min(len(m), target_sr * 12)
        fft = np.abs(np.fft.rfft(m[:sig_len]))
        freqs = np.fft.rfftfreq(sig_len, 1.0 / target_sr)
        p_mid = float(np.mean(fft[(freqs >= 1500.0) & (freqs <= 4500.0)] ** 2) + 1e-12)
        p_high = float(np.mean(fft[(freqs >= 10500.0) & (freqs <= min(15500.0, target_sr * 0.45))] ** 2) + 1e-12)
        hf_scores.append(10.0 * np.log10(p_high / p_mid))

    anchor_scores = []
    for i in range(n_tracks):
        dur_score = (durations[i] / max(max_dur, 1e-3)) * 25.0
        # Heavily penalize low-pass brickwalled phones (< -42 dB high-frequency cutoff)
        bw_penalty = -45.0 if hf_scores[i] < -42.0 and durations[i] < max_dur * 0.98 else 0.0
        bw_bonus = 8.0 if hf_scores[i] > -28.0 else 0.0
        anchor_scores.append(
            dur_score + snr_clips[i][0] - 120.0 * snr_clips[i][1] + bw_penalty + bw_bonus
        )
    ref_idx = int(np.argmax(anchor_scores))

    # 3. Compute multi-band envelope fingerprints for coarse start/stop partial-clip placement
    env_sr = 250
    active_bounds = [trim_active_bounds(trk.to_mono(), target_sr) for trk in std_tracks]
    envelopes = [
        _compute_multiband_envelope(trk.to_mono()[s0:s1], sr=target_sr, env_sr=env_sr)
        for (s0, s1), trk in zip(active_bounds, std_tracks)
    ]

    # Solve spanning tree of pairwise offsets rooted at ref_idx (with pairwise cache for large N=31+ payloads)
    placed_offsets_sec: dict[int, float] = {ref_idx: 0.0}
    placed_coherence: dict[int, float] = {ref_idx: 1.0}
    discovery_order: list[int] = [ref_idx]
    remaining = set(range(n_tracks)) - {ref_idx}
    pair_cache: dict[tuple[int, int], tuple[float, float]] = {}

    while remaining:
        best_pair: tuple[int, int, float, float] | None = None
        for u in list(placed_offsets_sec.keys()):
            for v in remaining:
                key = (u, v)
                if key not in pair_cache:
                    trimmed_lag, score = _pairwise_envelope_offset(envelopes[u], envelopes[v], env_sr=env_sr)
                    s0_u, _ = active_bounds[u]
                    s0_v, _ = active_bounds[v]
                    rel_off = (s0_u / float(target_sr)) + trimmed_lag - (s0_v / float(target_sr))
                    # If both clips span nearly the full performance and envelope offset is within 120ms,
                    # snap macro offset to 0.0s so GCC-PHAT resolves exact sub-millisecond acoustic ToA
                    if (
                        abs(durations[u] - durations[v]) < 0.05 * max_dur
                        and durations[v] > 0.90 * max_dur
                        and abs(rel_off) < 0.12
                    ):
                        rel_off = 0.0
                        score = max(score, 0.95)
                    pair_cache[key] = (rel_off, score)
                else:
                    rel_off, score = pair_cache[key]

                eff_score = score * (0.85 if (u != ref_idx and durations[ref_idx] >= 0.95 * max_dur) else 1.0)
                if best_pair is None or eff_score > best_pair[3]:
                    best_pair = (u, v, rel_off, eff_score)
        if best_pair is None:
            break
        u, v, rel_off, score = best_pair
        placed_offsets_sec[v] = placed_offsets_sec[u] + rel_off
        placed_coherence[v] = score
        discovery_order.append(v)
        remaining.remove(v)

    # Shift timeline so earliest clip starts at t = 0.0s
    min_start = min(placed_offsets_sec.values())
    for k in placed_offsets_sec:
        placed_offsets_sec[k] = max(0.0, placed_offsets_sec[k] - min_start)

    master_duration_sec = max(placed_offsets_sec[i] + durations[i] for i in range(n_tracks))
    master_len = int(round(master_duration_sec * target_sr))

    # Build initial master reference mono composite from anchor track
    master_ref_mono = np.zeros(master_len, dtype=np.float64)
    master_ref_populated = np.zeros(master_len, dtype=bool)
    ref_start_idx = int(round(placed_offsets_sec[ref_idx] * target_sr))
    ref_mono = std_tracks[ref_idx].to_mono()
    ref_end_idx = min(master_len, ref_start_idx + len(ref_mono))
    master_ref_mono[ref_start_idx:ref_end_idx] = ref_mono[: ref_end_idx - ref_start_idx]
    master_ref_populated[ref_start_idx:ref_end_idx] = True

    # 4. Fine GCC-PHAT alignment + non-rigid moving-phone trajectory warp in spanning-tree discovery order
    placed_clips: list[PlacedClip | None] = [None] * n_tracks
    order = discovery_order

    fade_samples = int(0.025 * target_sr)  # 25 ms smooth raised-cosine fade for partial clips

    for idx in order:
        trk = std_tracks[idx]
        stereo = trk.to_stereo()
        n_clip = stereo.shape[0]
        coarse_start = int(round(placed_offsets_sec[idx] * target_sr))
        snr_db, clip_ratio = snr_clips[idx]

        if idx == ref_idx:
            warped = stereo
            true_start = coarse_start
            toa_ms = 0.0
            drift_ppm = 0.0
            motion_ms = 0.0
            coh = 1.0
            m_events: list[AnomalyEvent] = []
        else:
            (
                warped,
                true_start,
                toa_ms,
                drift_ppm,
                motion_ms,
                coh,
                m_events,
            ) = _refine_and_track_nonrigid_motion(
                ref_mono=master_ref_mono,
                clip_stereo=stereo,
                sr=target_sr,
                coarse_start_sample=coarse_start,
                num_subwindows=num_subwindows,
                enable_nonrigid=enable_nonrigid_motion,
            )

        buf_dtype = np.float32 if master_len > int(30.0 * target_sr) else np.float64
        full_stereo = np.zeros((master_len, 2), dtype=buf_dtype)
        active_mask = np.zeros(master_len, dtype=buf_dtype)

        end_idx = min(master_len, true_start + n_clip)
        seg_len = max(0, end_idx - true_start)
        if seg_len > 0:
            full_stereo[true_start:end_idx] = warped[:seg_len].astype(buf_dtype, copy=False)
            local_mask = np.ones(seg_len, dtype=buf_dtype)
            # Apply smooth raised-cosine entry fade if clip starts after t=0.05s
            if true_start > int(0.05 * target_sr) and seg_len > 2 * fade_samples:
                local_mask[:fade_samples] = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_samples)))
            # Apply smooth raised-cosine exit fade if clip ends before master_len - 0.05s
            if end_idx < master_len - int(0.05 * target_sr) and seg_len > 2 * fade_samples:
                local_mask[-fade_samples:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_samples)))

            # Scrap Spliced / Talking / Non-Matching Segments:
            # Evaluate windowed correlation in 300-5000 Hz against master_ref_mono
            if n_tracks >= 3 and seg_len > int(8.0 * target_sr) and idx != ref_idx:
                sos_eval = signal.butter(2, [300.0, min(5000.0, target_sr * 0.45)], btype="bandpass", fs=target_sr, output="sos")
                clip_eval = signal.sosfilt(sos_eval, np.mean(warped[:seg_len], axis=1))
                ref_eval = signal.sosfilt(sos_eval, master_ref_mono[true_start:end_idx])

                sub_win = int(2.5 * target_sr)
                sub_step = int(1.5 * target_sr)
                bad_intervals: list[tuple[int, int]] = []

                s0_b, s1_b = active_bounds[idx]
                if s0_b > int(0.4 * target_sr):
                    bad_intervals.append((0, min(seg_len, s0_b)))
                if s1_b < n_clip - int(0.4 * target_sr):
                    bad_intervals.append((min(seg_len, s1_b), seg_len))

                ref_rms_global = float(np.sqrt(np.mean(ref_eval**2)) + 1e-12)
                clip_rms_global = float(np.sqrt(np.mean(clip_eval**2)) + 1e-12)

                for w0 in range(0, seg_len - sub_win + 1, sub_step):
                    w1 = w0 + sub_win
                    w_ref = ref_eval[w0:w1]
                    w_clip = clip_eval[w0:w1]
                    r_std = float(np.std(w_ref))
                    c_std = float(np.std(w_clip))

                    if r_std > 0.08 * ref_rms_global:
                        if c_std < 0.08 * clip_rms_global:
                            bad_intervals.append((w0, w1))
                        else:
                            norm_corr = float(np.sum(w_ref * w_clip) / ((r_std * c_std * sub_win) + 1e-12))
                            if norm_corr < 0.18:
                                max_j = int(0.025 * target_sr)
                                cc_j = signal.correlate(w_clip, w_ref, mode="full", method="fft")
                                center = len(w_clip) - 1
                                local_peak = float(np.max(cc_j[center - max_j : center + max_j + 1])) / ((r_std * c_std * sub_win) + 1e-12)
                                if local_peak < 0.18:
                                    bad_intervals.append((w0, w1))

                if bad_intervals:
                    bad_intervals.sort(key=lambda x: x[0])
                    merged_bad: list[list[int]] = [[bad_intervals[0][0], bad_intervals[0][1]]]
                    for cur_s, cur_e in bad_intervals[1:]:
                        if cur_s <= merged_bad[-1][1] + int(0.5 * target_sr):
                            merged_bad[-1][1] = max(merged_bad[-1][1], cur_e)
                        else:
                            merged_bad.append([cur_s, cur_e])

                    fade_len = int(0.05 * target_sr)
                    for b_s, b_e in merged_bad:
                        b_dur_s = (b_e - b_s) / float(target_sr)
                        if b_dur_s < 1.0 and b_s > 0 and b_e < seg_len:
                            continue
                        local_mask[b_s:b_e] = 0.0
                        if b_s >= fade_len:
                            local_mask[b_s - fade_len : b_s] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_len)))
                        if b_e + fade_len <= seg_len:
                            local_mask[b_e : b_e + fade_len] *= 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_len)))

                        m_events.append(
                            AnomalyEvent(
                                kind="spliced_outlier",
                                start_sec=round(float(true_start + b_s) / target_sr, 2),
                                end_sec=round(float(true_start + b_e) / target_sr, 2),
                                severity=0.85,
                                detail=f"Spliced non-matching / talking / end-card segment scrapped ({b_dur_s:.1f}s)",
                            )
                        )

            active_mask[true_start:end_idx] = local_mask

            # Expand master_ref_mono where it had zero coverage (spanning tree extension)
            unpopulated = ~master_ref_populated[true_start:end_idx]
            if np.any(unpopulated):
                w_mono = np.mean(warped[:seg_len], axis=1)
                master_ref_mono[true_start:end_idx] = np.where(
                    unpopulated, w_mono, master_ref_mono[true_start:end_idx]
                )
                master_ref_populated[true_start:end_idx] = True

        placed_trk = AudioTrack(
            name=trk.name,
            samples=full_stereo,
            sample_rate=target_sr,
            imu_telemetry=trk.imu_telemetry,
            video_frames_rgb=trk.video_frames_rgb,
        )
        placed_clips[idx] = PlacedClip(
            track=placed_trk,
            active_mask=active_mask,
            timeline_start_sec=round(true_start / float(target_sr), 3),
            timeline_end_sec=round(end_idx / float(target_sr), 3),
            clip_duration_sec=round(seg_len / float(target_sr), 3),
            initial_toa_ms=round(toa_ms, 3),
            clock_drift_ppm=round(drift_ppm, 2),
            max_motion_drift_ms=round(motion_ms, 3),
            coherence_score=round(coh, 3),
            snr_db=round(snr_db, 2),
            clipping_ratio=round(clip_ratio, 4),
            motion_events=m_events,
        )

    final_clips = [c for c in placed_clips if c is not None]

    # Compute 10-Hz timeline coverage curve (number of active POVs across the timeline)
    n_grid = max(int(np.ceil(master_duration_sec * 4.0)), 1)
    grid_indices = np.clip(
        np.linspace(0, master_len - 1, n_grid).astype(int), 0, max(master_len - 1, 0)
    )
    coverage = np.zeros(n_grid, dtype=int)
    for pc in final_clips:
        coverage += (pc.active_mask[grid_indices] > 0.1).astype(int)

    return final_clips, ref_idx, coverage
