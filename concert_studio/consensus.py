"""Stage 2 & Stage 4: Multi-POV True-Data Wavefront Consensus & Multi-Resolution STFT Fusion."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from concert_studio.anomalies import (
    classify_and_equalize_speaker_zones,
    compute_tf_anomaly_reliability_masks,
)
from concert_studio.models import AnomalyEvent, AudioTrack, IMUTelemetry


def match_ltas_eq(
    track_samples: np.ndarray,
    ref_samples: np.ndarray,
    n_fft: int = 2048,
    max_boost_db: float = 18.0,
) -> np.ndarray:
    """Match Long-Term Average Spectrum (LTAS) of a POV to a reference spectrum."""
    hop = n_fft // 4
    out = np.zeros_like(track_samples)
    channels = 1 if track_samples.ndim == 1 else track_samples.shape[1]

    for ch in range(channels):
        x = track_samples if track_samples.ndim == 1 else track_samples[:, ch]
        r = ref_samples if ref_samples.ndim == 1 else ref_samples[:, ch]

        _, _, Zx = signal.stft(x, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, Zr = signal.stft(r, nperseg=n_fft, noverlap=n_fft - hop)

        ltas_x = np.sqrt(np.mean(np.abs(Zx) ** 2, axis=1)) + 1e-12
        ltas_r = np.sqrt(np.mean(np.abs(Zr) ** 2, axis=1)) + 1e-12

        raw_gain = ltas_r / ltas_x
        smoothed_gain = ndimage.gaussian_filter1d(raw_gain, sigma=4.0)
        max_lin = 10.0 ** (max_boost_db / 20.0)
        min_lin = 10.0 ** (-max_boost_db / 20.0)
        clamped_gain = np.clip(smoothed_gain, min_lin, max_lin)[:, None]

        _, y = signal.istft(Zx * clamped_gain, nperseg=n_fft, noverlap=n_fft - hop)
        y_trimmed = y[: len(x)]
        if len(y_trimmed) < len(x):
            y_trimmed = np.pad(y_trimmed, (0, len(x) - len(y_trimmed)))

        if track_samples.ndim == 1:
            out = y_trimmed
        else:
            out[:, ch] = y_trimmed

    return out


def _single_res_wavefront_fusion(
    eq_matched_stack: list[np.ndarray],
    active_masks: list[np.ndarray],
    imu_list: list[IMUTelemetry | None],
    sr: int,
    ref_idx: int,
    mad_threshold: float,
    n_fft: int,
    enable_anomaly_masking: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[list[AnomalyEvent]]]:
    """Perform reliability-weighted, phase-coherent wavefront fusion at a single STFT resolution.

    Automatically chunks long multi-minute concert timelines into 24-second overlap-add windows
    so 30+ POVs can be fused in bounded memory (<400 MB) without allocating multi-GB 3D STFT tensors.
    """
    target_len = eq_matched_stack[0].shape[0]
    n_povs = len(eq_matched_stack)
    chunk_samples = int(24.0 * sr)
    overlap_samples = int(1.0 * sr)

    if target_len > int(32.0 * sr):
        consensus_full = np.zeros((target_len, 2), dtype=np.float64)
        outliers_full = np.zeros((target_len, 2), dtype=np.float64)
        weight_acc = np.zeros(target_len, dtype=np.float64)
        all_events: list[list[AnomalyEvent]] = [[] for _ in range(n_povs)]
        step = chunk_samples - overlap_samples

        for s0 in range(0, target_len, step):
            s1 = min(target_len, s0 + chunk_samples)
            b_len = s1 - s0
            if b_len < n_fft * 2:
                break
            sub_eq = [st[s0:s1] for st in eq_matched_stack]
            sub_act = [am[s0:s1] for am in active_masks]
            c_blk, o_blk, ev_blk = _single_res_wavefront_fusion_core(
                eq_matched_stack=sub_eq,
                active_masks=sub_act,
                imu_list=imu_list,
                sr=sr,
                ref_idx=ref_idx,
                mad_threshold=mad_threshold,
                n_fft=n_fft,
                enable_anomaly_masking=enable_anomaly_masking,
            )
            win = np.ones(b_len, dtype=np.float64)
            if s0 > 0 and b_len > overlap_samples:
                win[:overlap_samples] = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, overlap_samples)))
            if s1 < target_len and b_len > overlap_samples:
                win[-overlap_samples:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, overlap_samples)))
            consensus_full[s0:s1] += c_blk * win[:, None]
            outliers_full[s0:s1] += o_blk * win[:, None]
            weight_acc[s0:s1] += win

            t_off = s0 / float(sr)
            for i in range(n_povs):
                for ev in ev_blk[i]:
                    all_events[i].append(
                        AnomalyEvent(
                            kind=ev.kind,
                            start_sec=round(ev.start_sec + t_off, 3),
                            end_sec=round(ev.end_sec + t_off, 3),
                            severity=ev.severity,
                            detail=ev.detail,
                        )
                    )
            if s1 >= target_len:
                break

        valid = weight_acc > 1e-8
        consensus_full[valid] /= weight_acc[valid, None]
        outliers_full[valid] /= weight_acc[valid, None]

        # Coalesce overlapping/adjacent chunk events and cap to top 12 salient events per POV for clean reporting
        for i in range(n_povs):
            if len(all_events[i]) > 12:
                all_events[i] = sorted(
                    sorted(all_events[i], key=lambda e: e.severity, reverse=True)[:12],
                    key=lambda e: e.start_sec,
                )
        return consensus_full, outliers_full, all_events

    return _single_res_wavefront_fusion_core(
        eq_matched_stack=eq_matched_stack,
        active_masks=active_masks,
        imu_list=imu_list,
        sr=sr,
        ref_idx=ref_idx,
        mad_threshold=mad_threshold,
        n_fft=n_fft,
        enable_anomaly_masking=enable_anomaly_masking,
    )


def _single_res_wavefront_fusion_core(
    eq_matched_stack: list[np.ndarray],
    active_masks: list[np.ndarray],
    imu_list: list[IMUTelemetry | None],
    sr: int,
    ref_idx: int,
    mad_threshold: float,
    n_fft: int,
    enable_anomaly_masking: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[list[AnomalyEvent]]]:
    """Core single-block STFT wavefront fusion."""
    target_len = eq_matched_stack[0].shape[0]
    n_povs = len(eq_matched_stack)
    hop = n_fft // 4

    consensus_stereo = np.zeros((target_len, 2), dtype=np.float64)
    outliers_stereo = np.zeros((target_len, 2), dtype=np.float64)
    collected_events: list[list[AnomalyEvent]] = [[] for _ in range(n_povs)]

    for ch in range(2):
        stfts = []
        freqs = None
        frame_times = None
        for st in eq_matched_stack:
            f_arr, t_arr, Z = signal.stft(st[:, ch], fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            freqs = f_arr
            frame_times = t_arr
            stfts.append(Z)
        Z_stack = np.stack(stfts, axis=0)  # (N, F, T_frames)
        mags = np.abs(Z_stack)
        _, n_bins, n_frames = Z_stack.shape

        # Interpolate sample-level active_masks onto STFT frame times
        t_samples = np.arange(target_len, dtype=np.float64) / float(sr)
        active_frame_masks = np.zeros((n_povs, n_frames), dtype=np.float64)
        for i in range(n_povs):
            active_frame_masks[i] = np.interp(
                frame_times, t_samples, active_masks[i], left=0.0, right=0.0
            )

        # Compute 2D anomaly reliability masks (mic bumps, shakes, finger occlusion, IMU spikes)
        if enable_anomaly_masking and ch == 0:
            tf_anom_masks, anom_events = compute_tf_anomaly_reliability_masks(
                stft_mags=mags,
                active_frame_masks=active_frame_masks,
                freqs=freqs,
                frame_times_sec=frame_times,
                imu_list=imu_list,
            )
            collected_events = anom_events
        elif enable_anomaly_masking:
            tf_anom_masks, _ = compute_tf_anomaly_reliability_masks(
                stft_mags=mags,
                active_frame_masks=active_frame_masks,
                freqs=freqs,
                frame_times_sec=frame_times,
                imu_list=imu_list,
            )
        else:
            tf_anom_masks = np.ones_like(mags)

        # Compute per-bin Median and MAD across ACTIVE POVs at each frame
        act_2d = active_frame_masks > 0.15
        # Ensure at least reference POV is marked active on boundary STFT frames
        empty_frames = ~np.any(act_2d, axis=0)
        if np.any(empty_frames):
            act_2d[ref_idx, empty_frames] = True
        act_3d = act_2d[:, None, :]
        masked_mags = np.where(act_3d, mags, np.nan)
        med_mag = np.nanmedian(masked_mags, axis=0, keepdims=True)
        med_mag = np.nan_to_num(med_mag, nan=0.0)
        mad = np.nanmedian(np.abs(masked_mags - med_mag), axis=0, keepdims=True)
        mad = np.nan_to_num(mad, nan=0.0) + 1e-9

        deviation = (mags - med_mag) / (1.4826 * mad)
        # Reject positive localized crowd spikes (screaming fans, hand claps)
        crowd_inlier_weight = np.where(
            deviation > mad_threshold,
            np.exp(-2.5 * (deviation - mad_threshold) ** 2),
            1.0,
        )
        # Downweight deep dropouts/nulls
        crowd_inlier_weight = np.where(
            deviation < -mad_threshold * 1.5,
            0.22,
            crowd_inlier_weight,
        )

        # Anisotropic 2D Spectro-Temporal Persistence Smoothing (eliminates watery "musical noise"
        # gain-jump artifacts across adjacent STFT frames and harmonic bins)
        for i in range(n_povs):
            crowd_inlier_weight[i] = ndimage.gaussian_filter(
                crowd_inlier_weight[i], sigma=(1.0, 1.4)
            )

        # Combine active clip mask + crowd outlier mask + mechanical/occlusion anomaly mask
        raw_weights = (
            active_frame_masks[:, None, :]
            * crowd_inlier_weight
            * tf_anom_masks
        )
        weight_sum = np.sum(raw_weights, axis=0, keepdims=True)
        # Fallback to equal weighting across active POVs if all weights drop near zero
        fallback = active_frame_masks[:, None, :] / (
            np.sum(active_frame_masks[:, None, :], axis=0, keepdims=True) + 1e-12
        )
        weights = np.where(weight_sum > 1e-8, raw_weights / (weight_sum + 1e-12), fallback)

        # Winner-Take-Most Micro-Mosaic (Stop Stacking!)
        # When large arrays of POVs are present (n_povs >= 12 or payload with 20+ phones),
        # focus fusion strictly onto the top-3 cleanest inlier POVs at each (f, t) cell.
        # For smaller arrays (n_povs < 12), keep proportional inliers to preserve array SNR gain.
        if n_povs >= 6:
            k_top = 3 if n_povs >= 12 else max(3, int(round(n_povs * 0.70)))
            top_thresh = np.partition(weights, -k_top, axis=0)[-k_top : -k_top + 1, :, :]
            weights_sharp = np.where(weights >= top_thresh, weights, 0.0)
            weights_sharp = weights_sharp ** 1.5
            for i in range(n_povs):
                weights_sharp[i] = ndimage.gaussian_filter(weights_sharp[i], sigma=(1.0, 6.0))
            w_sharp_sum = np.sum(weights_sharp, axis=0, keepdims=True)
            weights = np.where(w_sharp_sum > 1e-8, weights_sharp / (w_sharp_sum + 1e-12), weights)

        # Phase-Coherent Wavefront Alignment (Zero Comb-Filtering & Shimmer Elimination):
        # Multi-phone STFT phase averaging causes comb-filtering and phase cancellation above 2.5 kHz
        # because phones at different distances differ by centimeters, destroying hi-hat and cymbal phase!
        # Solution: Lock phase strictly to the highest-weight active POV anchor in highs and bass.
        # When magnitude is fused from clean inliers and phase is locked to the physical anchor mic,
        # transients and high frequencies remain 100% natural, crisp, punchy, and shimmer-free.
        unit_phasors = Z_stack / (mags + 1e-12)
        mean_phasor = np.sum(weights * unit_phasors, axis=0)
        phasor_norm = np.abs(mean_phasor)

        # Inter-POV Circular Phase Concentration R(f, t) Wiener Post-Filter:
        smooth_phase_coherence = ndimage.gaussian_filter(phasor_norm, sigma=(1.2, 2.0))
        wiener_coherence_gain = np.clip(0.88 + 0.12 * smooth_phase_coherence, 0.88, 1.0)

        # Continuous Dominant Inlier POV (with hysteresis to prevent mic-switching phase chatter):
        # We select the highest-weight inlier POV at each time frame, using 20% hysteresis so that
        # phase stays locked to a single physical mic unless that mic is genuinely occluded or distorted.
        time_weights = np.mean(weights, axis=1)  # (N, T_frames)
        time_weights_smooth = ndimage.gaussian_filter1d(time_weights, sigma=8.0, axis=1)
        best_pov_t = np.zeros(n_frames, dtype=int)
        curr_p = int(np.argmax(time_weights_smooth[:, 0]))
        for t_idx in range(n_frames):
            cand = int(np.argmax(time_weights_smooth[:, t_idx]))
            if time_weights_smooth[cand, t_idx] > 1.25 * time_weights_smooth[curr_p, t_idx]:
                curr_p = cand
            best_pov_t[t_idx] = curr_p

        anchor_phasor = unit_phasors[
            best_pov_t[None, :],
            np.arange(n_bins)[:, None],
            np.arange(n_frames)[None, :],
        ]

        # Below 220 Hz: lock phase strictly to anchor (eliminates multi-mic sub-bass beating/warble).
        # Above 2500 Hz: lock phase strictly to anchor (eliminates comb-filtering, flanging & high-end shimmer).
        # Mid band (220 Hz - 2500 Hz): use coherent mean phasor when concentration is high, fallback to anchor.
        mid_coherent = np.where(phasor_norm > 0.35, mean_phasor / (phasor_norm + 1e-12), anchor_phasor)
        lock_to_anchor = (freqs < 220.0) | (freqs >= 2500.0)
        coherent_phasor = np.where(lock_to_anchor[:, None], anchor_phasor, mid_coherent)

        # True-Data Magnitude Reconstruction (Weighted inlier magnitude, zero hallucination)
        consensus_mag = np.sum(weights * mags, axis=0) * wiener_coherence_gain
        Z_consensus = consensus_mag * coherent_phasor

        active_count = np.maximum(np.sum(act_3d, axis=0), 1)
        Z_mean_active = np.sum(np.where(act_3d, Z_stack, 0.0), axis=0) / active_count
        Z_outliers = Z_mean_active - Z_consensus

        _, y_cons = signal.istft(Z_consensus, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        _, y_out = signal.istft(Z_outliers, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

        if len(y_cons) < target_len:
            y_cons = np.pad(y_cons, (0, target_len - len(y_cons)))
        if len(y_out) < target_len:
            y_out = np.pad(y_out, (0, target_len - len(y_out)))

        consensus_stereo[:, ch] = y_cons[:target_len]
        outliers_stereo[:, ch] = y_out[:target_len]

    return consensus_stereo, outliers_stereo, collected_events


def fuse_multiview_consensus(
    aligned_tracks: list[AudioTrack],
    ref_idx: int = 0,
    mad_threshold: float = 1.65,
    n_fft: int = 2048,
    active_masks: list[np.ndarray] | None = None,
    stft_resolutions: list[int] | None = None,
    enable_anomaly_masking: bool = True,
) -> tuple[AudioTrack, AudioTrack]:
    """Fuse aligned POV tracks via Multi-Zone Speaker EQ & Multi-Resolution Wavefront Consensus."""
    track_cons, track_out, _, _ = fuse_multiview_consensus_with_diagnostics(
        aligned_tracks=aligned_tracks,
        ref_idx=ref_idx,
        mad_threshold=mad_threshold,
        n_fft=n_fft,
        active_masks=active_masks,
        stft_resolutions=stft_resolutions,
        enable_anomaly_masking=enable_anomaly_masking,
    )
    return track_cons, track_out


def fuse_multiview_consensus_with_diagnostics(
    aligned_tracks: list[AudioTrack],
    ref_idx: int = 0,
    mad_threshold: float = 1.65,
    n_fft: int = 2048,
    active_masks: list[np.ndarray] | None = None,
    stft_resolutions: list[int] | None = None,
    enable_anomaly_masking: bool = True,
) -> tuple[AudioTrack, AudioTrack, list[str], list[list[AnomalyEvent]]]:
    """Full v2 consensus returning (consensus_track, outliers_track, speaker_zones, anomaly_events_per_pov)."""
    if not aligned_tracks:
        raise ValueError("aligned_tracks cannot be empty")
    if len(aligned_tracks) == 1:
        single = aligned_tracks[0].to_stereo()
        sr0 = aligned_tracks[0].sample_rate
        return (
            AudioTrack(name="consensus", samples=single, sample_rate=sr0),
            AudioTrack(name="outliers", samples=np.zeros_like(single), sample_rate=sr0),
            ["foh_balanced"],
            [[]],
        )

    sr = aligned_tracks[0].sample_rate
    target_len = aligned_tracks[ref_idx].samples.shape[0]
    n_povs = len(aligned_tracks)

    raw_stack = [
        trk.samples[:target_len] if (trk.samples.ndim == 2 and trk.samples.shape[1] == 2) else trk.to_stereo()[:target_len]
        for trk in aligned_tracks
    ]
    if active_masks is None:
        masks_list = [np.ones(target_len, dtype=np.float32) for _ in range(n_povs)]
    else:
        masks_list = [np.asarray(m)[:target_len] for m in active_masks]

    imu_list = [trk.imu_telemetry for trk in aligned_tracks]

    # Step 1: Classify speaker tuning zones & equalize toward True-Data Composite Venue Spectrum
    eq_matched_stack, speaker_zones = classify_and_equalize_speaker_zones(
        raw_stack, masks_list, sr=sr, n_fft=n_fft
    )

    resolutions = stft_resolutions if stft_resolutions else [n_fft]
    cons_accum = np.zeros((target_len, 2), dtype=np.float64)
    out_accum = np.zeros((target_len, 2), dtype=np.float64)
    primary_events: list[list[AnomalyEvent]] = [[] for _ in range(n_povs)]

    for r_idx, res_fft in enumerate(resolutions):
        c_part, o_part, evs = _single_res_wavefront_fusion(
            eq_matched_stack=eq_matched_stack,
            active_masks=masks_list,
            imu_list=imu_list,
            sr=sr,
            ref_idx=ref_idx,
            mad_threshold=mad_threshold,
            n_fft=int(res_fft),
            enable_anomaly_masking=enable_anomaly_masking,
        )
        cons_accum += c_part / float(len(resolutions))
        out_accum += o_part / float(len(resolutions))
        if r_idx == 0:
            primary_events = evs

    return (
        AudioTrack(name="spatial_consensus", samples=cons_accum, sample_rate=sr),
        AudioTrack(name="spatial_outliers_removed", samples=out_accum, sample_rate=sr),
        speaker_zones,
        primary_events,
    )
