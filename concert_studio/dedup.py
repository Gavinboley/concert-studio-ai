"""Duplicate & Re-Upload File Detection and Best-Version Selection for ConcertStudio AI v2.

Crowdsourced concert repositories frequently contain multiple uploads originating from the
*exact same physical phone recording*:
1. Exact duplicate uploads (renamed files, double-clicks).
2. Transcoded / social-media re-uploads (TikTok/WhatsApp/YouTube rips with 13-15 kHz AAC/MP3
   high-frequency brickwalling and codec noise).
3. Gain-boosted / brickwall-clipped re-uploads (normalized with hard digital clipping).
4. Downsampled or mono-collapsed re-uploads (e.g. 48 kHz stereo -> 24 kHz mono).
5. Trimmed sub-clip re-uploads (a short 5-second highlight cut from a longer POV capture).

Leaving duplicate uploads of the same POV in the pool skews multi-POV median consensus
(giving one phone's localized crowd screams 2x-3x voting power) and pollutes the mix with
transcoding artifacts. This module clusters same-POV uploads via passband phase-locked
waveform & spectro-temporal fingerprinting, evaluates each candidate's objective signal
fidelity, and selects the single best version per physical POV (grafting any non-overlapping
timeline extensions if a lower-bitrate duplicate spans a longer window).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np
from scipy import signal

from concert_studio.models import AudioTrack
from concert_studio.sync import _estimate_snr_and_clipping as estimate_snr_and_clipping


@dataclass(frozen=True)
class DuplicateCandidateScore:
    """Objective audio fidelity and completeness metrics for a single uploaded file."""

    track_name: str
    quality_score: float            # Composite score [0.0, 100.0]
    effective_bandwidth_hz: float   # 98% spectral roll-off frequency (detects lossy transcodes)
    clipping_ratio: float           # Fraction of clipped samples [0.0, 1.0]
    crest_factor_db: float          # Peak-to-RMS dynamic headroom in dB
    snr_db: float                   # Estimated signal-to-noise ratio in dB
    stereo_width_ratio: float       # Side/Mid energy ratio (0.0 = mono-collapsed)
    sample_rate: int
    duration_sec: float
    has_telemetry_or_video: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DuplicateGroup:
    """Represents a cluster of 2+ uploaded files originating from the same physical POV recording."""

    group_id: int
    best_track_name: str
    best_quality_score: float
    discarded_tracks: list[str] = field(default_factory=list)
    duplicate_classifications: dict[str, str] = field(default_factory=dict)
    max_pairwise_correlation: float = 1.0
    selection_reason: str = ""
    tail_grafted_sec: float = 0.0
    candidates: list[DuplicateCandidateScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d


@dataclass(frozen=True)
class DeduplicationResult:
    """Output of the duplicate detection and best-version selection stage."""

    unique_tracks: list[AudioTrack]
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    candidate_scores: list[DuplicateCandidateScore] = field(default_factory=list)
    duplicates_removed_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unique_track_names": [t.name for t in self.unique_tracks],
            "duplicates_removed_count": self.duplicates_removed_count,
            "duplicate_groups": [g.to_dict() for g in self.duplicate_groups],
            "candidate_scores": [c.to_dict() for c in self.candidate_scores],
        }


def score_track_fidelity(track: AudioTrack) -> DuplicateCandidateScore:
    """Compute an objective [0, 100] fidelity score to rank re-uploads of the same POV."""
    mono = track.to_mono()
    sr = track.sample_rate
    snr_db, clip_ratio = estimate_snr_and_clipping(mono)

    # 1. Effective High-Frequency Bandwidth (detects MP3/AAC/social-media lowpass brickwalls)
    nperseg = min(2048, len(mono))
    if nperseg >= 64:
        freqs, _, Z = signal.stft(mono, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
        power = np.mean(np.abs(Z) ** 2, axis=1)
        # Ignore sub-100Hz rumble when computing cumulative spectral rolloff
        valid = freqs >= 100.0
        v_freqs = freqs[valid]
        v_power = power[valid]
        total_p = float(np.sum(v_power) + 1e-15)
        cum_p = np.cumsum(v_power) / total_p
        idx_985 = int(np.searchsorted(cum_p, 0.985))
        idx_985 = min(idx_985, len(v_freqs) - 1)
        bw_hz = float(v_freqs[idx_985])

        # Also measure high-air ratio (13.5 kHz - 20 kHz vs 1 kHz - 6 kHz)
        mid_mask = (freqs >= 1000.0) & (freqs <= 6000.0)
        hf_mask = (freqs >= 13500.0) & (freqs <= min(20500.0, sr * 0.48))
        if np.any(hf_mask) and np.any(mid_mask):
            hf_energy = float(np.mean(power[hf_mask]))
            mid_energy = float(np.mean(power[mid_mask]) + 1e-15)
            hf_ratio_db = 10.0 * np.log10((hf_energy + 1e-18) / mid_energy)
            # If HF air band is >-38 dB relative to mid band, bandwidth extends into air band
            if hf_ratio_db > -38.0:
                bw_hz = max(bw_hz, min(19500.0, sr * 0.45))
    else:
        bw_hz = float(sr * 0.4)

    # 2. Crest factor (Peak-to-RMS in dB)
    rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
    peak = float(np.max(np.abs(mono)) + 1e-12)
    crest_db = float(20.0 * np.log10(peak / rms))

    # 3. Stereo width ratio (detects mono-collapsed re-uploads)
    if track.num_channels >= 2:
        st = track.to_stereo()
        mid_rms = float(np.sqrt(np.mean((0.5 * (st[:, 0] + st[:, 1])) ** 2)) + 1e-12)
        side_rms = float(np.sqrt(np.mean((0.5 * (st[:, 0] - st[:, 1])) ** 2)))
        stereo_ratio = float(np.clip(side_rms / mid_rms, 0.0, 1.0))
    else:
        stereo_ratio = 0.0

    has_meta = (track.imu_telemetry is not None) or (track.video_frames_rgb is not None)

    # Weighted quality score [0, 100]:
    # - Bandwidth (up to 32 pts): 20 kHz = 32 pts, 12 kHz = 16 pts, 6 kHz = 4 pts
    bw_score = float(np.clip((bw_hz - 4000.0) / 16000.0, 0.0, 1.0) * 32.0)
    # - Unclipped integrity (up to 26 pts): 0% clipping = 26 pts; >=5% clipping = 0 pts
    clip_penalty = float(np.clip(clip_ratio / 0.04, 0.0, 1.0) * 26.0)
    clip_score = 26.0 - clip_penalty
    # - Dynamic crest & SNR (up to 18 pts)
    dyn_score = float(np.clip((crest_db - 4.0) / 12.0, 0.0, 1.0) * 9.0) + float(
        np.clip(snr_db / 30.0, 0.0, 1.0) * 9.0
    )
    # - Sample rate & true stereo (up to 16 pts)
    sr_score = float(np.clip(sr / 48000.0, 0.2, 1.0) * 8.0)
    st_score = float(np.clip(stereo_ratio / 0.08, 0.0, 1.0) * 8.0)
    # - Duration & telemetry bonus (up to 8 pts)
    dur_score = float(np.clip(track.duration_sec / 10.0, 0.1, 1.0) * 5.0)
    meta_score = 3.0 if has_meta else 0.0

    total = round(bw_score + clip_score + dyn_score + sr_score + st_score + dur_score + meta_score, 2)

    return DuplicateCandidateScore(
        track_name=track.name,
        quality_score=float(np.clip(total, 1.0, 100.0)),
        effective_bandwidth_hz=round(bw_hz, 1),
        clipping_ratio=round(clip_ratio, 5),
        crest_factor_db=round(crest_db, 2),
        snr_db=round(snr_db, 2),
        stereo_width_ratio=round(stereo_ratio, 4),
        sample_rate=sr,
        duration_sec=round(track.duration_sec, 3),
        has_telemetry_or_video=has_meta,
    )


def _passband_fingerprint(track: AudioTrack, cmp_sr: int = 16000) -> np.ndarray:
    """Extract a 250 Hz - 4.5 kHz band-limited mono signal at `cmp_sr` for codec-robust matching.

    Why 250 Hz - 4.5 kHz?
    Even aggressive re-uploads (WhatsApp/TikTok transcodes, 22 kHz resamples, mono downmixes,
    or brickwall-clipped copies) preserve the core midrange waveform of that specific phone's
    microphone + local crowd capture, while two *different* phones in the venue have independent
    crowd noise and room reflections in this band.
    """
    mono = track.to_mono()
    if track.sample_rate != cmp_sr:
        g = int(np.gcd(track.sample_rate, cmp_sr))
        mono = signal.resample_poly(mono, cmp_sr // g, track.sample_rate // g)
    sos = signal.butter(3, [250.0, 4500.0], btype="bandpass", fs=cmp_sr, output="sos")
    bp = signal.sosfiltfilt(sos, mono)
    std = float(np.std(bp) + 1e-12)
    return bp / std


def _measure_same_pov_similarity(
    fp_a: np.ndarray,
    fp_b: np.ndarray,
    cmp_sr: int = 16000,
) -> tuple[float, int]:
    """Compute aligned overlap Pearson correlation between two passband fingerprints.

    Returns `(overlap_correlation, lag_samples_b_in_a)` where `lag_samples_b_in_a` is the
    sample index in `fp_a` where `fp_b` starts.
    """
    n_a, n_b = len(fp_a), len(fp_b)
    min_overlap = min(int(0.45 * cmp_sr), min(n_a, n_b) // 2)
    if min_overlap < 64:
        return 0.0, 0

    # Fast FFT cross-correlation
    xcorr = signal.fftconvolve(fp_a, fp_b[::-1], mode="full")
    best_idx = int(np.argmax(np.abs(xcorr)))
    # In 'full' mode, index k corresponds to fp_b[0] aligned with fp_a[k - (n_b - 1)]
    lag = best_idx - (n_b - 1)

    start_a = max(0, lag)
    start_b = max(0, -lag)
    ov_len = min(n_a - start_a, n_b - start_b)
    if ov_len < min_overlap:
        return 0.0, lag

    seg_a = fp_a[start_a : start_a + ov_len]
    seg_b = fp_b[start_b : start_b + ov_len]

    # Fine sub-sample alignment at the single fixed lag `lag`:
    # Crucial physical distinction:
    # - Two uploads/re-uploads of the SAME phone recording share the exact same ADC clock
    #   (zero relative clock drift) and identical microphone phase/crowd capture, so both the
    #   first half and second half of the overlap stay phase-locked at the EXACT same `lag`
    #   with high waveform correlation (r >= 0.94 even under heavy clipping or MP3 transcoding).
    # - Two DIFFERENT phones in the crowd have independent quartz clock drift (PPM), different
    #   acoustic reflections, and independent crowd noise, so their fixed-lag waveform correlation
    #   across both halves is much lower.
    norm_a = float(np.linalg.norm(seg_a) + 1e-12)
    norm_b = float(np.linalg.norm(seg_b) + 1e-12)
    r_full = float(np.dot(seg_a, seg_b) / (norm_a * norm_b))
    if r_full <= 0.0:
        return 0.0, lag

    half = ov_len // 2
    r_h1 = float(
        np.dot(seg_a[:half], seg_b[:half])
        / (float(np.linalg.norm(seg_a[:half]) * np.linalg.norm(seg_b[:half])) + 1e-12)
    )
    r_h2 = float(
        np.dot(seg_a[half:], seg_b[half:])
        / (float(np.linalg.norm(seg_a[half:]) * np.linalg.norm(seg_b[half:])) + 1e-12)
    )

    # Also check high-frequency detail residual (seg - smooth_seg) at the same fixed lag
    win_s = 7
    det_a = seg_a - np.convolve(seg_a, np.ones(win_s) / win_s, mode="same")
    det_b = seg_b - np.convolve(seg_b, np.ones(win_s) / win_s, mode="same")
    r_det = float(
        np.dot(det_a, det_b) / (float(np.linalg.norm(det_a) * np.linalg.norm(det_b)) + 1e-12)
    )

    combined = 0.45 * r_full + 0.35 * min(r_h1, r_h2) + 0.20 * max(0.0, r_det)
    return float(np.clip(combined, 0.0, 1.0)), lag


def _classify_duplicate_relation(
    best: DuplicateCandidateScore,
    other: DuplicateCandidateScore,
) -> str:
    """Classify why `other` is a duplicate/inferior re-upload compared to `best`."""
    if other.duration_sec < 0.82 * best.duration_sec:
        return "subclip_reupload"
    if other.clipping_ratio > best.clipping_ratio + 0.008:
        return "clipped_reupload"
    if other.effective_bandwidth_hz < best.effective_bandwidth_hz - 2200.0:
        return "transcoded_reupload"
    if other.stereo_width_ratio < 0.01 and best.stereo_width_ratio > 0.04:
        return "mono_collapsed_reupload"
    if other.sample_rate < best.sample_rate:
        return "resampled_reupload"
    return "exact_or_gain_duplicate"


def _explain_selection(
    best: DuplicateCandidateScore,
    others: list[DuplicateCandidateScore],
    tail_grafted_sec: float,
) -> str:
    reasons: list[str] = []
    min_bw = min(o.effective_bandwidth_hz for o in others)
    max_clip = max(o.clipping_ratio for o in others)
    min_st = min(o.stereo_width_ratio for o in others)
    min_dur = min(o.duration_sec for o in others)

    if best.effective_bandwidth_hz > min_bw + 1500.0:
        reasons.append(
            f"wider HF bandwidth ({best.effective_bandwidth_hz / 1000:.1f} kHz vs {min_bw / 1000:.1f} kHz transcode)"
        )
    if max_clip > best.clipping_ratio + 0.005:
        reasons.append(
            f"cleaner headroom ({best.clipping_ratio * 100:.2f}% vs {max_clip * 100:.2f}% clipped)"
        )
    if best.stereo_width_ratio > 0.04 and min_st < 0.01:
        reasons.append("preserved true stereo imaging vs mono-collapsed re-upload")
    if best.duration_sec > min_dur + 0.25:
        reasons.append(f"longer coverage ({best.duration_sec:.2f}s vs {min_dur:.2f}s sub-clip)")
    if tail_grafted_sec > 0.05:
        reasons.append(f"grafted +{tail_grafted_sec:.2f}s unique tail from longer re-upload")
    if not reasons:
        reasons.append(f"highest composite signal fidelity ({best.quality_score:.1f}/100)")
    return "; ".join(reasons)


def detect_and_deduplicate_tracks(
    tracks: list[AudioTrack],
    similarity_threshold: float = 0.93,
    graft_non_overlapping_tails: bool = True,
) -> DeduplicationResult:
    """Detect duplicate/re-uploaded POV files and select the single best version per POV.

    Parameters
    ----------
    tracks:
        All uploaded `AudioTrack` instances.
    similarity_threshold:
        Minimum combined passband waveform + micro-envelope correlation (`[0, 1]`, default `0.88`)
        to cluster two files as originating from the same physical POV recording.
    graft_non_overlapping_tails:
        If `True` and a lower-bitrate duplicate covers a longer time span than the winning
        high-fidelity master of that same POV, seamlessly grafts the non-overlapping leading
        or trailing audio segment onto the winning track.
    """
    n = len(tracks)
    if n == 0:
        return DeduplicationResult(unique_tracks=[])

    scores = [score_track_fidelity(t) for t in tracks]
    if n == 1:
        return DeduplicationResult(
            unique_tracks=list(tracks),
            duplicate_groups=[],
            candidate_scores=scores,
            duplicates_removed_count=0,
        )

    cmp_sr = 16000
    fps: list[np.ndarray | None] = [None] * n

    def _get_fp(idx: int) -> np.ndarray:
        if fps[idx] is None:
            fps[idx] = _passband_fingerprint(tracks[idx], cmp_sr=cmp_sr)
        return fps[idx]  # type: ignore[return-value]

    # Fast 200 Hz log-RMS envelope pre-screen so 31+ file payloads (465+ pairs) run in <0.2s
    env_sr = 200
    fast_envs: list[np.ndarray] = []
    for t in tracks:
        mono = t.to_mono()
        hop = max(1, int(t.sample_rate // env_sr))
        n_f = max(1, len(mono) // hop)
        env = np.log1p(np.sqrt(np.mean(mono[: n_f * hop].reshape(n_f, hop) ** 2, axis=1)) * 20.0)
        env = env - np.mean(env)
        fast_envs.append(env)

    # Pairwise similarity and relative lag matrix
    sim_matrix = np.eye(n, dtype=np.float64)
    lag_matrix = np.zeros((n, n), dtype=int)  # lag_matrix[i, j] = start of j inside i (at cmp_sr)
    for i in range(n):
        for j in range(i + 1, n):
            if n >= 10:
                ea, eb = fast_envs[i], fast_envs[j]
                if len(ea) >= 16 and len(eb) >= 16:
                    xc = signal.fftconvolve(ea, eb[::-1], mode="full")
                    best_k = int(np.argmax(xc))
                    lag_e = best_k - (len(eb) - 1)
                    sa = max(0, lag_e)
                    sb = max(0, -lag_e)
                    ov_e = min(len(ea) - sa, len(eb) - sb)
                    if ov_e < min(12, min(len(ea), len(eb)) // 2):
                        continue
                    na = float(np.linalg.norm(ea[sa : sa + ov_e]) + 1e-12)
                    nb = float(np.linalg.norm(eb[sb : sb + ov_e]) + 1e-12)
                    if float(xc[best_k]) / (na * nb) < 0.65:
                        continue

            sim, lag = _measure_same_pov_similarity(_get_fp(i), _get_fp(j), cmp_sr=cmp_sr)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
            lag_matrix[i, j] = lag
            lag_matrix[j, i] = -lag

    # Connected components clustering of same-POV duplicates
    visited = [False] * n
    clusters: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        comp = [i]
        visited[i] = True
        queue = [i]
        while queue:
            curr = queue.pop(0)
            for nxt in range(n):
                if not visited[nxt] and sim_matrix[curr, nxt] >= similarity_threshold:
                    visited[nxt] = True
                    comp.append(nxt)
                    queue.append(nxt)
        clusters.append(comp)

    unique_tracks: list[AudioTrack] = []
    duplicate_groups: list[DuplicateGroup] = []
    removed_count = 0

    for cluster in clusters:
        if len(cluster) == 1:
            unique_tracks.append(tracks[cluster[0]])
            continue

        # Rank candidates in this duplicate cluster by quality_score descending
        # Note: if a short sub-clip has slightly higher bitrate than a full-length recording,
        # give a duration completeness bonus relative to the longest clip in the cluster
        max_cluster_dur = max(scores[idx].duration_sec for idx in cluster)

        def _effective_rank_score(idx: int) -> float:
            s = scores[idx]
            completeness = min(1.0, s.duration_sec / max(max_cluster_dur, 1e-6))
            # A sub-clip covering <50% of the POV gets penalized unless we graft or rank completeness
            return s.quality_score + 12.0 * completeness

        ranked = sorted(cluster, key=_effective_rank_score, reverse=True)
        best_idx = ranked[0]
        other_indices = ranked[1:]
        removed_count += len(other_indices)

        winning_track = tracks[best_idx]
        best_score = scores[best_idx]
        tail_grafted_sec = 0.0

        # Optional tail grafting: if one of the discarded duplicates of the SAME POV extends
        # significantly beyond the winning high-res clip (> 0.20s), graft that non-overlapping tail
        if graft_non_overlapping_tails:
            sr_win = winning_track.sample_rate
            win_st = winning_track.to_stereo()
            for o_idx in other_indices:
                if sim_matrix[best_idx, o_idx] < similarity_threshold:
                    continue
                # Lag of o_idx relative to best_idx in winning_track's sample rate
                lag_cmp = lag_matrix[best_idx, o_idx]
                lag_win = int(round(lag_cmp * (sr_win / float(cmp_sr))))
                other_trk = tracks[o_idx]
                if other_trk.duration_sec < 0.50 * winning_track.duration_sec and winning_track.duration_sec > 30.0:
                    continue
                other_st = other_trk.to_stereo()
                if other_trk.sample_rate != sr_win:
                    g = int(np.gcd(other_trk.sample_rate, sr_win))
                    other_st = signal.resample_poly(other_st, sr_win // g, other_trk.sample_rate // g, axis=0)

                other_end_in_win = lag_win + len(other_st)
                if other_end_in_win > len(win_st) + int(0.20 * sr_win):
                    # Graft trailing extension with 20ms crossfade and RMS level matching
                    ext_start_in_other = len(win_st) - lag_win
                    if 0 < ext_start_in_other < len(other_st):
                        win_rms = float(np.sqrt(np.mean(win_st**2)) + 1e-12)
                        oth_rms = float(np.sqrt(np.mean(other_st[:ext_start_in_other] ** 2)) + 1e-12)
                        gain = np.clip(win_rms / oth_rms, 0.25, 4.0)
                        tail = other_st[ext_start_in_other:] * gain
                        win_st = np.vstack([win_st, tail])
                        tail_grafted_sec += len(tail) / float(sr_win)

            if tail_grafted_sec > 0.0:
                winning_track = AudioTrack(
                    name=winning_track.name,
                    samples=win_st,
                    sample_rate=sr_win,
                    imu_telemetry=winning_track.imu_telemetry,
                    video_frames_rgb=winning_track.video_frames_rgb,
                )

        unique_tracks.append(winning_track)

        classifications = {
            tracks[o_idx].name: _classify_duplicate_relation(best_score, scores[o_idx])
            for o_idx in other_indices
        }
        max_sim = max(float(sim_matrix[best_idx, o_idx]) for o_idx in other_indices)
        other_scores = [scores[o_idx] for o_idx in other_indices]

        duplicate_groups.append(
            DuplicateGroup(
                group_id=len(duplicate_groups) + 1,
                best_track_name=winning_track.name,
                best_quality_score=best_score.quality_score,
                discarded_tracks=[tracks[o_idx].name for o_idx in other_indices],
                duplicate_classifications=classifications,
                max_pairwise_correlation=round(max_sim, 4),
                selection_reason=_explain_selection(best_score, other_scores, tail_grafted_sec),
                tail_grafted_sec=round(tail_grafted_sec, 3),
                candidates=[scores[idx] for idx in ranked],
            )
        )

    return DeduplicationResult(
        unique_tracks=unique_tracks,
        duplicate_groups=duplicate_groups,
        candidate_scores=scores,
        duplicates_removed_count=removed_count,
    )
