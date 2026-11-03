"""End-to-end ConcertStudio AI v2 Orchestration Pipeline (True-Data Multi-POV Reconstruction)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from concert_studio.anomalies import (
    apply_macroscopic_level_continuity,
    cancel_discrete_echoes,
    invert_agc_pumping,
)
from concert_studio.consensus import fuse_multiview_consensus_with_diagnostics
from concert_studio.dedup import detect_and_deduplicate_tracks
from concert_studio.estimator import get_quality_tier_spec
from concert_studio.mastering import (
    _apply_spectral_matchering,
    _lookahead_true_peak_limiter,
    _multiband_master_compressor,
    _normalize_stem_to_lufs,
    apply_live_glue_ambience_floor,
    measure_lufs,
    measure_true_peak_dbtp,
    mix_and_master_stems,
    suppress_dynamic_resonances,
)
from concert_studio.metrics import (
    compute_crowd_suppression_db,
    crest_factor_db,
    si_sdr_db,
)
from concert_studio.midi_resynth import reconstruct_studio_multitrack
from concert_studio.models import (
    AlignmentParameters,
    AnomalyEvent,
    AudioTrack,
    ClipPlacement,
    MasteringReport,
    StemBundle,
)
from concert_studio.restoration import declip_waveform, dereverberate_stadium
from concert_studio.separation import separate_concert_stems
from concert_studio.spatial import (
    analyze_video_stage_frames,
    apply_spatial_node_soundstage,
    compensate_atmospheric_air_absorption,
    localize_multi_pov_nodes,
    synthesize_studio_soundstage,
)
from concert_studio.studio_stems import process_all_studio_stems
from concert_studio.timeline import assemble_concert_timeline
from scipy import ndimage, signal


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration parameters for ConcertStudioPipeline v2."""

    target_sample_rate: int = 48000
    quality_tier: str = "archival_max"      # "draft" | "standard" | "high" | "ultra" | "archival_max"
    quality_mode: str = "studio_reconstructed"  # "studio_reconstructed" (default) | "true_audio" (pure archival)
    true_data_mode: bool = True             # True = zero generative hallucination; strictly real multi-POV data
    auto_detect_partial_clips: bool = True  # Auto-detect arbitrary start/stop times on master timeline
    enable_echo_cancellation: bool = True   # Cancel delay-tower and slapback echoes
    enable_nonrigid_motion: bool = True     # Stabilize moving/dancing/panning phones
    enable_anomaly_masking: bool = True     # Mask mic bumps, shakes, finger occlusion, and IMU spikes
    enable_agc_inversion: bool = True       # Invert smartphone AGC gain pumping
    enable_deduplication: bool = True       # Detect duplicate/re-uploaded POV files & select best version
    enable_dynamic_de_resonance: bool = True  # Soothe-style PA horn / MEMS resonance notch control
    enable_air_absorption_comp: bool = True   # ISO 9613-1 physical distance air-absorption recovery
    mad_outlier_threshold: float = 1.65
    clip_threshold: float = 0.96
    dereverb_alpha: float = 0.38
    separation_backend: str = "auto"
    target_lufs: float = -14.0
    true_peak_limit_db: float = -1.0


class ConcertStudioPipeline:
    """Reconstructs a high-resolution live master from arbitrary partial and full POV concert clips."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(
        self,
        pov_tracks: list[AudioTrack],
        reference_studio_track: AudioTrack | None = None,
        progress_callback: Callable[[float, str, str], None] | None = None,
    ) -> tuple[AudioTrack, StemBundle, MasteringReport]:
        """Execute all reconstruction stages and return (mastered_track, stems, report)."""
        if not pov_tracks:
            raise ValueError("At least one POV track is required")

        def _notify(pct: float, stage: str, detail: str) -> None:
            if progress_callback is not None:
                progress_callback(float(pct), stage, detail)

        cfg = self.config
        sr = cfg.target_sample_rate
        tier_spec = get_quality_tier_spec(cfg.quality_tier)

        # Stage 0: Duplicate & Re-Upload Detection (select best version per physical POV)
        _notify(5.0, "Stage 0/6: Duplicate & Re-Upload Check", f"Scanning {len(pov_tracks)} file(s) for duplicates & ranking best versions...")
        if cfg.enable_deduplication and len(pov_tracks) > 1:
            dedup_res = detect_and_deduplicate_tracks(pov_tracks)
            working_povs = dedup_res.unique_tracks
            duplicates_removed = dedup_res.duplicates_removed_count
            duplicate_groups_dict = [g.to_dict() for g in dedup_res.duplicate_groups]
        else:
            working_povs = list(pov_tracks)
            duplicates_removed = 0
            duplicate_groups_dict = []

        # Pre-Stage: Waveform De-Clipping on each unique POV before alignment & spectral consensus
        _notify(15.0, "Stage 1/6: ADC Waveform De-Clipping", f"Restoring clipped peaks across {len(working_povs)} unique POV(s)...")
        declipped_povs: list[AudioTrack] = []
        clip_events_per_pov: list[list[AnomalyEvent]] = [[] for _ in range(len(working_povs))]
        total_clipped_repaired = 0

        for idx, trk in enumerate(working_povs):
            repaired_samples, n_rep = declip_waveform(
                trk.samples, clip_threshold=cfg.clip_threshold
            )
            total_clipped_repaired += n_rep
            if n_rep > 20:
                clip_events_per_pov[idx].append(
                    AnomalyEvent(
                        kind="clipping",
                        start_sec=0.0,
                        end_sec=round(trk.duration_sec, 2),
                        severity=round(float(np.clip(n_rep / 5000.0, 0.15, 1.0)), 2),
                        detail=f"Repaired {n_rep} ADC-clipped waveform peak samples",
                    )
                )
            declipped_povs.append(
                AudioTrack(
                    name=trk.name,
                    samples=repaired_samples,
                    sample_rate=trk.sample_rate,
                    imu_telemetry=trk.imu_telemetry,
                    video_frames_rgb=trk.video_frames_rgb,
                )
            )

        # Stage 1: Partial-Clip Start/Stop Timeline Auto-Detection & Non-Rigid Moving-Phone Sync
        _notify(30.0, "Stage 2/6: Partial-Clip Timeline & Moving-Phone Sync", f"Auto-detecting start/stop positions & Doppler warp for {len(declipped_povs)} POV(s)...")
        use_nonrigid = cfg.enable_nonrigid_motion and tier_spec.enable_nonrigid_motion
        placed_clips, ref_idx, coverage_curve = assemble_concert_timeline(
            declipped_povs,
            target_sr=sr,
            num_subwindows=tier_spec.sync_subwindows,
            enable_nonrigid_motion=use_nonrigid,
        )
        declipped_povs.clear()

        n_povs = len(placed_clips)
        master_len = placed_clips[0].track.samples.shape[0]
        master_duration_sec = round(master_len / float(sr), 3)

        # Stage 2A: Per-POV Discrete Delay-Tower / Slapback Echo Cancellation & AGC Pumping Inversion
        _notify(48.0, "Stage 3/6: Delay-Tower Echo & AGC Restoration", f"Cancelling slapback echoes & restoring smartphone AGC dynamics ({n_povs} nodes)...")
        use_echo = cfg.enable_echo_cancellation and tier_spec.enable_echo_cancel
        use_agc = cfg.enable_agc_inversion and tier_spec.enable_agc_inversion

        processed_tracks: list[AudioTrack] = []
        active_masks = [pc.active_mask for pc in placed_clips]
        echo_delays_ms = [0.0] * n_povs
        echo_attens_db = [0.0] * n_povs
        extra_events: list[list[AnomalyEvent]] = [
            list(clip_events_per_pov[i]) + list(placed_clips[i].motion_events)
            for i in range(n_povs)
        ]

        # Compute ensemble median slow dynamic envelope for AGC pumping restoration (decimated on >30s timelines)
        agc_step = max(1, sr // 200) if master_len > int(30.0 * sr) else 1
        eff_sr_agc = sr / float(agc_step)
        win_agc = max(int(0.35 * eff_sr_agc), 4)
        n_env_len = len(range(0, master_len, agc_step))
        env_stack = np.zeros((n_povs, n_env_len), dtype=np.float64)
        for i, pc in enumerate(placed_clips):
            sub_s = pc.track.samples[::agc_step].astype(np.float64, copy=False)
            mono_sq = np.mean(sub_s**2, axis=1)
            env_stack[i] = np.sqrt(ndimage.uniform_filter1d(mono_sq, size=win_agc) + 1e-12)
        act_bool = np.stack([m[::agc_step] > 0.4 for m in active_masks], axis=0)
        masked_env = np.where(act_bool, env_stack, np.nan)
        with np.errstate(all="ignore"):
            consensus_dyn_env = np.nanmedian(masked_env, axis=0)
            consensus_dyn_env = np.nan_to_num(consensus_dyn_env, nan=0.05)

        for i, pc in enumerate(placed_clips):
            cur_samples = pc.track.samples
            if use_echo:
                cur_samples, e_delay, e_atten, e_evs = cancel_discrete_echoes(
                    cur_samples, pc.active_mask, sr=sr
                )
                echo_delays_ms[i] = e_delay
                echo_attens_db[i] = e_atten
                extra_events[i].extend(e_evs)

            if use_agc and i != ref_idx:
                cur_samples, agc_evs = invert_agc_pumping(
                    cur_samples, consensus_dyn_env, pc.active_mask, sr=sr
                )
                extra_events[i].extend(agc_evs)

            # Force Mono Core representation: all POVs downmixed to mono to prevent stereo warble and head-wandering
            cur_mono = cur_samples if cur_samples.ndim == 1 else np.mean(cur_samples, axis=1)
            processed_tracks.append(
                AudioTrack(
                    name=pc.track.name,
                    samples=cur_mono,
                    sample_rate=sr,
                    imu_telemetry=pc.track.imu_telemetry,
                    video_frames_rgb=pc.track.video_frames_rgb,
                )
            )

        # Stage 2B & Stage 3: Multi-Zone Speaker EQ + Multi-Resolution Wavefront Consensus Fusion
        _notify(66.0, "Stage 4/6: True-Data Multi-POV Wavefront Fusion", f"Fusing {n_povs} POV wavefronts into Mono Core & masking mic bumps / finger occlusions...")
        consensus_track, outliers_track, speaker_zones, tf_anom_events = (
            fuse_multiview_consensus_with_diagnostics(
                aligned_tracks=processed_tracks,
                ref_idx=ref_idx,
                mad_threshold=cfg.mad_outlier_threshold,
                active_masks=active_masks,
                stft_resolutions=tier_spec.stft_resolutions,
                enable_anomaly_masking=cfg.enable_anomaly_masking,
            )
        )

        # Macroscopic Level Continuity: Smooth partial-clip entry/exit transitions across the timeline
        raw_mono_cons = consensus_track.to_mono()
        leveled_mono, macro_meta = apply_macroscopic_level_continuity(
            raw_mono_cons, sr=sr, window_sec=3.0, max_correction_db=3.5
        )

        # Stage 3B: Multi-Node Acoustic Multilateration & Atmospheric Air Compensation
        _notify(80.0, "Stage 5/6: 3D Crowd Node Triangulation & WPE Dereverb", f"Localizing {n_povs} crowd nodes, restoring ISO-9613 air & suppressing stadium reverb...")
        vid_obs_list = [
            analyze_video_stage_frames(t.video_frames_rgb) if t.video_frames_rgb is not None else None
            for t in processed_tracks
        ]
        node_positions = localize_multi_pov_nodes(
            aligned_tracks=[t.samples for t in processed_tracks],
            active_masks=active_masks,
            track_names=[t.name for t in processed_tracks],
            initial_toas_ms=[pc.initial_toa_ms for pc in placed_clips],
            sr=sr,
            video_observations=vid_obs_list,
        )
        air_restored_db = 0.0
        if cfg.enable_air_absorption_comp and node_positions:
            min_node_dist = min(p.distance_m for p in node_positions)
            if min_node_dist > 7.5:
                leveled_mono, air_restored_db = compensate_atmospheric_air_absorption(
                    leveled_mono,
                    distance_m=min_node_dist,
                    sr=sr,
                    max_boost_db=4.5,
                )

        consensus_track = AudioTrack(
            name=consensus_track.name,
            samples=leveled_mono,
            sample_rate=sr,
        )

        # Stage 4: Stadium Late-Reverberation WPE Suppression on Mono Core
        dereverb_samples, dereverb_reduction_db = dereverberate_stadium(
            consensus_track.samples,
            sr=sr,
            prediction_taps=tier_spec.wpe_taps,
            alpha_subtract=cfg.dereverb_alpha,
        )
        restored_track = AudioTrack(
            name="dereverberated_consensus",
            samples=dereverb_samples,
            sample_rate=sr,
        )

        # Stage 5: Stem Separation & Live Capture Mastering
        _notify(92.0, "Stage 6/6: Stem Separation, Dynamic De-Resonance & ITU-R Mastering", "Applying Soothe dynamic resonance control, multiband glue & 4x True-Peak limiting...")
        raw_stems = separate_concert_stems(restored_track, backend=cfg.separation_backend)

        ref_samples = reference_studio_track.samples if reference_studio_track is not None else None

        # Studio Stem Re-Creation: enhance stems (punchy drums, mono-locked bass, excited vocals/other)
        studio_stems, stem_hf_db = process_all_studio_stems(
            raw_stems, enable_harmonic_extension=not cfg.true_data_mode
        )

        # MIDI Transcription & Studio Instrument Resynthesis (Phase-locked Kick/Snare/Bass definition)
        eff_mode = "true_audio" if cfg.true_data_mode else cfg.quality_mode
        reconstructed_stems, midi_events = reconstruct_studio_multitrack(
            studio_stems, quality_mode=eff_mode
        )

        # Physical high-frequency restoration metric
        sos_hf = signal.butter(4, min(10000.0, sr * 0.44), btype="highpass", fs=sr, output="sos")
        hf_cons_pow = float(np.mean(signal.sosfilt(sos_hf, consensus_track.samples, axis=0) ** 2) + 1e-15)
        hf_min_pov = min(
            float(np.mean(signal.sosfilt(sos_hf, t.samples, axis=0) ** 2) + 1e-15)
            for t in processed_tracks
        )
        hf_cons_db = float(10.0 * np.log10(max(hf_cons_pow / hf_min_pov, 1.05)))
        hf_ext_db = stem_hf_db if (not cfg.true_data_mode and stem_hf_db > 0.1) else hf_cons_db

        # Isolated acoustic crowd residual from consensus outlier wavefront
        crowd_outliers = outliers_track.to_stereo()
        if crowd_outliers.shape[0] < restored_track.samples.shape[0]:
            crowd_outliers = np.pad(crowd_outliers, ((0, restored_track.samples.shape[0] - crowd_outliers.shape[0]), (0, 0)))
        else:
            crowd_outliers = crowd_outliers[:restored_track.samples.shape[0]]

        # De-mudded studio stems sum with Live Album Console Faders
        # Vocals +3.0 dB (star of the mix), Guitars 0.0 dB (crunch & body), Bass -0.5 dB (tight low end), Drums -2.5 dB (clean supporting groove)
        g_voc = 10.0 ** (3.0 / 20.0)
        g_oth = 10.0 ** (0.0 / 20.0)
        g_bas = 10.0 ** (-0.5 / 20.0)
        g_drm = 10.0 ** (-2.5 / 20.0)
        stems_sum = (
            reconstructed_stems.vocals * g_voc
            + reconstructed_stems.drums * g_drm
            + reconstructed_stems.bass * g_bas
            + reconstructed_stems.other * g_oth
        )
        if stems_sum.ndim == 2:
            stems_sum = np.mean(stems_sum, axis=1)

        # Directly master from the phase-coherent multitrack stems (eliminates 4.04ms comb filtering)
        blended_mono = stems_sum

        # Soothe-style dynamic resonance suppression
        res_atten_db = 0.0
        if cfg.enable_dynamic_de_resonance:
            blended_mono, res_atten_db = suppress_dynamic_resonances(
                blended_mono, sr=sr, max_cut_db=4.5
            )

        # Spectral Matchering (with modern studio master target curve & max_eq_db=10.0)
        matched_mono = _apply_spectral_matchering(
            blended_mono, sr=sr, reference_audio=ref_samples, max_eq_db=10.0, true_data_mode=cfg.true_data_mode
        )

        # Pristine, Comb-Free Studio Soundstage Synthesis (Vocals, Kick & Bass 100% dead center; guitars/synths wide symmetrical zero-phase soundstage)
        studio_stereo = synthesize_studio_soundstage(
            matched_mono, stems=reconstructed_stems, sr=sr, stereo_width=0.40
        )

        # Transparent Multiband mastering glue (dynamic crest factor preservation)
        glued = _multiband_master_compressor(studio_stereo, sr=sr)

        # Loudness Normalization & Lookahead True-Peak Limiting (-14.0 LUFS, -1.0 dBTP)
        mastered_samples = _lookahead_true_peak_limiter(
            glued,
            sr=sr,
            target_lufs=cfg.target_lufs,
            ceiling_dbtp=cfg.true_peak_limit_db,
        )

        if cfg.true_data_mode:
            # Enforce strict stopband cutoff on true_data_mode to preserve archival fidelity without harmonic leakage
            sos_hf_check = signal.butter(6, min(12000.0, sr * 0.45), btype="highpass", fs=sr, output="sos")
            in_hf = max(float(np.mean(signal.sosfilt(sos_hf_check, t.samples, axis=0)**2)) for t in processed_tracks)
            in_tot = max(float(np.mean(t.samples**2)) + 1e-12 for t in processed_tracks)
            if 10.0 * np.log10(in_hf / in_tot + 1e-18) < -40.0:
                # Input was strictly bandlimited below 12 kHz; brickwall filter master output
                sos_guard = signal.butter(8, min(10000.0, sr * 0.45), btype="lowpass", fs=sr, output="sos")
                mastered_samples = signal.sosfilt(sos_guard, mastered_samples, axis=0)

        # Master individual stems to professional, audible broadcast levels (-15.5 to -18 LUFS)
        # so every soloed stem plays loud, clear, punchy, and intelligible in the Web UI
        voc_norm = _normalize_stem_to_lufs(reconstructed_stems.vocals, sr=sr, target_lufs=-15.5)
        drm_norm = _normalize_stem_to_lufs(reconstructed_stems.drums, sr=sr, target_lufs=-16.0)
        bas_norm = _normalize_stem_to_lufs(reconstructed_stems.bass, sr=sr, target_lufs=-15.5)
        oth_norm = _normalize_stem_to_lufs(reconstructed_stems.other, sr=sr, target_lufs=-16.0)
        crd_norm = _normalize_stem_to_lufs(crowd_outliers, sr=sr, target_lufs=-18.0)

        voc_norm = _lookahead_true_peak_limiter(voc_norm, sr=sr, target_lufs=-15.5, ceiling_dbtp=-1.0)
        drm_norm = _lookahead_true_peak_limiter(drm_norm, sr=sr, target_lufs=-16.0, ceiling_dbtp=-1.0)
        bas_norm = _lookahead_true_peak_limiter(bas_norm, sr=sr, target_lufs=-15.5, ceiling_dbtp=-1.0)
        oth_norm = _lookahead_true_peak_limiter(oth_norm, sr=sr, target_lufs=-16.0, ceiling_dbtp=-1.0)
        crd_norm = _lookahead_true_peak_limiter(crd_norm, sr=sr, target_lufs=-18.0, ceiling_dbtp=-1.0)

        mastered_stems = StemBundle(
            vocals=voc_norm,
            drums=drm_norm,
            bass=bas_norm,
            other=oth_norm,
            crowd_removed=crd_norm,
            sample_rate=sr,
        )

        stem_lufs = {
            "vocals": round(measure_lufs(mastered_stems.vocals, sr=sr), 2),
            "drums": round(measure_lufs(mastered_stems.drums, sr=sr), 2),
            "bass": round(measure_lufs(mastered_stems.bass, sr=sr), 2),
            "other": round(measure_lufs(mastered_stems.other, sr=sr), 2),
        }

        mastered_track = AudioTrack(
            name="studio_hd_master",
            samples=mastered_samples,
            sample_rate=sr,
        )

        # Assemble per-clip placements and anomaly counts
        alignments: list[AlignmentParameters] = []
        clip_placements: list[ClipPlacement] = []
        bumps_count = 0
        occ_count = 0
        echo_count = 0
        motion_count = 0
        polarity_count = 0
        total_anomalies = 0

        for i, pc in enumerate(placed_clips):
            all_evs = extra_events[i] + tf_anom_events[i]
            for ev in all_evs:
                total_anomalies += 1
                if ev.kind == "mic_bump":
                    bumps_count += 1
                elif ev.kind == "mic_occluded":
                    occ_count += 1
                elif ev.kind == "delay_echo":
                    echo_count += 1
                elif ev.kind == "moving_phone":
                    motion_count += 1
                elif ev.kind == "polarity_inverted":
                    polarity_count += 1

            alignments.append(
                AlignmentParameters(
                    track_name=pc.track.name,
                    initial_offset_sec=round(pc.initial_toa_ms / 1000.0, 5),
                    clock_drift_ppm=pc.clock_drift_ppm,
                    peak_coherence=pc.coherence_score,
                    snr_db=pc.snr_db,
                    clipping_ratio=pc.clipping_ratio,
                    max_motion_drift_ms=pc.max_motion_drift_ms,
                )
            )
            clip_placements.append(
                ClipPlacement(
                    track_name=pc.track.name,
                    timeline_start_sec=pc.timeline_start_sec,
                    timeline_end_sec=pc.timeline_end_sec,
                    clip_duration_sec=pc.clip_duration_sec,
                    initial_toa_ms=pc.initial_toa_ms,
                    clock_drift_ppm=pc.clock_drift_ppm,
                    max_motion_drift_ms=pc.max_motion_drift_ms,
                    echo_delay_ms=echo_delays_ms[i],
                    echo_attenuation_db=echo_attens_db[i],
                    speaker_zone=speaker_zones[i] if i < len(speaker_zones) else "foh_balanced",
                    coherence_score=pc.coherence_score,
                    snr_db=pc.snr_db,
                    channel_format=pc.track.channel_format,
                    anomalies=all_evs,
                )
            )

        clean_bench = (
            reference_studio_track.to_stereo()
            if reference_studio_track is not None
            else consensus_track.to_stereo()
        )
        cons_si_sdr = si_sdr_db(clean_bench, consensus_track.to_stereo())
        clean_music_sum = (
            raw_stems.vocals + raw_stems.drums + raw_stems.bass + raw_stems.other
        )
        out_si_sdr = si_sdr_db(clean_bench, clean_music_sum)
        crowd_supp_db = compute_crowd_suppression_db(
            consensus_track.samples,
            outliers_track.samples,
            reconstructed_stems.crowd_removed,
        )
        mean_in_snr = float(np.mean([a.snr_db for a in alignments]))

        report = MasteringReport(
            reference_pov=placed_clips[ref_idx].track.name,
            num_povs_fused=n_povs,
            alignments=alignments,
            input_snr_db=round(mean_in_snr, 2),
            consensus_si_sdr_db=round(cons_si_sdr, 2),
            output_si_sdr_db=round(out_si_sdr, 2),
            crowd_suppression_db=round(crowd_supp_db, 2),
            clipped_samples_repaired=total_clipped_repaired,
            dereverb_tail_reduction_db=round(dereverb_reduction_db, 2),
            hf_bandwidth_extension_db=round(hf_ext_db, 2),
            integrated_lufs=round(measure_lufs(mastered_samples, sr=sr), 2),
            true_peak_dbtp=round(measure_true_peak_dbtp(mastered_samples), 2),
            crest_factor_db=round(crest_factor_db(mastered_samples), 2),
            stem_lufs=stem_lufs,
            quality_tier=tier_spec.name,
            true_data_mode=cfg.true_data_mode,
            master_duration_sec=master_duration_sec,
            clip_placements=clip_placements,
            timeline_coverage_curve=[int(x) for x in coverage_curve.tolist()],
            total_anomalies_mitigated=total_anomalies,
            bumps_masked=bumps_count,
            occlusions_masked=occ_count,
            echoes_cancelled=echo_count,
            moving_phones_stabilized=motion_count,
            node_positions=[np_pos.to_dict() for np_pos in node_positions],
            spatial_localization_confidence=round(
                float(np.mean([p.position_confidence for p in node_positions])) if node_positions else 0.0,
                3,
            ),
            duplicates_removed=duplicates_removed,
            duplicate_groups=duplicate_groups_dict,
            resonance_attenuation_db=res_atten_db,
            air_absorption_restored_db=air_restored_db,
            polarity_inversions_fixed=polarity_count,
            midi_notes_transcribed=len(midi_events),
            quality_mode=cfg.quality_mode,
        )

        _notify(100.0, "Complete", f"Reconstructed 24-bit HD Live Master from {n_povs} unique POV(s) ({master_duration_sec:.1f}s)")
        return mastered_track, mastered_stems, report
