"""Tests for ConcertStudio AI v2.5 Acoustic Realism, Soundstage & Artifact Suppression Upgrades."""

from __future__ import annotations

import numpy as np
from scipy import signal

from concert_studio.mastering import (
    apply_live_glue_ambience_floor,
    suppress_dynamic_resonances,
)
from concert_studio.models import AudioTrack
from concert_studio.pipeline import ConcertStudioPipeline, PipelineConfig
from concert_studio.spatial import (
    SpatialNodePosition,
    apply_spatial_node_soundstage,
    compensate_atmospheric_air_absorption,
)
from concert_studio.synthetic import generate_synthetic_concert


def test_soothe_dynamic_resonance_suppressor_attenuates_harsh_horn_whistle() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.5, sr=sr, num_povs=2, seed=801)
    clean_mix = scenario.studio_reference.to_stereo()

    # Inject a harsh 3.6 kHz narrowband PA horn whistling resonance
    t = np.arange(len(clean_mix), dtype=np.float64) / float(sr)
    whistle = 0.22 * np.sin(2.0 * np.pi * 3600.0 * t)[:, None]
    harsh_mix = clean_mix + whistle

    smoothed, max_cut_db = suppress_dynamic_resonances(harsh_mix, sr=sr, max_cut_db=4.5)
    assert max_cut_db > 1.5

    # Measure energy in narrow 3.4-3.8 kHz notch before and after
    sos_bp = signal.butter(3, [3400.0, 3800.0], btype="bandpass", fs=sr, output="sos")
    before_rms = float(np.sqrt(np.mean(signal.sosfilt(sos_bp, harsh_mix, axis=0) ** 2)))
    after_rms = float(np.sqrt(np.mean(signal.sosfilt(sos_bp, smoothed, axis=0) ** 2)))
    assert after_rms < before_rms * 0.85


def test_iso_9613_air_absorption_compensation_restores_distant_hf_with_snr_gate() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.5, sr=sr, num_povs=2, seed=802)
    st = scenario.studio_reference.to_stereo()

    # Make first 0.5s near-silent room floor so we can verify the SNR gate does not boost silence
    st[: int(0.5 * sr)] *= 0.002

    restored, mean_boost_db = compensate_atmospheric_air_absorption(
        st, distance_m=38.0, sr=sr, max_boost_db=5.5
    )
    assert mean_boost_db > 0.4

    # High-frequency energy (>8 kHz) on active musical section (1.0s - 2.5s) should increase
    sos_hf = signal.butter(3, 8000.0, btype="highpass", fs=sr, output="sos")
    s_act = int(1.0 * sr)
    hf_before = float(np.sqrt(np.mean(signal.sosfilt(sos_hf, st[s_act:], axis=0) ** 2)))
    hf_after = float(np.sqrt(np.mean(signal.sosfilt(sos_hf, restored[s_act:], axis=0) ** 2)))
    assert hf_after > hf_before * 1.08


def test_automatic_polarity_inversion_detection_prevents_bass_cancellation() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.5, sr=sr, num_povs=2, seed=803)
    pov1 = scenario.pov_tracks[0]
    # Invert polarity (180-deg phase flip) on POV 2
    pov2_inverted = AudioTrack(
        name="pov_02_inverted_cable",
        samples=-1.0 * scenario.pov_tracks[1].to_stereo(),
        sample_rate=sr,
    )

    pipeline = ConcertStudioPipeline(
        PipelineConfig(quality_tier="draft", enable_deduplication=False)
    )
    mastered, _, report = pipeline.run([pov1, pov2_inverted])
    assert report.polarity_inversions_fixed >= 1
    assert float(np.sqrt(np.mean(mastered.samples**2))) > 0.05


def test_blumlein_3band_spatial_soundstage_and_live_glue_floor() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(duration_sec=2.5, sr=sr, num_povs=3, seed=804)
    cons = scenario.studio_reference.to_stereo()
    tracks = [t.to_stereo() for t in scenario.pov_tracks]
    masks = [np.ones(len(cons), dtype=np.float64) for _ in tracks]

    nodes = [
        SpatialNodePosition(
            node_id=1, track_name="left_node", x_meters=-8.0, y_meters=12.0, z_meters=0.0,
            distance_m=14.4, azimuth_deg=-33.0, elevation_zone="mid_floor",
            placement_source="audio_multilateration", spatial_pan_lr=-0.65, position_confidence=0.85,
        ),
        SpatialNodePosition(
            node_id=2, track_name="center_node", x_meters=0.0, y_meters=10.0, z_meters=0.0,
            distance_m=10.0, azimuth_deg=0.0, elevation_zone="floor_pit",
            placement_source="audio_multilateration", spatial_pan_lr=0.0, position_confidence=0.90,
        ),
        SpatialNodePosition(
            node_id=3, track_name="right_node", x_meters=8.5, y_meters=13.0, z_meters=0.0,
            distance_m=15.5, azimuth_deg=33.0, elevation_zone="mid_floor",
            placement_source="audio_multilateration", spatial_pan_lr=0.65, position_confidence=0.85,
        ),
    ]

    staged = apply_spatial_node_soundstage(
        cons, aligned_tracks=tracks, active_masks=masks, node_positions=nodes, spatial_blend=0.25, sr=sr
    )
    assert staged.shape == cons.shape

    glued = apply_live_glue_ambience_floor(staged, sr=sr, floor_dbfs=-42.0)
    assert glued.shape == staged.shape


def test_31_file_partial_clip_payload_and_progress_bar() -> None:
    sr = 24000
    # Generate 28 unique POVs (with many partial clips starting/stopping at random places) + 3 duplicate re-uploads = 31 files
    scenario = generate_synthetic_concert(
        duration_sec=2.5,
        sr=sr,
        num_povs=28,
        seed=931,
        include_partial_clips=True,
        include_physical_anomalies=True,
    )
    payload_31 = list(scenario.pov_tracks)

    # Add 3 degraded duplicate re-uploads of POV 0, POV 1, POV 2 to reach 31 files
    for k in range(3):
        src = scenario.pov_tracks[k]
        clipped_dup = AudioTrack(
            name=f"{src.name}_reupload_dup_{k}",
            samples=np.clip(src.to_stereo() * 2.1, -0.98, 0.98),
            sample_rate=sr,
        )
        payload_31.append(clipped_dup)

    assert len(payload_31) == 31

    progress_log: list[tuple[float, str]] = []

    def _on_progress(pct: float, stage: str, _detail: str) -> None:
        progress_log.append((pct, stage))

    pipeline = ConcertStudioPipeline(
        PipelineConfig(
            target_sample_rate=sr,
            quality_tier="draft",
            true_data_mode=True,
            enable_deduplication=True,
        )
    )
    mastered, stems, report = pipeline.run(
        payload_31,
        reference_studio_track=scenario.studio_reference,
        progress_callback=_on_progress,
    )

    # Verify all 31 files were processed: 3 duplicates removed, 28 unique partial/full POVs fused
    assert report.duplicates_removed == 3
    assert report.num_povs_fused == 28
    assert len(report.clip_placements) == 28
    assert len(report.node_positions) == 28
    # Verify progress bar reached 100% monotonically
    assert len(progress_log) >= 6
    assert progress_log[0][0] == 5.0
    assert progress_log[-1][0] == 100.0
    assert mastered.duration_sec >= 2.2

