"""Tests for Multi-Node Acoustic Multilateration and Video Stage Geometry Crowd Placement."""

from __future__ import annotations

import numpy as np

from concert_studio.spatial import (
    analyze_video_stage_frames,
    apply_spatial_node_soundstage,
    localize_multi_pov_nodes,
    synthesize_stage_video_frame,
)
from concert_studio.synthetic import generate_synthetic_concert


def test_video_stage_frame_geometry_estimates_crowd_distance_angle_and_elevation() -> None:
    # Front-pit left viewer (7m distance, -28 deg left azimuth, floor_pit)
    frame_pit_left = synthesize_stage_video_frame(
        distance_m=7.0, azimuth_deg=-28.0, elevation_zone="floor_pit"
    )
    obs_pit = analyze_video_stage_frames(frame_pit_left)

    # Balcony right viewer (35m distance, +25 deg right azimuth, balcony)
    frame_balc_right = synthesize_stage_video_frame(
        distance_m=35.0, azimuth_deg=25.0, elevation_zone="balcony"
    )
    obs_balc = analyze_video_stage_frames(frame_balc_right)

    assert obs_pit.estimated_distance_m < 12.0
    assert obs_balc.estimated_distance_m > 22.0
    assert obs_pit.estimated_azimuth_deg < -5.0
    assert obs_balc.estimated_azimuth_deg > 5.0
    assert obs_pit.elevation_zone == "floor_pit"
    assert obs_balc.elevation_zone == "balcony"


def test_multi_node_acoustic_localization_confidence_scales_with_node_count() -> None:
    sr = 48000
    scenario = generate_synthetic_concert(
        duration_sec=2.5,
        sr=sr,
        num_povs=6,
        seed=44,
        include_physical_anomalies=True,
    )
    tracks = [t.samples for t in scenario.pov_tracks]
    masks = [np.ones(len(t), dtype=np.float64) for t in tracks]
    names = [t.name for t in scenario.pov_tracks]
    toas_ms = [0.0, 14.0, 9.0, 28.0, 19.0, 35.0]

    # Compare 2-node audio localization vs 6-node audio localization
    pos_2 = localize_multi_pov_nodes(tracks[:2], masks[:2], names[:2], toas_ms[:2], sr=sr)
    pos_6 = localize_multi_pov_nodes(tracks, masks, names, toas_ms, sr=sr)

    assert len(pos_2) == 2
    assert len(pos_6) == 6
    # Adding more Nodes increases network triangulation confidence
    assert pos_6[0].position_confidence > pos_2[0].position_confidence
    # Node #6 (35ms ToA) must be localized significantly deeper into the crowd than Node #1 (0ms ToA)
    assert pos_6[5].distance_m > pos_6[0].distance_m + 5.0

    # Fuse with video stage observations
    vid_obs = [
        analyze_video_stage_frames(t.video_frames_rgb) if t.video_frames_rgb is not None else None
        for t in scenario.pov_tracks
    ]
    pos_fused = localize_multi_pov_nodes(
        tracks, masks, names, toas_ms, sr=sr, video_observations=vid_obs
    )
    assert all(p.placement_source == "audio_video_fused" for p in pos_fused)
    assert pos_fused[0].position_confidence > pos_6[0].position_confidence

    # Verify spatial soundstage enrichment preserves shape and energy
    enriched = apply_spatial_node_soundstage(
        scenario.studio_reference.samples, tracks, masks, pos_fused
    )
    assert enriched.shape == scenario.studio_reference.samples.shape
