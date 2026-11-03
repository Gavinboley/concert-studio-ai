"""Tests for Automated Multi-Camera Concert Video Director."""

from __future__ import annotations

import pytest
import numpy as np

from concert_studio.director import ConcertVideoDirector, VideoDirectorPlan


def test_concert_video_director():
    clips = [
        {"name": "cam_pit_center.mov", "start_sec": 0.0, "end_sec": 120.0, "azimuth_deg": 0.0, "distance_m": 5.0, "stability": 0.9},
        {"name": "cam_balcony_wide.mov", "start_sec": 0.0, "end_sec": 120.0, "azimuth_deg": -5.0, "distance_m": 25.0, "stability": 0.95},
        {"name": "cam_stage_left_fan.mov", "start_sec": 30.0, "end_sec": 90.0, "azimuth_deg": -25.0, "distance_m": 12.0, "stability": 0.8},
    ]

    # 120s concert at 120 BPM
    voc = np.zeros(120)
    voc[15:45] = 1.0  # vocals singing between 15s and 45s

    plan = ConcertVideoDirector.direct_concert(
        clip_roster=clips,
        total_duration_sec=120.0,
        vocal_energy_1hz=voc,
        tempo_bpm=120.0,
    )

    assert plan.num_shots >= 10
    assert plan.total_duration_sec == 120.0
    assert len(plan.camera_roster) == 3
    assert plan.shots[0].start_sec == 0.0
    assert plan.shots[-1].end_sec <= 120.0

    # Verify no zero-duration shots
    for s in plan.shots:
        assert s.duration_sec >= 1.0
        assert s.shot_type in ("close_up_lead", "stage_wide", "side_stage_left", "side_stage_right", "crowd_energy")

    edl = plan.export_edl()
    assert "TITLE: ConcertStudio_MultiCam" in edl
    assert "FCM: NON-DROP FRAME" in edl
    assert "001" in edl
