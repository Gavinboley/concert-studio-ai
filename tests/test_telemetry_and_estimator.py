"""Tests for MP4/MOV camm IMU telemetry extraction and Quality-Tier processing estimator."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pytest

from concert_studio.estimator import QUALITY_TIERS, estimate_processing_plan
from concert_studio.telemetry import (
    build_mp4_with_camm_telemetry,
    compute_imu_stability_mask,
    extract_imu_from_mp4_bytes,
    parse_imu_json_sidecar,
)


def test_mp4_camm_binary_imu_telemetry_roundtrip() -> None:
    t = np.linspace(0.0, 2.0, 50)
    accel_xyz = np.column_stack([np.zeros(50), np.zeros(50), np.full(50, 9.81)])
    gyro_xyz = np.zeros((50, 3))
    # Inject a sharp phone bump/shake at t = 1.0s (indices 23..27)
    accel_xyz[23:27, 0] = 14.5
    gyro_xyz[23:27, 1] = 4.8

    mp4_bytes = build_mp4_with_camm_telemetry(t, accel_xyz, gyro_xyz)
    imu = extract_imu_from_mp4_bytes(mp4_bytes)
    assert imu is not None
    assert len(imu.timestamps_sec) == 50
    assert float(np.max(imu.accel_magnitude[23:27])) > 6.0
    assert float(np.max(imu.gyro_magnitude[23:27])) > 4.0

    frame_t = np.linspace(0.0, 2.0, 100)
    weights, events = compute_imu_stability_mask(imu, frame_t)
    assert len(weights) == 100
    assert float(np.min(weights[45:55])) < 0.25
    assert float(np.mean(weights[:20])) > 0.98
    assert any(ev.kind == "mic_bump" for ev in events)


def test_parse_imu_json_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "clip1.imu.json"
    sidecar.write_text(
        json.dumps(
            {
                "timestamps_sec": [0.0, 0.5, 1.0, 1.5],
                "accel_xyz": [[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [8.0, 0.0, 9.81], [0.0, 0.0, 9.81]],
                "gyro_xyz": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 0.0]],
            }
        ),
        encoding="utf-8",
    )
    imu = parse_imu_json_sidecar(sidecar)
    assert len(imu.timestamps_sec) == 4
    assert imu.accel_magnitude[2] > 2.0
    assert pytest.approx(imu.gyro_magnitude[2], rel=1e-3) == 3.5


def test_estimate_processing_plan_scales_with_tiers_and_file_count() -> None:
    tier_order = ["draft", "standard", "high", "ultra", "archival_max"]
    estimates = [
        estimate_processing_plan(num_files=8, avg_duration_sec=120.0, quality_tier=t)
        for t in tier_order
    ]

    for i in range(len(estimates) - 1):
        assert estimates[i + 1].estimated_processing_sec > estimates[i].estimated_processing_sec
        assert estimates[i + 1].expected_resolution_score > estimates[i].expected_resolution_score

    # Verify scaling with file count (2 files vs 10 files vs 25 files)
    est_2 = estimate_processing_plan(num_files=2, avg_duration_sec=120.0, quality_tier="high")
    est_10 = estimate_processing_plan(num_files=10, avg_duration_sec=120.0, quality_tier="high")
    est_25 = estimate_processing_plan(num_files=25, avg_duration_sec=120.0, quality_tier="high")

    assert est_2.estimated_processing_sec < est_10.estimated_processing_sec < est_25.estimated_processing_sec
    assert est_2.expected_snr_gain_db < est_10.expected_snr_gain_db < est_25.expected_snr_gain_db
    assert est_2.expected_resolution_score < est_10.expected_resolution_score < est_25.expected_resolution_score
