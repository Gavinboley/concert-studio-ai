"""Video container (MP4/MOV camm/gpmd) & JSON IMU accelerometer/gyroscope telemetry extractor."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import numpy as np

from concert_studio.models import AnomalyEvent, IMUTelemetry


def parse_imu_json_sidecar(path_or_data: str | Path | dict) -> IMUTelemetry:
    """Parse IMU accelerometer and gyroscope telemetry from a JSON file or dict."""
    if isinstance(path_or_data, (str, Path)):
        raw = json.loads(Path(path_or_data).read_text(encoding="utf-8"))
    else:
        raw = path_or_data

    t = np.asarray(raw.get("timestamps_sec", []), dtype=np.float64)
    if "accel_magnitude" in raw and "gyro_magnitude" in raw:
        accel_mag = np.asarray(raw["accel_magnitude"], dtype=np.float64)
        gyro_mag = np.asarray(raw["gyro_magnitude"], dtype=np.float64)
    else:
        # Compute dynamic acceleration |a - g| and angular velocity |w| from 3-axis vectors
        accel_xyz = np.asarray(raw.get("accel_xyz", []), dtype=np.float64)
        gyro_xyz = np.asarray(raw.get("gyro_xyz", []), dtype=np.float64)
        if accel_xyz.ndim == 2 and accel_xyz.shape[1] == 3:
            total_g = np.linalg.norm(accel_xyz, axis=1)
            gravity_est = float(np.median(total_g)) if len(total_g) > 0 else 9.81
            accel_mag = np.abs(total_g - gravity_est)
        else:
            accel_mag = np.zeros_like(t)
        if gyro_xyz.ndim == 2 and gyro_xyz.shape[1] == 3:
            gyro_mag = np.linalg.norm(gyro_xyz, axis=1)
        else:
            gyro_mag = np.zeros_like(t)

    return IMUTelemetry(
        timestamps_sec=t,
        accel_magnitude=accel_mag,
        gyro_magnitude=gyro_mag,
    )


def build_mp4_with_camm_telemetry(
    timestamps_sec: np.ndarray,
    accel_xyz: np.ndarray,
    gyro_xyz: np.ndarray,
) -> bytes:
    """Build a valid ISO Base Media (MP4/MOV) binary buffer containing a `camm` IMU metadata box.

    Follows the Google/Apple Camera Motion Metadata (`camm`) packet layout:
    Each `camm` sample record consists of:
      [timestamp_sec: float64, packet_type: uint16, ax: float32, ay: float32, az: float32, gx: float32, gy: float32, gz: float32]
    """
    t = np.asarray(timestamps_sec, dtype=np.float64).ravel()
    acc = np.asarray(accel_xyz, dtype=np.float32).reshape(-1, 3)
    gyr = np.asarray(gyro_xyz, dtype=np.float32).reshape(-1, 3)
    n = len(t)

    payload = bytearray()
    payload.extend(struct.pack(">I", n))
    for i in range(n):
        payload.extend(
            struct.pack(
                "<dHffffff",
                float(t[i]),
                2,  # packet_type 2 = combined 6-axis IMU sample
                float(acc[i, 0]),
                float(acc[i, 1]),
                float(acc[i, 2]),
                float(gyr[i, 0]),
                float(gyr[i, 1]),
                float(gyr[i, 2]),
            )
        )

    camm_box = struct.pack(">I4s", 8 + len(payload), b"camm") + bytes(payload)
    udta_box = struct.pack(">I4s", 8 + len(camm_box), b"udta") + camm_box
    moov_box = struct.pack(">I4s", 8 + len(udta_box), b"moov") + udta_box
    ftyp_box = struct.pack(">I4s4sI4s", 20, b"ftyp", b"isom", 512, b"isom")
    return ftyp_box + moov_box


def extract_imu_from_mp4_bytes(data: bytes) -> IMUTelemetry | None:
    """Scan an MP4/MOV binary stream for `camm` or `gpmd` camera motion IMU atoms."""
    idx = data.find(b"camm")
    if idx < 4:
        return None
    box_size = struct.unpack(">I", data[idx - 4 : idx])[0]
    payload = data[idx + 4 : idx - 4 + box_size]
    if len(payload) < 4:
        return None
    count = struct.unpack(">I", payload[:4])[0]
    rec_size = struct.calcsize("<dHffffff")
    if len(payload) < 4 + count * rec_size or count == 0:
        return None

    t_list = np.zeros(count, dtype=np.float64)
    acc_xyz = np.zeros((count, 3), dtype=np.float64)
    gyr_xyz = np.zeros((count, 3), dtype=np.float64)
    offset = 4
    for i in range(count):
        ts, _ptype, ax, ay, az, gx, gy, gz = struct.unpack(
            "<dHffffff", payload[offset : offset + rec_size]
        )
        t_list[i] = ts
        acc_xyz[i] = (ax, ay, az)
        gyr_xyz[i] = (gx, gy, gz)
        offset += rec_size

    total_g = np.linalg.norm(acc_xyz, axis=1)
    gravity_est = float(np.median(total_g)) if count > 0 else 9.81
    accel_mag = np.abs(total_g - gravity_est)
    gyro_mag = np.linalg.norm(gyr_xyz, axis=1)
    return IMUTelemetry(
        timestamps_sec=t_list,
        accel_magnitude=accel_mag,
        gyro_magnitude=gyro_mag,
    )


def compute_imu_stability_mask(
    imu: IMUTelemetry | None,
    frame_times_sec: np.ndarray,
    accel_bump_threshold: float = 3.5,
    gyro_shake_threshold: float = 2.2,
) -> tuple[np.ndarray, list[AnomalyEvent]]:
    """Interpolate IMU telemetry onto STFT frames and return (stability_weights, anomaly_events).

    When a phone is violently shaken or bumped (high |a - g| or high angular velocity),
    `stability_weights` drops toward 0.05 so steady POVs carry those frames.
    """
    n_frames = len(frame_times_sec)
    if imu is None or len(imu.timestamps_sec) == 0:
        return np.ones(n_frames, dtype=np.float64), []

    accel_f = np.interp(frame_times_sec, imu.timestamps_sec, imu.accel_magnitude, left=0.0, right=0.0)
    gyro_f = np.interp(frame_times_sec, imu.timestamps_sec, imu.gyro_magnitude, left=0.0, right=0.0)

    accel_excess = np.maximum(accel_f - accel_bump_threshold, 0.0) / max(accel_bump_threshold, 1e-6)
    gyro_excess = np.maximum(gyro_f - gyro_shake_threshold, 0.0) / max(gyro_shake_threshold, 1e-6)
    combined_stress = accel_excess + 0.8 * gyro_excess

    weights = np.clip(np.exp(-2.0 * combined_stress), 0.05, 1.0)

    events: list[AnomalyEvent] = []
    impact_mask = combined_stress > 0.25
    if np.any(impact_mask):
        padded = np.concatenate([[False], impact_mask, [False]])
        starts = np.flatnonzero(padded[1:] & ~padded[:-1])
        ends = np.flatnonzero(~padded[1:] & padded[:-1])
        dt = float(frame_times_sec[1] - frame_times_sec[0]) if n_frames > 1 else 0.02
        for s, e in zip(starts, ends):
            t0 = float(frame_times_sec[s])
            t1 = float(frame_times_sec[min(e - 1, n_frames - 1)] + dt)
            sev = float(np.clip(np.max(combined_stress[s:e]) / 2.5, 0.1, 1.0))
            peak_a = float(np.max(accel_f[s:e]))
            peak_g = float(np.max(gyro_f[s:e]))
            events.append(
                AnomalyEvent(
                    kind="mic_bump",
                    start_sec=round(t0, 3),
                    end_sec=round(t1, 3),
                    severity=round(sev, 2),
                    detail=f"IMU motion spike (accel={peak_a:.1f} m/s^2, gyro={peak_g:.1f} rad/s)",
                )
            )

    return weights, events
