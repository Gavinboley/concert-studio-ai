"""Automated Multi-Camera Concert Video Director for ConcertStudio AI v2.

Analyzes time-aligned video POVs, spatial stage nodes, and musical beats/vocal cues
to synthesize an intelligent, professional multi-camera Edit Decision List (EDL / FCPXML)
dynamically switching angles between Pit, Center Stage, Close-ups, and Balcony wides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np


@dataclass(frozen=True)
class CameraShot:
    """Individual cut in the multi-camera concert video edit."""

    shot_index: int
    camera_name: str
    start_sec: float
    end_sec: float
    duration_sec: float
    shot_type: str            # "close_up_lead" | "stage_wide" | "crowd_energy" | "side_stage_left" | "side_stage_right"
    switch_reason: str        # "vocal_lead_entrance" | "rhythmic_downbeat_cut" | "camera_shake_recovery" | "wide_dynamic_drop"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoDirectorPlan:
    """Complete multi-camera directorial cut list and export interfaces."""

    total_duration_sec: float
    num_shots: int
    camera_roster: list[str]
    shots: list[CameraShot]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_duration_sec": round(self.total_duration_sec, 2),
            "num_shots": self.num_shots,
            "camera_roster": self.camera_roster,
            "shots": [s.to_dict() for s in self.shots],
        }

    def export_edl(self, title: str = "ConcertStudio_MultiCam") -> str:
        """Export standard CMX 3600 Edit Decision List (EDL) for Premiere / DaVinci Resolve."""
        lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
        for s in self.shots:
            # Convert seconds to SMPTE TC (HH:MM:SS:FF at 30 fps)
            tc_in = self._sec_to_smpte(s.start_sec)
            tc_out = self._sec_to_smpte(s.end_sec)
            reel = s.camera_name.split(".")[0][:8].upper()
            lines.append(f"{s.shot_index:03d}  {reel:<8} V     C        {tc_in} {tc_out} {tc_in} {tc_out}")
            lines.append(f"* FROM CLIP NAME: {s.camera_name}")
            lines.append(f"* COMMENT: ShotType={s.shot_type} Reason={s.switch_reason}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def _sec_to_smpte(cls, sec: float, fps: int = 30) -> str:
        total_frames = int(round(sec * fps))
        ff = total_frames % fps
        ss = (total_frames // fps) % 60
        mm = (total_frames // (fps * 60)) % 60
        hh = total_frames // (fps * 3600)
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


class ConcertVideoDirector:
    """Intelligent automated director for concert multi-camera switching."""

    @classmethod
    def direct_concert(
        cls,
        clip_roster: list[dict[str, Any]],        # [{"name": str, "start_sec": float, "end_sec": float, "azimuth_deg": float, "distance_m": float, "stability": float}]
        total_duration_sec: float,
        vocal_energy_1hz: np.ndarray | None = None,
        tempo_bpm: float = 120.0,
        min_shot_duration_sec: float = 2.5,
        max_shot_duration_sec: float = 7.5,
    ) -> VideoDirectorPlan:
        """Synthesize a rhythmic, musical multi-cam cut list across the concert timeline."""
        if not clip_roster or total_duration_sec <= 0.0:
            return VideoDirectorPlan(total_duration_sec=0.0, num_shots=0, camera_roster=[], shots=[])

        # Beat grid for musical cutting (cut on bar boundaries: every 4 beats = 240 / BPM sec)
        bar_duration = (60.0 / tempo_bpm) * 4.0
        n_bars = max(int(np.ceil(total_duration_sec / bar_duration)), 1)
        grid_cut_points = [float(i * bar_duration) for i in range(n_bars + 1) if float(i * bar_duration) < total_duration_sec]
        if not grid_cut_points or grid_cut_points[-1] < total_duration_sec - 1.0:
            grid_cut_points.append(total_duration_sec)

        # Classify camera shot types by distance & azimuth
        cam_types: dict[str, str] = {}
        cam_weights: dict[str, float] = {}
        for c in clip_roster:
            name = c["name"]
            dist = float(c.get("distance_m", 12.0))
            az = float(c.get("azimuth_deg", 0.0))
            stab = float(c.get("stability", 0.85))

            if dist < 8.0:
                cam_types[name] = "close_up_lead"
            elif dist > 20.0:
                cam_types[name] = "stage_wide"
            elif az < -15.0:
                cam_types[name] = "side_stage_left"
            elif az > 15.0:
                cam_types[name] = "side_stage_right"
            else:
                cam_types[name] = "crowd_energy"

            cam_weights[name] = stab

        shots: list[CameraShot] = []
        curr_time = 0.0
        shot_idx = 1
        last_cam: str | None = None

        while curr_time < total_duration_sec - 0.5:
            # Find cameras active at curr_time
            active = [c for c in clip_roster if c["start_sec"] <= curr_time <= c["end_sec"]]
            if not active:
                # Find next earliest camera start
                future = [c for c in clip_roster if c["start_sec"] > curr_time]
                if not future:
                    break
                curr_time = min(c["start_sec"] for c in future)
                active = [c for c in clip_roster if c["start_sec"] <= curr_time <= c["end_sec"]]
                if not active:
                    break

            # Check if vocals are singing right now
            sec_int = int(min(curr_time, len(vocal_energy_1hz) - 1)) if vocal_energy_1hz is not None and len(vocal_energy_1hz) > 0 else 0
            is_vocal_active = bool(vocal_energy_1hz[sec_int] > 0.35) if vocal_energy_1hz is not None and len(vocal_energy_1hz) > 0 else False

            # Select best camera
            candidate_cams = [c for c in active if c["name"] != last_cam] or active

            if is_vocal_active:
                # Prefer close-up lead or center
                preferred = [c for c in candidate_cams if cam_types.get(c["name"]) in ("close_up_lead", "crowd_energy")]
                chosen = (preferred or candidate_cams)[0]
                reason = "vocal_lead_entrance"
            else:
                # Instrumental / solos / wide
                preferred = [c for c in candidate_cams if cam_types.get(c["name"]) in ("stage_wide", "side_stage_left", "side_stage_right")]
                chosen = (preferred or candidate_cams)[0]
                reason = "rhythmic_downbeat_cut"

            # Determine shot length: align to next musical bar boundary within [min_shot_duration, max_shot_duration]
            max_avail_sec = chosen["end_sec"] - curr_time
            target_end = curr_time + min(max_shot_duration_sec, max_avail_sec)

            # Snap to closest grid cut point
            valid_cuts = [pt for pt in grid_cut_points if curr_time + min_shot_duration_sec <= pt <= curr_time + max_avail_sec]
            if valid_cuts:
                shot_end = min(valid_cuts[0], total_duration_sec)
            else:
                shot_end = min(curr_time + min(max_shot_duration_sec, max_avail_sec), total_duration_sec)

            if shot_end <= curr_time + 0.5:
                shot_end = min(curr_time + 2.0, total_duration_sec)

            dur = round(shot_end - curr_time, 2)
            shots.append(
                CameraShot(
                    shot_index=shot_idx,
                    camera_name=chosen["name"],
                    start_sec=round(curr_time, 2),
                    end_sec=round(shot_end, 2),
                    duration_sec=dur,
                    shot_type=cam_types.get(chosen["name"], "stage_wide"),
                    switch_reason=reason,
                )
            )

            last_cam = chosen["name"]
            curr_time = shot_end
            shot_idx += 1

        all_names = sorted(list({c["name"] for c in clip_roster}))
        return VideoDirectorPlan(
            total_duration_sec=round(total_duration_sec, 2),
            num_shots=len(shots),
            camera_roster=all_names,
            shots=shots,
        )
