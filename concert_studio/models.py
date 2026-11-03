"""Core data structures and contracts for ConcertStudio AI v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class IMUTelemetry:
    """Smartphone accelerometer and gyroscope motion telemetry aligned to clip time."""

    timestamps_sec: np.ndarray   # 1D float64 (M,) in clip-local seconds
    accel_magnitude: np.ndarray  # 1D float64 (M,) in m/s^2 (deviation from 1g gravity)
    gyro_magnitude: np.ndarray   # 1D float64 (M,) in rad/s angular velocity

    def __post_init__(self) -> None:
        t = np.asarray(self.timestamps_sec, dtype=np.float64).ravel()
        a = np.asarray(self.accel_magnitude, dtype=np.float64).ravel()
        g = np.asarray(self.gyro_magnitude, dtype=np.float64).ravel()
        if not (len(t) == len(a) == len(g)):
            raise ValueError("IMUTelemetry arrays must have equal length")
        object.__setattr__(self, "timestamps_sec", t)
        object.__setattr__(self, "accel_magnitude", a)
        object.__setattr__(self, "gyro_magnitude", g)


@dataclass(frozen=True)
class AnomalyEvent:
    """Detected acoustic or mechanical anomaly interval on a specific POV clip."""

    kind: str       # "mic_bump" | "mic_occluded" | "clipping" | "delay_echo" | "moving_phone" | "crowd_scream" | "agc_pumping"
    start_sec: float
    end_sec: float
    severity: float  # [0.0, 1.0]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioTrack:
    """Represents an in-memory mono or stereo audio signal normalized to [-1.0, 1.0]."""

    name: str
    samples: np.ndarray  # float64 array of shape (T,) or (T, 2)
    sample_rate: int = 48000
    imu_telemetry: IMUTelemetry | None = None
    video_frames_rgb: np.ndarray | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples)
        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float64)
        if arr.ndim not in (1, 2):
            raise ValueError(f"AudioTrack samples must be 1D or 2D, got shape {arr.shape}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        object.__setattr__(self, "samples", arr)

    @property
    def duration_sec(self) -> float:
        return float(self.samples.shape[0]) / float(self.sample_rate)

    @property
    def num_channels(self) -> int:
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    @property
    def is_true_stereo(self) -> bool:
        """Return True if track has 2 distinct channels with non-trivial stereo difference energy."""
        if self.samples.ndim != 2 or self.samples.shape[1] < 2:
            return False
        diff = self.samples[:, 0] - self.samples[:, 1]
        sum_ch = self.samples[:, 0] + self.samples[:, 1]
        diff_power = float(np.mean(diff**2))
        sum_power = float(np.mean(sum_ch**2)) + 1e-12
        return (diff_power / sum_power) > 0.001

    @property
    def channel_format(self) -> str:
        """Return 'mono' | 'dual_mono' | 'stereo'."""
        if self.samples.ndim == 1 or self.samples.shape[1] == 1:
            return "mono"
        return "stereo" if self.is_true_stereo else "dual_mono"

    def to_mono(self) -> np.ndarray:
        """Return a 1D float64 mono downmix."""
        if self.samples.ndim == 1:
            return self.samples.copy()
        return np.mean(self.samples, axis=1)

    def to_stereo(self) -> np.ndarray:
        """Return a 2D (T, 2) float64 stereo signal."""
        if self.samples.ndim == 2 and self.samples.shape[1] == 2:
            return self.samples.copy()
        mono = self.to_mono()
        return np.column_stack([mono, mono])


@dataclass(frozen=True)
class AlignmentParameters:
    """Estimated temporal offset and clock-drift parameters for a single POV recording."""

    track_name: str
    initial_offset_sec: float
    clock_drift_ppm: float
    peak_coherence: float
    snr_db: float
    clipping_ratio: float = 0.0
    max_motion_drift_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClipPlacement:
    """Placement, acoustic profile, and detected anomalies for a full or partial POV clip."""

    track_name: str
    timeline_start_sec: float
    timeline_end_sec: float
    clip_duration_sec: float
    initial_toa_ms: float
    clock_drift_ppm: float
    max_motion_drift_ms: float
    echo_delay_ms: float
    echo_attenuation_db: float
    speaker_zone: str
    coherence_score: float
    snr_db: float
    channel_format: str = "stereo"
    anomalies: list[AnomalyEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


@dataclass(frozen=True)
class ProcessingEstimate:
    """Estimated processing wall-clock time, resolution gain, and active passes for a quality tier."""

    quality_tier: str
    num_files: int
    total_input_duration_sec: float
    estimated_master_duration_sec: float
    estimated_processing_sec: float
    expected_snr_gain_db: float
    expected_resolution_score: float
    stft_resolutions: list[int] = field(default_factory=list)
    active_passes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StemBundle:
    """Separated musical stems and isolated crowd residual at studio sample rate."""

    vocals: np.ndarray        # (T, 2) float64
    drums: np.ndarray         # (T, 2) float64
    bass: np.ndarray          # (T, 2) float64
    other: np.ndarray         # (T, 2) float64
    crowd_removed: np.ndarray # (T, 2) float64
    sample_rate: int = 48000

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "vocals": self.vocals,
            "drums": self.drums,
            "bass": self.bass,
            "other": self.other,
            "crowd_removed": self.crowd_removed,
        }


@dataclass(frozen=True)
class MasteringReport:
    """Objective quality telemetry across the ConcertStudio AI v2 reconstruction pipeline."""

    reference_pov: str
    num_povs_fused: int
    alignments: list[AlignmentParameters] = field(default_factory=list)
    input_snr_db: float = 0.0
    consensus_si_sdr_db: float = 0.0
    output_si_sdr_db: float = 0.0
    crowd_suppression_db: float = 0.0
    clipped_samples_repaired: int = 0
    dereverb_tail_reduction_db: float = 0.0
    hf_bandwidth_extension_db: float = 0.0
    integrated_lufs: float = -14.0
    true_peak_dbtp: float = -1.0
    crest_factor_db: float = 12.0
    stem_lufs: dict[str, float] = field(default_factory=dict)
    quality_tier: str = "ultra"
    true_data_mode: bool = True
    master_duration_sec: float = 0.0
    clip_placements: list[ClipPlacement] = field(default_factory=list)
    timeline_coverage_curve: list[int] = field(default_factory=list)
    total_anomalies_mitigated: int = 0
    bumps_masked: int = 0
    occlusions_masked: int = 0
    echoes_cancelled: int = 0
    moving_phones_stabilized: int = 0
    node_positions: list[dict[str, Any]] = field(default_factory=list)
    spatial_localization_confidence: float = 0.0
    duplicates_removed: int = 0
    duplicate_groups: list[dict[str, Any]] = field(default_factory=list)
    resonance_attenuation_db: float = 0.0
    air_absorption_restored_db: float = 0.0
    polarity_inversions_fixed: int = 0
    midi_notes_transcribed: int = 0
    quality_mode: str = "studio_reconstructed"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["clip_placements"] = [c.to_dict() for c in self.clip_placements]
        return d
