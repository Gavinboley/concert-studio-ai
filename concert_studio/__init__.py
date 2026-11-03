"""ConcertStudio AI v2: Multi-POV True-Data Live Concert Reconstruction Engine."""

from concert_studio.estimator import QUALITY_TIERS, estimate_processing_plan
from concert_studio.models import (
    AlignmentParameters,
    AnomalyEvent,
    AudioTrack,
    ClipPlacement,
    IMUTelemetry,
    MasteringReport,
    ProcessingEstimate,
    StemBundle,
)
from concert_studio.pipeline import ConcertStudioPipeline, PipelineConfig

__all__ = [
    "AlignmentParameters",
    "AnomalyEvent",
    "AudioTrack",
    "ClipPlacement",
    "ConcertStudioPipeline",
    "IMUTelemetry",
    "MasteringReport",
    "PipelineConfig",
    "ProcessingEstimate",
    "QUALITY_TIERS",
    "StemBundle",
    "estimate_processing_plan",
]
__version__ = "0.2.0"
