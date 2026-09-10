"""
Preprocessing and in-vehicle calibration package for Intelligent Dead Reckoning.
Provides:
- VehicleFrameAligner: Auto attitude leveling (gravity) and forward heading alignment.
- VibrationFilter: Butterworth filtering and pothole shock clamping.
- ZeroVelocityDetector: ZUPT / ZARU detection for vehicle stops.
"""

from .alignment import VehicleFrameAligner, AlignmentResult
from .filters import VibrationFilter
from .zupt import ZeroVelocityDetector, ZuptState

__all__ = [
    "VehicleFrameAligner",
    "AlignmentResult",
    "VibrationFilter",
    "ZeroVelocityDetector",
    "ZuptState",
]
