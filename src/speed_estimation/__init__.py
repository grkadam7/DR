"""
Phase 3: AI/ML Speed & Kinematic Estimation package.
Provides IMU-to-speed dataset builder, model architecture, training loop,
ONNX export, and inference engine.
"""

from .dataset import SpeedDataset, build_windows
from .model import SpeedNet
from .trainer import SpeedModelTrainer
from .inference import SpeedInferenceEngine

__all__ = [
    "SpeedDataset",
    "build_windows",
    "SpeedNet",
    "SpeedModelTrainer",
    "SpeedInferenceEngine",
]
