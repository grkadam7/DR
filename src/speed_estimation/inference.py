"""
ONNX Export & Real-Time Inference Engine for SpeedNet.

Exports trained PyTorch model to ONNX, then wraps an ONNXRuntime
inference session for low-latency streaming speed prediction.

Addresses Requirements: SPEED-02 (< 20ms), SPEED-01 (deployment)
"""

import os
import time
import json
from typing import Optional, Dict, Any
import numpy as np


class OnnxExporter:
    """Exports a trained SpeedNet checkpoint to ONNX format."""

    def __init__(self, checkpoint_dir: str = "models"):
        self.checkpoint_dir = checkpoint_dir

    def export(
        self,
        output_path: str = "models/speednet.onnx",
        window_size: int = 50,
        opset_version: int = 11,
    ) -> str:
        """
        Load best checkpoint and export to ONNX.

        Args:
            output_path: Path where the .onnx file is saved.
            window_size: Sequence length the model was trained with.
            opset_version: ONNX opset version.
        Returns:
            Path to the exported .onnx file.
        """
        import torch
        from .model import SpeedNet

        ckpt_path = os.path.join(self.checkpoint_dir, "speednet_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train the model first.")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = SpeedNet(window_size=window_size)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Dummy input: (batch=1, channels=6, time=window_size)
        dummy = torch.randn(1, 6, window_size)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["imu_window"],
            output_names=["speed_ms"],
            dynamic_axes={
                "imu_window": {0: "batch_size"},
                "speed_ms": {0: "batch_size"},
            },
        )
        print(f"  ✓ ONNX model exported to: {output_path}")
        return output_path


class SpeedInferenceEngine:
    """
    Real-time speed inference engine using ONNX Runtime.
    Wraps a sliding buffer of IMU samples, applies normalization,
    and returns speed predictions in m/s with latency measurement.
    """

    def __init__(
        self,
        onnx_path: str = "models/speednet.onnx",
        norm_stats_path: Optional[str] = "models/norm_stats.json",
        window_size: int = 50,
    ):
        """
        Args:
            onnx_path: Path to the exported .onnx model file.
            norm_stats_path: JSON file with per-channel mean/sigma for normalization.
            window_size: Number of IMU samples per inference window.
        """
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"]
            )
        except ImportError:
            raise ImportError(
                "onnxruntime is required for inference. Install it with:\n"
                "  pip install onnxruntime"
            )

        self.window_size = window_size
        self._buffer: list = []   # rolling buffer of [ax, ay, az, gy, gp, gr] samples

        # Load normalization stats
        self.mu = np.zeros((1, 6, 1), dtype=np.float32)
        self.sigma = np.ones((1, 6, 1), dtype=np.float32)
        if norm_stats_path and os.path.exists(norm_stats_path):
            with open(norm_stats_path) as f:
                stats = json.load(f)
            self.mu = np.array(stats["mu"], dtype=np.float32).reshape(1, 6, 1)
            self.sigma = np.array(stats["sigma"], dtype=np.float32).reshape(1, 6, 1)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def push_sample(
        self, ax: float, ay: float, az: float,
        gy_yaw: float, gy_pitch: float, gy_roll: float
    ) -> Optional[Dict[str, Any]]:
        """
        Push a single IMU sample into the rolling buffer.
        Returns speed prediction (dict) once the buffer is full, otherwise None.

        Returns:
            {"speed_ms": float, "speed_kmh": float, "latency_ms": float}
        """
        self._buffer.append([ax, ay, az, gy_yaw, gy_pitch, gy_roll])
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        if len(self._buffer) < self.window_size:
            return None

        return self._infer()

    def infer_window(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Direct inference on a pre-built (window_size, 6) numpy array.

        Returns:
            {"speed_ms": float, "speed_kmh": float, "latency_ms": float}
        """
        if window.shape != (self.window_size, 6):
            raise ValueError(f"Expected window shape ({self.window_size}, 6), got {window.shape}")
        self._buffer = window.tolist()
        return self._infer()

    def _infer(self) -> Dict[str, Any]:
        """Run ONNX inference on current buffer."""
        window = np.array(self._buffer, dtype=np.float32)  # (W, 6)
        # Channels-first and batch dim: (1, 6, W)
        x = window.T[np.newaxis, :, :]     # (1, 6, W)
        x = (x - self.mu) / self.sigma     # normalize

        t0 = time.perf_counter()
        result = self.session.run([self.output_name], {self.input_name: x})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        speed_ms = float(np.clip(result[0][0, 0], 0.0, None))
        return {
            "speed_ms": speed_ms,
            "speed_kmh": speed_ms * 3.6,
            "latency_ms": latency_ms,
        }

    def benchmark_latency(self, n_iters: int = 200) -> Dict[str, float]:
        """
        Measure inference latency over n_iters random windows.

        Returns:
            {"mean_ms": float, "p95_ms": float, "p99_ms": float, "max_ms": float}
        """
        latencies = []
        dummy = np.random.randn(self.window_size, 6).astype(np.float32)
        for _ in range(n_iters):
            res = self.infer_window(dummy)
            latencies.append(res["latency_ms"])

        arr = np.array(latencies)
        return {
            "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }
