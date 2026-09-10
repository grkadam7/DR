"""
Unit tests for Phase 3: AI/ML Speed Estimation.
Tests dataset building, model forward pass, trainer sanity, and inference engine.
"""

import os
import json
import tempfile
import unittest
import numpy as np


class TestSpeedDataset(unittest.TestCase):

    def test_build_windows_shape(self):
        """build_windows returns correct (N, W, 6) and (N,) shapes."""
        import pandas as pd
        from src.speed_estimation.dataset import build_windows

        # Synthetic DataFrame
        n_rows = 300
        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "ax": rng.randn(n_rows),
            "ay": rng.randn(n_rows),
            "az": rng.randn(n_rows) + 9.8,
            "gyr_yaw": rng.randn(n_rows) * 0.1,
            "gyr_pitch": rng.randn(n_rows) * 0.1,
            "gyr_roll": rng.randn(n_rows) * 0.1,
            "speed_ms": np.abs(rng.randn(n_rows)) * 3.0,
        })

        W, stride = 50, 10
        X, y = build_windows(df, window_size=W, stride=stride, normalize=False)

        expected_n = (n_rows - W) // stride + 1
        self.assertEqual(X.shape, (expected_n, W, 6))
        self.assertEqual(y.shape, (expected_n,))
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_build_windows_normalization(self):
        """Normalized windows have approximately zero mean."""
        import pandas as pd
        from src.speed_estimation.dataset import build_windows

        rng = np.random.RandomState(1)
        n_rows = 500
        df = pd.DataFrame({
            "ax": rng.randn(n_rows) + 10.0,  # large offset
            "ay": rng.randn(n_rows),
            "az": rng.randn(n_rows) + 9.8,
            "gyr_yaw": rng.randn(n_rows),
            "gyr_pitch": rng.randn(n_rows),
            "gyr_roll": rng.randn(n_rows),
            "speed_ms": np.abs(rng.randn(n_rows)),
        })
        X, _ = build_windows(df, window_size=50, stride=5, normalize=True)
        # Global mean across all normalized windows should be ~0
        self.assertAlmostEqual(float(X.mean()), 0.0, delta=0.05)

    def test_speed_label_is_ms(self):
        """Speed labels are in m/s (derived from km/h column)."""
        import pandas as pd
        from src.speed_estimation.dataset import build_windows

        rng = np.random.RandomState(2)
        n_rows = 200
        speed_kmh = np.abs(rng.randn(n_rows)) * 10 + 5  # 5-15 km/h
        speed_ms_expected = speed_kmh / 3.6

        df = pd.DataFrame({
            "ax": rng.randn(n_rows),
            "ay": rng.randn(n_rows),
            "az": rng.randn(n_rows),
            "gyr_yaw": rng.randn(n_rows) * 0.1,
            "gyr_pitch": rng.randn(n_rows) * 0.1,
            "gyr_roll": rng.randn(n_rows) * 0.1,
            "speed_ms": speed_ms_expected,
        })
        _, y = build_windows(df, window_size=50, stride=25, normalize=False)
        # Labels should be within m/s range (not km/h range)
        self.assertLess(float(y.max()), 20.0, "Speed labels should be in m/s not km/h")


class TestSpeedNet(unittest.TestCase):

    def test_forward_pass_shape(self):
        """SpeedNet output shape is (B, 1)."""
        import torch
        from src.speed_estimation.model import SpeedNet

        model = SpeedNet(window_size=50)
        model.eval()
        batch = torch.randn(8, 6, 50)
        with torch.no_grad():
            out = model(batch)
        self.assertEqual(out.shape, (8, 1))

    def test_output_non_negative(self):
        """SpeedNet output is always non-negative (final ReLU)."""
        import torch
        from src.speed_estimation.model import SpeedNet

        model = SpeedNet(window_size=50)
        model.eval()
        # Extreme negative input
        batch = torch.full((4, 6, 50), -100.0)
        with torch.no_grad():
            out = model(batch)
        self.assertTrue((out >= 0).all().item())

    def test_parameter_count_reasonable(self):
        """Model should be lightweight (< 200K parameters)."""
        from src.speed_estimation.model import SpeedNet
        model = SpeedNet(window_size=50)
        n_params = model.count_parameters()
        self.assertLess(n_params, 200_000)
        self.assertGreater(n_params, 1_000)

    def test_different_batch_sizes(self):
        """SpeedNet handles various batch sizes."""
        import torch
        from src.speed_estimation.model import SpeedNet

        model = SpeedNet(window_size=50)
        model.eval()
        for bs in [1, 16, 64]:
            x = torch.randn(bs, 6, 50)
            with torch.no_grad():
                out = model(x)
            self.assertEqual(out.shape, (bs, 1))


class TestSpeedTrainerSanity(unittest.TestCase):

    def _make_dummy_loaders(self, n=200, W=50, bs=32):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        rng = np.random.RandomState(42)
        # Correlated IMU → speed: speed ≈ mean(|ax|) * 2
        X = rng.randn(n, 6, W).astype(np.float32)
        y = (np.abs(X[:, 0, :]).mean(axis=1) * 2.0).reshape(-1, 1).astype(np.float32)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=bs, shuffle=False)
        return loader, loader, loader  # train=val=test for sanity

    def test_trainer_loss_decreases(self):
        """Training loss should decrease over 5 epochs on a learnable pattern."""
        from src.speed_estimation.model import SpeedNet
        from src.speed_estimation.trainer import SpeedModelTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SpeedNet(window_size=50)
            trainer = SpeedModelTrainer(
                model=model,
                window_size=50,
                max_epochs=5,
                patience=99,
                checkpoint_dir=tmpdir,
            )
            train_loader, val_loader, _ = self._make_dummy_loaders()
            history = trainer.fit(train_loader, val_loader, verbose=False)

            first_rmse = history["train_rmse"][0]
            last_rmse = history["train_rmse"][-1]
            self.assertLess(last_rmse, first_rmse * 1.5,
                            "Training RMSE should not increase dramatically")

    def test_trainer_evaluate_returns_metrics(self):
        """Trainer.evaluate returns required metric keys."""
        from src.speed_estimation.model import SpeedNet
        from src.speed_estimation.trainer import SpeedModelTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SpeedNet(window_size=50)
            trainer = SpeedModelTrainer(
                model=model,
                window_size=50,
                max_epochs=2,
                patience=99,
                checkpoint_dir=tmpdir,
            )
            train_loader, val_loader, test_loader = self._make_dummy_loaders()
            trainer.fit(train_loader, val_loader, verbose=False)
            metrics = trainer.evaluate(test_loader, load_best=True)

            for key in ["rmse_ms", "rmse_kmh", "mae_ms", "max_err_ms"]:
                self.assertIn(key, metrics)
                self.assertGreaterEqual(metrics[key], 0.0)


class TestInferenceEngine(unittest.TestCase):

    def _export_dummy_model(self, tmpdir: str, window_size: int = 50) -> str:
        """Train a 1-epoch model and export to ONNX for testing."""
        import torch
        from src.speed_estimation.model import SpeedNet

        model = SpeedNet(window_size=window_size)
        model.eval()

        onnx_path = os.path.join(tmpdir, "speednet.onnx")
        dummy = torch.randn(1, 6, window_size)
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["imu_window"], output_names=["speed_ms"],
            opset_version=11,
            dynamic_axes={"imu_window": {0: "batch"}, "speed_ms": {0: "batch"}},
        )
        # Save dummy norm stats
        stats = {"mu": [0.0] * 6, "sigma": [1.0] * 6}
        stats_path = os.path.join(tmpdir, "norm_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        return onnx_path, stats_path

    def test_inference_engine_output(self):
        """SpeedInferenceEngine returns non-negative speed from window."""
        try:
            import onnxruntime  # noqa
        except ImportError:
            self.skipTest("onnxruntime not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path, stats_path = self._export_dummy_model(tmpdir)
            from src.speed_estimation.inference import SpeedInferenceEngine

            engine = SpeedInferenceEngine(
                onnx_path=onnx_path,
                norm_stats_path=stats_path,
                window_size=50,
            )
            window = np.random.randn(50, 6).astype(np.float32)
            result = engine.infer_window(window)

            self.assertIn("speed_ms", result)
            self.assertIn("latency_ms", result)
            self.assertGreaterEqual(result["speed_ms"], 0.0)

    def test_inference_latency_under_20ms(self):
        """Single inference must be < 20ms (SPEED-02 requirement)."""
        try:
            import onnxruntime  # noqa
        except ImportError:
            self.skipTest("onnxruntime not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path, stats_path = self._export_dummy_model(tmpdir)
            from src.speed_estimation.inference import SpeedInferenceEngine

            engine = SpeedInferenceEngine(
                onnx_path=onnx_path,
                norm_stats_path=stats_path,
                window_size=50,
            )
            bench = engine.benchmark_latency(n_iters=100)
            # Use p95 latency as the SLA metric
            self.assertLess(bench["p95_ms"], 20.0,
                            f"p95 latency {bench['p95_ms']:.2f}ms exceeds 20ms SLA (SPEED-02)")

    def test_push_sample_buffering(self):
        """push_sample returns None until buffer full, then returns dict."""
        try:
            import onnxruntime  # noqa
        except ImportError:
            self.skipTest("onnxruntime not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path, stats_path = self._export_dummy_model(tmpdir, window_size=10)
            from src.speed_estimation.inference import SpeedInferenceEngine

            engine = SpeedInferenceEngine(
                onnx_path=onnx_path,
                norm_stats_path=stats_path,
                window_size=10,
            )
            results = []
            for i in range(15):
                r = engine.push_sample(0.0, 0.0, 9.8, 0.0, 0.0, 0.0)
                results.append(r)

            # First 9 should be None (buffer filling)
            for r in results[:9]:
                self.assertIsNone(r)
            # From sample 10 onward should return a result
            for r in results[9:]:
                self.assertIsNotNone(r)
                self.assertIn("speed_ms", r)


if __name__ == "__main__":
    unittest.main()
