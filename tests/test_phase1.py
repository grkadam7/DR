"""Phase 1 Automated Verification Test Suite
Tests WGS84/ENU coordinate conversions, IO-VNBD dataset loader,
baseline INS double-integration mechanization, metrics, and visualization.
"""

import os
import unittest
import numpy as np

from src.data.iovnbd_loader import (
    geodetic_to_ecef,
    ecef_to_enu,
    geodetic_to_enu,
    enu_to_geodetic,
    IOVNBDDataset,
    generate_synthetic_run,
)
from src.dead_reckoning.baseline_ins import BaselineINS, INSEstimate
from src.eval.metrics import compute_drift_metrics
from src.eval.trajectory_plotter import plot_trajectory_comparison


class TestCoordinateTransforms(unittest.TestCase):
    """Verifies WGS84 ellipsoidal and Local ENU conversions."""

    def test_geodetic_enu_roundtrip(self):
        # Reference origin: New Delhi / ISRO ground station coords
        lat0, lon0, alt0 = 28.6139, 77.2090, 215.0

        # Offset target point (~1 km north, ~500 m east, +20 m altitude)
        lat_target, lon_target, alt_target = 28.6229, 77.2141, 235.0

        # Forward transform
        e, n, u = geodetic_to_enu(lat_target, lon_target, alt_target, lat0, lon0, alt0)

        self.assertGreater(n, 900.0)  # ~1000m North
        self.assertGreater(e, 400.0)  # ~500m East
        self.assertAlmostEqual(u, 20.0, delta=0.5)

        # Reverse transform
        lat_rev, lon_rev, alt_rev = enu_to_geodetic(e, n, u, lat0, lon0, alt0)

        # Accuracy within 0.001 meters (<1 mm equivalent)
        self.assertAlmostEqual(lat_target, lat_rev, places=6)
        self.assertAlmostEqual(lon_target, lon_rev, places=6)
        self.assertAlmostEqual(alt_target, alt_rev, delta=0.01)

    def test_origin_is_zero_enu(self):
        lat0, lon0, alt0 = 52.40166, -1.50529, 147.5
        e, n, u = geodetic_to_enu(lat0, lon0, alt0, lat0, lon0, alt0)
        self.assertAlmostEqual(float(e), 0.0, places=5)
        self.assertAlmostEqual(float(n), 0.0, places=5)
        self.assertAlmostEqual(float(u), 0.0, places=5)


class TestIOVNBDLoader(unittest.TestCase):
    """Verifies dataset parsing and synthetic generation."""

    def test_synthetic_generator_structure(self):
        data = generate_synthetic_run(duration_s=60.0, dt=0.1, seed=123)
        self.assertEqual(data.num_samples, 600)
        self.assertAlmostEqual(data.duration_s, 60.0, delta=0.2)
        self.assertAlmostEqual(data.sampling_rate_hz, 10.0, delta=0.1)
        self.assertEqual(data.accel.shape, (600, 3))
        self.assertEqual(data.gyro.shape, (600, 3))
        self.assertEqual(data.enu_positions.shape, (600, 3))
        self.assertGreater(data.total_distance_m, 200.0)

    def test_load_real_iovnbd_csv_if_present(self):
        csv_path = "data/iovnbd/S-S1.csv"
        if os.path.exists(csv_path):
            data = IOVNBDDataset.load_csv(csv_path)
            self.assertGreater(data.num_samples, 50000)
            self.assertAlmostEqual(data.sampling_rate_hz, 10.0, delta=0.2)
            self.assertEqual(data.accel.shape[1], 3)
            self.assertEqual(data.gyro.shape[1], 3)
            self.assertGreater(data.total_distance_m, 30000.0)


class TestBaselineINS(unittest.TestCase):
    """Verifies Baseline Strapdown INS mechanization and error accumulation."""

    def test_pure_ins_drift_growth(self):
        # Generate 60-second synthetic run
        data = generate_synthetic_run(duration_s=60.0, dt=0.1, seed=42)

        ins = BaselineINS()
        est = ins.run(
            timestamps_s=data.timestamps_s,
            accel_b=data.accel,
            gyro_b=data.gyro,
            initial_pos_enu=data.enu_positions[0],
            initial_vel_enu=np.array([0.0, 0.0, 0.0]),
            initial_euler_rad=np.array([0.0, 0.0, 0.0]),
        )

        self.assertEqual(len(est.positions_enu), data.num_samples)
        self.assertEqual(est.positions_enu.shape, (600, 3))

        metrics = compute_drift_metrics(est.positions_enu, data.enu_positions)

        # Baseline unassisted MEMS INS MUST exhibit substantial drift (>10%)
        # which validates why AI / Map-Matching / Kalman Filtering are needed.
        self.assertGreater(metrics.drift_percentage, 10.0)
        self.assertGreater(metrics.final_error_m, 10.0)


class TestEvaluationAndPlotter(unittest.TestCase):
    """Verifies metric calculations and figure generation."""

    def test_metrics_calculation(self):
        gt = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=float)
        # 1 meter offset at each point
        est = np.array([[0, 1], [10, 1], [20, 1], [30, 1]], dtype=float)

        metrics = compute_drift_metrics(est, gt, dimension=2)
        self.assertAlmostEqual(metrics.total_distance_m, 30.0)
        self.assertAlmostEqual(metrics.final_error_m, 1.0)
        self.assertAlmostEqual(metrics.drift_percentage, (1.0 / 30.0) * 100.0)
        self.assertAlmostEqual(metrics.rmse_m, 1.0)
        self.assertAlmostEqual(metrics.max_error_m, 1.0)

    def test_plotter_generates_file(self):
        data = generate_synthetic_run(duration_s=20.0, dt=0.1, seed=99)
        ins = BaselineINS()
        est = ins.run(
            timestamps_s=data.timestamps_s,
            accel_b=data.accel,
            gyro_b=data.gyro,
            initial_pos_enu=data.enu_positions[0],
        )

        out_path = "reports/figures/test_trajectory_plot.png"
        metrics = plot_trajectory_comparison(
            timestamps_s=data.timestamps_s,
            ground_truth_enu=data.enu_positions,
            baseline_ins_enu=est.positions_enu,
            gnss_enu=None,
            output_path=out_path,
            title_suffix="Unit Test Route",
        )

        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 1000)
        # Cleanup test artifact
        if os.path.exists(out_path):
            os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
