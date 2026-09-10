"""
Unit tests for Phase 2: In-Vehicle Alignment, Signal Filtering, and ZUPT Detection.
"""

import unittest
import numpy as np

from src.preprocessing.alignment import VehicleFrameAligner
from src.preprocessing.filters import VibrationFilter
from src.preprocessing.zupt import ZeroVelocityDetector
from src.dead_reckoning.calibrated_ins import CalibratedDeadReckoning


class TestPhase2AlignmentAndPreprocessing(unittest.TestCase):

    def setUp(self):
        self.g = 9.80665
        self.sampling_rate = 10.0

    def test_gravity_leveling_flat(self):
        """Test leveling when phone is already flat on table."""
        aligner = VehicleFrameAligner(sampling_rate=self.sampling_rate, gravity_nominal=self.g)
        # 10 samples of flat phone
        accel_flat = np.tile([0.0, 0.0, self.g], (10, 1))
        success = aligner.calibrate_static(accel_flat)
        self.assertTrue(success)

        res = aligner.get_result()
        self.assertAlmostEqual(res.pitch_deg, 0.0, places=1)
        self.assertAlmostEqual(res.roll_deg, 0.0, places=1)

        # Transformed vector should have zero kinematic acceleration
        a_kin, _ = aligner.transform_imu(np.array([0.0, 0.0, self.g]), np.zeros(3))
        np.testing.assert_allclose(a_kin, [0.0, 0.0, 0.0], atol=1e-3)

    def test_gravity_leveling_tilted(self):
        """Test leveling when phone is tilted 30 deg pitch on dashboard mount."""
        theta_rad = np.radians(30.0)
        # Measured specific force when tilted pitch theta:
        # ax = -g * sin(theta), ay = 0, az = g * cos(theta)
        accel_tilted = np.tile([
            -self.g * np.sin(theta_rad),
            0.0,
            self.g * np.cos(theta_rad)
        ], (20, 1))

        aligner = VehicleFrameAligner(sampling_rate=self.sampling_rate, gravity_nominal=self.g)
        success = aligner.calibrate_static(accel_tilted)
        self.assertTrue(success)

        res = aligner.get_result()
        self.assertAlmostEqual(res.pitch_deg, 30.0, delta=1.0)
        self.assertAlmostEqual(res.roll_deg, 0.0, delta=1.0)

        # Transformed specific force must align with vertical vehicle axis [0, 0, g]
        a_leveled = aligner.R_level @ accel_tilted[0]
        np.testing.assert_allclose(a_leveled, [0.0, 0.0, self.g], atol=1e-3)

    def test_forward_axis_alignment(self):
        """Test dynamic yaw alignment detecting forward longitudinal vehicle axis."""
        aligner = VehicleFrameAligner(sampling_rate=self.sampling_rate, gravity_nominal=self.g)
        # Static leveling first
        aligner.calibrate_static(np.tile([0.0, 0.0, self.g], (10, 1)))

        # Vehicle accelerates forward with a surge along phone's +X axis
        n_surge = 20
        surge_accel = np.zeros((n_surge, 3))
        surge_accel[:, 0] = 1.5  # 1.5 m/s^2 forward surge
        surge_accel[:, 2] = self.g

        success = aligner.align_forward_axis(surge_accel)
        self.assertTrue(success)

        res = aligner.get_result()
        self.assertTrue(res.is_calibrated)
        self.assertAlmostEqual(res.yaw_deg, 0.0, delta=2.0)

    def test_vibration_filter_attenuation(self):
        """Test Butterworth low-pass filter attenuates high-frequency vibration."""
        fs = 50.0  # 50 Hz edge IMU
        v_filter = VibrationFilter(sampling_rate=fs, cutoff_freq=4.0)

        t = np.linspace(0, 2.0, int(2.0 * fs))
        # 1 Hz vehicle motion + 20 Hz engine vibration
        clean_motion = 2.0 * np.sin(2 * np.pi * 1.0 * t)
        vibration = 3.0 * np.sin(2 * np.pi * 20.0 * t)
        noisy_signal = clean_motion + vibration

        filtered = v_filter.filter_batch(noisy_signal, clamp_potholes=False)

        # High frequency energy must be reduced by at least 80%
        residual_vibration = np.std(filtered - clean_motion)
        original_vibration = np.std(vibration)
        self.assertLess(residual_vibration, 0.2 * original_vibration)

    def test_pothole_spike_clamping(self):
        """Test Hampel outlier filter clamps sudden isolated pothole shock spikes."""
        v_filter = VibrationFilter(sampling_rate=10.0)
        signal = np.full(30, 2.0)
        # Inject pothole shock spike
        signal[15] = 38.5  # extreme bump

        clamped = v_filter.filter_spikes_hampel(signal)
        self.assertLess(clamped[15], 5.0)
        self.assertAlmostEqual(clamped[15], 2.0, delta=0.5)

    def test_zupt_detector(self):
        """Test ZUPT classification on stationary vs moving segments."""
        zupt = ZeroVelocityDetector(sampling_rate=self.sampling_rate, nominal_gravity=self.g)

        # 1. Stationary segment: acceleration = g, gyro = 0 with minimal noise
        n_stat = 15
        stat_accel = np.tile([0.0, 0.0, self.g], (n_stat, 1)) + np.random.normal(0, 0.02, (n_stat, 3))
        stat_gyro = np.zeros((n_stat, 3)) + np.random.normal(0, 0.005, (n_stat, 3))
        t_stat = np.arange(n_stat) * 0.1

        mask_stat, _ = zupt.detect_batch(stat_accel, stat_gyro, t_stat)
        # After initial buffer, must be detected as stationary
        self.assertTrue(np.any(mask_stat))
        self.assertTrue(mask_stat[-1])

        # 2. Moving segment: vehicle driving with acceleration and turns
        n_mov = 15
        mov_accel = np.tile([1.5, 0.5, self.g], (n_mov, 1)) + np.random.normal(0, 0.5, (n_mov, 3))
        mov_gyro = np.tile([0.0, 0.0, 0.3], (n_mov, 1))
        t_mov = np.arange(n_mov) * 0.1

        mask_mov, _ = zupt.detect_batch(mov_accel, mov_gyro, t_mov)
        self.assertFalse(mask_mov[-1])

    def test_calibrated_ins_runs(self):
        """Test complete calibrated INS pipeline on synthetic data."""
        n = 100
        t = np.arange(n) * 0.1
        accel = np.tile([0.0, 0.0, self.g], (n, 1))
        gyro = np.zeros((n, 3))

        ins = CalibratedDeadReckoning(sampling_rate=self.sampling_rate)
        pos, vel, head, stops = ins.propagate(t, accel, gyro)

        self.assertEqual(pos.shape, (n, 3))
        self.assertEqual(vel.shape, (n, 3))
        self.assertEqual(len(head), n)
        self.assertEqual(len(stops), n)


if __name__ == "__main__":
    unittest.main()
