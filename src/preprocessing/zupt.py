"""
Zero Velocity Update (ZUPT) and Zero Angular Rate Update (ZARU) Detector.
Detects stationary periods (traffic stops, red lights, parking) from IMU dynamics,
freezes velocity drift, and estimates online sensor biases.

Addresses Requirement: FILTER-02.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class ZuptState:
    """State output of the ZUPT detector at the current epoch."""
    is_stationary: bool
    stationary_duration_sec: float
    accel_variance: float
    gyro_magnitude: float
    gyro_bias_estimate: np.ndarray  # 3-element [bx, by, bz] (rad/s)
    accel_bias_estimate: np.ndarray  # 3-element [bx, by, bz] (m/s^2)


class ZeroVelocityDetector:
    """
    Generalized Likelihood Ratio Test (GLRT) / Multi-Condition Detector
    to detect vehicle stops and clamp accumulated dead reckoning errors.
    """

    def __init__(
        self,
        sampling_rate: float = 10.0,
        window_size_sec: float = 0.8,
        accel_var_threshold: float = 0.12,     # (m/s^2)^2
        accel_mag_diff_threshold: float = 0.40, # | ||a|| - g | < threshold (m/s^2)
        gyro_mag_threshold: float = 0.08,       # rad/s (~4.5 deg/s)
        min_consecutive_stops: int = 3,         # Min detections before triggering stop
        nominal_gravity: float = 9.80665,
    ):
        """
        Args:
            sampling_rate: IMU sampling frequency in Hz.
            window_size_sec: Window length in seconds for computing motion statistics.
            accel_var_threshold: Maximum variance of acceleration magnitude for stationary state.
            accel_mag_diff_threshold: Max allowed deviation of ||a|| from nominal gravity.
            gyro_mag_threshold: Maximum angular velocity magnitude for stationary state.
            min_consecutive_stops: Minimum consecutive stationary epochs to activate ZUPT.
            nominal_gravity: Expected Earth gravitational acceleration (m/s^2).
        """
        self.sampling_rate = sampling_rate
        self.window_size = max(3, int(sampling_rate * window_size_sec))
        self.accel_var_threshold = accel_var_threshold
        self.accel_mag_diff_threshold = accel_mag_diff_threshold
        self.gyro_mag_threshold = gyro_mag_threshold
        self.min_consecutive_stops = min_consecutive_stops
        self.g_nominal = nominal_gravity

        # Buffers for rolling window
        self._accel_buf = []
        self._gyro_buf = []

        # State tracking
        self.consecutive_stationary_count = 0
        self.stationary_duration_sec = 0.0
        self.is_currently_stopped = False

        # Online estimated sensor biases
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

    def update_sample(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: Optional[float] = None
    ) -> ZuptState:
        """
        Update detector state with a single time step IMU measurement.
        
        Args:
            accel: 3-element [ax, ay, az] in m/s^2.
            gyro: 3-element [wx, wy, wz] in rad/s.
            dt: Time step duration in seconds (defaults to 1 / sampling_rate).
        Returns:
            ZuptState with stationary classification and bias estimates.
        """
        if dt is None:
            dt = 1.0 / self.sampling_rate

        self._accel_buf.append(accel)
        self._gyro_buf.append(gyro)

        if len(self._accel_buf) > self.window_size:
            self._accel_buf.pop(0)
            self._gyro_buf.pop(0)

        # If buffer not filled yet, return non-stationary
        if len(self._accel_buf) < self.window_size:
            return ZuptState(
                is_stationary=False,
                stationary_duration_sec=0.0,
                accel_variance=999.0,
                gyro_magnitude=999.0,
                gyro_bias_estimate=self.gyro_bias.copy(),
                accel_bias_estimate=self.accel_bias.copy(),
            )

        a_win = np.array(self._accel_buf)
        w_win = np.array(self._gyro_buf)

        # Condition 1: Variance of acceleration magnitude
        a_mags = np.linalg.norm(a_win, axis=1)
        a_var = float(np.var(a_mags))

        # Condition 2: Mean acceleration magnitude close to gravity
        a_mean_mag = float(np.mean(a_mags))
        a_mag_diff = abs(a_mean_mag - self.g_nominal)

        # Condition 3: Gyroscope angular velocity magnitude
        w_mags = np.linalg.norm(w_win, axis=1)
        w_mean_mag = float(np.mean(w_mags))

        # Stationary test
        cond_accel_var = a_var < self.accel_var_threshold
        cond_accel_mag = a_mag_diff < self.accel_mag_diff_threshold
        cond_gyro = w_mean_mag < self.gyro_mag_threshold

        is_stop_epoch = cond_accel_var and cond_accel_mag and cond_gyro

        if is_stop_epoch:
            self.consecutive_stationary_count += 1
            if self.consecutive_stationary_count >= self.min_consecutive_stops:
                self.is_currently_stopped = True
                self.stationary_duration_sec += dt
                # Online update of gyro bias (exponential smoothing)
                current_gyro_mean = np.mean(w_win, axis=0)
                alpha = 0.05
                self.gyro_bias = (1.0 - alpha) * self.gyro_bias + alpha * current_gyro_mean
            else:
                self.is_currently_stopped = False
                self.stationary_duration_sec = 0.0
        else:
            self.consecutive_stationary_count = 0
            self.is_currently_stopped = False
            self.stationary_duration_sec = 0.0

        return ZuptState(
            is_stationary=self.is_currently_stopped,
            stationary_duration_sec=self.stationary_duration_sec,
            accel_variance=a_var,
            gyro_magnitude=w_mean_mag,
            gyro_bias_estimate=self.gyro_bias.copy(),
            accel_bias_estimate=self.accel_bias.copy(),
        )

    def detect_batch(
        self,
        accel_arr: np.ndarray,
        gyro_arr: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch processing of full dataset runs to return a stationary boolean mask.
        
        Args:
            accel_arr: Array of shape (N, 3).
            gyro_arr: Array of shape (N, 3).
            timestamps: Array of shape (N,) in seconds.
        Returns:
            stationary_mask: Boolean array of shape (N,) where True indicates stationary.
            gyro_biases: Array of shape (N, 3) tracking the online gyro bias estimate.
        """
        n = len(accel_arr)
        stationary_mask = np.zeros(n, dtype=bool)
        gyro_biases = np.zeros((n, 3))

        # Reset state
        self._accel_buf.clear()
        self._gyro_buf.clear()
        self.consecutive_stationary_count = 0
        self.stationary_duration_sec = 0.0
        self.is_currently_stopped = False
        self.gyro_bias = np.zeros(3)

        for i in range(n):
            dt = timestamps[i] - timestamps[i - 1] if i > 0 else 1.0 / self.sampling_rate
            if dt <= 0 or dt > 1.0:
                dt = 1.0 / self.sampling_rate

            state = self.update_sample(accel_arr[i], gyro_arr[i], dt)
            stationary_mask[i] = state.is_stationary
            gyro_biases[i] = state.gyro_bias_estimate

        return stationary_mask, gyro_biases
