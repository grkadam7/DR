"""
Digital Vibration and Pothole Shock Filtering Module.
Suppresses chassis vibrations, engine harmonics, and transient pothole shocks
from smartphone and edge IMU measurements.

Addresses Requirement: FILTER-01.
"""

from typing import Optional, Union
import numpy as np
from scipy.signal import butter, lfilter, filtfilt


class VibrationFilter:
    """
    Applies low-pass Butterworth filtering and Hampel outlier clamping
    to isolate macro vehicle navigation acceleration from mechanical vibration.
    """

    def __init__(
        self,
        sampling_rate: float = 10.0,
        cutoff_freq: float = 3.0,
        filter_order: int = 2,
        hampel_window: int = 5,
        hampel_n_sigmas: float = 3.0,
    ):
        """
        Args:
            sampling_rate: IMU sampling frequency in Hz.
            cutoff_freq: Low-pass filter cutoff frequency in Hz (typically 2.0 - 4.0 Hz).
            filter_order: Butterworth filter order.
            hampel_window: Half-window size for pothole spike detection.
            hampel_n_sigmas: Outlier threshold multiplier based on MAD.
        """
        self.sampling_rate = sampling_rate
        self.cutoff_freq = min(cutoff_freq, 0.45 * sampling_rate)
        self.filter_order = filter_order
        self.hampel_window = hampel_window
        self.hampel_n_sigmas = hampel_n_sigmas

        # Design low-pass Butterworth coefficients
        nyquist = 0.5 * sampling_rate
        normal_cutoff = self.cutoff_freq / nyquist
        self.b, self.a = butter(self.filter_order, normal_cutoff, btype="low", analog=False)

        # Real-time state registers for 3 channels
        self.zi_accel = None
        self.zi_gyro = None

    def filter_spikes_hampel(self, signal_1d: np.ndarray) -> np.ndarray:
        """
        Detects and clamps sudden transient pothole shocks using a Hampel filter
        (Median Absolute Deviation outlier detection).
        
        Args:
            signal_1d: 1D array of sensor values.
        Returns:
            Cleaned 1D array with spikes clamped to local median threshold.
        """
        n = len(signal_1d)
        if n < self.hampel_window * 2 + 1:
            return signal_1d.copy()

        cleaned = signal_1d.copy()
        k = 1.4826  # Scale factor for Gaussian distribution

        for i in range(self.hampel_window, n - self.hampel_window):
            window = signal_1d[i - self.hampel_window : i + self.hampel_window + 1]
            med = np.median(window)
            mad = np.median(np.abs(window - med))
            threshold = max(self.hampel_n_sigmas * k * mad, 1.0)

            if np.abs(signal_1d[i] - med) > threshold:
                # Clamp spike to local median
                cleaned[i] = med

        return cleaned

    def filter_batch(
        self,
        signal: np.ndarray,
        clamp_potholes: bool = True,
        zero_phase: bool = True
    ) -> np.ndarray:
        """
        Batch filter for 1D or (N, 3) 2D sensor arrays.
        
        Args:
            signal: Array of shape (N,) or (N, 3).
            clamp_potholes: Whether to apply Hampel pothole spike rejection first.
            zero_phase: If True, uses filtfilt (forward-backward zero phase distortion).
        Returns:
            Filtered sensor array of the same shape.
        """
        if len(signal) < 10:
            return signal.copy()

        is_1d = signal.ndim == 1
        if is_1d:
            data = signal[:, np.newaxis]
        else:
            data = signal.copy()

        n_samples, n_channels = data.shape
        filtered = np.zeros_like(data)

        for ch in range(n_channels):
            channel_data = data[:, ch]
            if clamp_potholes:
                channel_data = self.filter_spikes_hampel(channel_data)

            # Butterworth low-pass
            padlen = min(15, len(channel_data) - 1)
            if zero_phase and len(channel_data) > 3 * self.filter_order:
                filtered[:, ch] = filtfilt(self.b, self.a, channel_data, padlen=padlen)
            else:
                filtered[:, ch] = lfilter(self.b, self.a, channel_data)

        return filtered[:, 0] if is_1d else filtered

    def filter_realtime_sample(
        self,
        sample: np.ndarray,
        is_gyro: bool = False
    ) -> np.ndarray:
        """
        Single-sample real-time filter for on-device streaming execution.
        
        Args:
            sample: 3-element vector [x, y, z].
            is_gyro: True if gyroscope sample, False for accelerometer.
        Returns:
            Filtered 3-element vector.
        """
        from scipy.signal import lfilter_zi

        if is_gyro:
            if self.zi_gyro is None:
                self.zi_gyro = np.array([lfilter_zi(self.b, self.a) * s for s in sample]).T
            filtered, self.zi_gyro = lfilter(self.b, self.a, sample[:, np.newaxis], zi=self.zi_gyro)
            return filtered[:, 0]
        else:
            if self.zi_accel is None:
                self.zi_accel = np.array([lfilter_zi(self.b, self.a) * s for s in sample]).T
            filtered, self.zi_accel = lfilter(self.b, self.a, sample[:, np.newaxis], zi=self.zi_accel)
            return filtered[:, 0]
