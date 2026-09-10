"""
IMU Windowed Dataset Builder for Speed Estimation.

Reads IO-VNBD CSV data, constructs overlapping sliding windows of
raw 6-DOF IMU measurements, and pairs them with GPS-derived forward speed
labels for supervised training.

Addresses Requirements: SPEED-01 (data pipeline side)
"""

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column mappings for IO-VNBD CSV
# ---------------------------------------------------------------------------
COL_TIME_MS = "TIME SINCE START (ms)"
COL_SPEED_KMH = " GPS SPEED (Kmh)"
COL_ACCEL_X = " ACCELEROMETER X (m/s\xb2) "
COL_ACCEL_Y = " ACCELEROMETER Y (m/s\xb2)"
COL_ACCEL_Z = " ACCELEROMETER Z (m/s\xb2)"
COL_GYRO_YAW = " GYROSCOPE Yaw (rad/s)"
COL_GYRO_PITCH = " GYROSCOPE Pitch (rad/s)"
COL_GYRO_ROLL = " GYROSCOPE Roll (rad/s)"

# Feature channels: [ax, ay, az, gy_yaw, gy_pitch, gy_roll]
N_FEATURES = 6
KMH_TO_MS = 1.0 / 3.6


def load_iovnbd_csv(csv_path: str) -> pd.DataFrame:
    """Load IO-VNBD CSV with correct encoding and strip whitespace from column names."""
    df = pd.read_csv(csv_path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    # Re-map cleaned column names
    rename_map = {
        "ACCELEROMETER X (m/s\xb2)": "ax",
        "ACCELEROMETER Y (m/s\xb2)": "ay",
        "ACCELEROMETER Z (m/s\xb2)": "az",
        "GYROSCOPE Yaw (rad/s)": "gyr_yaw",
        "GYROSCOPE Pitch (rad/s)": "gyr_pitch",
        "GYROSCOPE Roll (rad/s)": "gyr_roll",
        "GPS SPEED (Kmh)": "speed_kmh",
        "TIME SINCE START (ms)": "time_ms",
    }
    df = df.rename(columns=rename_map)
    # Drop rows with NaN in feature or label columns
    needed = ["ax", "ay", "az", "gyr_yaw", "gyr_pitch", "gyr_roll", "speed_kmh", "time_ms"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    # Speed in m/s
    df["speed_ms"] = df["speed_kmh"] * KMH_TO_MS
    return df


def build_windows(
    df: pd.DataFrame,
    window_size: int = 50,
    stride: int = 10,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct overlapping sliding windows from the IO-VNBD DataFrame.

    Args:
        df: Loaded IO-VNBD DataFrame with standardized column names.
        window_size: Number of IMU timesteps per window.
        stride: Step between consecutive windows.
        normalize: If True, standardize each feature channel to zero-mean unit-variance
                   (statistics computed from training data only; for test use stored stats).
    Returns:
        X: (N_windows, window_size, 6) float32 feature array [ax,ay,az,gy,gp,gr]
        y: (N_windows,) float32 label array — forward speed in m/s at window center
    """
    feature_cols = ["ax", "ay", "az", "gyr_yaw", "gyr_pitch", "gyr_roll"]
    features = df[feature_cols].values.astype(np.float32)   # (T, 6)
    speeds = df["speed_ms"].values.astype(np.float32)        # (T,)

    T = len(features)
    windows_X: List[np.ndarray] = []
    windows_y: List[float] = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows_X.append(features[start:end])
        # Label: speed at center of window
        center = (start + end) // 2
        windows_y.append(float(speeds[center]))

    X = np.stack(windows_X, axis=0)  # (N, W, 6)
    y = np.array(windows_y, dtype=np.float32)  # (N,)

    if normalize:
        # Per-channel standardization across entire batch
        mu = X.mean(axis=(0, 1), keepdims=True)
        sigma = X.std(axis=(0, 1), keepdims=True) + 1e-6
        X = (X - mu) / sigma

    return X, y


class SpeedDataset:
    """
    PyTorch-compatible dataset wrapper for windowed IMU speed data.
    Supports train/val/test splits by sequential time ordering.
    """

    def __init__(
        self,
        csv_path: str,
        window_size: int = 50,
        stride: int = 5,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        seed: int = 42,
    ):
        self.window_size = window_size
        self.stride = stride
        self.train_frac = train_frac
        self.val_frac = val_frac

        df = load_iovnbd_csv(csv_path)
        X_all, y_all = build_windows(df, window_size=window_size, stride=stride, normalize=False)

        # Sequential split (do NOT shuffle — time series)
        n = len(X_all)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        X_train = X_all[:n_train]
        y_train = y_all[:n_train]
        X_val = X_all[n_train : n_train + n_val]
        y_val = y_all[n_train : n_train + n_val]
        X_test = X_all[n_train + n_val :]
        y_test = y_all[n_train + n_val :]

        # Fit normalization on train split only
        self.mu = X_train.mean(axis=(0, 1), keepdims=True)
        self.sigma = X_train.std(axis=(0, 1), keepdims=True) + 1e-6

        self.X_train = (X_train - self.mu) / self.sigma
        self.X_val = (X_val - self.mu) / self.sigma
        self.X_test = (X_test - self.mu) / self.sigma
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.norm_stats = {"mu": self.mu.squeeze().tolist(), "sigma": self.sigma.squeeze().tolist()}

    def get_torch_loaders(self, batch_size: int = 256):
        """Return (train_loader, val_loader, test_loader) DataLoader objects."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        def make_loader(X, y, shuffle):
            ds = TensorDataset(
                torch.from_numpy(X.transpose(0, 2, 1)),  # (N, 6, W) — channels first
                torch.from_numpy(y).unsqueeze(1)          # (N, 1)
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

        return (
            make_loader(self.X_train, self.y_train, shuffle=True),
            make_loader(self.X_val, self.y_val, shuffle=False),
            make_loader(self.X_test, self.y_test, shuffle=False),
        )
