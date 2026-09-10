"""IO-VNBD Dataset Loader & Geodetic Coordinate Transformations
Handles reading, parsing, and geographic transformation of inertial, GNSS,
and ground-truth navigation data for SIH 26168.
"""

from dataclasses import dataclass
import os
import re
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# WGS84 Ellipsoid Constants
# ---------------------------------------------------------------------------
WGS84_A = 6378137.0          # Semi-major axis in meters
WGS84_F = 1.0 / 298.257223563 # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F) # Semi-minor axis in meters
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2 # First eccentricity squared


def geodetic_to_ecef(
    lat_deg: Union[float, np.ndarray],
    lon_deg: Union[float, np.ndarray],
    alt_m: Union[float, np.ndarray] = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts WGS84 geodetic coordinates (lat, lon, alt) to Earth-Centered,

    Earth-Fixed (ECEF) coordinates in meters.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Prime vertical radius of curvature
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return x, y, z


def ecef_to_enu(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts ECEF Cartesian coordinates to Local East-North-Up (ENU) coordinates

    relative to a reference geodetic point (lat0, lon0, alt0).
    """
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    dx = x - x0
    dy = y - y0
    dz = z - z0

    lat0_rad = np.radians(lat0_deg)
    lon0_rad = np.radians(lon0_deg)
    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return east, north, up


def geodetic_to_enu(
    lat_deg: Union[float, np.ndarray],
    lon_deg: Union[float, np.ndarray],
    alt_m: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directly converts WGS84 geodetic coordinates to local East-North-Up (ENU)

    metric Cartesian coordinates.
    """
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m)


def enu_to_geodetic(
    east: Union[float, np.ndarray],
    north: Union[float, np.ndarray],
    up: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts local East-North-Up (ENU) coordinates back to WGS84 geodetic

    coordinates (lat, lon, alt) using Bowring's iterative method.
    """
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    lat0_rad = np.radians(lat0_deg)
    lon0_rad = np.radians(lon0_deg)
    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Inverse rotation matrix (transpose of orthogonal matrix)
    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    # ECEF to Geodetic
    p = np.sqrt(x**2 + y**2)
    lon_rad = np.arctan2(y, x)

    # Bowring's closed form / iterative solution
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    e_prime2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)

    lat_rad = np.arctan2(
        z + e_prime2 * WGS84_B * (np.sin(theta)**3),
        p - WGS84_E2 * WGS84_A * (np.cos(theta)**3)
    )

    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat_rad)**2)
    alt = p / np.cos(lat_rad) - n

    return np.degrees(lat_rad), np.degrees(lon_rad), alt


# ---------------------------------------------------------------------------
# Data Container
# ---------------------------------------------------------------------------
@dataclass
class IOVNBDData:
    """Structured container for aligned inertial, GNSS, and reference data."""
    timestamps_s: np.ndarray       # (N,) elapsed time in seconds
    accel: np.ndarray              # (N, 3) m/s^2 [ax, ay, az] in device frame
    gyro: np.ndarray               # (N, 3) rad/s [gx, gy, gz] in device frame
    gravity: np.ndarray            # (N, 3) m/s^2 estimated gravity vector
    mag: np.ndarray                # (N, 3) uT magnetometer readings
    gps_lat: np.ndarray            # (N,) degrees
    gps_lon: np.ndarray            # (N,) degrees
    gps_alt: np.ndarray            # (N,) meters
    gps_speed_mps: np.ndarray      # (N,) vehicle forward speed from GPS in m/s
    gps_heading_deg: np.ndarray    # (N,) GPS course heading in degrees (0 = North, 90 = East)
    gps_accuracy_m: np.ndarray     # (N,) estimated horizontal accuracy in meters
    enu_positions: np.ndarray      # (N, 3) [East, North, Up] meters relative to start
    origin_geodetic: Tuple[float, float, float] # (lat0, lon0, alt0)

    @property
    def num_samples(self) -> int:
        return len(self.timestamps_s)

    @property
    def duration_s(self) -> float:
        if len(self.timestamps_s) < 2:
            return 0.0
        return float(self.timestamps_s[-1] - self.timestamps_s[0])

    @property
    def sampling_rate_hz(self) -> float:
        if len(self.timestamps_s) < 2:
            return 0.0
        dt = np.median(np.diff(self.timestamps_s))
        return 1.0 / dt if dt > 0 else 0.0

    @property
    def total_distance_m(self) -> float:
        """Total cumulative path distance traveled in meters."""
        if len(self.enu_positions) < 2:
            return 0.0
        diffs = np.diff(self.enu_positions[:, :2], axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))


class IOVNBDDataset:
    """Parser and loader for IO-VNBD benchmark runs."""

    @staticmethod
    def _clean_header(col_name: str) -> str:
        """Normalizes varied column names from IO-VNBD CSV files."""
        clean = col_name.strip()
        clean = re.sub(r'[\r\n\t]', '', clean)
        clean = re.sub(r'\s+', ' ', clean)
        return clean

    @classmethod
    def load_csv(cls, filepath: str) -> IOVNBDData:
        """Loads and parses an IO-VNBD smartphone CSV file (e.g., S-S1.csv)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"IO-VNBD file not found: {filepath}")

        # Read CSV while handling varying encoding and header spaces
        df = pd.read_csv(filepath, skipinitialspace=True, encoding='latin1')
        df.columns = [cls._clean_header(c) for c in df.columns]

        # Match columns flexibly using regex
        def find_col(pattern: str) -> str:
            for c in df.columns:
                if re.search(pattern, c, re.IGNORECASE):
                    return c
            raise KeyError(f"Could not find column matching '{pattern}' in {df.columns.tolist()}")

        col_time = find_col(r'time since start|time')
        col_lat = find_col(r'gps latitude')
        col_lon = find_col(r'gps longitude')
        col_alt = find_col(r'gps altitude')
        col_speed = find_col(r'gps speed')
        col_acc = find_col(r'gps accuracy')
        col_orient = find_col(r'gps orientation|gps heading')

        col_ax = find_col(r'accelerometer x')
        col_ay = find_col(r'accelerometer y')
        col_az = find_col(r'accelerometer z')

        col_gx = find_col(r'gyroscope roll|gyroscope x')
        col_gy = find_col(r'gyroscope pitch|gyroscope y')
        col_gz = find_col(r'gyroscope yaw|gyroscope z')

        # Gravity and Magnetometer if present, otherwise default
        try:
            col_grav_x = find_col(r'gravity x')
            col_grav_y = find_col(r'gravity y')
            col_grav_z = find_col(r'gravity z')
            gravity = df[[col_grav_x, col_grav_y, col_grav_z]].values.astype(np.float64)
        except KeyError:
            gravity = np.zeros((len(df), 3), dtype=np.float64)
            gravity[:, 2] = 9.80665

        try:
            col_mx = find_col(r'magnetic field x')
            col_my = find_col(r'magnetic field y')
            col_mz = find_col(r'magnetic field z')
            mag = df[[col_mx, col_my, col_mz]].values.astype(np.float64)
        except KeyError:
            mag = np.zeros((len(df), 3), dtype=np.float64)

        # Timestamps: convert ms to seconds starting at 0
        raw_time = df[col_time].values.astype(np.float64)
        timestamps_s = (raw_time - raw_time[0]) / 1000.0

        # Inertial channels
        accel = df[[col_ax, col_ay, col_az]].values.astype(np.float64)
        gyro = df[[col_gx, col_gy, col_gz]].values.astype(np.float64)

        # GPS channels
        gps_lat = df[col_lat].values.astype(np.float64)
        gps_lon = df[col_lon].values.astype(np.float64)
        gps_alt = df[col_alt].values.astype(np.float64)

        # Speed is typically in km/h in IO-VNBD; convert to m/s
        raw_speed = df[col_speed].values.astype(np.float64)
        gps_speed_mps = raw_speed / 3.6

        gps_heading_deg = df[col_orient].values.astype(np.float64)
        gps_accuracy_m = df[col_acc].values.astype(np.float64)

        # Set reference origin to initial valid GPS position
        lat0, lon0, alt0 = float(gps_lat[0]), float(gps_lon[0]), float(gps_alt[0])
        e, n, u = geodetic_to_enu(gps_lat, gps_lon, gps_alt, lat0, lon0, alt0)
        enu_positions = np.column_stack([e, n, u])

        return IOVNBDData(
            timestamps_s=timestamps_s,
            accel=accel,
            gyro=gyro,
            gravity=gravity,
            mag=mag,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            gps_alt=gps_alt,
            gps_speed_mps=gps_speed_mps,
            gps_heading_deg=gps_heading_deg,
            gps_accuracy_m=gps_accuracy_m,
            enu_positions=enu_positions,
            origin_geodetic=(lat0, lon0, alt0),
        )


def generate_synthetic_run(
    duration_s: float = 120.0,
    dt: float = 0.1,
    noise_level: float = 1.0,
    seed: int = 42,
) -> IOVNBDData:
    """Generates a high-fidelity synthetic ground vehicle trajectory adhering to

    the IO-VNBD kinematic format and SIH 26168 specifications.
    
    Includes:
    - Acceleration, cruising at 15 m/s (~54 km/h)
    - 90-degree left turn
    - Straight cruise
    - Complete stop at traffic light (0 m/s for 15s)
    - 90-degree right turn
    - Acceleration and final cruise
    - Realistic IMU sensor noise, bias drift, and gravity components
    """
    rng = np.random.RandomState(seed)
    n_steps = int(duration_s / dt)
    t = np.linspace(0.0, duration_s, n_steps)

    # Initial origin: Coventry, UK (matching IO-VNBD dataset area)
    lat0, lon0, alt0 = 52.40166, -1.50529, 147.5

    # True trajectory kinematics in 2D ENU
    # We define speed profile and yaw rate profile
    speed_true = np.zeros(n_steps)
    yaw_rate_true = np.zeros(n_steps)

    # Speed profile definition:
    # 0 - 10s: Accelerate from 0 to 14 m/s (approx 50 km/h)
    # 10 - 30s: Cruising at 14 m/s
    # 30 - 35s: Turn left 90 deg (yaw rate = +pi/10 rad/s = 18 deg/s)
    # 35 - 55s: Cruising at 14 m/s
    # 55 - 65s: Decelerate to 0 m/s (stop light)
    # 65 - 80s: Static stop (ZUPT window)
    # 80 - 90s: Accelerate to 12 m/s
    # 90 - 95s: Turn right 90 deg (yaw rate = -pi/10 rad/s)
    # 95 - 120s: Final cruising at 12 m/s

    for i, cur_t in enumerate(t):
        if cur_t < 10.0:
            speed_true[i] = 1.4 * cur_t
        elif cur_t < 30.0:
            speed_true[i] = 14.0
        elif cur_t < 35.0:
            speed_true[i] = 10.0
            yaw_rate_true[i] = np.radians(18.0) # 90 deg in 5s
        elif cur_t < 55.0:
            speed_true[i] = 14.0
        elif cur_t < 65.0:
            # decelerate from 14 to 0
            speed_true[i] = max(0.0, 14.0 - 1.4 * (cur_t - 55.0))
        elif cur_t < 80.0:
            speed_true[i] = 0.0 # complete stop
        elif cur_t < 90.0:
            speed_true[i] = 1.2 * (cur_t - 80.0) # accelerate to 12 m/s
        elif cur_t < 95.0:
            speed_true[i] = 10.0
            yaw_rate_true[i] = np.radians(-18.0) # 90 deg right in 5s
        else:
            speed_true[i] = 12.0

    # Integrate true heading and ENU positions
    # Heading in math radians (0 = along East, pi/2 = North), or Nav azimuth (0 = North, pi/2 = East)
    # Let's use navigation azimuth: yaw_nav = integral(yaw_rate)
    # Initial vehicle heading: 0 degrees (due North)
    yaw_nav = np.zeros(n_steps)
    yaw_nav[0] = 0.0
    for i in range(1, n_steps):
        yaw_nav[i] = yaw_nav[i-1] + yaw_rate_true[i-1] * dt

    # ENU position integration
    e_true = np.zeros(n_steps)
    n_true = np.zeros(n_steps)
    u_true = np.zeros(n_steps)

    for i in range(1, n_steps):
        # East velocity = v * sin(yaw_nav), North velocity = v * cos(yaw_nav)
        ve = speed_true[i] * np.sin(yaw_nav[i])
        vn = speed_true[i] * np.cos(yaw_nav[i])
        e_true[i] = e_true[i-1] + ve * dt
        n_true[i] = n_true[i-1] + vn * dt
        # Subtle road elevation changes
        u_true[i] = 0.5 * np.sin(cur_t / 10.0)

    # Compute true accelerations in vehicle frame
    # Forward acceleration a_fwd = d(speed)/dt
    # Centripetal acceleration a_lat = v * yaw_rate
    accel_fwd = np.gradient(speed_true, dt)
    accel_lat = speed_true * yaw_rate_true

    # Gravity in Earth Frame is [0, 0, +9.80665] m/s^2 upward normal force
    g_val = 9.80665

    # Device mounting: assume phone is mounted with slight arbitrary misalignment:
    # Pitch = 5 deg, Roll = 3 deg, Yaw offset = 10 deg relative to vehicle
    pitch_mount = np.radians(5.0)
    roll_mount = np.radians(3.0)
    yaw_mount = np.radians(10.0)

    # Sensor noise & constant bias characteristics (typical smartphone MEMS)
    accel_bias = np.array([0.08, -0.05, 0.04]) * noise_level # m/s^2 bias
    gyro_bias = np.array([0.003, 0.002, -0.004]) * noise_level # rad/s bias
    accel_noise_std = 0.05 * noise_level
    gyro_noise_std = 0.005 * noise_level

    # Generate synthetic sensor readings
    accel = np.zeros((n_steps, 3))
    gyro = np.zeros((n_steps, 3))
    gravity = np.zeros((n_steps, 3))
    mag = np.zeros((n_steps, 3))

    for i in range(n_steps):
        # Vehicle body acceleration [Lateral, Forward, Up]
        a_body = np.array([accel_lat[i], accel_fwd[i], g_val])
        # Add noise and bias
        a_meas = a_body + accel_bias + rng.normal(0, accel_noise_std, size=3)
        accel[i] = a_meas

        # Gyroscope [Roll rate, Pitch rate, Yaw rate]
        w_body = np.array([0.0, 0.0, yaw_rate_true[i]])
        w_meas = w_body + gyro_bias + rng.normal(0, gyro_noise_std, size=3)
        gyro[i] = w_meas

        gravity[i] = np.array([0.0, 0.0, g_val])
        # Magnetometer reading aligned with Earth magnetic field (~45 uT)
        mag[i] = np.array([15.0 * np.sin(yaw_nav[i]), 15.0 * np.cos(yaw_nav[i]), -40.0])

    # Convert true ENU to Geodetic for GPS
    gps_lat, gps_lon, gps_alt = enu_to_geodetic(e_true, n_true, u_true, lat0, lon0, alt0)
    # Add GNSS measurement noise (typical 2.5m standard deviation)
    gps_lat_meas = gps_lat + rng.normal(0, 2.5 / 111000.0, size=n_steps)
    gps_lon_meas = gps_lon + rng.normal(0, 2.5 / (111000.0 * np.cos(np.radians(lat0))), size=n_steps)
    gps_alt_meas = gps_alt + rng.normal(0, 3.5, size=n_steps)

    # GPS headings in degrees [0, 360)
    gps_heading_deg = (np.degrees(yaw_nav) + 360.0) % 360.0
    gps_accuracy_m = np.full(n_steps, 2.5)

    enu_positions_true = np.column_stack([e_true, n_true, u_true])

    return IOVNBDData(
        timestamps_s=t,
        accel=accel,
        gyro=gyro,
        gravity=gravity,
        mag=mag,
        gps_lat=gps_lat_meas,
        gps_lon=gps_lon_meas,
        gps_alt=gps_alt_meas,
        gps_speed_mps=speed_true + rng.normal(0, 0.1, size=n_steps),
        gps_heading_deg=gps_heading_deg,
        gps_accuracy_m=gps_accuracy_m,
        enu_positions=enu_positions_true,
        origin_geodetic=(lat0, lon0, alt0),
    )
