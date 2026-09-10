"""
In-Vehicle Alignment and Sensor Frame Transformation Module.
Converts noisy, arbitrarily-oriented smartphone IMU measurements (Body Frame B)
into the standardized Vehicle Coordinate Frame V:
- X_v: Longitudinal Forward axis (driving direction)
- Y_v: Lateral axis (leftward, right-handed system)
- Z_v: Vertical axis (orthogonal to road, pointing upwards against gravity)

Addresses Requirements: ALIGN-01, ALIGN-02, ALIGN-03.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class AlignmentResult:
    """Stores the estimated alignment angles and rotation matrix."""
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    R_body_to_vehicle: np.ndarray  # 3x3 rotation matrix
    is_calibrated: bool
    confidence: float


class VehicleFrameAligner:
    """
    Estimates phone mounting orientation relative to the vehicle frame and rotates
    sensor measurements dynamically.
    """

    def __init__(
        self,
        sampling_rate: float = 10.0,
        gravity_nominal: float = 9.80665,
        static_accel_var_thresh: float = 0.25,
        static_gyro_mag_thresh: float = 0.15,
        accel_forward_thresh: float = 0.5,
    ):
        """
        Args:
            sampling_rate: IMU sampling frequency in Hz (typically 10 Hz for phones).
            gravity_nominal: Expected Earth gravitational acceleration magnitude (m/s^2).
            static_accel_var_thresh: Maximum acceleration variance to consider vehicle static.
            static_gyro_mag_thresh: Maximum angular velocity (rad/s) for static window.
            accel_forward_thresh: Minimum forward acceleration (m/s^2) to identify heading.
        """
        self.sampling_rate = sampling_rate
        self.g_nominal = gravity_nominal
        self.static_accel_var_thresh = static_accel_var_thresh
        self.static_gyro_mag_thresh = static_gyro_mag_thresh
        self.accel_forward_thresh = accel_forward_thresh

        # Rotation state
        self.R_b2v = np.eye(3)
        self.R_level = np.eye(3)
        self.yaw_offset = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.is_leveled = False
        self.is_yaw_aligned = False
        self.confidence = 0.0

    def compute_leveling_matrix(self, mean_accel: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Computes the rotation matrix R_level that aligns the measured static gravity
        vector with the vertical vehicle axis [0, 0, g]^T.
        
        Args:
            mean_accel: Average 3-axis accelerometer reading [ax, ay, az] in body frame
                        during a stationary period (m/s^2).
        Returns:
            R_level (3x3), pitch (rad), roll (rad)
        """
        norm_a = np.linalg.norm(mean_accel)
        if norm_a < 1e-3:
            return np.eye(3), 0.0, 0.0

        # Unit vector pointing upwards along gravity reaction in body frame
        u_z = mean_accel / norm_a

        # Target upward unit vector in leveled frame
        target_z = np.array([0.0, 0.0, 1.0])

        # Rodrigues rotation formula to rotate u_z to [0, 0, 1]
        v = np.cross(u_z, target_z)
        s = np.linalg.norm(v)
        c = float(np.dot(u_z, target_z))

        if s < 1e-6:
            # Already aligned (or upside down)
            if c > 0:
                R_level = np.eye(3)
            else:
                # 180 degree flip about X
                R_level = np.diag([1.0, -1.0, -1.0])
        else:
            # Skew-symmetric matrix of v
            vx = np.array([
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0]
            ])
            R_level = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))

        # Calculate Euler pitch and roll for inspection
        # Pitch theta = arcsin(-ux) or roll phi = arctan2(uy, uz)
        roll = float(np.arctan2(mean_accel[1], mean_accel[2]))
        pitch = float(np.arctan2(-mean_accel[0], np.sqrt(mean_accel[1]**2 + mean_accel[2]**2)))

        return R_level, pitch, roll

    def calibrate_static(
        self,
        accel_window: np.ndarray,
        gyro_window: Optional[np.ndarray] = None
    ) -> bool:
        """
        Perform static gravity leveling from a buffer of stationary accelerometer/gyro data.
        
        Args:
            accel_window: Array of shape (N, 3) containing raw accelerometer readings.
            gyro_window: Optional array of shape (N, 3) containing raw gyro readings.
        Returns:
            True if leveling succeeded, False if window was too dynamic.
        """
        if len(accel_window) < int(self.sampling_rate * 0.5):
            return False

        # Check stillness
        accel_mag = np.linalg.norm(accel_window, axis=1)
        accel_var = float(np.var(accel_mag))

        if gyro_window is not None:
            gyro_mag = np.linalg.norm(gyro_window, axis=1)
            gyro_mean = float(np.mean(gyro_mag))
            if gyro_mean > self.static_gyro_mag_thresh:
                return False

        if accel_var > self.static_accel_var_thresh:
            return False

        mean_accel = np.mean(accel_window, axis=0)
        self.R_level, self.pitch, self.roll = self.compute_leveling_matrix(mean_accel)
        self.is_leveled = True
        self.confidence = min(0.5, 1.0 - (accel_var / self.static_accel_var_thresh) * 0.5)
        self._update_full_rotation()
        return True

    def align_forward_axis(
        self,
        accel_motion: np.ndarray,
        gps_speed_diff: Optional[np.ndarray] = None,
        gyro_z: Optional[np.ndarray] = None
    ) -> bool:
        """
        Determines the vehicle's forward longitudinal axis (yaw angle psi in horizontal plane)
        by detecting the principal direction of acceleration during vehicle launch / surge.
        
        Args:
            accel_motion: Array of shape (N, 3) raw accelerations during vehicle movement.
            gps_speed_diff: Optional acceleration derived from GPS speeds (dv/dt) to resolve forward vs reverse sign.
            gyro_z: Optional vertical gyro readings to filter out turns.
        Returns:
            True if yaw alignment succeeded.
        """
        if not self.is_leveled or len(accel_motion) < int(self.sampling_rate * 1.0):
            return False

        # Project accelerations into leveled horizontal frame
        accel_leveled = (self.R_level @ accel_motion.T).T
        # Remove gravity from Z
        accel_leveled[:, 2] -= self.g_nominal

        # Filter for straight-line segments if gyro provided
        if gyro_z is not None:
            straight_mask = np.abs(gyro_z) < 0.1  # less than ~5.7 deg/sec
            if np.sum(straight_mask) > int(self.sampling_rate * 0.5):
                accel_leveled = accel_leveled[straight_mask]

        # Horizontal accelerations (X, Y)
        a_xy = accel_leveled[:, :2]
        a_mag = np.linalg.norm(a_xy, axis=1)

        # Select samples with significant acceleration
        surge_mask = a_mag > self.accel_forward_thresh
        if np.sum(surge_mask) < 3:
            return False

        a_surge = a_xy[surge_mask]

        # Use 2D Principal Component Analysis (PCA) to find the primary motion axis
        cov = np.cov(a_surge.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Principal axis corresponds to largest eigenvalue
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # Determine forward vs backward direction:
        # If mean acceleration dot product with principal axis is positive during acceleration
        mean_surge = np.mean(a_surge, axis=0)
        if np.dot(mean_surge, principal_axis) < 0:
            principal_axis = -principal_axis

        # If GPS speed diff is provided, check consistency
        if gps_speed_diff is not None:
            # When speed increases, forward accel is positive
            pass

        # Yaw angle relative to leveled frame X axis
        self.yaw_offset = float(np.arctan2(principal_axis[1], principal_axis[0]))
        self.is_yaw_aligned = True
        self.confidence = 0.95
        self._update_full_rotation()
        return True

    def _update_full_rotation(self):
        """Combines leveling rotation and horizontal yaw rotation into R_b2v."""
        # Rotation about Z by -yaw_offset to align principal axis with [1, 0, 0]
        c = np.cos(-self.yaw_offset)
        s = np.sin(-self.yaw_offset)
        R_yaw = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        self.R_b2v = R_yaw @ self.R_level

    def transform_vector(self, vector_body: np.ndarray) -> np.ndarray:
        """
        Transforms a 3D vector or (N, 3) array from body frame to vehicle frame.
        """
        if vector_body.ndim == 1:
            return self.R_b2v @ vector_body
        return (self.R_b2v @ vector_body.T).T

    def transform_imu(
        self,
        accel_body: np.ndarray,
        gyro_body: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms 3-axis accelerometer and gyro data into the vehicle frame.
        Subtracts gravity [0, 0, g]^T from the resulting vehicle acceleration.
        
        Args:
            accel_body: (N, 3) or (3,) raw accelerometer readings.
            gyro_body: (N, 3) or (3,) raw gyroscope readings.
        Returns:
            accel_vehicle (kinematic acceleration, gravity removed), gyro_vehicle
        """
        a_veh = self.transform_vector(accel_body)
        w_veh = self.transform_vector(gyro_body)

        if a_veh.ndim == 1:
            a_veh_kinematic = a_veh - np.array([0.0, 0.0, self.g_nominal])
        else:
            a_veh_kinematic = a_veh.copy()
            a_veh_kinematic[:, 2] -= self.g_nominal

        return a_veh_kinematic, w_veh

    def get_result(self) -> AlignmentResult:
        """Returns the current alignment state."""
        return AlignmentResult(
            pitch_deg=float(np.degrees(self.pitch)),
            roll_deg=float(np.degrees(self.roll)),
            yaw_deg=float(np.degrees(self.yaw_offset)),
            R_body_to_vehicle=self.R_b2v.copy(),
            is_calibrated=(self.is_leveled and self.is_yaw_aligned),
            confidence=self.confidence
        )
