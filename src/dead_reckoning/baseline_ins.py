"""Baseline Pure Double-Integration Inertial Navigation System (INS)
Demonstrates the fundamental sensor drift error explosion in low-cost MEMS IMUs
when operating without external speed updates (OBD-II), kinematic constraints, or AI.
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional


@dataclass
class INSEstimate:
    """Estimated navigation states over time."""
    timestamps_s: np.ndarray       # (N,) seconds
    positions_enu: np.ndarray      # (N, 3) East, North, Up in meters
    velocities_enu: np.ndarray     # (N, 3) Ve, Vn, Vu in m/s
    orientations_rad: np.ndarray   # (N, 3) Roll, Pitch, Yaw in radians
    accelerations_enu: np.ndarray  # (N, 3) Net linear accelerations in navigation frame


class BaselineINS:
    """Standard strapdown inertial navigation algorithm with 6-DOF mechanization."""

    def __init__(self, gravity_magnitude: float = 9.80665):
        self.g_val = gravity_magnitude
        self.g_n = np.array([0.0, 0.0, self.g_val]) # Navigation frame gravity [E, N, U]

    @staticmethod
    def _euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Computes Direction Cosine Matrix (DCM) from body frame to navigation frame (ENU)

        using Z-Y-X (Yaw-Pitch-Roll) Euler angle sequence.
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # R_b^n: transforms a vector from Body frame to Navigation frame
        # R = R_z(yaw) * R_y(pitch) * R_x(roll)
        r11 = cy * cp
        r12 = cy * sp * sr - sy * cr
        r13 = cy * sp * cr + sy * sr

        r21 = sy * cp
        r22 = sy * sp * sr + cy * cr
        r23 = sy * sp * cr - cy * sr

        r31 = -sp
        r32 = cp * sr
        r33 = cp * cr

        return np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])

    @staticmethod
    def _quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
        """Converts unit quaternion [qw, qx, qy, qz] to 3x3 rotation matrix R_b^n."""
        qw, qx, qy, qz = q
        return np.array([
            [1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy - qw*qz), 2.0*(qx*qz + qw*qy)],
            [2.0*(qx*qy + qw*qz), 1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz - qw*qx)],
            [2.0*(qx*qz - qw*qy), 2.0*(qy*qz + qw*qx), 1.0 - 2.0*(qx**2 + qy**2)]
        ])

    @staticmethod
    def _dcm_to_euler(r: np.ndarray) -> Tuple[float, float, float]:
        """Extracts Roll, Pitch, Yaw (radians) from Direction Cosine Matrix R_b^n."""
        pitch = -np.arcsin(np.clip(r[2, 0], -1.0, 1.0))
        roll = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(r[1, 0], r[0, 0])
        return roll, pitch, yaw

    @staticmethod
    def _quaternion_update(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """Updates unit quaternion using angular velocity vector omega [wx, wy, wz] over dt."""
        norm_w = np.linalg.norm(omega)
        if norm_w < 1e-12:
            return q

        theta = norm_w * dt
        axis = omega / norm_w
        sin_half = np.sin(theta / 2.0)
        cos_half = np.cos(theta / 2.0)

        # Delta quaternion dq = [cos(theta/2), sin(theta/2)*axis]
        dq = np.array([cos_half, sin_half * axis[0], sin_half * axis[1], sin_half * axis[2]])

        # Quaternion multiplication: q_next = q * dq
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = dq

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        q_next = np.array([w, x, y, z])
        return q_next / np.linalg.norm(q_next)

    def run(
        self,
        timestamps_s: np.ndarray,
        accel_b: np.ndarray,
        gyro_b: np.ndarray,
        initial_pos_enu: Optional[np.ndarray] = None,
        initial_vel_enu: Optional[np.ndarray] = None,
        initial_euler_rad: Optional[np.ndarray] = None,
    ) -> INSEstimate:
        """Executes full strapdown INS mechanization across time series.

        
        Args:
            timestamps_s: (N,) timestamps in seconds
            accel_b: (N, 3) specific force measured by accelerometer in body frame [m/s^2]
            gyro_b: (N, 3) angular rate measured by gyroscope in body frame [rad/s]
            initial_pos_enu: (3,) initial [East, North, Up] position in meters
            initial_vel_enu: (3,) initial [Ve, Vn, Vu] velocity in m/s
            initial_euler_rad: (3,) initial [Roll, Pitch, Yaw] in radians
            
        Returns:
            INSEstimate containing calculated positions, velocities, attitudes, and accelerations.
        """
        n_samples = len(timestamps_s)
        if n_samples == 0:
            raise ValueError("Timestamps array is empty")

        pos = np.zeros((n_samples, 3), dtype=np.float64)
        vel = np.zeros((n_samples, 3), dtype=np.float64)
        euler = np.zeros((n_samples, 3), dtype=np.float64)
        acc_n = np.zeros((n_samples, 3), dtype=np.float64)

        # Set initial conditions
        if initial_pos_enu is not None:
            pos[0] = initial_pos_enu
        if initial_vel_enu is not None:
            vel[0] = initial_vel_enu

        # Initial attitude
        if initial_euler_rad is not None:
            roll, pitch, yaw = initial_euler_rad
        else:
            # Estimate coarse leveling from first accelerometer reading
            a0 = accel_b[0]
            norm_a = np.linalg.norm(a0)
            if norm_a > 1e-3:
                pitch = np.arctan2(-a0[0], np.sqrt(a0[1]**2 + a0[2]**2))
                roll = np.arctan2(a0[1], a0[2])
            else:
                pitch, roll = 0.0, 0.0
            yaw = 0.0

        euler[0] = [roll, pitch, yaw]
        dcm = self._euler_to_dcm(roll, pitch, yaw)

        # Convert initial DCM to quaternion [qw, qx, qy, qz]
        tr = np.trace(dcm)
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            qw = 0.25 / s
            qx = (dcm[2, 1] - dcm[1, 2]) * s
            qy = (dcm[0, 2] - dcm[2, 0]) * s
            qz = (dcm[1, 0] - dcm[0, 1]) * s
        else:
            if dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
                s = 2.0 * np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
                qw = (dcm[2, 1] - dcm[1, 2]) / s
                qx = 0.25 * s
                qy = (dcm[0, 1] + dcm[1, 0]) / s
                qz = (dcm[0, 2] + dcm[2, 0]) / s
            elif dcm[1, 1] > dcm[2, 2]:
                s = 2.0 * np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
                qw = (dcm[0, 2] - dcm[2, 0]) / s
                qx = (dcm[0, 1] + dcm[1, 0]) / s
                qy = 0.25 * s
                qz = (dcm[1, 2] + dcm[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
                qw = (dcm[1, 0] - dcm[0, 1]) / s
                qx = (dcm[0, 2] + dcm[2, 0]) / s
                qy = (dcm[1, 2] + dcm[2, 1]) / s
                qz = 0.25 * s

        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        q /= np.linalg.norm(q)

        # Initial acceleration resolution
        f_b0 = accel_b[0]
        f_n0 = dcm @ f_b0
        acc_n[0] = f_n0 - self.g_n

        # Strapdown integration loop
        for k in range(1, n_samples):
            dt = timestamps_s[k] - timestamps_s[k-1]
            if dt <= 0.0:
                dt = 0.01

            # 1. Update orientation quaternion from gyroscope measurements
            w_b = gyro_b[k-1]
            q = self._quaternion_update(q, w_b, dt)
            dcm = self._quaternion_to_dcm(q)
            euler[k] = self._dcm_to_euler(dcm)

            # 2. Transform specific force to navigation frame and remove gravity
            f_b = accel_b[k]
            f_n = dcm @ f_b
            a_lin = f_n - self.g_n
            acc_n[k] = a_lin

            # 3. Trapezoidal integration for velocity
            vel[k] = vel[k-1] + 0.5 * (acc_n[k-1] + acc_n[k]) * dt

            # 4. Trapezoidal integration for position
            pos[k] = pos[k-1] + 0.5 * (vel[k-1] + vel[k]) * dt

        return INSEstimate(
            timestamps_s=timestamps_s,
            positions_enu=pos,
            velocities_enu=vel,
            orientations_rad=euler,
            accelerations_enu=acc_n,
        )
