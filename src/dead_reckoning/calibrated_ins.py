"""
Calibrated Inertial Dead Reckoning with Preprocessing & ZUPT.
Applies:
1. Auto-leveling (gravity rotation) and forward axis alignment into Vehicle Frame.
2. Digital vibration filtering and pothole shock clamping.
3. Zero Velocity Update (ZUPT) and Zero Angular Rate Update (ZARU) resets.

Demonstrates the performance improvement over baseline raw INS.
"""

from typing import Optional, Tuple
import numpy as np

from src.preprocessing.alignment import VehicleFrameAligner, AlignmentResult
from src.preprocessing.filters import VibrationFilter
from src.preprocessing.zupt import ZeroVelocityDetector


class CalibratedDeadReckoning:
    """
    Inertial Dead Reckoning engine incorporating coordinate alignment,
    vibration filtering, and ZUPT drift resets.
    """

    def __init__(
        self,
        sampling_rate: float = 10.0,
        enable_filtering: bool = True,
        enable_zupt: bool = True,
        enable_alignment: bool = True,
        gravity: float = 9.80665,
    ):
        self.sampling_rate = sampling_rate
        self.enable_filtering = enable_filtering
        self.enable_zupt = enable_zupt
        self.enable_alignment = enable_alignment
        self.g = gravity

        self.aligner = VehicleFrameAligner(sampling_rate=sampling_rate, gravity_nominal=gravity)
        self.filter = VibrationFilter(sampling_rate=sampling_rate, cutoff_freq=3.0)
        self.zupt_detector = ZeroVelocityDetector(sampling_rate=sampling_rate, nominal_gravity=gravity)

    def calibrate(
        self,
        accel_raw: np.ndarray,
        gyro_raw: np.ndarray,
        initial_static_sec: float = 2.0,
        initial_motion_sec: float = 5.0
    ) -> AlignmentResult:
        """
        Calibrate mounting attitude from the first static seconds, then forward axis.
        """
        n_static = max(10, int(self.sampling_rate * initial_static_sec))
        static_accel = accel_raw[:n_static]
        static_gyro = gyro_raw[:n_static]

        # 1. Leveling
        self.aligner.calibrate_static(static_accel, static_gyro)

        # 2. Forward axis alignment
        n_motion = min(len(accel_raw), int(self.sampling_rate * (initial_static_sec + initial_motion_sec)))
        if n_motion > n_static:
            motion_accel = accel_raw[n_static:n_motion]
            motion_gyro_z = gyro_raw[n_static:n_motion, 2]
            self.aligner.align_forward_axis(motion_accel, gyro_z=motion_gyro_z)

        return self.aligner.get_result()

    def propagate(
        self,
        timestamps: np.ndarray,
        accel_raw: np.ndarray,
        gyro_raw: np.ndarray,
        initial_enu_pos: Optional[np.ndarray] = None,
        initial_heading_deg: Optional[float] = None,
        initial_velocity: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagate trajectory using calibrated and filtered IMU readings with ZUPT resets.
        
        Returns:
            positions (N, 3), velocities (N, 3), headings_deg (N,), stationary_mask (N,)
        """
        n = len(timestamps)
        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))
        headings = np.zeros(n)
        stationary_mask = np.zeros(n, dtype=bool)

        if initial_enu_pos is not None:
            positions[0] = initial_enu_pos.copy()
        if initial_velocity is not None:
            velocities[0] = initial_velocity.copy()
        if initial_heading_deg is not None:
            headings[0] = float(np.radians(initial_heading_deg))

        # 1. Preprocessing: Filtering
        if self.enable_filtering:
            accel_clean = self.filter.filter_batch(accel_raw, clamp_potholes=True)
            gyro_clean = self.filter.filter_batch(gyro_raw, clamp_potholes=False)
        else:
            accel_clean = accel_raw.copy()
            gyro_clean = gyro_raw.copy()

        # 2. Coordinate Alignment into Vehicle Frame
        if self.enable_alignment:
            accel_veh, gyro_veh = self.aligner.transform_imu(accel_clean, gyro_clean)
        else:
            accel_veh = accel_clean.copy()
            accel_veh[:, 2] -= self.g
            gyro_veh = gyro_clean.copy()

        # 3. ZUPT Stationary Detection
        if self.enable_zupt:
            stationary_mask, gyro_biases = self.zupt_detector.detect_batch(accel_raw, gyro_raw, timestamps)
        else:
            gyro_biases = np.zeros((n, 3))

        # 4. Sequential Dead Reckoning Propagation
        current_heading = headings[0]
        current_vel = velocities[0].copy()
        current_pos = positions[0].copy()

        for i in range(1, n):
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0.0 or dt > 1.0:
                dt = 1.0 / self.sampling_rate

            is_stopped = stationary_mask[i]

            if is_stopped:
                # ZUPT: clamp velocity to zero, freeze position
                current_vel = np.zeros(3)
                # Gyroscope bias removal (ZARU)
                omega_z = gyro_veh[i, 2] - gyro_biases[i, 2]
                # During complete stop, heading does not change
            else:
                # Heading integration using yaw rate
                omega_z = gyro_veh[i, 2] - gyro_biases[i, 2]
                current_heading += omega_z * dt
                # Normalize heading to [-pi, pi]
                current_heading = (current_heading + np.pi) % (2.0 * np.pi) - np.pi

                # Forward acceleration in horizontal vehicle frame
                # a_veh[i, 0] is forward acceleration
                a_forward = accel_veh[i, 0]

                # Convert forward/lateral acceleration to East-North navigation frame
                # In ENU: East is X_enu (heading 90 deg / pi/2), North is Y_enu (heading 0)
                # Heading psi measured clockwise from North:
                # East velocity:  v_east  = v_forward * sin(psi)
                # North velocity: v_north = v_forward * cos(psi)
                a_east = a_forward * np.sin(current_heading)
                a_north = a_forward * np.cos(current_heading)
                a_up = accel_veh[i, 2]

                # Velocity update
                current_vel[0] += a_east * dt
                current_vel[1] += a_north * dt
                current_vel[2] += a_up * dt

                # Bound vertical velocity drift
                current_vel[2] = np.clip(current_vel[2], -1.0, 1.0)

            # Position update (Trapezoidal)
            current_pos += current_vel * dt

            positions[i] = current_pos.copy()
            velocities[i] = current_vel.copy()
            headings[i] = float(np.degrees(current_heading))

        return positions, velocities, headings, stationary_mask
