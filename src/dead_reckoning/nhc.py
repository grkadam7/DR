"""Lightweight non-holonomic constraint (NHC) correction for vehicle DR."""

from dataclasses import dataclass
import numpy as np


@dataclass
class NHCCorrection:
    """Corrected vehicle-frame velocity and the applied residual."""
    velocity: np.ndarray
    residual: np.ndarray


class NonHolonomicConstraint:
    """Enforce the vehicle assumption v_y ~= 0 and v_z ~= 0.

    This is deliberately a small deterministic layer.  It can be used by the
    current DR engine and later replaced/embedded as a measurement update in
    the GNSS/INS EKF without changing the public interface.
    """

    def __init__(self, lateral_std: float = 0.25, vertical_std: float = 0.20):
        if lateral_std <= 0 or vertical_std <= 0:
            raise ValueError("NHC standard deviations must be positive")
        self.lateral_std = float(lateral_std)
        self.vertical_std = float(vertical_std)

    def correct(self, velocity_vehicle: np.ndarray) -> NHCCorrection:
        """Return velocity with lateral and vertical components constrained to zero."""
        v = np.asarray(velocity_vehicle, dtype=float).reshape(3).copy()
        residual = v.copy()
        v[1] = 0.0
        v[2] = 0.0
        return NHCCorrection(velocity=v, residual=residual)

    def apply_batch(self, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply NHC to an ``(N, 3)`` velocity array."""
        values = np.asarray(velocities, dtype=float).copy()
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("velocities must have shape (N, 3)")
        residual = values.copy()
        values[:, 1] = 0.0
        values[:, 2] = 0.0
        return values, residual
