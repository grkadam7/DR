import numpy as np

from src.dead_reckoning.nhc import NonHolonomicConstraint


def test_nhc_zeroes_lateral_and_vertical_velocity():
    nhc = NonHolonomicConstraint()
    corrected = nhc.correct(np.array([12.0, 2.5, -0.7]))
    np.testing.assert_allclose(corrected.velocity, [12.0, 0.0, 0.0])
    np.testing.assert_allclose(corrected.residual, [12.0, 2.5, -0.7])


def test_nhc_batch_preserves_forward_velocity():
    nhc = NonHolonomicConstraint()
    values, residual = nhc.apply_batch(
        np.array([[5.0, 1.0, 0.2], [8.0, -2.0, 1.5]])
    )
    np.testing.assert_allclose(values, [[5.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    np.testing.assert_allclose(residual, [[5.0, 1.0, 0.2], [8.0, -2.0, 1.5]])
