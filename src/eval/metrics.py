"""Navigation Trajectory Evaluation Metrics
Calculates Absolute Trajectory Error (ATE), Root Mean Square Error (RMSE),
and cumulative drift percentage against total distance for SIH 26168.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class TrajectoryMetrics:
    """Quantitative performance metrics comparing estimated trajectory with ground truth."""
    total_distance_m: float       # Total ground-truth distance traveled in meters
    final_error_m: float          # Error at final timestamp in meters
    drift_percentage: float       # (final_error / total_distance) * 100 %
    rmse_m: float                 # Root Mean Square Error over entire trajectory
    max_error_m: float            # Peak error observed
    mean_error_m: float           # Average error along trajectory
    median_error_m: float         # Median error along trajectory
    errors_over_time: np.ndarray  # (N,) instantaneous Euclidean error at each step

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "Total Distance (m)": round(self.total_distance_m, 2),
            "Final Error (m)": round(self.final_error_m, 2),
            "Drift Percentage (%)": round(self.drift_percentage, 2),
            "RMSE (m)": round(self.rmse_m, 2),
            "Max Error (m)": round(self.max_error_m, 2),
            "Mean Error (m)": round(self.mean_error_m, 2),
            "Median Error (m)": round(self.median_error_m, 2),
        }

    def print_summary(self, label: str = "Trajectory Metrics") -> None:
        print(f"\n==================== {label} ====================")
        print(f"Total Trajectory Distance : {self.total_distance_m:.2f} m")
        print(f"Final Positional Error    : {self.final_error_m:.2f} m")
        print(f"Drift Percentage (Target <10%): {self.drift_percentage:.2f} %")
        print(f"Root Mean Square Error    : {self.rmse_m:.2f} m")
        print(f"Maximum Positional Error  : {self.max_error_m:.2f} m")
        print(f"Mean Positional Error     : {self.mean_error_m:.2f} m")
        print("========================================================\n")


def compute_drift_metrics(
    estimated_enu: np.ndarray,
    ground_truth_enu: np.ndarray,
    dimension: int = 2,
) -> TrajectoryMetrics:
    """Computes rigorous metric errors between estimated and ground-truth positions.

    
    Args:
        estimated_enu: (N, 3) or (N, 2) estimated coordinates in meters [E, N, (U)]
        ground_truth_enu: (N, 3) or (N, 2) ground-truth coordinates in meters
        dimension: 2 for horizontal (E, N) drift; 3 for 3D (E, N, U) drift
        
    Returns:
        TrajectoryMetrics with drift percentage, RMSE, and error progression.
    """
    if len(estimated_enu) != len(ground_truth_enu):
        raise ValueError(
            f"Length mismatch: est has {len(estimated_enu)}, gt has {len(ground_truth_enu)}"
        )

    est = estimated_enu[:, :dimension]
    gt = ground_truth_enu[:, :dimension]

    # Instantaneous Euclidean distance error at each timestamp
    errors = np.linalg.norm(est - gt, axis=1)

    # Ground-truth total path distance
    gt_diffs = np.diff(gt, axis=0)
    total_dist = float(np.sum(np.linalg.norm(gt_diffs, axis=1)))
    if total_dist <= 1e-6:
        total_dist = 1e-6 # prevent division by zero for static runs

    final_error = float(errors[-1])
    drift_pct = (final_error / total_dist) * 100.0
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_err = float(np.max(errors))
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))

    return TrajectoryMetrics(
        total_distance_m=total_dist,
        final_error_m=final_error,
        drift_percentage=drift_pct,
        rmse_m=rmse,
        max_error_m=max_err,
        mean_error_m=mean_err,
        median_error_m=median_err,
        errors_over_time=errors,
    )
