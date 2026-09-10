"""Trajectory Visualizer & Comparison Plotter
Generates publication-quality 2D/3D trajectory plots and error curves
demonstrating baseline INS exponential drift vs ground truth for SIH 26168.
"""

import os
from typing import Optional, List
import matplotlib
# Use Agg backend for headless server execution
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .metrics import compute_drift_metrics, TrajectoryMetrics


def plot_trajectory_comparison(
    timestamps_s: np.ndarray,
    ground_truth_enu: np.ndarray,
    baseline_ins_enu: np.ndarray,
    gnss_enu: Optional[np.ndarray] = None,
    output_path: str = "reports/figures/baseline_drift_comparison.png",
    title_suffix: str = "IO-VNBD Benchmark Vehicle Run",
) -> TrajectoryMetrics:
    """Generates and saves a dual-panel figure:

    Panel 1: 2D Spatial Trajectory (East vs North in meters)
    Panel 2: Cumulative Position Error vs Time (showing exponential drift)
    
    Returns:
        TrajectoryMetrics for the baseline INS.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics_ins = compute_drift_metrics(baseline_ins_enu, ground_truth_enu)
    metrics_gnss = (
        compute_drift_metrics(gnss_enu, ground_truth_enu)
        if gnss_enu is not None
        else None
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    fig.patch.set_facecolor('#0f172a') # Slate 900 dark theme

    for ax in axes:
        ax.set_facecolor('#1e293b') # Slate 800
        ax.grid(True, linestyle='--', alpha=0.3, color='#94a3b8')
        ax.tick_params(colors='#cbd5e1', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#475569')

    # -------------------------------------------------------------
    # Panel 1: 2D Trajectory (East vs North)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_title(f"2D Trajectory Comparison | {title_suffix}", fontsize=13, color='#f8fafc', fontweight='bold', pad=12)
    ax1.set_xlabel("East Position (meters)", fontsize=11, color='#e2e8f0')
    ax1.set_ylabel("North Position (meters)", fontsize=11, color='#e2e8f0')

    # Ground truth
    ax1.plot(
        ground_truth_enu[:, 0],
        ground_truth_enu[:, 1],
        color='#10b981', # Emerald
        linewidth=2.5,
        label=f"Ground Truth Reference ({metrics_ins.total_distance_m:.1f} m)",
        zorder=5,
    )

    # Raw GNSS
    if gnss_enu is not None:
        ax1.scatter(
            gnss_enu[:, 0],
            gnss_enu[:, 1],
            color='#38bdf8', # Sky blue
            s=8,
            alpha=0.6,
            label=f"Standard GNSS (RMSE: {metrics_gnss.rmse_m:.2f} m)",
            zorder=3,
        )

    # Baseline INS
    ax1.plot(
        baseline_ins_enu[:, 0],
        baseline_ins_enu[:, 1],
        color='#f43f5e', # Rose red
        linewidth=2.0,
        linestyle='--',
        label=f"Baseline INS (Drift: {metrics_ins.drift_percentage:.1f}%)",
        zorder=4,
    )

    # Start & End Markers
    ax1.scatter(
        ground_truth_enu[0, 0],
        ground_truth_enu[0, 1],
        color='#3b82f6',
        s=90,
        edgecolor='#ffffff',
        linewidth=1.5,
        label="Start Point (t=0s)",
        zorder=6,
    )
    ax1.scatter(
        ground_truth_enu[-1, 0],
        ground_truth_enu[-1, 1],
        color='#10b981',
        marker='*',
        s=140,
        edgecolor='#ffffff',
        label="GT End Point",
        zorder=6,
    )
    ax1.scatter(
        baseline_ins_enu[-1, 0],
        baseline_ins_enu[-1, 1],
        color='#f43f5e',
        marker='X',
        s=120,
        edgecolor='#ffffff',
        label=f"INS End Point ({metrics_ins.final_error_m:.1f} m error)",
        zorder=6,
    )

    ax1.axis('equal')
    legend1 = ax1.legend(loc='best', framealpha=0.85, facecolor='#1e293b', edgecolor='#475569')
    for text in legend1.get_texts():
        text.set_color('#f8fafc')

    # -------------------------------------------------------------
    # Panel 2: Error Progression Over Time
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_title("Absolute Positional Error Growth Over Time", fontsize=13, color='#f8fafc', fontweight='bold', pad=12)
    ax2.set_xlabel("Elapsed Time (seconds)", fontsize=11, color='#e2e8f0')
    ax2.set_ylabel("Euclidean Position Error (meters)", fontsize=11, color='#e2e8f0')

    # Baseline INS error
    ax2.plot(
        timestamps_s,
        metrics_ins.errors_over_time,
        color='#f43f5e',
        linewidth=2.2,
        label=f"Raw INS Drift Error (Max: {metrics_ins.max_error_m:.1f} m)",
    )

    # GNSS error baseline
    if metrics_gnss is not None:
        ax2.plot(
            timestamps_s,
            metrics_gnss.errors_over_time,
            color='#38bdf8',
            alpha=0.7,
            linewidth=1.5,
            label=f"GNSS Receiver Error (Mean: {metrics_gnss.mean_error_m:.2f} m)",
        )

    # Target Drift Benchmark Threshold (10% of total distance)
    drift_target_10pct = 0.10 * metrics_ins.total_distance_m
    ax2.axhline(
        y=drift_target_10pct,
        color='#eab308', # Yellow
        linestyle=':',
        linewidth=2.0,
        label=f"SIH <10% Target Bound ({drift_target_10pct:.1f} m)",
    )

    ax2.fill_between(
        timestamps_s,
        0,
        drift_target_10pct,
        color='#eab308',
        alpha=0.08,
        label="Acceptable Drift Region",
    )

    legend2 = ax2.legend(loc='upper left', framealpha=0.85, facecolor='#1e293b', edgecolor='#475569')
    for text in legend2.get_texts():
        text.set_color('#f8fafc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[Plotter] Saved publication figure to: {output_path}")
    return metrics_ins


def main():
    """Runs baseline dead reckoning on a synthetic benchmark run and generates figure."""
    from ..data.iovnbd_loader import generate_synthetic_run
    from ..dead_reckoning.baseline_ins import BaselineINS

    print("[Phase 1] Generating benchmark dataset run...")
    data = generate_synthetic_run(duration_s=120.0, dt=0.1)
    print(f"[Phase 1] Trajectory generated: {data.num_samples} samples, {data.duration_s:.1f}s, {data.total_distance_m:.1f}m distance")

    print("[Phase 1] Executing pure double-integration Baseline INS...")
    ins = BaselineINS()
    est = ins.run(
        timestamps_s=data.timestamps_s,
        accel_b=data.accel,
        gyro_b=data.gyro,
        initial_pos_enu=data.enu_positions[0],
        initial_vel_enu=np.array([0.0, 0.0, 0.0]),
        initial_euler_rad=np.array([0.0, 0.0, 0.0]),
    )

    print("[Phase 1] Computing metrics and generating plots...")
    output_png = "reports/figures/baseline_ins_drift.png"
    metrics = plot_trajectory_comparison(
        timestamps_s=data.timestamps_s,
        ground_truth_enu=data.enu_positions,
        baseline_ins_enu=est.positions_enu,
        gnss_enu=None,
        output_path=output_png,
        title_suffix="Synthetic Urban Route (120s, 2 Turns, 1 Stop)",
    )
    metrics.print_summary("Baseline INS Dead Reckoning Performance")


if __name__ == "__main__":
    main()
