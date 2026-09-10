"""Phase 1 Comprehensive Benchmark Runner
Runs baseline unassisted INS dead reckoning on:
1. Realistic synthetic benchmark trajectory (with stop-and-go and turns)
2. Real-world IO-VNBD smartphone dataset run (S-S1.csv)
Generates trajectory comparison plots and quantitative drift metrics for SIH review.
"""

import os
import numpy as np
from src.data.iovnbd_loader import IOVNBDDataset, generate_synthetic_run
from src.dead_reckoning.baseline_ins import BaselineINS
from src.eval.trajectory_plotter import plot_trajectory_comparison


def run_benchmark():
    os.makedirs("reports/figures", exist_ok=True)
    ins = BaselineINS()

    # -------------------------------------------------------------
    # 1. Synthetic Benchmark Run (120s urban trip)
    # -------------------------------------------------------------
    print("\n--- [1/2] Evaluating on Synthetic Benchmark Run ---")
    syn_data = generate_synthetic_run(duration_s=120.0, dt=0.1, seed=42)
    print(f"Route: {syn_data.num_samples} points, {syn_data.duration_s:.1f}s, {syn_data.total_distance_m:.1f}m distance")

    est_syn = ins.run(
        timestamps_s=syn_data.timestamps_s,
        accel_b=syn_data.accel,
        gyro_b=syn_data.gyro,
        initial_pos_enu=syn_data.enu_positions[0],
        initial_vel_enu=np.array([0.0, 0.0, 0.0]),
        initial_euler_rad=np.array([0.0, 0.0, 0.0]),
    )

    syn_metrics = plot_trajectory_comparison(
        timestamps_s=syn_data.timestamps_s,
        ground_truth_enu=syn_data.enu_positions,
        baseline_ins_enu=est_syn.positions_enu,
        gnss_enu=None,
        output_path="reports/figures/baseline_ins_synthetic.png",
        title_suffix="Synthetic Urban Route (120s, 2 Turns, 1 Stop)",
    )
    syn_metrics.print_summary("Synthetic Route: Baseline INS Drift")

    # -------------------------------------------------------------
    # 2. Real-world IO-VNBD Smartphone Dataset Run (First 300s window)
    # -------------------------------------------------------------
    real_csv = "data/iovnbd/S-S1.csv"
    if os.path.exists(real_csv):
        print("\n--- [2/2] Evaluating on Real IO-VNBD Smartphone Dataset (S-S1.csv) ---")
        real_data = IOVNBDDataset.load_csv(real_csv)

        # Evaluate on the first 300 seconds (3,000 samples at 10 Hz)
        sample_window = min(3000, real_data.num_samples)
        t_slice = real_data.timestamps_s[:sample_window]
        t_slice = t_slice - t_slice[0]
        accel_slice = real_data.accel[:sample_window]
        gyro_slice = real_data.gyro[:sample_window]
        gt_slice = real_data.enu_positions[:sample_window]
        gt_slice = gt_slice - gt_slice[0] # local origin at window start

        initial_speed = real_data.gps_speed_mps[0]
        initial_heading_rad = np.radians(real_data.gps_heading_deg[0])
        v0_e = initial_speed * np.sin(initial_heading_rad)
        v0_n = initial_speed * np.cos(initial_heading_rad)

        est_real = ins.run(
            timestamps_s=t_slice,
            accel_b=accel_slice,
            gyro_b=gyro_slice,
            initial_pos_enu=gt_slice[0],
            initial_vel_enu=np.array([v0_e, v0_n, 0.0]),
            initial_euler_rad=np.array([0.0, 0.0, initial_heading_rad]),
        )

        real_metrics = plot_trajectory_comparison(
            timestamps_s=t_slice,
            ground_truth_enu=gt_slice,
            baseline_ins_enu=est_real.positions_enu,
            gnss_enu=None,
            output_path="reports/figures/baseline_ins_real_iovnbd.png",
            title_suffix="Real IO-VNBD S-S1 Smartphone Dataset (300s Window)",
        )
        real_metrics.print_summary("Real IO-VNBD: Baseline INS Drift")

    print("[Phase 1 Benchmark Complete] All plots saved to reports/figures/.")


if __name__ == "__main__":
    run_benchmark()
