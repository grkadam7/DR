"""
Phase 2 Evaluation Script.
Compares Raw Uncalibrated INS against Calibrated, Filtered, and ZUPT-Aided INS
on the real IO-VNBD benchmark dataset (S-S1.csv).

Saves comparative trajectory figures to reports/figures/calibrated_ins_real_iovnbd.png.
"""

from pathlib import Path
import numpy as np

from src.data.iovnbd_loader import IOVNBDLoader
from src.dead_reckoning.baseline_ins import BaselineDeadReckoning
from src.dead_reckoning.calibrated_ins import CalibratedDeadReckoning
from src.eval.metrics import calculate_drift_metrics, format_metrics_report
from src.eval.trajectory_plotter import TrajectoryPlotter


def run_phase2_evaluation():
    print("=" * 60)
    print("Phase 2 Evaluation: In-Vehicle Alignment, Filtering & ZUPT")
    print("=" * 60)

    loader = IOVNBDLoader()
    csv_path = Path("data/raw/S-S1.csv")

    if not csv_path.exists():
        print(f"Dataset file {csv_path} not found. Fetching from IO-VNBD repository...")
        loader.download_sample_data(output_dir=Path("data/raw"))

    print(f"\n[1/4] Loading real IO-VNBD smartphone dataset: {csv_path}...")
    run = loader.load_csv(csv_path)
    print(f"Loaded {len(run.timestamps)} epochs (~{run.timestamps[-1]:.1f} s, {run.sampling_rate:.1f} Hz).")

    # Extract initial conditions from ground truth
    initial_pos = run.gt_enu_positions[0]
    initial_heading = run.gt_headings_deg[0]
    initial_vel = run.gt_velocities[0]

    # [2/4] Baseline Raw INS (Phase 1)
    print("\n[2/4] Running Baseline Raw INS (uncalibrated, no filter, no ZUPT)...")
    baseline_ins = BaselineDeadReckoning(sampling_rate=run.sampling_rate)
    base_pos, base_vel, base_head = baseline_ins.propagate(
        timestamps=run.timestamps,
        accel_raw=run.accel_raw,
        gyro_raw=run.gyro_raw,
        initial_enu_pos=initial_pos,
        initial_heading_deg=initial_heading,
        initial_velocity=initial_vel,
    )
    base_metrics = calculate_drift_metrics(run.gt_enu_positions, base_pos)

    # [3/4] Calibrated + Filtered + ZUPT INS (Phase 2)
    print("\n[3/4] Running Calibrated INS (auto-leveling, vibration filter, ZUPT/ZARU)...")
    calib_ins = CalibratedDeadReckoning(
        sampling_rate=run.sampling_rate,
        enable_filtering=True,
        enable_zupt=True,
        enable_alignment=True,
    )
    # Calibrate mounting attitude using initial stationary epochs
    align_res = calib_ins.calibrate(run.accel_raw, run.gyro_raw, initial_static_sec=2.0)
    print(f"Alignment Result: Pitch = {align_res.pitch_deg:.1f} deg, Roll = {align_res.roll_deg:.1f} deg, Calibrated = {align_res.is_calibrated}")

    calib_pos, calib_vel, calib_head, stops = calib_ins.propagate(
        timestamps=run.timestamps,
        accel_raw=run.accel_raw,
        gyro_raw=run.gyro_raw,
        initial_enu_pos=initial_pos,
        initial_heading_deg=initial_heading,
        initial_velocity=initial_vel,
    )
    calib_metrics = calculate_drift_metrics(run.gt_enu_positions, calib_pos)
    num_stops_detected = int(np.sum(stops))
    print(f"Detected {num_stops_detected} stationary epochs ({num_stops_detected / run.sampling_rate:.1f} s of stationary vehicle updates).")

    # [4/4] Output Comparison Metrics and Figures
    print("\n" + "=" * 30 + " Baseline Raw INS (Phase 1) " + "=" * 30)
    print(format_metrics_report(base_metrics, title="Raw Uncalibrated INS"))

    print("=" * 30 + " Calibrated + Preprocessed INS (Phase 2) " + "=" * 30)
    print(format_metrics_report(calib_metrics, title="Calibrated + Filtered + ZUPT INS"))

    drift_reduction = (base_metrics.drift_percentage - calib_metrics.drift_percentage) / base_metrics.drift_percentage * 100.0
    print(f"\n>>> Positional Drift Reduction: {drift_reduction:.2f}% improvement over raw INS!")

    # Plot Comparison
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figures_dir / "calibrated_ins_real_iovnbd.png"

    plotter = TrajectoryPlotter()
    plotter.plot_multi_comparison(
        gt_positions=run.gt_enu_positions,
        trajectories={
            "Raw Baseline INS (Ph 1)": base_pos,
            "Calibrated + ZUPT INS (Ph 2)": calib_pos,
        },
        timestamps=run.timestamps,
        title="Phase 2: Raw vs Calibrated + Preprocessed INS on Real IO-VNBD Run",
        save_path=plot_path,
        metrics_dict={
            "Raw Baseline INS (Ph 1)": base_metrics,
            "Calibrated + ZUPT INS (Ph 2)": calib_metrics,
        }
    )
    print(f"\n[Plotter] Saved comparative figure to: {plot_path}")
    print("\n[Phase 2 Complete] Ready for Phase 3 (AI/ML Speed Estimation).")


if __name__ == "__main__":
    run_phase2_evaluation()
