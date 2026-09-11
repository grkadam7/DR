"""Generate a real IMU-driven GNSS-outage replay from IO-VNBD.

This adapter uses the repository's CalibratedDeadReckoning engine. It keeps
raw CSV recordings out of GitHub and exports only the compact replay needed by
the browser demo.

Example:
  python tools/generate_imu_dr_replay.py \
      --smartphone S-S2.csv \
      --output mobile/demo_data.js \
      --outage-start 30 \
      --outage-duration 60

The first few seconds are used for vehicle-frame calibration. At outage start,
the engine is initialized from the last GNSS position, speed and heading, then
propagates using accelerometer + gyroscope data only. GNSS is restored after
the requested outage duration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running this file directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.iovnbd_loader import IOVNBDDataset, geodetic_to_enu, enu_to_geodetic
from src.dead_reckoning.calibrated_ins import CalibratedDeadReckoning


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smartphone", required=True)
    p.add_argument("--output", default="mobile/demo_data.js")
    p.add_argument("--outage-start", type=float, default=30.0)
    p.add_argument("--outage-duration", type=float, default=60.0)
    p.add_argument("--max-points", type=int, default=2500)
    p.add_argument("--initial-static-sec", type=float, default=2.0)
    p.add_argument("--initial-motion-sec", type=float, default=5.0)
    args = p.parse_args()

    data = IOVNBDDataset.load_csv(args.smartphone)
    t = data.timestamps_s
    if len(t) < 20:
        raise ValueError("Not enough samples for IMU DR replay")

    # The loader already converts IO-VNBD GPS speed to m/s.
    start_t = float(args.outage_start)
    end_t = start_t + float(args.outage_duration)
    outage_mask = (t >= start_t) & (t <= end_t)
    if not np.any(outage_mask):
        raise ValueError("Requested outage interval is outside the recording")

    start_i = int(np.flatnonzero(outage_mask)[0])
    end_i = int(np.flatnonzero(outage_mask)[-1])
    if start_i < int(10 * (args.initial_static_sec + args.initial_motion_sec)):
        raise ValueError("Outage starts too early for the requested calibration window")

    # Calibrate once from the beginning, then run the IMU engine only over the
    # outage segment, initialized at the last known GNSS state.
    engine = CalibratedDeadReckoning(sampling_rate=10.0)
    engine.calibrate(
        data.accel,
        data.gyro,
        initial_static_sec=args.initial_static_sec,
        initial_motion_sec=args.initial_motion_sec,
    )

    lat0, lon0, alt0 = data.origin_geodetic
    start_e, start_n, start_u = geodetic_to_enu(
        data.gps_lat[start_i], data.gps_lon[start_i], data.gps_alt[start_i],
        lat0, lon0, alt0,
    )
    start_pos = np.array([float(start_e), float(start_n), float(start_u)])

    psi = np.radians(float(data.gps_heading_deg[start_i]))
    v = float(data.gps_speed_mps[start_i])
    initial_velocity = np.array([v * np.sin(psi), v * np.cos(psi), 0.0])

    seg = slice(start_i, end_i + 1)
    dr_pos, dr_vel, dr_heading, _ = engine.propagate(
        t[seg], data.accel[seg], data.gyro[seg],
        initial_enu_pos=start_pos,
        initial_heading_deg=float(data.gps_heading_deg[start_i]),
        initial_velocity=initial_velocity,
    )

    records = []
    for j, idx in enumerate(range(start_i, end_i + 1)):
        dr_lat, dr_lon, dr_alt = enu_to_geodetic(
            dr_pos[j, 0], dr_pos[j, 1], dr_pos[j, 2], lat0, lon0, alt0
        )
        records.append({
            "t": round(float(t[idx]), 3),
            "lat": round(float(data.gps_lat[idx]), 8),
            "lon": round(float(data.gps_lon[idx]), 8),
            "speedMps": round(float(data.gps_speed_mps[idx]), 3),
            "speedKmh": round(float(data.gps_speed_mps[idx]) * 3.6, 2),
            "headingDeg": round(float(data.gps_heading_deg[idx]) % 360.0, 2),
            "drLat": round(float(dr_lat), 8),
            "drLon": round(float(dr_lon), 8),
            "drSpeedMps": round(float(np.linalg.norm(dr_vel[j, :2])), 3),
            "drHeadingDeg": round(float(dr_heading[j]) % 360.0, 2),
            "drActive": True,
        })

    # Include a sparse GNSS context before/after outage so the UI can show the
    # complete route while using the IMU-generated path only in the outage.
    keep = np.linspace(0, len(t) - 1, min(args.max_points, len(t))).round().astype(int)
    by_index = {idx: i for i, idx in enumerate(range(start_i, end_i + 1))}
    for idx in keep:
        if idx in by_index:
            continue
        records.append({
            "t": round(float(t[idx]), 3),
            "lat": round(float(data.gps_lat[idx]), 8),
            "lon": round(float(data.gps_lon[idx]), 8),
            "speedMps": round(float(data.gps_speed_mps[idx]), 3),
            "speedKmh": round(float(data.gps_speed_mps[idx]) * 3.6, 2),
            "headingDeg": round(float(data.gps_heading_deg[idx]) % 360.0, 2),
            "drLat": None, "drLon": None,
            "drSpeedMps": None, "drHeadingDeg": None,
            "drActive": False,
        })

    records.sort(key=lambda x: x["t"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "IO-VNBD + CalibratedDeadReckoning",
        "dr_source": "accelerometer + gyroscope",
        "outage_start_s": start_t,
        "outage_end_s": end_t,
        "note": "DR fields are generated by the repository IMU engine; GNSS lat/lon remain the reference trajectory.",
        "points": records,
    }
    output.write_text(
        "// Generated by tools/generate_imu_dr_replay.py; do not edit manually.\n"
        "window.DEMO_TRAJECTORY = " + json.dumps(payload, separators=(",", ":")) + ";\n"
        "window.REPLAY_SOURCE = 'IO-VNBD + IMU DR';\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} replay points to {output}")
    print(f"IMU DR outage: {start_t:.1f}s -> {end_t:.1f}s ({end_t-start_t:.1f}s)")


if __name__ == "__main__":
    main()
