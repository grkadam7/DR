# AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation (IDR-INS)
*SIH Problem Statement ID: 26168 | Organization: Indian Space Research Organisation (ISRO)*

## What This Is
An edge-deployable software engine and mobile application that transforms standalone smartphones and external IMU sensors into an Intelligent Dead Reckoning (IDR) navigation system with seamless GNSS fusion. When GNSS outages occur (tunnels, underpasses, multi-level parking, deep urban canyons, jamming), the system instantly transitions to inertial tracking (INS), estimating vehicle speed and road trajectory without requiring any physical connection to the vehicle's internal computer or OBD-II port, and seamlessly transitions back when GNSS returns.

## Core Value
Restrict positional drift to less than 10% of total distance travelled during GNSS signal blackouts using low-cost noisy MEMS smartphone sensors without OBD-II feeds, ensuring uninterrupted, lane-level navigation across Indian road conditions.

## Business & Stakeholder Context
- **Organization**: Indian Space Research Organisation (ISRO), Department of Space
- **Target Beneficiaries**: Commercial logistics, ride-hailing (Ola/Uber/Rapido), quick commerce (Blinkit/Zepto/Swiggy), emergency first responders, and millions of Indian two-wheeler/four-wheeler drivers without factory-fitted wheel INS.
- **Success Metrics**:
  - Positional drift < 10% of distance during GNSS outages (e.g. < 5m over 50m in < 1 min; < 100m over 1km at 60 km/h).
  - Continuous 10 Hz position update rate on mobile; up to 200 Hz update rate on edge engine using high-grade/FOG IMU data.
  - Sub-100 millisecond seamless transition between GNSS-aided INS and Dead Reckoning.
  - Accurate forward speed prediction solely from IMU noise and vibrations.

## Requirements

### Validated
(None yet — initialize development to validate)

### Active
- [ ] **REQ-ALIGN**: In-Vehicle Alignment & Auto-Calibration Engine to estimate pitch, roll, and yaw relative to vehicle driving direction regardless of phone orientation.
- [ ] **REQ-SPEED**: AI/ML Speed & Vibration Filter to dynamically filter road noise/potholes/engine idling and estimate forward velocity purely from IMU.
- [ ] **REQ-MAP**: Advanced Map-Matching Filter leveraging offline OpenStreetMap (OSM) data and Non-Holonomic Constraints (NHC) to snap drifting trajectories back to the road grid.
- [ ] **REQ-FUSION**: GNSS+INS Fusion Algorithm (EKF/UKF + AI residual error compensation) for continuous optimal state estimation.
- [ ] **REQ-OUTAGE**: Millisecond-level GNSS Deficit Handler for instant transition between GNSS-aided and pure Dead Reckoning modes.
- [ ] **REQ-EDGE**: Edge-deployable core engine (C++/Python/ONNX) capable of running up to 200 Hz with external/FOG IMU sensors.
- [ ] **REQ-MOBILE**: Cross-platform mobile navigation application with real-time map UI, smooth vehicle tracking, and on-device inference at 10 Hz.
- [ ] **REQ-EVAL**: Benchmark validation on the IO-VNBD dataset (`https://github.com/onyekpeu/IO-VNBD`) producing position trajectory plots and drift error analysis for SIH proposal submission.

### Out of Scope
- **OBD-II / CAN Bus Hardware Integration**: Solution must be strictly non-invasive; relying on vehicle internal computer defeats accessibility for older vehicles and two-wheelers.
- **Continuous Cloud Processing**: Real-time navigation and inference cannot depend on active 4G/5G connectivity inside tunnels or remote valleys; all inference and map-matching must function offline on-device.

## Context
- **Sensor Challenges**: Consumer-grade smartphone MEMS IMUs suffer from run-to-run bias, thermo-mechanical drift, scale factor errors, chassis vibration, engine harmonics, and sudden shocks from Indian road potholes.
- **Kinematic Constraints**: Land vehicles obey Non-Holonomic Constraints (NHC)—vehicles do not slide sideways or fly upwards. Applying NHC ($v_y \approx 0, v_z \approx 0$ in vehicle frame) dramatically dampens perpendicular drift.
- **Dataset**: IO-VNBD (Inertial and Odometry benchmark dataset for ground vehicle positioning) provides ground truth odometry, GNSS trajectories, and raw IMU readings for rigorous model training and evaluation.

## Constraints
- **Hardware**: Must run on standard commercial Android/iOS smartphones without external sensors for the mobile tier, and standard edge compute (e.g. Raspberry Pi / Jetson / x86) for the edge tier.
- **Latency**: Position update $\le 100$ ms on mobile (10 Hz); $\le 5$ ms on edge (200 Hz).
- **Offline Capability**: Offline map tiles and routing graphs must be pre-bundled or cached locally.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Hybrid Architecture (Cloud Train, Edge Infer) | Complex deep learning models (temporal CNN/LSTM/Transformer) can be trained on large datasets (IO-VNBD) and exported as quantized ONNX/TFLite models for sub-millisecond mobile inference. | — Pending |
| Coordinate System Normalization | Phone frame must be dynamically rotated to Vehicle Frame (Forward, Lateral, Vertical) using gravity vector (accelerometer static window) and principal acceleration/gyro axis during initial motion. | — Pending |
| Multi-Stage Fusion (Physics EKF + AI Residual) | Pure physics EKF/UKF provides baseline stability and kinematic bounds; AI models predict forward speed and sensor bias residuals to prevent drift explosion. | — Pending |

---
*Last updated: 2026-09-10 after SIH Problem Statement ID 26168 Ingestion*
