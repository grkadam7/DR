# SIH Problem Statement: ID 26168
## AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation

- **Organization**: Indian Space Research Organisation (ISRO)
- **Department**: Department of Space / Indian Space Research Organisation
- **Category**: Software
- **Theme**: Smart Vehicles
- **Dataset Reference**: [IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning](https://github.com/onyekpeu/IO-VNBD)

---

### 1. Background
Vehicle logistics, ride-hailing services, quick commerce, and emergency responders heavily rely on smartphone-based navigation apps (such as Google Maps or MapmyIndia) powered by GNSS (GPS/Galileo/NavIC etc.). However, when a vehicle enters a long underground tunnel/underpass, a multi-level parking lot, a dense forested highway, or a deep urban canyon surrounded by skyscrapers, GNSS connectivity drops entirely.

GNSS signals are inherently weak and vulnerable to structural blockage (urban canyons, dense foliage, tunnels, deep valleys) and unintentional electromagnetic interferences from a variety of sources such as jamming. This causes navigation apps to freeze, jump erratically, or miscalculate upcoming turns, leading to missed exits, delivery delays, and safety hazards.

In these environments, systems must rely on self-contained Inertial Navigation Systems (INS) built from Inertial Measurement Units (IMUs) (accelerometers and gyroscopes) to calculate position via dead reckoning during GNSS outages and switch back to GNSS-aided INS after blackouts. While INS is immune to external jamming, low-cost tactical or MEMS-grade IMUs suffer from inherent sensor biases, deterministic errors, and thermo-mechanical noise.

While modern high-end cars possess factory-fitted, wheel-connected Inertial Navigation Systems (INS), the vast majority of vehicles on Indian roads—including commercial trucks, older cars, and millions of two-wheelers (motorcycles/scooters)—rely solely on the driver's smartphone mounted on the dashboard or placed in a mobile holder.

Using a smartphone's internal MEMS IMU (accelerometer and gyroscope) to track vehicle position via dead reckoning during a GNSS blackout is highly challenging:
- The smartphone is subjected to severe chassis vibrations, engine harmonics, sudden braking, and road potholes.
- Without an external speedometer feed from the vehicle's OBD-II port, calculating distance and velocity exclusively from consumer-grade smartphone sensors results in exponential error accumulation, causing the estimated location to drift away within seconds.

To overcome these challenges, there is an urgent need for AI-ML enhanced dead reckoning and sensor fusion (GNSS+INS) techniques that integrate AI and machine learning models with real-time correction strategies.

---

### 2. Description
The goal is to develop a lightweight, edge-deployable software engine and mobile application that transforms a standalone smartphone into an **Intelligent Dead Reckoning (IDR)** system with GNSS Fusion. When a GNSS outage occurs, the application must instantly transition to inertial tracking (INS), maintaining lane-level accuracy without requiring any physical connection to the vehicle's internal computer and seamlessly switch back to a GNSS-aided INS solution.

To bypass the need for an external speedometer, the solution must employ AI/ML models trained on vehicle kinematics to accurately predict vehicle speed and acceleration profiles solely from the smartphone's noisy accelerometer/gyro inputs. It must dynamically detect and filter out non-navigation motions such as engine idling vibrations, pothole shocks, bumps, and accidental phone misalignments on the mount.

Furthermore, the navigation engine should implement a smart **Map-Matching Filter**. By overlaying the inertial trajectory onto an offline map database (e.g., OpenStreetMap), the system should use the road layout as a constraint. For instance, it can apply Non-Holonomic Constraints (NHC), assuming a car cannot slide sideways or fly upwards, to dramatically snap the drifting IMU path back onto the actual road grid.

Also, the GNSS+INS fusion algorithm should employ AI/ML techniques to develop an AI-based fusion model to mitigate drift errors and provide accurate position.

**Generalizability**: The final solution and AI/ML models developed must not be constricted to smartphone IMU sensor data alone (mobile application). These algorithms and models must also work with external IMU sensor data (edge-deployable software engine, e.g., high-precision FOG-based IMUs).

---

### 3. Dataset Details
- **IO-VNBD**: Inertial and Odometry benchmark dataset for ground vehicle positioning (`https://github.com/onyekpeu/IO-VNBD`).
- This dataset must be used to train and test the models and submit for screening of proposals.
- Teams are required to include the preliminary AI models and the results of the position plot inferenced from the subset of the IO-VNBD dataset as part of their proposals submitted for evaluation. During the screening process, more datasets will be provided for further evaluation of the AI models.

---

### 4. The On-Device Workflow
Dead reckoning and GNSS fusion algorithms are hybrid. Complex training happens in the cloud/desktop a priori, while inference happens on the smartphone/edge device:

1. **Model Training (Cloud/Desktop)**:
   - Train AI-ML models using IMU sensor datasets collected using a smartphone mounted on vehicles or open-source datasets (e.g., IO-VNBD).
   - Bundle offline map databases (e.g., OpenStreetMap).
2. **On-Device Execution (Smartphone / Edge)**:
   - Export trained lightweight models to the smartphone/edge runtime.
   - Receive live inputs from the IMU (accelerometer, gyroscope, magnetometer/compass) and GNSS (when available).
   - Remove sensor noise & bias, predict corrections, perform map-matching, and compute continuous positions for both dead reckoning and GNSS+INS fusion.

---

### 5. Expected Technical Capabilities
1. **In-Vehicle Alignment & Calibration Engine**:
   - Algorithmic module that automatically determines the phone's pitch, roll, and yaw relative to the vehicle's driving direction, regardless of mount angle or orientation.
2. **AI Speed & Vibration Filter**:
   - Deep-learning or statistical signal-processing model running locally on the phone to filter high-frequency road noise/potholes and directly estimate vehicle forward velocity from IMU signals.
3. **Advanced Map-Matching & Kinematic Constraints**:
   - Framework (e.g., AI-ML framework or Unscented Kalman Filter + Hidden Markov Map Matching) that binds the calculated position to known road networks and geometric paths during a dropout, incorporating Non-Holonomic Constraints (NHC).
4. **GNSS+INS Fusion Engine**:
   - Innovative AI-based Sensor Fusion Algorithm combining GNSS & IMU measurements to eliminate drift errors and provide accurate position and velocity.
5. **Seamless GNSS Deficit Handler**:
   - Instant transition mechanism between GNSS-aided INS and dead reckoning within milliseconds of GNSS signal blackout and vice-versa.
6. **Real-time Navigation Interface**:
   - Functional mobile application UI displaying a smooth, uninterrupted vehicle icon with seamless navigation.

---

### 6. Performance Benchmarks
- **Dead Reckoning Drift**:
  - Restrict positional drift to **< 10% of total distance travelled** during GNSS blackout.
  - Examples:
    - Drift **< 5 meters** over a **50m** GNSS-denied environment in **< 1 minute**.
    - Drift **< 100 meters** over a **1 km** GNSS-denied environment at a speed of **60 km/h** in tunnels/underground metro.
- **Update Rates**:
  - **Mobile application**: **10 Hz** position update rate.
  - **Edge-deployable software engine**: Up to **200 Hz** for FOG-based IMU sensor data.
