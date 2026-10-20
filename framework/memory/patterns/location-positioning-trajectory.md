# Pattern: location-positioning-trajectory

## When this applies
- GNSS / smartphone / vehicle positioning tasks with raw pseudorange, carrier phase, CN0
- Trajectory regression with Haversine / distance-based metrics
- Small trip counts (~150) with strong temporal autocorrelation within trips
- Train/test splits by location or device with potential distribution shift

## Proven patterns
- **Use domain libraries, never reimplement**: `gnss_lib_py`, `filterpy`, `pymap3d`, `georinex` collapse 1300-line scipy solvers into ~100–300 lines (smartphone-decimeter-2022 run 20260420_070725 burned full budget on a 1303-line FGO that only reached 3.73 m val)
- **Kalman RTS backward smoother as pre-processor**: best run (20260216_203619, 5.4877 m) and all competitive candidates used it; removes WLS bias and temporal autocorrelation
- **Savitzky-Golay post-smoothing (window=11, poly=2)**: stable 20–30% reduction in 95th-percentile tail across runs 20260304_091243 and 20260304_213518
- **CN0-weighted aggregations** beat uniform means for signal-quality features in every smartphone-decimeter-2022 run
- **GBDT (LightGBM/XGBoost) with Huber loss** beats neural nets when trips < 200 (BiLSTM killed for overfit, MLP timed out)
- **GroupKFold by trip ID** is mandatory; date-only grouping leaks (Candidate 4 in run 20260304_091243: 2.78 m CV → 12.99 m test)
- **Separate lat/lon models** avoid coordinate-axis coupling (confirmed run 20260304_091243)
- **Sample weighting by 1/AccuracyMeters or Kalman innovation covariance** consistently improves final 10% of performance

## Common pitfalls
- **Reimplementing GNSS primitives from scratch** exhausts coder budget before depth techniques (cycle-slip, ambiguity resolution, IMU fusion) can be added
- **CV→test collapse from distribution shift**: best CV 2.78–3.2 m → test 5.5–5.9 m across all 3 smartphone-decimeter-2022 runs; LAX-urban train vs MTV/SVL-open-sky test never addressed
- **Linearized pseudorange geometry projection** is numerically fragile (eliminated pre-implementation in run 20260304_213518)
- **Hardcoded Kalman process noise** (vel_noise=4.0 vs optimal ~1.0) cost 0.13 m in run 20260216_203619
- **Neural nets on small trip counts** overfit or timeout (BiLSTM, MLP both failed)
- **Reviser/agent-limit exhaustion**: runs 20260216, 20260304_091243, 20260304_213518 all hit 9/10 call limits or reviser crashes before final tuning

## Model selection guide
- Trips < 200 → LightGBM/XGBoost + Huber, NOT neural nets
- Need sub-3m → physics pipeline (FGO via `gnss_lib_py`) + ML residual correction ensemble
- Urban canyon / multipath dominant → UKF or particle filter (`filterpy`) over EKF
- Default backbone: WLS baseline → Kalman RTS smoother → GBDT residual → Savitzky-Golay post-smooth

## Feature engineering patterns
- Per-epoch: CN0 mean/std/min/percentiles (weighted), sat count per constellation, elevation stats
- Geometry: GDOP/PDOP proxy from satellite positions
- Per-satellite (for re-WLS): CN0, elevation, constellation flag, ISRB
- Sample weight: AccuracyMeters or Kalman innovation covariance
- Atmospheric (iono/tropo) and IMU columns are available but underexplored — likely upside

## Tuning patterns
- Huber α=0.9 for LightGBM beats MSE on pseudorange outliers
- Kalman process noise is the dominant lever; tune adaptively by velocity magnitude (constant-velocity breaks at stops/turns)
- Strong regularization mandatory at trip-count ~150 (Candidate 4 collapse trace)
- Quantile regression directly on 50th/95th matches the metric better than mean + post-hoc smoothing (untested but indicated)
- Ensemble best-physics + best-ML candidate (never executed across 3 runs — likely free gain)
