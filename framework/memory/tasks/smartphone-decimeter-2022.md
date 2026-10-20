# Task Memory: smartphone-decimeter-2022

## Available domain libraries (pre-installed, use freely)

The following GNSS / state-estimation libraries are installed in the `rl` conda env. Import and use them directly — do NOT run `pip install` (prohibited by system rules), and do NOT re-implement their functionality from scratch in scipy / numpy. This is the key change compared to earlier runs on this task.

- **`gnss_lib_py`** (Stanford, v1.0+): batteries-included GNSS utilities. Has EKF, WLS, factor-graph helpers, GPS constants, ephemeris parsing, coordinate transforms. Start here for any state-estimation pipeline.
- **`georinex`**: parse RINEX observation / navigation files if the raw data requires it.
- **`pymap3d`**: ECEF ↔ ENU ↔ geodetic (lat/lon/alt) coordinate transforms. Replaces hand-rolled WGS84 conversions.
- **`filterpy`**: production Kalman / EKF / UKF / particle-filter implementations. Use this instead of writing a Kalman filter from scratch.

**Why this matters**: prior runs hit a per-candidate code-budget ceiling writing filters and FGO skeletons from scratch in scipy.optimize (one run produced a 1300-line factor-graph implementation reaching only ~3.7 m val). With these libraries, the same pipeline is ~100–300 lines, leaving room to add cycle-slip detection, IMU fusion, or ambiguity resolution within a single coder session.

## Task profile
- Type: Regression (position correction)
- Dataset: 151 training trips from smartphones (Pixel4/5/6Pro, Samsung, Xiaomi) across locations (LAX urban canyon, MTV/SVL open-sky); raw GNSS measurements (~258K epochs, ~30 satellites/epoch); modalities include pseudorange, carrier phase, CN0, elevation, ionospheric/tropospheric delays
- Metric: Average Haversine distance in meters (lower is better); scoring uses mean of 50th + 95th percentile
- Medal thresholds: Bronze ≤ 3.068 m; Gold ~1.229 m (T. Suzuki 2022 1st-place reference)

## Best result
- Score: 5.4877 m | Medal: none | Run: 20260216_203619
- Approach: Kalman RTS smoother with velocity consistency checks (ECEF-to-ENU conversion) achieved best validation (3.201 m) but test-set distribution shift (urban LAX in train vs. open-sky MTV/SVL in test) inflated submission score. LightGBM and hybrid re-WLS candidates also produced valid submissions but scored worse.

## Score progression
| Run | Score | Medal | Key change |
|-----|-------|-------|------------|
| 20260216_203619 | 5.4877 m | none | Kalman RTS smoother + LightGBM correction + hybrid re-WLS; reviser crashed before final grading |
| 20260304_091243 | 5.8538 m | none | EKF + RTS + dual XGBoost (lat/lon) with Huber loss + Savitzky-Golay post-smoothing; Candidate 4 collapsed from 2.78 m CV → 12.99 m test |
| 20260304_213518 | 5.8486 m | none | WLS baseline + LightGBM (Huber α=0.9) + Savitzky-Golay (window=11, poly=2); GroupKFold by trip ID; sample weighting by 1/AccuracyMeters |

**No improvement trajectory observed** — all three runs cluster at 5.5–5.9 m, far above the 3.068 m bronze threshold. Best CV scores (2.78–3.2 m) fail to transfer to test set.

## Proven strategies
- **Kalman RTS backward smoother** as pre-processor: consistently removes temporal autocorrelation and systematic WLS bias; appears in all competitive candidates
- **Savitzky-Golay post-smoothing** (window=11, polyorder=2): stable across runs, reduces 95th-percentile tail errors by 20–30% vs. independent-epoch correction
- **CN0-weighted signal aggregation**: CN0DbHz-weighted mean/std/min/percentiles consistently outperform uniform averaging across all runs
- **GBDT over neural nets**: LightGBM and XGBoost with Huber loss reliably beat MLP/BiLSTM given only 151 trips; domain features + tabular methods dominate
- **GroupKFold by trip ID**: essential to prevent leakage; phones from same trip share GNSS environment and multipath patterns. Grouping by date alone is insufficient
- **Separate lat/lon models**: avoids coupling coordinate-axis errors (confirmed in run 20260304_091243)
- **Sample weighting by 1/AccuracyMeters or Kalman innovation uncertainty**: reduces influence of high-error WLS epochs, improves parameter learning

## Failure patterns
- **Re-implementing GNSS primitives from scratch exhausts coder budget**: run 4 (20260420_070725) C2 wrote a 1303-line scipy.optimize factor-graph solver from scratch; it works (val 3.73 m, barely better than the 3.76 m baseline) but leaves no coder-session budget for cycle-slip detection, ambiguity resolution, or IMU fusion — the techniques that actually separate 3.7 m from <3 m. With `gnss_lib_py` / `filterpy` / `pymap3d` now installed, the skeleton should be 100–300 lines and the remaining session budget should go toward depth (see below).
- **Persistent train→test distribution shift**: best CV scores (2.78–3.2 m) degrade to 5.5–5.9 m on test. LAX urban canyon drives dominate training; test likely has more open-sky MTV/SVL drives. No run has solved this.
- **Neural nets timeout or overfit**: BiLSTM killed for overfitting risk (151 trips); MLP timed out at 1200s twice due to per-satellite data loading bottleneck
- **GroupKFold misconfiguration**: Candidate 4 (run 2) achieved 2.78 m CV but collapsed to 12.99 m on full test — broken GroupKFold strategy and weak regularization
- **Reviser/pipeline failures**: run 1 reviser crashed (`judge_output.json does not exist`); runs 2–3 hit 9/10 agent call limits before completing tuning
- **Kalman process noise left untuned**: hardcoded vel_noise=4.0 vs. optimal ~1.0 contributed 0.13 m gap in run 1; no run achieved adaptive per-drive tuning
- **Linearized pseudorange geometry projection** numerically fragile (eliminated pre-implementation in run 3)

## Feature engineering playbook
- **Per-epoch signal quality**: CN0DbHz (mean, std, min, percentiles weighted by signal strength), satellite count per constellation (GPS/GLONASS/Galileo), elevation statistics
- **Geometry**: GDOP/PDOP proxy from satellite positions, constellation-specific handling
- **Atmospheric**: ionospheric and tropospheric delay columns available but not fully validated in any run
- **Per-satellite features** (for re-WLS): CN0, elevation, constellation flag, ISRB — grounds correction in signal physics
- **WLS baseline quality**: AccuracyMeters as sample weight; WLS baseline errors range 2.1–5.5 m across 151 trips
- **Temporal context**: Savitzky-Golay or Kalman smoother captures trajectory continuity; independent-epoch features alone insufficient for 95th percentile

## Tuning insights
- **Huber loss** (α=0.9 for LightGBM, default for XGBoost) handles pseudorange outliers better than MSE
- **LightGBM quick-test** with n_estimators=10 gave 3.645 m; full 2000-tree run never executed — potential unrealized gain
- **Kalman filter process noise** is the dominant tuning lever for the physics-based pipeline; constant-velocity model breaks at stops/turns; adaptive noise based on velocity magnitude helps
- **Kalman innovation covariance** as sample weight for ML training improved final 10% of performance
- **Regularization critical for small dataset**: Candidate 4's collapse (2.78→12.99 m) traced to weak regularization + CV leak

## Open questions
Most of these were previously blocked by per-candidate code-budget; the new libraries unlock them.

- **Factor graph / bundle adjustment via `gnss_lib_py`**: it ships with FGO helpers; use those instead of scipy.optimize from scratch. 1st-place used FGO — expected necessary for bronze.
- **Time-differenced carrier phase (TDCP)**: `gnss_lib_py` exposes carrier-phase primitives; cycle-slip detection is the piece that has to be written, but the surrounding machinery is now library code.
- **Particle filter / UKF** via `filterpy.monte_carlo` and `filterpy.kalman.UnscentedKalmanFilter` — already implemented, just parameterize. Good for non-Gaussian multipath spikes in urban canyon.
- **IMU fusion**: `gnss_lib_py` has loose-coupling helpers; the data columns include IMU readings (accel, gyro) — prior runs didn't touch them.
- **Phone-model stratification**: separate models or noise covariance per device (Pixel5 vs. Samsung S20 Ultra have different antenna patterns); only mentioned, never tested.
- **Ensemble Kalman + LightGBM**: combining best physics candidate with best ML candidate — natural combination, never executed.
- **Quantile regression**: directly predict 50th and 95th percentile (actual metric) instead of mean correction + post-hoc smoothing.
- **Distribution shift mitigation**: no run has addressed the fundamental train (LAX-heavy) → test (MTV/SVL-heavy) domain gap; domain adaptation or route-type features needed.
