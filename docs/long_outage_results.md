# Matched-model long-outage results

The unchanged EKF becomes overconfident in a narrow position direction during
a 20-second clean GPS outage, even with matched prior and noise. Gating then
prevents useful clean fixes from correcting some runs. Similar median trajectory
errors hide this failure: four gated runs never meet the settling criterion.
This follow-up adds measurements and evidence, not a localization improvement.

## What ran

Protocol commit `959df37` preceded evaluation. Implementation commit
`0694ea9c4d7255bd5f513f14e49e6f3d7803d5b2` was clean at execution. All
120 planned C++ replays completed: 30 seeds (83001–83030), paired nominal and
dropout inputs, basic and gated EKF. Zero numerical/output failures, unstarted
runs, or undefined full/position NEES samples. These seeds are now consumed.

The [protocol](long_outage_protocol.md) fixes a 32-second run, GPS removal in
[2,22) s, clean return at 22 s, and ten seconds of follow-up. Truth uses independent
Python Euler–Maruyama integration at 10 ms, with an initial state drawn from the
filter prior and matching speed/rate/bias diffusion. Sensor streams are independent;
each dropout recording deletes exactly 100 nominal GPS rows. The six-state filter,
Euler propagation/Jacobian, Joseph update, noise settings and gate are unchanged.
The C++ command receives observations and filter configuration only.

Position RMSE is `sqrt(mean(dx**2 + dy**2))` on the common 100 Hz post-event grid.
Heading differences wrap to [-pi,pi). Outage metrics use [2,22); the last outage
error is measured at 21.99 s. Values below are medians over all 30 seeds; N/A means
no outage. Do not read the tiny median differences as meaningful improvement.

| Case | Method | Whole RMSE (m) | Outage RMSE (m) | End-outage error (m) | Final error (m) | Heading RMSE (rad) |
|---|---|---:|---:|---:|---:|---:|
| Nominal | Basic | 0.204832 | N/A | N/A | 0.100850 | 0.044612 |
| Nominal | Gated | 0.204442 | N/A | N/A | 0.112863 | 0.046904 |
| Dropout | Basic | 1.370987 | 1.722691 | 3.198576 | 0.197046 | 0.201024 |
| Dropout | Gated | 1.370861 | 1.722691 | 3.368554 | 0.230186 | 0.215738 |

| Case | Method | Whole RMSE range (m) | Maximum final error (m) | GPS accepted / rejected | First return accepted | Settled |
|---|---|---:|---:|---:|---:|---:|
| Nominal | Basic | 0.123–0.293 | 0.265 | 4830 / 0 | N/A | N/A |
| Nominal | Gated | 0.122–0.298 | 0.291 | 4783 / 47 | N/A | N/A |
| Dropout | Basic | 0.240–7.735 | 1.594 | 1830 / 0 | 30/30 | 30/30 |
| Dropout | Gated | 0.240–10.482 | 26.544 | 1607 / 223 | 23/30 | 26/30 |

All GPS here is clean, with Gaussian measurement noise; rejection counts are
observations in this experiment, not a general detection-rate claim. Basic and
gated inputs match. Two gated runs differ before dropout because their early
clean measurements were gated; the modes need not have identical outage priors.

Settling means <=1 m for two elapsed seconds on the 10 ms grid within [22,32].
Gated failures were seeds 83019, 83021, 83024 and 83029. Seeds 83021, 83024 and
83029 rejected all 51 fixes from 22–32 s; their final errors were 9.411, 26.544,
and 22.617 m. Seed 83019 accepted 16, rejected 35 and ended at 2.412 m. Basic
seeds 83019 and 83029 met the dwell criterion but ended above 1 m (1.594 and
1.282 m). Thus even the basic method's 30/30 settling result is not durable recovery.

## Error versus reported uncertainty

At each fixed time, the ensemble contains one value per independent seed.
For 30 runs the nominal Gaussian 95% mean-NEES reference is [1.3494,2.7766]
for 2D position, [4.8247,7.3015] for all six states. These are approximate
nonlinear-EKF references; gated results are descriptive. Times are correlated,
and this is not a multiple-testing or uncertainty-calibration certificate.

Selected basic-EKF values, with every predefined point saved in `ensemble.csv`:

| Case / phase | Time (s) | Position mean NEES | Full mean NEES | Inside own nominal 95% position ellipse |
|---|---:|---:|---:|---:|
| Both, post-event | 1.99 | 1.917 | 5.988 | 28/30 |
| Dropout, post-event | 10 | 1.843 | 7.121 | 28/30 |
| Dropout, post-event | 15 | 8.111 | 17.636 | 23/30 |
| Dropout, pre-GPS | 22 | 23.176 | 82.209 | 19/30 |
| Nominal, pre-GPS | 22 | 1.847 | 6.035 | 29/30 |
| Dropout, post-event | 22 | 19.062 | 78.095 | 19/30 |
| Dropout, post-event | 32 | 11.799 | 32.538 | 20/30 |

The nominal basic position ensemble slightly exceeds the reference at 15 s
(2.949); its other fixed-time comparisons, including the separate pre-GPS
sample, lie inside. The dropout mismatch grows substantially before GPS returns,
so corrupt fixes or a GPS correction are not necessary to cause it.

At 22 s pre-GPS, basic dropout mean position NEES splits into **22.279 along
each run's narrow covariance direction and 0.897 along its wide direction**.
The median NEES is 3.228, so large failures strongly affect the mean. The median
heading standard deviation is 0.626 rad. In a post-hoc numerical inspection,
seed 83029 has position variances 0.0518 and 179.4108 m² along its principal
axes, with NEES contributions 211.862 and 0.855. A large major axis does not
make the narrow direction safe. Plotted illustrative traces retain the
preselected seed 83001; it settles and does not represent the worst runs.

The aggregate world-frame geometry can misleadingly look conservative:

```text
basic dropout, 22 s pre-GPS (30 independent samples)
mean position error (m)       [0.4373, 1.7079]
centered sample covariance   [[21.4416, 8.6920], [8.6920, 21.9970]] m²
uncentered second moment     [[20.9181, 9.1491], [9.1491, 24.1808]] m²
mean reported P              [[23.6387, 6.5568], [6.5568, 51.7524]] m²
aggregate whitened ratios    [0.3955, 0.9043]
```

Both aggregate ratios are below one, yet mean per-run NEES is 23.176. There is
no contradiction: averaging differently oriented P matrices loses which error
belongs to which narrow direction. Mean of `eᵀ solve(P,e)` is not a quadratic
statistic computed using mean P. Centered covariance also excludes mean error;
the second moment deliberately retains it. Keep all three comparisons.

This evidence points toward accumulated nonlinear approximation and uncertainty
representation as the next issue to isolate. Matching prior/noise and passing
equation audits do not prove the EKF's distribution is Gaussian or that every
implementation assumption is correct. Positive covariance and low final median
position error are insufficient evidence of credible uncertainty.

## Reproduction and artifacts

Local environment: macOS 15.7 arm64, Apple Clang 17.0.0.17000604, C++17,
CMake 3.31.6, Eigen 3.4.0, GoogleTest 1.17.0; Python 3.11.9, NumPy 2.2.6,
SciPy 1.15.3, matplotlib 3.10.3 and pytest 8.3.5 in `.venv`.

```bash
.venv/bin/python tools/run_long_outage.py --binary build/navtrace_replay \
  --output results/long-outage-study
```

That path now exists; use a new directory to reproduce, not overwrite. The
single-worker Debug study took 118.677 s and recorded 170.260 MB retained
(manifest's count before its final update; this is study cost, not estimator
latency). [Build/test commands and mutation proof](long_outage_validation.md)
record 23 C++ tests, 56 Python tests and a sanitizer smoke. No failures/skips;
260 third-party Python deprecation warnings. Hosted CI was not verified here.

Actual artifacts, kept out of Git:

- [Run manifest](../results/long-outage-study/manifest.json), with all commands,
  source/input/output hashes, compiler and dependency versions. Binary SHA-256:
  `f619bf2a98d03171380949d88e7579c46eaab0f8974c0eab75e89470bc1ca3b2`.
- [Every run's metrics](../results/long-outage-study/metrics.csv),
  [summary](../results/long-outage-study/summary.csv),
  [fixed-time ensemble](../results/long-outage-study/ensemble.csv),
  [geometry](../results/long-outage-study/geometry.json), and
  [return events](../results/long-outage-study/return-events.csv).
- [Trajectories](../results/long-outage-study/trajectories.png),
  [position/heading errors](../results/long-outage-study/errors.png),
  [ensemble NEES](../results/long-outage-study/ensemble-nees.png), and
  [pre-GPS geometry](../results/long-outage-study/return-geometry.png).
  All four figures were visually inspected; trajectory and geometry axes use
  equal spatial scales, and plots retain full error ranges.
- `seed-*/inputs/` contains paired observations, separate truth, annotations and
  generator provenance. Full C++ state/P and GPS diagnostics are `.csv.gz`.
- [Independent artifact review](../results/long-outage-validation/artifact-review.json):
  verified all 409 artifact hashes, 240 uncompressed-output hashes, and exact
  observation pairing for all 30 seeds. No original M1–M3 artifacts changed.

## Stop and next task

No estimator bug was exposed in this task; the deliberately introduced offline
centering bug was caught and removed. No recovery mechanism, resets, covariance
inflation, retuning or new dependency was added. Sources changed are the study
runner, geometry/plot tools, focused tests, CI smoke and documentation. The
protocol and tested implementation were committed before consuming seeds;
the results report is a subsequent local commit on `feat/long-outage-study`.

The single next task is an independent **offline conditional Monte Carlo
prediction reference** through the outage, starting from a saved Gaussian
posterior and conditioning on the same wheel/gyro observations. Freeze sampling
and convergence checks first. This can isolate how the EKF approximates the
nonlinear distribution before changing production covariance or GPS recovery.
No implementation of that task is included here. Stop for Ryan's review.
