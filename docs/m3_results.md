# M3 results: uncertainty, observability, and rejected GPS return

M3 is implemented and evaluated. **M4 was not attempted.** The evidence explains
measurement lockout, but does not justify choosing an absolute position source
or an automatic reset under the available sensor information. This remains an
offline synthetic experiment, with no real-world validation or safety claim.

## Preserved history and actual execution

Started clean at `32aaf85`, branch `feat/gps-fault-evaluation`, no remote. Work is
on `feat/consistency-and-recovery`. Protocol commit `f63d0f9` preceded all new
seeds; diagnostics commit `6b72d54` and frozen study commit `7c486d2` preceded
reserved evaluation. The full study and final M2 diagnosis ran from clean
`7c486d2`. Core `src/ekf.cpp`, filter.conf, M2 protocol/configuration and original
results are unchanged. The only C++ runtime addition is diagnostic snapshots
and GPS-history timing, not new estimator behavior. The separate linearization
helper is offline and may receive nominal truth; replay never does.

The [protocol](m3_protocol.md) executed exactly **170/170** planned replays:
140 stress runs (10 independent seeds × 7 paired conditions × 2 original EKFs)
and 30 sampled-prior model-matched sanity runs. Zero numerical, output, audit,
or missing-run failures; no exclusions and no undefined covariance statistics.
All 1,660 fixed-time records were present. One worker took **267.815 s** and
retained **361.2 MB**, including losslessly compressed full trajectories and
GPS snapshots. These are experiment costs, not real-time or latency benchmarks.
The three-run development smoke took 4.277 s/10.2 MB; linear
projection was 242 s/578 MB. No study size was changed after observing outcomes.
Seeds 81001–81010 and 82001–82030 are now consumed evidence.

Artifacts (ignored by Git):

- [Original frozen run manifest](../results/consistency-study/manifest.json),
  [summary](../results/consistency-study/summary.csv),
  [paired differences](../results/consistency-study/paired-comparisons.csv),
  [fixed-time statistics](../results/consistency-study/fixed-time-statistics.csv),
  [ensemble records](../results/consistency-study/ensemble.csv).
- [Corrected per-run metric labels](../results/m3-corrected-metrics/metrics.csv)
  and [correction provenance](../results/m3-corrected-metrics/manifest.json).
  The labeling correction below does not change the RMSE, NEES, or recovery
  results reported here.
- [M2 event table](../results/m3-diagnosis/event-table.csv),
  [all GPS audits](../results/m3-diagnosis/all-gps-audit.csv),
  [sensor correction decomposition](../results/m3-diagnosis/sensor-corrections-after-45s.csv),
  [reproduction record](../results/m3-diagnosis/reproduction.json).
- [M2 uncertainty figure](../results/m3-diagnosis/dropout_offset-uncertainty.png)
  and [position ellipses](../results/m3-diagnosis/dropout_offset-ellipses.png).
  M2 seed 70002 is diagnostic consumed data, not new validation.
- Predefined illustrative seed 81001: [early offset](../results/consistency-study/early_offset-uncertainty.png),
  [weak-start clean return](../results/consistency-study/weak_late_clean-uncertainty.png),
  [weak-start position ellipses](../results/consistency-study/weak_late_clean-ellipses.png).
  Every stress condition has its own full-scale uncertainty plot. Symlog NIS/NEES
  plots show all magnitudes, and NIS lines break across GPS absence.

## Reproduction and event-level explanation

Before edits, nominal and early-outage seed 70002/basic+gated produced four
byte-identical output pairs. With instrumentation, **all 12** nominal/early-loss
replays across consumed M2 seeds reproduced trajectory bytes and every original
GPS column exactly. The worst M2 final error remains **48.044 m**, not replaced
by M3's different scenarios. All three M2 gated failure runs still reject all
76 clean GPS fixes from 45–60 s.

Independent Python checked **2,472 GPS events**. Maximum discrepancies were
zero for residual/S, 1.99e-13 for the posterior mean, 1.67e-15 for Joseph P,
and 5.72e-10 absolute for NIS (including large NIS). Pre-snapshots occur after
prediction and any same-time gyro/wheel updates, before the GPS correction.

For M2 seed 70002, directions of the full 2x2 S explain the decisions:

| Event | S eigenvalues m² | Residual projections m | Squared directional contributions | NIS/decision |
|---|---|---|---|---|
| 40 s, first offset fix | 1.20151, 969.36754 | 1.45379, −16.73221 | 1.75904, 0.28881 | 2.04786, accepted |
| 45 s, first clean fix | 0.372240, 0.379092 | signs/basis in event table | 56.43615, 112.58970 | 169.02585, rejected |

The 40 s correction is Δposition=(14.25097,−8.81508) m, Δyaw=−0.54619 rad,
Δrate=−0.0115373 and Δbias=+0.0115373 rad/s. The bad absolute fix does **not**
worsen every state component: yaw error improves +0.42585→−0.12034 rad and bias
error improves −0.0117642→−0.000226881 rad/s. Position remains displaced while
covariance contracts. By 44.8 s position error is 7.3922 m, marginal position
P eigenvalues are 0.01205/0.01830 m², and position NEES is 3380.63. The following
clean observation is inconsistent with that confident displaced prediction.

M2 seed 70001 takes a different route: it accepts no offset fixes. At 39.99 s,
position error is 23.0900 m and position NEES 47.669 despite P eigenvalues
1.09774/951.10462 m². At the 45 s clean return, the narrow S direction contributes
75.04197 of NIS 75.48385; the broad direction contributes only 0.44188. Trace or
largest variance alone hides the mismatch. This is compatible with inadequate
local Gaussian geometry after large nonlinear drift, not proof of a numerical
covariance-collapse bug.

Rejected GPS applies exactly zero state/P correction. For seed 70002 during
45–60 s, the 76 GPS corrections are zero; 1,501 gyro updates contribute summed
absolute yaw corrections 0.03633 rad and position correction path 0.000241 m;
301 wheel updates contribute position correction path 0.37311 m. These are
absolute path sums, not net damage measures. Propagation and scalar updates
continue; rejection prevents an absolute correction rather than repeatedly
injecting a bad GPS correction.

## Observability is separate from covariance consistency

Actual production F matrices were exported on independent analytical nominal
paths. Whitened H Phi D uses process transitions only. Fixed scales are
[1 m,1 m,.1 rad,1 m/s,.1 rad/s,.03 rad/s], rank tolerance 1e-9*s_max.
[Full singular values and weak directions](../results/consistency-study/observability/observability.json)
record both horizons and sensor schedules.

| Nominal motion | All sensors rank, 2/20 s | No GPS rank, 2/20 s | Smallest singular value, all sensors 2→20 s |
|---|---:|---:|---:|
| Stationary | 4 / 4 | 2 / 2 | zero / zero |
| Straight v=1.5 | 6 / 6 | 2 / 2 | 0.04045 → 3.08201 |
| Turning v=1.5, rate=.12 | 6 / 6 | 2 / 2 | 0.04055 → 3.37456 |

Gyro-only rank is 1: rate+bias. Straight moving GPS can distinguish heading and
rate through time-dependent lateral position sensitivity; turning is not
necessary for local rank. Two seconds is still weak. One weak straight-2 s
physical direction is approximately [0,−.01268,.02764,0,−.02761,.02761], which
nearly cancels gyro rate+bias while hiding curvature in heading/position.
Stationary GPS cannot identify heading or the rate/bias difference. No zero-rate
observation is injected from a near-zero wheel measurement. These deterministic
sensitivities ignore process noise and prior covariance and do not establish
global observability or statistical calibration.

At 1.99 s in consumed seeds 70001/70002, rate/bias correlation is about −.97377
and bias std .02850 rad/s. Yaw errors/stds are .08058/.09128 and −.02349/.09119
rad; bias errors −.01938 and −.01176 rad/s. These particular initial errors do
not suggest a restrictive prior, but remain weakly separated. Individual
fixed-prior NEES values are descriptive, not independent calibration trials.

## Predefined 30-run matched sanity ensemble

The independent Python generator samples x0 from the stated Gaussian prior and
uses the same discrete Euler–Maruyama diffusion, with independent process and
sensor streams. Only basic EKF is evaluated here. All statistics are after the
last same-time update; N=30 independent runs at each fixed time, not pooled time.

| Time s | Mean full NEES, d=6 | Mean marginal position NEES, d=2 | Nominal 95% position ellipse coverage |
|---|---:|---:|---:|
| 0 | 4.827 | 1.463 | 30/30 |
| 0.5 | 5.277 | 1.505 | 30/30 |
| 1 | 5.520 | 1.725 | 30/30 |
| 2 | 5.663 | 1.634 | 30/30 |

Approximate 95% mean-NEES references from chi-square(N*d)/N are [4.825,7.301]
for full state and [1.349,2.777] for position. The values lie inside these
references. This is short-horizon sanity evidence, not proof of consistency.
Times share the same runs and are correlated. Thirty realizations and 100%
observed coverage do not establish a true 100% coverage rate. This check changes
both prior sampling and process realization relative to the main stress cases;
it does not isolate prior mismatch versus process mismatch or long-outage effects.

## Ten-run paired stress results

Whole-run RMSE in meters: median [minimum,maximum]. Counts mean meeting the
predefined <=1 m sampled dwell for two seconds, not permanent recovery.

| Condition | Basic EKF RMSE | Gated EKF RMSE | Met dwell: basic / gated (each out of 10) |
|---|---:|---:|---:|
| Nominal | .163 [.135,.198] | .155 [.135,.197] | Not applicable |
| Early clean return | 1.128 [.283,2.549] | 1.128 [.281,2.550] | 10 / 10 |
| Early offset return | 3.097 [2.831,3.779] | 16.921 [11.053,20.285] | 10 / 0 |
| Later clean return | .321 [.158,1.171] | .335 [.159,.937] | 10 / 10 |
| Later offset return | 2.785 [2.688,3.088] | .442 [.150,1.341] | 0 / 10 |
| Weak-start clean return | 3.475 [2.066,5.600] | 3.786 [2.061,17.068] | 10 / 7 |
| Weak-start offset return | 4.171 [3.031,6.075] | 5.520 [2.971,11.056] | 0 / 3 |

Gated RMSE is worse in 10/10 early-offset runs, better in 10/10 later-offset
runs, and worse in 9/10 for each weak-start case. It is worse in 4/10 nominal,
7/10 early-clean and 5/10 later-clean runs. Median differences do not replace
[all paired differences](../results/consistency-study/paired-comparisons.csv).
The worst new final error is 42.853 m (gated early-offset seed 81001).

The gate accepts 212/250 offset fixes after early loss versus 0/250 after later
loss; with weak pre-loss motion it accepts 50/250. Clean rejections are respectively
1661/1760,19/1760,569/1760. In weak-start clean return it rejects 439/2010 clean
fixes and three runs miss the dwell entirely, without any injected GPS fault.
Thus offset acceptance is one failure route, not a necessary cause of lockout.

Available clean-return follow-up is 38 s early-clean,33 s early-offset,20 s
later/weak clean,15 s later/weak offset. Non-recoveries remain null. Early-clean
qualifying starts are 0–.2 s for both methods; early-offset basic starts span
4.70–26.00 s, while all ten gated runs remain unobserved through 33 s. All ten
later-offset gated runs already satisfy the limit at return, confirmed 2 s
later. No mean over only successful runs is used as an overall recovery time.

**Eight runs meet the dwell but finish above 1 m:** weak-clean basic seeds
81001/81010, gated 81004/81007/81008/81009, and weak-offset gated 81002/81006.
This is permitted by the preserved M2 criterion. Do not call those durable
recoveries. Final/max error and traces remain available alongside dwell status.

At fixed 39.99 s, gated weak-start position NEES mean is 443.06 and coverage
0/10, versus later original-path clean NEES 1.99 and coverage 10/10. At 60 s,
early-offset gated NEES mean is 6220.88 and coverage 0/10. These are descriptive
mismatch/fault observations, not formal Gaussian hypothesis-test results.

## Evidence checkpoint and M4 decision

- **Confirmed:** short pre-loss motion weakly informs heading/rate/bias; GPS
  absence removes direct absolute information; accepted offset fixes can produce
  a confident displaced position; repeated gate rejection supplies no GPS
  correction; full covariance direction, not its trace, determines NIS.
- **Supported contributors:** weak pre-loss conditioning and inadequate local
  uncertainty geometry after nonlinear drift. Weak-clean failures show that bad
  GPS is not required. The model's stationary interval supplies little heading
  information even though the prior still returns a finite uncertainty.
- **Not supported as the cause:** a discovered C++ Jacobian, Joseph, timestamp,
  NIS timing, or rejection-update implementation defect. Nor do the inspected
  initial errors support a simply over-restrictive initial prior.
- **Not isolated:** relative contributions of deterministic prior mismatch,
  Brownian process mismatch, abrupt stops/restarts, and large-error linearization.
  A first-order straight-motion position sensitivity has zero x response to a
  small heading perturbation, whereas the exact displacement includes
  L*(cos(delta_yaw)-1); this illustrates nonlinear radial error, but is not a
  quantitative causal decomposition of the curved M2 path.

**Path D: finish M3; do not force M4.** The gate's limitation is established,
but a trustworthy absolute source is not identified by these inputs. Smooth,
consistent GPS can still carry a common offset. An automatic reacquisition policy
would add an operational trust assumption; the study does not establish one.
There is no new method to promote or recommend as default. Original gated/basic
methods remain experimental. No reset, inflation, new noise/prior setting, or
recovery state machine was implemented.

The single next task is a **matched-model 20-second clean-outage Monte Carlo
study with freshly reserved seeds**, comparing marginal covariance geometry to
the independent empirical position-error distribution. Hold the sampled prior
and stochastic process assumptions matched; introduce no GPS faults. This tests
long-horizon nonlinear covariance approximation separately from the present
fixed-prior analytical stress before selecting a recovery intervention.

## Verification, bugs, environment, and commands

Environment: macOS 15.7 arm64, Apple Clang 17.0.0.17000604, C++17, CMake 3.31.6,
Eigen 3.4.0, GoogleTest 1.17.0, Python 3.11.9, NumPy 2.2.6, matplotlib 3.10.3,
pytest 8.3.5, offline SciPy 1.15.3. Dependencies remain project-local/pinned.

- **23/23 CTest and 48/48 pytest** pass in Debug and ASan/UBSan; no skips.
  Project C++ builds pass six warning flags including -Werror. No sanitizer
  findings; three actual study replays ran under sanitizers. Release build and
  three-replay smoke also passed. Smoke pose metrics match across all builds.
- New tests cover independent NEES, singular matrices, directional NIS,
  pre/post timing, marginal covariance, elapsed irregular/gapped dwell,
  Jacobian product order, units, ambiguity, paired generation, prefix causality,
  complete run/timestamp matrices, and preserved failure denominators.
- Mutating marginal position covariance to the full precision block made a
  hand case return 4/3 instead of 1; the regression failed. Reversing process
  Jacobian multiplication also failed its regression. Both were restored.
- Final review found a real **offline labeling bug**: no-outage nominal/matched
  rows reused whole-run values under outage_* names. A new actual-CLI regression
  failed (201 matched samples versus expected zero), then passed after fix
  commit `089ba4f`. Fifty rows are relabeled in the separate corrected artifact:
  outage_samples=0 and other outage fields=null. The original frozen metrics
  and manifest remain intact. Every other field was verified identical; no
  reserved-seed replay was repeated or retuned for this correction.
- Python reports **220 third-party matplotlib/pyparsing deprecation warnings**
  when the new plots are exercised; none are suppressed. No numerical warnings.
  Hosted CI and Linux execution remain unverified. CI includes development smoke
  only, not evaluation seeds.
- Audit verified 600 retained artifact hashes,49 source hashes and340 decompressed
  output hashes; binary/helper hashes and paired input paths agree. Source hashes
  identify frozen `7c486d2`, before the explicit label fix and this report.
  Logs and audits are in `results/m3-validation/`.

From the repository root, with fresh result directories:

```bash
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python -m pytest -q
python tools/diagnose_failure.py --binary build/navtrace_replay --output results/diagnosis-rerun
python tools/run_consistency_study.py --binary build/navtrace_replay --output results/consistency-rerun
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DNAVTRACE_SANITIZE=ON
cmake --build build-sanitize --parallel 2
UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-sanitize --output-on-failure
UBSAN_OPTIONS=halt_on_error=1 NAVTRACE_BINARY=build-sanitize/navtrace_replay python -m pytest -q
UBSAN_OPTIONS=halt_on_error=1 python tools/run_consistency_study.py --binary build-sanitize/navtrace_replay --output results/sanitizer-rerun --smoke
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel 2
python tools/run_consistency_study.py --binary build-release/navtrace_replay --output results/release-rerun --smoke
```

Rerunning consumed seeds is reproduction, not fresh validation. No remotes were
added, no commits pushed, no real-world data used, and no recovery implementation
is implied by the branch name.
