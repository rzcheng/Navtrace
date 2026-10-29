# NavTrace

NavTrace is a small C++17 localization experiment for a simulated ground robot.
It combines local GPS positions, wheel speed, and gyroscope yaw rate, then
measures what happens when GPS disappears. It is an offline experimental
estimator, not a safety-qualified navigation system.

M0–M1 implements a six-state EKF, causal dead reckoning, optional GPS innovation
gating, deterministic CSV replay, analytical synthetic truth, independent Python
evaluation, plots, and focused tests. M2 adds a fixed GPS fault/return study
with reserved seeds, paired comparisons, GPS rejection accounting, and an
explicit position-settling criterion. Wheel slip, automatic recovery logic,
uncertainty calibration, benchmarking, datasets, and ROS integration remain
**not implemented**. Later diagnostic studies below preserve the same estimator
and tuning. The conditional outage reference is finished and awaiting review.

## Setup and run

Requires a C++17 compiler and Python 3.11. The full Python suite also uses
the pinned optional oracle backend on Linux x64 or macOS arm64. From the
project root in a fresh checkout:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --only-binary=:all: --require-hashes --no-deps \
  --target .deps/oracle -r requirements-oracle-ci-lock.txt
export PYTHONPATH="$PWD:$PWD/.deps/oracle${PYTHONPATH:+:$PYTHONPATH}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python -m pytest -q -W error::RuntimeWarning
python tools/run_demo.py --binary build/navtrace_replay --output results/my-demo
```

The isolated backend leaves the original requirements and virtualenv unchanged.
Other platforms need a separately verified wheel; do not skip oracle tests or
fall back to an unpinned source build. An existing research workspace may reuse
its verified isolated backend through `PYTHONPATH` without reinstalling it.
For the already configured workspace, start at `source .venv/bin/activate`.
Use a fresh output directory; the demo refuses to overwrite existing results.
`--overwrite` explicitly replaces only a marked NavTrace results directory.
The C++ executable always refuses existing output files. A failed replay can
leave partial CSVs; the demo's manifest must say `complete` before using results.

Installed Eigen >=3.4 and GoogleTest >=1.17 are preferred. Otherwise CMake fetches
the hash-verified **Eigen 3.4.0** and **GoogleTest 1.17.0** release archives into
the build tree. Downloads require network access on the first configure.
This workspace also caches those archives in ignored `.deps/`. See
[dependency records](docs/dependencies.md). All Python tools, including CMake,
are installed in `.venv`; no system Python changes are needed.

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DNAVTRACE_SANITIZE=ON
cmake --build build-sanitize --parallel 2
UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-sanitize --output-on-failure
NAVTRACE_BINARY=build-sanitize/navtrace_replay UBSAN_OPTIONS=halt_on_error=1 python -m pytest -q
```

## Inputs, outputs, and ownership

The default development recording lasts 60 s: gyro 100 Hz, wheel 20 Hz, GPS
5 Hz, with a constant simulated gyro bias of 0.015 rad/s. The matching dropout
file removes GPS during `[20,40)` s from the same noise realizations. These are
simulation choices, not claims about a real robot. No parameters were changed
after examining the development results. Seeds 70001–70003 were reserved from
M1 for the predefined [M2 protocol](docs/m2_protocol.md).

`scenarios/development.json` is generator-only. `scenarios/filter.conf` contains
the engine's initial prior and process/gate configuration. The engine receives
neither truth nor the actual bias, fault annotations, or outage schedule.
Both methods receive the same initial pose/speed prior. See [model](docs/model.md)
for frames, units, covariance assumptions, and observability limitations.

To invoke replay directly:

```sh
build/navtrace_replay sensors.csv scenarios/filter.conf ekf estimates.csv gps-diagnostics.csv
```

The method is `dr`, `ekf` (gate off), or `gated` (gate on, same noise settings).
Input CSV is strict, unquoted, LF-terminated, with this header:

```text
timestamp_us,sensor,value1,value2,cov11,cov12,cov22
```

Timestamps are integer microseconds. Same-time order is gyro, wheel, GPS;
duplicates and delayed events fail. GPS uses two position values and a symmetric
2x2 covariance. Scalar sensors use value1/cov11 and zero for other fields.
Each output row follows one event; evaluation takes the last simultaneous row
at each truth/gyro timestamp, without interpolation. Full EKF covariance and
minimum-eigenvalue/symmetry diagnostics are saved; DR covariance fields are blank.

The estimator, replay, generator, and evaluator are project implementations
created with Codex assistance. Eigen provides matrix algebra/decompositions,
GoogleTest provides C++ tests, NumPy supplies random generation/array operations,
and matplotlib draws the plots. No existing EKF/localization library is used.

## Measured development results

The real C++ demo artifacts are local and ignored by Git:
[metrics](results/first-demo/metrics.csv), [trajectory plot](results/first-demo/trajectories.png),
[error plots](results/first-demo/errors.png), and [manifest](results/first-demo/manifest.json).
Re-run the command above to create these artifacts on another checkout.
The manifest records versions, source/input/output/executable hashes, build
configuration, seed, Git revision/dirty state, and actual commands.

Whole-run position RMSE (meters):

| Recording | Dead reckoning | Basic EKF | Gated EKF |
|---|---:|---:|---:|
| Nominal | 15.939 | 0.115 | 0.132 |
| GPS dropout | 15.939 | 0.131 | 0.175 |

Gating was worse here: an ordinary noisy fix at 12.6 s exceeded the threshold.
The constant-bias, no-slip simulation is favorable to this model; one development
seed does not establish general performance. See [validation](docs/validation.md)
for full results, tests, and limitations. A macOS/Linux CI workflow is provided;
creating it is not evidence that hosted CI ran.

## GPS fault study (M2)

The [frozen protocol](scenarios/gps_faults.json) prescribes nominal GPS, dropout,
three isolated spikes, a sustained offset, and early dropout followed by offset
GPS on return. All cases for a seed use the same noise realizations. The runner
checks the M1 filter configuration hash, refuses existing result directories,
and never supplies fault annotations or truth to C++.

```sh
# Validate the tooling with the already-used development seed.
python tools/run_fault_study.py --binary build/navtrace_replay \
  --output results/fault-development --development-smoke
# Evaluate all three reserved seeds: 5 cases x 3 methods x 3 seeds = 45 replays.
python tools/run_fault_study.py --binary build/navtrace_replay \
  --output results/gps-fault-study
```

Outputs include `metrics.csv`, `summary.csv` (median/min/max across seeds),
`paired-comparisons.csv`, and `recovery.csv`, plus all C++ outputs, annotations,
plots, and a hashed manifest. Position settling requires error <=1 m for a
full 2 s after clean GPS returns. Acceptance delay and settling delay are
separate metrics. Failure to settle is saved explicitly with a null delay;
it is never averaged away as zero. See the protocol for observation windows
and the distinction between already within the limit and later recovery.

Use new seeds for future validation once these are inspected. The three-seed
study is evidence for these specified synthetic cases, not a general robustness
or detection-rate claim. CI runs the development variant only.

The [measured M2 report](docs/m2_results.md) records the completed 45-replay
experiment. Median position RMSE for basic/gated EKF was 0.354/0.201 m for
spikes and 5.697/0.378 m for sustained offsets. Early dropout with offset GPS on
return was worse with gating: 5.964/6.775 m, including a gated run at 22.000 m.
All three gated runs rejected every clean fix from 45–60 s in that case.
No parameters were retuned or recovery failures removed from the report.
See the [actual comparison plot](results/gps-fault-study/position-rmse.png)
and [return records](results/gps-fault-study/recovery.csv). Seeds 70001–70003
have now been evaluated and are no longer untouched validation data.

## M3 diagnosis and consistency study

M3 adds explicit GPS pre/post snapshots and offline full-state/marginal-position
NEES, nominal ellipse coverage, event-direction NIS, and local process sensitivity.
It keeps the estimator equations and all M1/M2 parameters unchanged. The
[predefined protocol](docs/m3_protocol.md) separates a sampled-prior, discrete
model-matched sanity ensemble from analytical, mismatched/faulted stress cases.

```bash
python tools/diagnose_failure.py --binary build/navtrace_replay --output results/m3-event-diagnosis
python tools/run_consistency_study.py --binary build/navtrace_replay --output results/consistency-study
```

Both commands require fresh output paths. Add `--smoke` to the study command for
three development replays without consuming reserved evaluation seeds. Install
updated `requirements-lock.txt` first; SciPy is an offline analysis dependency.
The C++ `navtrace_linearize` helper uses nominal paths only for offline analysis;
those paths are never passed to the replay estimator. Full raw study CSVs are
retained losslessly compressed. See [diagnostic conventions](docs/diagnostics.md).

The [measured M3 report](docs/m3_results.md) records 170 replays and preserves the
M2 failure. With early offset returns, median basic/gated RMSE was 3.097/16.921 m
and 10/0 of ten runs met the two-second settling criterion. With later offset
returns, the corresponding results were 2.785/0.442 m and 0/10. Three weak-motion
runs also failed to settle with entirely clean GPS. The short matched-model
sanity ensemble passed its approximate reference checks; that does not establish
long-outage consistency. **No recovery policy was added.** The M3 report records
the failure evidence and remaining limitations.

Historical reports retain their original commit IDs. After the commit-message
cleanup, use the [commit map](docs/commit-history.md) to locate the same source
snapshots in the renamed history.

## Long GPS outage follow-up

The [frozen experiment](docs/long_outage_protocol.md) tests uncertainty through
a clean 20-second GPS outage with prior, process noise and sensor noise matched
to the filter. It compares paired nominal/dropout recordings across 30 fresh
seeds, using the unchanged basic and gated C++ EKF. No recovery policy is added.

```bash
python tools/run_long_outage.py --binary build/navtrace_replay \
  --output results/long-outage-development --development-smoke
python tools/run_long_outage.py --binary build/navtrace_replay \
  --output results/long-outage-study
```

Both commands refuse existing output folders. Outputs include `metrics.csv`,
`summary.csv`, `ensemble.csv`, `geometry.json`, `return-events.csv`, full compressed
C++ trajectories/GPS diagnostics, paired inputs, truth, and a hashed manifest.
`trajectories.png`, `errors.png`, `ensemble-nees.png` and `return-geometry.png`
show actual estimates and independent errors. Aggregate error covariance and
mean reported covariance are geometric summaries; their agreement would not
by itself establish calibrated conditional uncertainty.

The [measured long-outage report](docs/long_outage_results.md) records 120 replays
without numerical/output failures. Basic/gated median outage-recording RMSE was
1.371/1.371 m, but only 26/30 gated runs met the settling criterion; three rejected
every clean returning fix. Basic position mean NEES reached 23.176 just before
return, versus the approximate reference [1.349,2.777]. See the actual
[uncertainty plot](results/long-outage-study/ensemble-nees.png) and
[error geometry](results/long-outage-study/return-geometry.png). Seeds 83001–83030
are consumed. The estimator and tuning remain unchanged; no recovery policy
has been added.

## Conditional outage reference

The [reference protocol](docs/conditional_reference_protocol.md) isolates the
distribution after GPS loss in two consumed recordings. An independent Python
linear-Gaussian path sampler conditions on the actual wheel/gyro observations,
then integrates position along each sampled path. It retains the full checkpoint
covariance and stops before GPS returns. Production C++ and tuning stay fixed.

```bash
python tools/run_conditional_reference.py --pilot --output results/reference-pilot
python tools/run_conditional_reference.py --pilot-results results/reference-pilot \
  --output results/reference-comparison
```

Use fresh output folders. This consumes the frozen local
`results/long-outage-study` artifacts; `--study PATH` selects another copy.
To reconstruct that baseline, run its documented experiment at commit `0694ea9`
in a separate checkout, then supply its results path. Do not reset current work.
Historical study commits such as `0694ea9` and their frozen generated inputs
are preserved in the original local research checkout. The published
integration starts from a reviewed source snapshot; it does not publish
that earlier local ancestry or the ignored study artifacts. A fresh public
clone can build and run the development tests, but cannot reproduce these
saved studies from the checkout alone. The historical commands above require
the preserved baseline checkout and inputs. Their hash checks remain strict.

The runner verifies baseline hashes, performs a bounded pilot, and retains
samples, distribution comparisons, convergence diagnostics and three plots.
Monte Carlo samples for one recording are not independent navigation trials.

The [measured reference report](docs/conditional_reference_results.md) records
18 runs, 90 endpoint distributions, and all 70 predefined numerical-resolution
checks passing. Immediately before GPS returns, the non-lockout/lockout cases
have reference-to-EKF mean differences of 0.266/1.096 m. Reference variance along
the EKF's narrow direction is 1.98/48.16 times the reported variance; only
84.92%/45.85% of samples fall inside its nominal 95% position ellipse. These are
conditional-distribution diagnostics for two consumed recordings, not empirical
coverage or evidence of a better navigation system. The
[position samples](results/conditional-reference-study/position-shape.png) show
the curved distribution that a single ellipse does not fully describe.
No production change follows automatically from this comparison.
