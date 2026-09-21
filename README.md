# NavTrace

NavTrace is a small C++17 localization experiment for a simulated ground robot.
It combines local GPS positions, wheel speed, and gyroscope yaw rate, then
measures what happens when GPS disappears. It is an offline experimental
estimator, not a safety-qualified navigation system.

M0–M1 implements a six-state EKF, causal dead reckoning, optional GPS innovation
gating, deterministic CSV replay, analytical synthetic truth, independent Python
evaluation, plots, and focused tests. Broader bad-GPS/reacquisition experiments,
held-out seeds, wheel slip, uncertainty calibration, benchmarking, datasets, and
ROS integration are **not implemented**. Stop here for review.

## Setup and run

Requires a C++17 compiler and Python 3.11+. On this host, Python is at
`/Users/ryancheng/.pyenv/versions/3.11.9/bin/python3.11`. From the project root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python -m pytest -q
python tools/run_demo.py --binary build/navtrace_replay --output results/my-demo
```

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
after examining the development results. Seeds 70001–70003 are reserved for
the next milestone and have not been evaluated.

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
Ryan should review the functions and decisions in the
[walkthrough](docs/walkthrough.md); generated code is not evidence of personal
authorship or understanding.

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
