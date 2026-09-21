# NavTrace instructions

Scope: an offline C++17 six-state planar EKF and deterministic sensor replay,
with independent Python synthetic data/evaluation. Ryan authorized continuing
after M0–M1 review. Current milestone M2 is GPS fault/return evaluation on
reserved seeds, with a frozen protocol and unchanged estimator parameters.
Stop after M2 for review; no automatic reset/inflation or other roadmap work.
No ROS, hardware, SLAM, web app, proprietary data, or other project changes.
Do not expand scope without Ryan's approval.

Build/test (activate `.venv` first):
`cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug`,
`cmake --build build --parallel 2`,
`ctest --test-dir build --output-on-failure`, `python -m pytest -q`.
Demo: `python tools/run_demo.py --binary build/navtrace_replay --output results/first-demo`.
M2: `python tools/run_fault_study.py --binary build/navtrace_replay --output results/gps-fault-study`.
Use `--development-smoke` to validate the study tooling without consuming held-out seeds.
Sanitizers: configure `build-sanitize` with `-DNAVTRACE_SANITIZE=ON`.

Use meters, seconds, radians, CCW heading from world +x, body forward signed
speed. Timestamps are integer microseconds; simultaneous order is gyro,
wheel, GPS. Never sort delayed inputs inside replay. Never give the engine
truth, fault labels, actual bias, future observations, or outage schedules.

Math changes require independent analytical/finite-difference tests, PSD and
symmetry checks, CTest, pytest, and a real C++ demo. Do not weaken tolerances,
clip eigenvalues, cherry-pick seeds, or invent improvements to pass checks.
Keep Eigen as the numerical dependency and GoogleTest as the C++ test framework.
Keep Python dependencies local; keep generated artifacts out of Git by default.
