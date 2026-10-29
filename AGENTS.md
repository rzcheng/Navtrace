# NavTrace instructions

Scope: an offline C++17 six-state planar EKF and deterministic sensor replay,
with independent Python synthetic data/evaluation. Ryan authorized continuing
after M2 review. M3 added event diagnostics, offline consistency,
and local observability under a predefined experiment. M4 is conditional on
an evidence checkpoint and a separately frozen design; do not force a policy.
Preserve all M1/M2 configurations, protocols, and measured failures.
M2 seeds 70001–70003 have now been evaluated; do not reuse them as untouched
validation data for future tuning. Preserve the failures in `docs/m2_results.md`.
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

M3 finished with a documented Path D decision; no M4 policy was implemented.
Read docs/m3_results.md before further changes.
Seeds 81001–81010 and 82001–82030 are consumed; never relabel them untouched.
Preserve original frozen results and the separate absent-outage labeling correction.
Ryan authorized the next task: the matched-model long-outage follow-up in
docs/long_outage_protocol.md. Keep the estimator and tuning fixed; stop after
that study for review. Do not add a recovery policy during this task.
The follow-up is finished; read docs/long_outage_results.md. Seeds 83001–83030
are consumed. Preserve the frozen protocol and measured covariance/gating
failures. Ryan has now authorized the bounded conditional Monte Carlo reference
in docs/conditional_reference_protocol.md. Keep production code and all tuning
fixed, use two consumed recordings, and stop after the comparison for review.
That comparison is finished; read docs/conditional_reference_results.md.
Preserve its frozen inputs, protocol, samples and measured mean/shape differences.
Do not implement a correction or recovery policy without a new reviewed scope.

Ryan approved issues #1–#3 on 2026-10-09: one offline recursive Gaussian
position-moment candidate and comparison against the frozen conditional
reference. Follow docs/position_moment_protocol.md after its freeze. Keep
production C++, all tuning, the gate, reference samples and old results fixed.
Use only consumed recordings 83001 and 83021 and their existing five endpoints.
No new sampling, recovery policy, candidate sweep or navigation experiment.
Stop after the bounded comparison and independent review.

The single position-moment comparison was attempted at 55bb5b1 and failed in
the evaluation tooling. Read docs/position_moment_results.md before further
work. Preserve the failed artifacts and partial unfavorable outputs. Ryan approved
the specific repair/relaunch scope on 2026-10-09: replace only the two metric
projections with direct contraction, add full-size synthetic regression tests,
independently review and validate, then make exactly one replacement launch
with the same protocol, candidate, inputs and budget in a fresh directory.
Follow docs/position_moment_repair_validation.md. Stop for review after that
comparison. Candidate changes and navigation experiments remain unauthorized.

The approved replacement comparison is finished at launch commit 89077fc.
Read docs/position_moment_repaired_results.md. All numerical reference checks
resolved, but severe narrow covariance error remains and ellipse mass worsened
in the selected lockout endpoint. Preserve both attempts and the unchanged
candidate. No additional experiment, candidate change, navigation evaluation,
production change or recovery policy is authorized without a new reviewed scope.

Project-manager wrap-up on 2026-10-17 closed issues #1–#3 after checking
their completed acceptance criteria. The fixed candidate stops here and
will not advance to navigation testing. Issue #4 and
docs/prediction_closure_proposal.md describe a smaller analytical diagnostic.
That proposal is design-only and still needs Ryan's scope approval before
new implementation or execution. Do not treat closure of #1–#3 as approval
of #4, a correction, another recording experiment, or production adoption.

Ryan's coordinator approved issue #4 on 2026-10-17 under Ryan's instruction
to make the pending choices. Follow the separately frozen scope in
docs/prediction_closure.md: one analytical note, 16 fixed synthetic fixtures,
independent review and the required regression checks. The earlier proposal
status is historical. Keep candidate and production code, tuning, prior
protocols and results fixed. Stop at the diagnostic report; no correction,
recovery policy, new recording experiment or navigation trial is approved.

Issue #4 execution is complete; read docs/prediction_closure.md for the
fixed-fixture results and review record. Prediction alone reproduced the
analytical covariance loss, without a candidate change. Preserve the frozen
protocol and all evidence. Stop at this diagnostic boundary: no further
experiment, representation change, correction or recovery policy is
authorized without a separately reviewed scope.

Personal study guides, learning walkthroughs, review questions, resume notes,
and ChatGPT handoffs must live outside this repository, including outside
ignored project folders. Use the sibling `../navtrace-study-notes/` directory.
Keep model specifications, build instructions, experiment protocols, validation
records, and measured results in the repository.

## Writing style

Commit messages should be all lowercase, short, and written in plain language.
Say what changed without packing in technical details or adding formal prefixes.
Examples: `add the first localization demo`, `fix outage labels in the results`.
Use straightforward comments, descriptive names, and easy-to-follow code.
Keep useful math and units in comments where they help explain the code. Do not
make code deliberately sloppy or change behavior just to make it look informal.
