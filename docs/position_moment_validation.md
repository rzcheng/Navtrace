# Position moment validation

This records validation for issues #1–#3 under the frozen
[protocol](position_moment_protocol.md). The protocol commit is `4c5f843`.
The saved-reference comparison is separate from these implementation checks.
No production C++, tuning, reference sampler, old protocol or old result changed.

## Independent checks

The candidate worker implemented only `tools/position_moments.py` and its tests.
A separate mathematical reviewer checked the formulas, full covariance terms,
Joseph update, discrete timing and causal reports without reading frozen samples.
The coordinator inspected the implementation and integration diff directly.

- The 66 focused candidate tests passed with RuntimeWarning treated as an error.
  Conditional-yaw numerical quadrature independently integrates the full six-state
  mean and covariance without reusing the characteristic-function formulas.
  Across its signed/zero-speed and yaw-variance fixtures, maximum mean and
  covariance discrepancies were 2.22e-15 and 4.44e-16.
- A temporary mutation dropped A from A*G*B.T. Six independent quadrature/cross
  checks failed. Restoring the identical module bytes restored all 66 passes.
- The independent reviewer compared 32 synthetic fixtures with the raw complex
  identities: maximum mean/covariance differences were 2.22e-16 / 1.11e-16.
  An independent linear latent recursion agreed within 1.74e-17 / 8.33e-17.
- The coordinator separately compared the full Gaussian-input formulas with
  64-node conditional-yaw integration for a correlated signed-speed fixture:
  maximum mean/covariance differences 8.88e-16 / 5.11e-15, with minimum covariance
  eigenvalue 0.0093724. This was a synthetic check, not a recording comparison.
- A known nonlinear conditioning example demonstrates the approximation:
  exact conditional position covariance depends on a later wheel residual;
  the projected Gaussian covariance update does not. Passing moment tests
  therefore does not prove exact recursive conditional inference.

The initial yaw-shift test used a 2e-15 absolute yaw bound. Eight unwrapped
additions produced a 2.66e-15 rounding difference. Before recording evaluation,
only that yaw assertion was replaced by `(2*n+3)*eps*M`, accounting for the two
n-addition recursions, initial 2*pi shift and comparison subtractions. Position
and covariance bounds stayed at 2e-15; frozen production and comparison
thresholds stayed unchanged. The initial failure and bound are retained in
`results/position-moment-validation/validation.json`.

## Comparison tooling review

A different worker implemented the runner, position metrics, plots and tests.
An independent read-only reviewer checked definitions, historical source and
artifact audits, candidate/input separation, all 90 planned rows, 810 metric
entries, 90 resolution outcomes, and budget/failure accounting.

Review found an invalid-candidate serialization bug: nonfinite arrays were
stored before validation and could prevent writing a failure manifest. Validation
now precedes storage; regression tests retain all 720 unaffected metric entries
when every candidate snapshot fails. A second review caught a CLI label treating
an existing-output refusal as a newly started failed comparison; the final
regression covers both stages.

Synthetic tests use deterministic arrays and a stub candidate. They check
independent metric arithmetic, full-sample versus batch covariance, unbiased
own-mean covariance versus displaced second moment, mixed covariance, pooled
MCSE, marginal ellipses, complete failures/unstarted entries, all input guards,
historical source hashes, and actual rendering of all four plots. All ten
spatial panels retain full sample ranges and equal axes.

A budget stop during whole-case candidate replay has no partial snapshot return:
the active case is failed and later unavailable reports remain not_started.
No metric computation resumes after a budget stop. This limitation is explicit;
it does not permit a retry or a larger budget.

## Integration evidence

Checks run from the integration worktree using the existing project-local
Python environment and unchanged Debug C++ build:

```
ctest --test-dir build --output-on-failure
python -m pytest -q -W error::RuntimeWarning
python tools/run_demo.py --binary build/navtrace_replay \
  --output results/position-moment-development-demo
```

Final focused evaluator tests passed 27/27, also rerun by its independent reviewer.
The final integrated pytest run passed 164/164 in 17.20 s with RuntimeWarning
treated as an error. The 260 existing third-party deprecation warnings remain.
All 66 preserved baseline source hashes matched before launch.

CTest passed 23/23. The first shell invocation did not find ctest on PATH;
using the existing environment's absolute executable resolved that tooling error.
The actual C++ development demo completed all six method/scenario runs. Gated
position RMSE was 0.132119 m nominal and 0.174847 m dropout; ungated values were
0.114672 m and 0.130551 m. This checks the unchanged production path; it is not
candidate performance. Both demo figures were inspected.

Logs are under `results/position-moment-integration-checks/`. Mutation and restored
candidate test logs are under `results/position-moment-validation/`. Generated
artifacts remain ignored. No sanitizer run was needed because no C++ code changed.
No hosted CI execution is claimed.

## Actual comparison outcome

The single launch at `55bb5b1` failed in the metric tooling despite these passing
small-fixture checks. Only N=512 rows completed; all medium/high rows raised a
local large-thin-matmul error. Read [position_moment_results.md](position_moment_results.md)
for full failure accounting, independently reproduced synthetic diagnosis,
partial unfavorable results and the required review before repair/relaunch.
This validation record is not a claim of a completed comparison.
