# Prediction closure diagnostic

Approved scope for [issue #4](https://github.com/rzcheng/Navtrace/issues/4),
2026-10-17. Ryan instructed his coordinator to make the pending choices; the
coordinator explicitly approved the analytical note and 16 fixed synthetic
tests described in [the proposal](prediction_closure_proposal.md). That
approval includes independent review and the repository validation checks.
It does not authorize a correction, recovery policy, real-recording experiment,
or navigation trial. The previous position-moment candidate remains stopped.

The sections through "Exclusions and decision" are the protocol to freeze
before implementation. Validation evidence will be appended afterward, without
changing this scope or its numerical bounds. The starting integration commit
is `9a23f71`; the original checkout remains at `c834bf3`.

## Question and bounded change


Does repeated Gaussian projection during prediction alone discard nonlinear
temporal dependence and produce covariance loss in the implemented candidate?
This work checks this mechanism in a fixed analytical fixture. It
does not attempt to assign a fraction of the recorded 83021 error to it.

Add only `tests/test_prediction_closure.py` and this technical derivation and
validation record. Freeze this scope, fixture matrix and tolerances in a local
commit after independent review and before implementation or execution. Keep `tools/position_moments.py` at SHA-256
`7bd81a4fdd92296919f7aaf2d1b32cacdbe4d1e7d880a5168eb2697e7e3d8104`.
The parent owns the protocol, integration, results and Git operations; one implementation subagent
owns only the new test file, and an independent reviewer is read-only.

The existing conditioning counterexample in `tests/test_position_moments.py`
shows that a projected Gaussian covariance misses measurement-residual
dependence. This new check isolates prediction, with no observations.

## Analytical prediction to verify

Use state `[px, py, theta, v, yaw_rate, gyro_bias]`. Initially position is zero,
signed speed v is deterministic, turn rate and bias are zero, and
`theta ~ N(0,s)` is the only uncertain component. Diffusion is zero and there
are no observations. Heading therefore stays the same random variable across
all N equal Euler steps, with h=T/N. The exact endpoint is
`[T*v*cos(theta), T*v*sin(theta), theta, v, 0, 0]` for every N.

Define

```
dx = v^2 * (1-exp(-s))^2 / 2
dy = v^2 * (1-exp(-2*s)) / 2
a  = v*s*exp(-s/2)
b  = v^2*s*exp(-s)
```

The exact mean is `[T*v*exp(-s/2), 0, 0, v, 0, 0]`. Its nonzero covariance
entries are `Pxx=T^2*dx`, `Pyy=T^2*dy`, `Py,theta=Ptheta,y=T*a`, and
`Ptheta,theta=s`. The other entries are zero.

In the stopped candidate, induction predicts the same mean and
position/heading cross-covariance, but

```
Pxx = T^2*dx/N
Pyy = T^2*(dy/N + (1-1/N)*b)
```

For x, symmetry gives `Cov(px,theta)=0` after every step. Gaussian projection
turns that zero covariance into independence, losing the nonlinear dependence
of accumulated px on cos(theta). The exact x variance contains diagonal terms
`N*h^2*dx` and cross-time terms `N*(N-1)*h^2*dx`. The recursion retains only
the first part. This is prediction-only loss; its exact mean is insufficient
to establish a correct uncertainty model.

The y formula retains some temporal dependence through `Cov(py,theta)`, so it
would be wrong to say the candidate drops all temporal covariance. For s>0,
its y/heading Schur complement is `T^2*(dy-b)/N`, with
`dy-b=v^2*exp(-s)*(sinh(s)-s)>=0`. Positive semidefiniteness therefore does not
establish accuracy. At s=0, every covariance entry is zero; no variance ratio
is defined. At N=1, the full one-step moments agree.

These are analytical predictions to check against the implementation, not
assumed experimental results. The protocol does not require a favorable
conclusion: retain a contradiction or failed check without changing the
fixture matrix, bounds or candidate.

## Derivation of the retained dependence

For a centered Gaussian heading, E[exp(i*k*theta)]=exp(-k^2*s/2).
Its derivative gives E[theta*sin(theta)]=s*exp(-s/2), while odd symmetry gives
E[theta*cos(theta)]=0. The double-angle identities give dx and dy above.
All increments use the same heading; multiplying the common displacement by
T gives the exact endpoint moments without a discretization error in this
fixture.

Let C_n=Cov(py,theta) after n projected steps. The Gaussian-input identity
Cov(py,v*sin(theta))=v*exp(-s/2)*C_n follows by conditioning py on theta.
Each step adds h*a to C_n, so C_n=n*h*a. Consequently

```
Pyy_(n+1) = Pyy_n + 2*n*h^2*b + h^2*dy
Pxx_(n+1) = Pxx_n + h^2*dx
```

Summing gives the stated candidate covariance. The missing variances are
`T^2*dx*(1-1/N)` in x and `T^2*(dy-b)*(1-1/N)` in y. Position/heading
cross-covariance and the latent marginal remain exact in this special case.
A smaller step therefore need not improve the approximation: for positive s,
the x variance tends to zero as N grows at fixed T, even though its exact
value is fixed. This statement does not recommend changing navigation step
sizes or establish a correction.

## Fixed acceptance and validation plan

Use the full Cartesian product T=0.1 s, v in {-2,+2} m/s, s in {0,0.4} rad^2,
and N in {1,2,4,8}: 16 deterministic fixtures. These are analytical test
parameters, not changes to navigation priors, diffusion, or step settings.

- Check the full six-state mean and 6x6 covariance, not just Pxx. Compare both
  one `predict(T, max_step_s=T/N)` call and N calls with interval T/N,
  each explicitly passing `max_step_s=T/N` to avoid default subdivision.
- Independently integrate the deterministic endpoint map over one shared
  heading with 64- and 128-node Gauss-Hermite quadrature. Compute centered
  products directly; do not reuse the candidate moment identities or import
  the conditional sampler. Check both orders against the exact full moments
  and each other with rtol=0 and absolute tolerance 5e-12 in each entry's units.
- Compare the actual recursion with the predicted full moments at the same
  tolerance. For s>0, also check Pxx_candidate/Pxx_exact=1/N within 5e-12
  absolute error. Require exact zero candidate covariance for s=0; quadrature
  keeps the stated tolerance. Retain the measured disagreement with exact
  endpoint variance as the expected failure mechanism,
  rather than interpreting passing regression tests as a correct estimator.
- Check symmetry and PSD using the existing scale-aware 1e-10 tolerance; no
  eigenvalue clipping, inflation, jitter, or tolerance relaxation. Confirm the
  unchanged latent marginal and signed position/heading cross-covariance.
- In an isolated in-memory mutation, zero only the position rows of the
  displacement cross-covariance before prediction. The full-covariance check
  must detect the mutation for N>1 and s>0. Keeping the position/heading
  cross-covariance while dropping its contribution to y variance can make the
  matrix non-PSD; rejection by existing covariance validation counts as
  detection. Do not bypass that check. If a matrix is returned, the independent
  full-covariance comparison must fail. Restore automatically; no source or
  recorded artifact is modified.
- Obtain independent analytical and final-diff review, run full pytest with
  RuntimeWarning as an error, CTest, and a real C++ development demo in a fresh
  output directory. Inspect its plots and preserve logs/source hashes in a
  fresh ignored validation directory. No C++ change is planned, so sanitizers
  are only relevant if a separately approved C++ change is requested.
- Report all fixtures and any failure without adjusting the frozen matrix or
  bounds. Stop at the diagnostic report; another candidate is not an automatic
  next step.

## Exclusions and decision

No candidate or production code changes, reference reruns, samples, recordings,
recording-specific decomposition, endpoint oracle, tuning, recovery, new
candidate, or navigation evaluation. Existing code, protocols, parameters and
results remain fixed. No new dependency is needed. This is a deterministic
mathematical diagnostic, not a fresh navigation validation set.

A full deterministic endpoint oracle is deferred: its totals largely duplicate
the resolved Monte Carlo moments, while a large exact cross-time term alone
would not quantify what the recursive approximation lost. Any later comparison
would need a separate definition of attribution and a separately reviewed scope.

The completion decision is whether the implemented projection exhibits the
predicted covariance loss in these fixtures. After reporting all evidence and
limitations, stop for review. Neither confirmation nor contradiction authorizes
an estimator change, a new fixture search, or another recording experiment.

## Validation results — 2026-10-17

The 16 fixed fixtures confirm the predicted mechanism in the unchanged
implementation. For positive heading variance, x variance is 1/N of the exact
endpoint value, while mean, position/heading cross-covariance and the latent
marginal remain correct within the frozen bounds. At N=8 this is an 87.5%
variance deficit. Positive semidefiniteness and exact one-step moments did not
prevent the accumulated covariance error.

The protocol was frozen at `c1b4c9a95b42f66b5a399a6a0979465055d0ed71`, with
SHA-256 `2b379c9093be95e4ce12ae0c1e3d81813aedf3d6787f6da33edc37eaadf712a2`,
before test implementation or execution. The frozen text above is unchanged;
this results section was appended after validation.

### All fixed fixtures

All 16 fixtures completed without a failed acceptance check or contradiction.
The rows below cover both signs, both heading variances and every prescribed N.
For s=0.4, exact Pxx is 0.002173777440919 m² and exact Pyy is
0.011013420717656 m² for every N and both speed signs. For s=0, exact covariance
is zero. Complete full means, exact/projected 6x6 covariances, both quadrature
orders, residuals and mutation outcomes are in `fixtures.json`; this table
shows the changing entries, rounded to 12 decimal places.

| v (m/s) | s (rad²) | N | mean px (m) | candidate Pxx (m²) | candidate Pyy (m²) | candidate Py,theta (m·rad) |
|---:|---:|---:|---:|---:|---:|---:|
| -2 | 0 | 1 | -0.200000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| -2 | 0 | 2 | -0.200000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| -2 | 0 | 4 | -0.200000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| -2 | 0 | 8 | -0.200000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| -2 | 0.4 | 1 | -0.163746150616 | 0.002173777441 | 0.011013420718 | -0.065498460246 |
| -2 | 0.4 | 2 | -0.163746150616 | 0.001086888720 | 0.010869270727 | -0.065498460246 |
| -2 | 0.4 | 4 | -0.163746150616 | 0.000543444360 | 0.010797195732 | -0.065498460246 |
| -2 | 0.4 | 8 | -0.163746150616 | 0.000271722180 | 0.010761158234 | -0.065498460246 |
| 2 | 0 | 1 | 0.200000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| 2 | 0 | 2 | 0.200000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| 2 | 0 | 4 | 0.200000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| 2 | 0 | 8 | 0.200000000000 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| 2 | 0.4 | 1 | 0.163746150616 | 0.002173777441 | 0.011013420718 | 0.065498460246 |
| 2 | 0.4 | 2 | 0.163746150616 | 0.001086888720 | 0.010869270727 | 0.065498460246 |
| 2 | 0.4 | 4 | 0.163746150616 | 0.000543444360 | 0.010797195732 | 0.065498460246 |
| 2 | 0.4 | 8 | 0.163746150616 | 0.000271722180 | 0.010761158234 | 0.065498460246 |

For s=0.4, the observed x variance ratios for N=1,2,4,8 were respectively
1, 0.5, 0.25 and 0.125, within 2.23e-16 of those values. The actual candidate
mean differed from the analytical mean by at most 2.78e-17, and its covariance
differed from the analytically predicted recursion by at most 3.47e-18.
The two public-call paths produced identical arrays. All eight zero-variance
fixtures returned exactly zero candidate covariance; their means agreed up to
ordinary floating-point summation.

Independent endpoint quadrature used the same random heading at every step.
The implementation tests use direct NumPy weighted centered products; the
parent's separate collector uses scalar `math.fsum` over the deterministic
endpoint map. It imports no test helper or reference sampler. In that collector,
the largest quadrature covariance error was 1.67e-16, and the largest mean or
between-order difference was 2.23e-16. These numerical checks support the
analytical calculation; agreement between quadrature orders is not a rigorous
integration error certificate.

The specified in-memory cross-covariance mutation was detected for all six
positive-variance, multi-step cases. The tests exercised both call paths
(12 mutation executions); the existing PSD check rejected each on the second
substep. The parent's separate six bulk-path mutations also produced PSD
rejection. No validation was bypassed. Automatic restoration and candidate
source hashes were checked; there was no persistent source mutation.

### Implementation, review and regression evidence

The implementation subagent owned only `tests/test_prediction_closure.py`.
Its focused run passed 22 tests: 16 fixed fixtures and six mutation tests.
The parent inspected that file and independently derived the covariance
recurrences, collected every numerical row, checked the final source scope,
and ran integration validation. An independent mathematical reviewer accepted
the protocol before its freeze and the completed test diff afterward. The
reviewer independently reran the 22 focused tests with RuntimeWarning as an
error (22 passed in 0.14 s), checked the source hashes, and reviewed the separate
collector and all saved outcomes. The reviewer also checked every report row,
residual maximum, hash, provenance statement and final stop boundary. No
mathematical, implementation or reporting blocker remained.

Commands used the original checkout's project-local Python/CMake environment.
The saved command list records absolute executable paths and return codes.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python -W error::RuntimeWarning -m pytest -q
python -W error::RuntimeWarning tools/run_demo.py \
  --binary build/navtrace_replay \
  --output results/prediction-closure-demo-20261017
python -W error::RuntimeWarning \
  results/prediction-closure-checks-20261017/check_fixtures.py
```

Debug configure/build succeeded, 23/23 CTest cases passed, and full pytest
passed 189 tests in 16.71 s without warnings. No C++ source changed, so no new
sanitizer run was needed. No hosted CI run is claimed.

The fresh C++ development demo completed all six nominal/dropout replay runs
using development seed 20260921. Its manifest records a clean launch at
`c1b4c9a`, before the new test file was written. Both trajectory and error plots
were inspected. The existing EKFs track the development trajectory, with
outage errors and dead-reckoning drift visible. The gated dropout end-window
error is 0.375 m versus 0.221 m for the basic EKF. This is regression evidence,
not a candidate navigation evaluation or a new validation seed.

After execution, all 86 files tracked at the protocol freeze still matched
their saved hashes. All 541 files across the four preserved study directories
also matched: 410 long-outage files, 109 conditional-reference files, and 11
files for each position-moment attempt. These counts include manifests and
marker files; the original two studies contain 517 artifacts excluding their
two manifests. All 59 source hashes and 24 artifact hashes in the new demo
manifest matched. Only this report and the administrative AGENTS boundary are
updated afterward, in addition to the new test file.

### Artifacts and limits

The fresh ignored evidence directory is
`results/prediction-closure-checks-20261017/`. It contains `frozen-protocol.md`,
`preservation-before.json`, `preservation-after-execution.json`, `commands.json`,
build/CTest/pytest/demo logs, `check_fixtures.py`, `fixtures.log`, and
`fixtures.json`. The demo and its figures are in
`results/prediction-closure-demo-20261017/`.

Key SHA-256 values:

```
tests/test_prediction_closure.py
  a43f8f81a63fbaa91f8e776b760ab7a20546e0249cc1bb613b2ba9be2e430964
fixtures.json
  aff1235723a64bb9f7817c463ece07d34fadcdbf6c0e78dd36be766b9c7a2530
check_fixtures.py
  621936ec8b9749cf41a3541b1727a7e06ad1e13d116b870a1d6935410ab99de4
```

This establishes a prediction-only mechanism in the prescribed constant-speed,
constant-heading model. It does not assign any fraction of recording 83021's
error to that mechanism, establish calibration, or supply an online remedy.
No frozen recording was replayed, no reference samples were recomputed, no
navigation parameters were changed, and no candidate was added or modified.
The smaller-step result is a warning about this approximation, not a proposed
navigation setting.

The diagnostic stops here. A useful next decision is whether to commission a
separate design-only review of a representation that retains nonlinear
position/heading dependence. That review would need to produce a concrete
candidate and bounded validation scope before any new implementation; this
milestone authorizes neither. A full recording endpoint oracle remains deferred.
