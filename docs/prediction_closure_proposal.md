# Proposed prediction closure check

Status: design proposal only, prepared on 2026-10-17. No implementation or
execution is authorized by this document. The next scope needs Ryan's review.
Tracked in
[issue #4](https://github.com/rzcheng/Navtrace/issues/4).

## Completed milestone

Issues #1–#3 are complete locally and were closed during project-manager
wrap-up on 2026-10-17. The fixed position-moment candidate stops at the boundary
recorded in [the replacement results](position_moment_repaired_results.md).
It will not advance to navigation testing. Local completion and issue closure
do not mean pushed, merged, released, or accepted for production.

The replacement completed 90 distributions, 810 metrics and 90 numerical
reference checks. At the 83021 endpoint, mean discrepancy fell from 1.096 m to
0.034 m, but ellipse mass fell from 45.85% to 22.37%, and reference narrow
variance remained 43.27 times the candidate's. Both the failed first attempt
and successful replacement remain preserved. The existing 167 Python tests,
23 CTest tests and C++ development demo passed during that milestone. They
were not rerun for this design proposal.

## Question and bounded change

Does repeated Gaussian projection during prediction alone discard nonlinear
temporal dependence and produce covariance loss in the implemented candidate?
The proposed work checks this mechanism in a fixed analytical fixture. It
does not attempt to assign a fraction of the recorded 83021 error to it.

Add only `tests/test_prediction_closure.py` and a technical derivation and
validation record in `docs/prediction_closure.md`. Before execution, freeze
this scope, fixture matrix and tolerances in a local commit after independent
review. Keep `tools/position_moments.py` at SHA-256
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

In the stopped candidate, the proposed induction gives the same mean and
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

These are analytical predictions, reviewed by inspection. They have not been
verified by new executable tests in this proposal.

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

The single requested decision is whether to approve this note-and-tests scope.
If approved, implementation can proceed within these bounds without another
scope request. Completion would establish or refute this prediction mechanism
for the fixed fixture; it would not establish a remedy or explain the full
recorded-case discrepancy.
