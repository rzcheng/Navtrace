# Conditional reference validation and launch decision

Before the full diagnostic, CTest passed **23/23** cases and pytest passed
**71/71** tests in 11.51 s with `RuntimeWarning` treated as an error. No skips.
The 260 matplotlib/pyparsing deprecation warnings are unchanged third-party
warnings. C++ was not changed; sanitizers were not rerun for this Python-only
milestone. Hosted CI and Linux execution are not claimed.

Fifteen new Python tests cover correlated six-state initial sampling, a
hand-computable Gaussian measurement update and normalized log density,
independent joint Gaussian conditioning across two times, analytical nonlinear
position moments without observations, zero diffusion, equal timestamps,
circular angles, invalid data, budget termination, and prefix causality.
A further analytical example conditions both initial x and past speed on a
later wheel measurement before integrating position. Boundary tests exclude
the already-used 1.99 s event and the returning GPS correction. Comparison
tests separate own-mean covariance from EKF-centered second moment, check
marginal-P ellipse mass, and retain failed/missing convergence entries.

Mutation: set the initial-position conditional gain to zero, removing its
correlations with latent states. The full correlated-prior sampling test fails;
restoring the source makes it pass. Logs and the restored SHA-256 are in
`results/conditional-reference-validation/`. No mutation remains in code.

The local NumPy 2.2.6 matrix-multiply path emitted divide/overflow/invalid
warnings on a moderate 4x4 matrix times finite standard-normal values, while
returning finite results. Direct `einsum` contraction returned identical values
without warnings. The sampler uses that direct contraction for its 2–4 row
sample arrays; warnings are not suppressed to pass tests. This is a local
numerical-library observation, not a demonstrated defect in NavTrace's C++ EKF.

## Pilot-based launch decision

Three retained tooling pilots use the same fixed recording 83001, N=512,
reference seed 20260922 and five report times. The first measured cost; a
second checked figure margins; the final pilot includes improved budget-failure
accounting and the last analytical test. No distribution settings or convergence
thresholds changed between them. Earlier pilot directories remain intact.

The authoritative pilot is `results/conditional-reference-pilot-approved`:
all five distributions completed, wall time **0.819 s**, peak RSS **163.625 MiB**.
The conservative estimate for the full 18-run study is about **57.5 s**, below
the frozen **600 s / 512 MiB** limits. Its latent means/covariances agree with
the saved C++ moments within 8e-15. Launch the frozen 512/2048/8192 counts,
three independent reference seeds, two cases; do not expand if unresolved.
The pilot's Monte Carlo outputs are not treated as held-out evidence.

```bash
.venv/bin/ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest -q -W error::RuntimeWarning
.venv/bin/python -W error::RuntimeWarning tools/run_conditional_reference.py \
  --pilot --output results/conditional-reference-pilot-approved
.venv/bin/python -W error::RuntimeWarning tools/run_conditional_reference.py \
  --pilot-results results/conditional-reference-pilot-approved \
  --output results/conditional-reference-study
```

Use new result paths on reproduction; the runner refuses overwrites. The
full diagnostic must start from a clean committed implementation. Baseline
artifacts and production source hashes are checked by the runner before use.

## Completed diagnostic and artifact audit

The full study ran from clean commit `44a1d19` on `feat/conditional-reference`:
18 runs, 90/90 distributions, zero failures or unstarted distributions.
Wall time was **9.242 s**, peak RSS **192.578 MiB**; retained output is
16,439,659 bytes. All **70** predefined stability/precision comparisons resolved;
none were inconclusive. The largest stability-limit fraction was 0.522 and the
largest precision-limit fraction was 0.769. See the
[results report](conditional_reference_results.md) for interpretation.

The post-run audit verified 108 reference artifact hashes, 63 source hashes
against the recorded commit, all 409 unchanged baseline artifact hashes, and
both actual 2,402-observation input prefixes. The original baseline manifest
also remains unchanged. All three saved plots were visually inspected.
The audit is saved separately in
`results/conditional-reference-validation/artifact-review.json`; frozen study
outputs were not rewritten. Only documentation changed after the full run.
