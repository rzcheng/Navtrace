# Observation-conditioned angular paths

Ryan approved the staged design and deterministic prototype in
[issue #9](https://github.com/rzcheng/Navtrace/issues/9) on 2026-10-18, after
accepting the reporting correction in #8. This note fixes one received-prefix
proposal before implementation. It authorizes no stochastic study, new recording,
reference comparison or production change. Existing designs and no-go results
remain frozen. The starting commit is `383a026`.

## Decision and limits

Use the exact Gaussian posterior of the active angular **primitive coordinates**
for a received observation prefix, then retain the exact conditional Gaussian
position/speed law for each supplied angular path. Rebuild from the checkpoint
for each prefix. This is a bounded batch diagnostic, not a persistent streaming
filter. It neither resamples an old ensemble nor multiplies by its old weights.
Previously returned reports remain unchanged.

A local gyro-adapted transition was considered algebraically and rejected for
this milestone: conditional on a fixed past angular path, the new angular noise
is independent of speed and the next wheel observation. Such a proposal cannot
correct the inherited angular/speed mismatch. Conditioning the whole received
prefix includes those checkpoint correlations. Wheel information can also move
new angular innovations indirectly after gyro conditioning; fixing the old path
would remove that route.

The exact proposal has constant algebraic importance ratios. This does not prove
that a finite set of paths gives accurate nonlinear position moments, resolves
the prior study's failures, or is affordable for long recordings. Supplied fixed
coordinates are deterministic fixtures, not independent navigation trials or a
new empirical ESS study. Batch memory and factorization cost grow with prefix
length. GPS, which depends on nonlinear position, is outside this derivation.

## Primitive representation

Use the existing physical split `a=(yaw, yaw_rate, gyro_bias)` and
`ell=(px, py, speed)`, with unwrapped yaw and the full correlated checkpoint.
The read-only `split_checkpoint` helper gives angular/conditional covariances,
active indices and conditional gain K. Construct La and J by Cholesky on those
returned active covariance blocks, embedded in their physical coordinates:

\[
a_0=\mu_a+L_a U_0,\qquad
\ell_0=\mu_\ell+K L_a U_0+J Z_0.
\]

Columns of La and J span the existing accepted coordinate-active supports.
U0 and Z0 are independent standard normal coordinates. Use the existing
conservative checkpoint support policy; reject unresolved rotated singular or
nearly singular active blocks. Do not introduce a pseudoinverse rank cutoff,
jitter, eigenvalue clipping or covariance inflation.

Append one independent standard coordinate to U for each positive rate or bias
diffusion at each positive substep, and to Z for each positive speed diffusion.
Multiply it by `sqrt(q*h)` in the physical map. Exactly zero diffusion adds no
coordinate. U and Z have independent standard normal priors. Empty blocks are
valid. A tiny positive diffusion remains positive; no new numerical rank choice
is made from its scale. Redundant yaw-path values are outputs, not independent
coordinates to invert.

At every old-state Euler substep,

\[
\theta_{j+1}=\theta_j+h\omega_j,\quad
\omega_{j+1}=\omega_j+\sqrt{q_\omega h}U_{\omega,j},\quad
b_{j+1}=b_j+\sqrt{q_bh}U_{b,j},
\]
\[
p_{j+1}=p_j+h v_j(\cos\theta_j,\sin\theta_j)^T,\quad
v_{j+1}=v_j+\sqrt{q_vh}Z_{v,j}.
\]

Thus angular states and speed are affine in their primitive coordinates.
For fixed U, the complete endpoint ell is affine in Z:

\[
\ell=b(U)+F(U)Z.
\]

Construct b and F using the old angle and old speed at each step, before adding
end-step diffusion. Initial p/speed correlations and position/speed correlations
created across steps stay in F. No integrated process noise or Gaussian position
projection is added.

## Received observations and exact proposal

For the received gyro/wheel prefix, assemble

\[
y=d+H_U U+H_Z Z+\epsilon,\qquad
\epsilon\sim\mathcal N(0,R).
\]

R is the diagonal matrix of the supplied strictly positive independent sensor
variances. Each gyro row selects the current rate plus bias and has a zero Z
row. Each wheel row selects current speed, including the angular checkpoint
term through K and all speed innovations received by its timestamp. In general,
wheel rows of HU are not zero. Integrating Z correlates observation rows; never
replace the resulting covariance with its diagonal.

Let r=y-d and V=R+HZ HZ^T. Then

\[
p(y\mid U)=\mathcal N(r;H_U U,V),\quad
C_U=(I+H_U^T V^{-1}H_U)^{-1},\quad
m_U=C_U H_U^T V^{-1}r.
\]

The proposal is q(U|y)=N(mU,CU). At supplied posterior-standard coordinates xi,
materialize U=mU+chol(CU)xi. This mapping accepts coordinates; the prototype
opens no RNG stream. The conditional linear primitives are

\[
m_Z(U)=H_Z^T V^{-1}(r-H_U U),\qquad
C_Z=I-H_Z^T V^{-1}H_Z.
\]

Calculate covariances with solves and Joseph/square-root identities; inverse
notation is mathematical only. In particular, with G=HZ^T V^{-1}, use
`CZ=(I-G*HZ)(I-G*HZ)^T+G*R*G^T`. The U covariance can be computed from the SPD
information factor and triangular solves. Apply the unchanged
`1e-12*max(1,lambda_max)` resolution test to the symmetric SPD matrices V, S,
the U information matrix, and materialized CU and CZ before their solves or
density factorizations. It is an eigenvalue test on those matrices, not on a
triangular factor. Unresolved conditioning is a reported refusal, not a repaired
answer. Empty observation blocks have log evidence and log likelihood zero;
empty primitive blocks have a unique empty coordinate, zero log density, and
empty identity covariance. Their determinant convention is one. All finite covariance outputs use
the unchanged `1e-10*max(1,max(abs(P)))` symmetry/PSD tolerance. Output singular
covariances need no factorization and must not be assigned a numerical rank.

For each path,

\[
m_\ell=b(U)+F(U)m_Z(U),\qquad C_\ell=F(U)C_ZF(U)^T.
\]

The angular endpoint is fixed within that component. Reconstruct full six-state
mixture moments using within-component covariance plus the full covariance of
component means, including angular/linear cross terms. These moments describe
the supplied finite mixture only; they do not replace its conditional paths.

## Importance identity and information boundary

With S=R+HU HU^T+HZ HZ^T, the exact evidence is N(r;0,S), and pointwise

\[
\log\phi(U)+\log p(y\mid U)-\log q(U\mid y)=\log p(y).
\]

Return the separate prior, likelihood, proposal and evidence terms so tests can
check this identity independently. The normalized component weights are 1/M.
Do not multiply another gyro/wheel likelihood, a previous prefix weight or a
extra Jacobian for the redundant physical path into that ratio. Every Gaussian
density includes its dimension, log determinant and normalization constant;
q(U) includes the determinant of the standard-coordinate-to-U transformation.
Densities are on active U coordinates, not a singular redundant angular path
vector. A later
change to the proposal would require a new importance derivation.

Each call receives only the checkpoint and the already received event prefix.
Timestamps are integer microseconds; intervals use the unchanged
`ceil(dt/max_step)` equal partition. A same-time gyro precedes wheel, with no
extra propagation. Reject duplicate, delayed, reordered, checkpoint-time or
unsupported sensor events. A gyro-prefix call cannot inspect the same-time wheel
that has not yet arrived. The following wheel-prefix call may recompute past
latent paths using that now-received wheel. No earlier returned output changes.
No requested report inserts a propagation boundary: report only the checkpoint
(empty prefix) or the final received event.

## Bounded prototype and deterministic checks

Add only `tools/observed_trajectory.py` and `tests/test_observed_trajectory.py`
after independent acceptance of this note. The immutable prefix interface
accepts the full checkpoint, fixed diffusion, ordered events and existing step/
gap configuration; evaluation accepts a finite M-by-dU coordinate array.
Expose the coordinate layout, full angular substep paths, conditional endpoint
means/covariances, importance terms and represented full moments. No truth,
reference sample, RNG, prior weight or recovery input is accepted.

Hard limits are 128 positive substeps, 64 received events and 2048 supplied
coordinate rows; default max step is 0.01 s and default max gap is 2 s, within
the existing allowed bounds. Check limits before large allocation. New fixtures
use at most eight substeps and fixed zero/positive/negative coordinates; they
are correctness tests, not a deterministic quadrature performance experiment.

Required independent checks, before any future experiment:

1. Hand-computed scalar Gaussian conditioning, including wheel information about
   correlated angular/speed checkpoint state, plus gyro-only rate/bias ambiguity.
2. A small full joint Gaussian information-form or batch-conditioning oracle for
   `(U,Z)` independent of the proposal's sequential conditional construction.
   Build the observation maps independently from the physical checkpoint plus
   one/two-step noise joint; do not reuse prototype HU/HZ, layout or affine-map
   helpers for expected values. Check full cross blocks, evidence and the
   decomposition of conditional moments, including induced cross-innovation
   covariance after two steps.
3. Direct old-state physical propagation at fixed primitive coordinates versus
   affine path maps; finite differences of the maps and Gaussian log densities.
   Detect replacing old angle/speed with the newly diffused values.
4. Pointwise importance identities at multiple fixed coordinates, nonzero
   residuals and correlated observation rows. Temporary omissions/double-counting
   of likelihood and removal of angular/speed correlation must fail the checks.
5. Symmetry/PSD, full six-state cross-covariance, zero-diffusion/deterministic and
   empty-coordinate limits, negative/zero speed, periodic displacement with
   unwrapped angle, and conservative invalid/unresolved input handling.
6. Gyro-only then same-time-wheel prefix causality, repeated deterministic calls,
   immutable prior outputs, strict event ordering and the hard resource bounds.

The main fixed fixture uses the already published O checkpoint factors and
diffusion in `trajectory_mixture_proposal.md`, without reading saved arrays.
Its four observations are gyro 0.09 with variance 0.02 squared, then wheel 1.4
with variance 0.1 squared at 10000 us; gyro 0.11 then wheel 1.6 with the same
respective variances at 20000 us. Posterior-standard rows are all zero, all +1
and alternating +1/-1. Independent fixtures may use exact scalar or zero
covariance limits, signed speed and deterministic basis directions. Test both
the marginalized importance identity above and the full joint identity
`log p(U)+log p(Z)+log p(Y|U,Z)-log q(U|Y)-log p(Z|U,Y)=log p(Y)`.
Omitting prior/proposal normalization constants must also be detected.
No old reference or study oracle supplies the new test's expected answer.

Use the existing numerical tolerances; independent finite differences may use
`1e-6` perturbations and `1e-7` absolute/relative derivative tolerance. Exact
Gaussian/map checks use `1e-10` absolute with no relaxed implementation target.
Record any implementation bug, retained failed attempt and independently reviewed
repair. A fixture or tolerance must not be changed to hide a failure.

## Validation, ownership and stop

The implementation agent owns only the two new Python files. The independent
reviewer owns no files. The parent owns this note, the validation report and a
separate proposed experiment, all validation runs, integration and Git actions.
Existing files remain unchanged; no branch switching or agent commits.

Reuse the reviewed command monitor without editing it, redirecting its CHECKS
and DEMO roots only within the calling process to fresh
`results/observed-trajectory-checks-v1/` and
`results/observed-trajectory-demo-v1/`. Count every command attempt against 900 s
aggregate execution, 2 GiB assessed per-invocation process-tree memory and
256 MiB retained validation/demo output. Preserve logs, failures and the monitor's
100 ms sampling/short-lived-descendant limitation. Independent acceptance of
this reuse is required before the first numerical command.

Run the new fixed-coordinate checks, Debug configure/build, CTest, full pytest
with RuntimeWarning fatal and the unchanged real C++ development demo using
its existing development seed 20260921. Existing fixed-seed sampler/quadrature
regressions are checks of preserved code; they do not open proposed-study streams
or constitute a new candidate study. Inspect the demo figures. No C++ source
change is planned, so no new sanitizer run is required. Do not claim hosted CI.

Verify all 95 baseline tracked files, all 1184 saved worktree artifacts and 579
older artifact hashes remain unchanged. Report actual outcomes and limitations,
freeze one bounded experiment proposal without executing it, make local lowercase
commits and stop for the next decision. No push, merge or release is authorized.
