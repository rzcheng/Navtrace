# Offline position moment protocol

Ryan approved issues [#1](https://github.com/rzcheng/Navtrace/issues/1),
[#2](https://github.com/rzcheng/Navtrace/issues/2), and
[#3](https://github.com/rzcheng/Navtrace/issues/3) on 2026-10-09. This protocol
freezes one offline Python candidate and one saved-reference comparison.
Commit this document and `scenarios/position_moments.json` after independent
review, before candidate implementation or evaluation. Independently review
and validate the implementation before launching the comparison.

Question: does exact Gaussian-input position prediction, followed by recursive
Gaussian projection and the existing wheel/gyro update, reduce the observed
position-moment errors enough to warrant a separately reviewed navigation
experiment? This is approximate conditional inference. An unfavorable or
inconclusive result still completes the study; no improvement is required to
report completion honestly.

## Fixed inputs and boundaries

Start from inspected commit `c834bf3`, retaining all preceding local commits.
The saved conditional reference was generated at `44a1d19`; its original
long-outage recordings were generated at `0694ea9`. Full commits, manifest
SHA-256 values, source hashes, and expected counts are pinned in the JSON.
Verify all 409 long-outage artifacts, all 108 reference artifacts, and all 90
saved sample arrays. Verify each manifest's source hashes against that
manifest's recorded Git commit, not against later documentation. Keep the
original manifests, protocols, code, samples and reported failures unchanged.

Use only consumed recordings 83001 and 83021. Both start from their complete
gated six-state Gaussian after the 1.99 s gyro update. It already includes the
last wheel observation at 1.95 s and GPS at 1.8 s. Do not reuse the checkpoint
event. Process the original subsequent gyro/wheel observations, in supplied
integer-microsecond timestamp order; do not sort. Equal-time order is gyro
then wheel. Duplicate sensors at a timestamp and out-of-order input are errors.
Report at 1.99, 5, 10, 15 and 22 s. Reports are post-event, including all
same-time gyro/wheel updates. At 22 s stop before returning GPS. The expected
post-checkpoint observation counts are 0, 362, 962, 1562 and 2402.

Only the evaluator may read frozen sample arrays and comparator snapshots
after the checkpoint. The candidate receives checkpoint mean/covariance,
observations, timestamps and unchanged configuration. No truth, true bias,
fault labels, outage schedule, samples, future observations or GPS enter it.
Changing observations after a report must not change that report.

Keep production C++, replay, Q, R, priors, Euler step partition, GPS gate and
recovery criteria fixed. No new recording, reference sampling, parameter fit,
alternative candidate, dependency, GPS-return update, recovery policy or
production change is included. All 83001--83030 recordings remain consumed.
Many conditional samples are not independent navigation trials.

## Gaussian-input prediction

State order is `[px, py, yaw, speed, yaw_rate, gyro_bias]`. Units are meters,
seconds and radians; heading is CCW from world +x and forward speed is signed.
Maintain a real, unwrapped yaw internally, initialized to the checkpoint's
stored angle. All covariances involving yaw use this same real-valued lift.
Only display yaw or compare angular means modulo 2*pi. A change of the input
yaw mean by 2*pi must leave position moments and covariance unchanged.

For Gaussian X with mean m and covariance P, write theta=yaw, v=speed,
s=P(theta,theta), c=P(v,theta), phi=exp(i*m_theta-s/2), and w=v*exp(i*theta).
Differentiating the Gaussian characteristic function gives

```
mu_w = E[w] = (m_v + i*c)*phi
Cov(X_j,w) = (P(j,v) - c*P(j,theta) + i*m_v*P(j,theta))*phi
E[abs(w)^2] = m_v^2 + P(v,v)
E[w^2] = ((m_v + 2*i*c)^2 + P(v,v))*exp(2*i*m_theta - 2*s)
```

Let g=[Re(w),Im(w)], mu_g=[Re(mu_w),Im(mu_w)], and take real/imaginary
columns of Cov(X,w) to form the 6x2 matrix G=Cov(X,g). With

```
a = E[abs(w)^2] - abs(mu_w)^2
b = E[w^2] - mu_w^2
D = Cov(g) = 0.5 * [[a+Re(b), Im(b)], [Im(b), a-Re(b)]]
```

set A=I6 with A(yaw,yaw_rate)=h, and B to a 6x2 zero matrix with h*I2 in
its position rows. The old-state Euler update and its full moments are

```
X_next = A*X + B*g + noise
m_next = A*m + B*mu_g
P_next = A*P*A.T + A*G*B.T + B*G.T*A.T + B*D*B.T + Qh
Qh = diag(0,0,0,qv*h,qomega*h,qbias*h)
```

All terms use the pre-step state. In particular A*G retains correlation with
next yaw theta+h*yaw_rate, including h*Cov(g,yaw_rate). End-step diffusion
does not directly contribute to the same step's position increment.

For elapsed dt, use exactly ceil(dt/max_step_s) equal steps h=dt/steps, as in
production; zero dt is a no-op. Reject negative/nonfinite dt, invalid config,
and gaps above max_gap_s. Preserve the original default 0.01 s and 2 s bounds.
Use the configuration and measurement variances in the frozen inputs without
retuning. Copy input arrays and never leave partial caller mutations on error.

These prediction moments are exact for the assumed Gaussian input and the
stated discrete map, up to floating-point error. Position becomes non-Gaussian
after propagation. Replace that joint distribution by the Gaussian with these
moments before the next step or scalar observation; do not call the recursion
an exact conditional filter or an exact endpoint calculation.

For wheel H selects speed; for gyro H selects yaw_rate+gyro_bias. Apply the
same residual, gain, and Joseph covariance equations as the existing model:

```
r = z-H*m; S = H*P*H.T+R; K = P*H.T/S
m_updated = m+K*r
P_updated = (I-K*H)*P*(I-K*H).T + K*R*K.T
```

R must be finite and positive. These are Gaussian conditioning equations for
the projected joint Gaussian, not exact conditioning of the nonlinear path
law. The latent 4x4 marginal follows the unchanged linear Gaussian filter and
must agree with the saved C++ latent marginal at every report (yaw mean modulo
2*pi). Use the existing 1e-9 absolute numerical agreement limit; record actual
errors. This is a code/model agreement check, not a calibration claim.

Check full covariance finiteness, symmetry and PSD after every step/update.
Use tolerance 1e-10*max(1,max(abs(P))) for symmetry and a negative eigenvalue
roundoff allowance, as in the production model. Only roundoff symmetrization
is permitted. No eigenvalue clipping, jitter, extra noise or inflation. Use
stable algebra, including cancellation-aware small-variance expressions where
needed, rather than loosening tolerances to conceal numerical defects.

## Candidate API and validation

The separate module is `tools/position_moments.py`:

```
predict(mean, covariance, dt_s, diffusion,
        max_step_s=.01, max_gap_s=2.) -> (mean, covariance)
observe(mean, covariance, sensor, value, variance) -> (mean, covariance)
run_candidate(mean, covariance, checkpoint_us, events, report_times_us,
              diffusion, max_step_s=.01, max_gap_s=2., check_budget=None)
    -> {timestamp_us: {'mean': ndarray(6), 'covariance': ndarray(6,6),
                       'observations': int}}
```

Events expose timestamp_us, sensor, value and variance, compatible with the
existing Observation dataclass. Reports are the checkpoint or actual
observation timestamps in these recordings. Save after all same-time events;
no extra report-triggered propagation or GPS support. Inputs are copied.
Invoke check_budget during event processing when supplied. Validate report
ordering/uniqueness and observation boundaries; never use later events to
compute an earlier snapshot.

Before frozen-recording comparison, independently review derivation and code.
Tests must cover:

- Full predicted mean/covariance against independent one-dimensional numerical
  integration over yaw. Conditional X given yaw is Gaussian; compute required
  polynomial/trigonometric conditional expectations without reusing the
  candidate characteristic-function helper. Include all position/latent terms.
- Finite-difference checks of the deterministic Euler map and the small-input-
  covariance limit against its Jacobian; include next-yaw/rate cross terms.
- Correlated position/yaw/speed/rate/bias, signed and zero mean speed,
  near-zero heading variance, deterministic limits, zero diffusion, zero dt,
  and invariance to yaw-mean shifts by 2*pi. Check symmetry and PSD explicitly.
- Independently hand-computed correlated scalar updates and a known example
  showing Gaussian closure error: conditioning a nonlinear displacement on a
  later wheel observation can change exact position covariance with the
  observed residual, while the Gaussian update covariance is residual-independent.
  Do not require false exact agreement in that fixture.
- Causal prefixes, unchanged earlier reports under changed later input,
  old-state increment/end-step diffusion timing, equal-time ordering,
  excluded checkpoint/GPS, copied inputs, invalid inputs and budget callbacks.
- A temporary mutation dropping an important cross term must fail an independent
  check. Record failure and successful restoration; retain no mutation in source.

Freeze tolerances from analytical/numerical error bounds before the recording
comparison; do not infer them from candidate performance. Run CTest, all pytest
with RuntimeWarning treated as error, and the real C++ development demo after
integration. No production C++ change is authorized; a separately approved
change would also require sanitizers.

## Complete saved-sample comparison

Evaluate the fixed candidate once per recording. At each of 10 case/time
points, save its full mean/P and the saved gated EKF comparator. Read all saved
counts 512, 2048 and 8192 and reference seeds 91001--91003: 90 sample-distribution
rows, with nine scalar metrics each (810 entries). No samples are regenerated.
There are 90 resolution outcomes: 2 recordings x 5 times x 9 metrics.

For each endpoint take ascending-eigenvalue orthonormal principal axes of the
original EKF marginal position covariance as a common basis. Fix each column's
sign so its largest-absolute component is positive (ties choose the first
component); use these same axes for all methods, counts and replicates at that
endpoint. Eigenvector signs themselves have no physical meaning. Retain the
full covariance so off-diagonal or rotation error cannot be hidden.

For position samples p, sample mean pbar, original EKF position mean m0 and
axes e_n,e_w, the nine reference metrics are:

| Metric | Definition | Units |
|---|---|---|
| mean_x_m, mean_y_m | full-sample coordinate means | m |
| variance_narrow_m2, variance_wide_m2 | unbiased own-mean sample variance along e_n/e_w (denominator N-1) | m^2 |
| covariance_mixed_m2 | unbiased sample covariance between the two centered projections (N-1) | m^2 |
| second_moment_narrow_m2, second_moment_wide_m2 | mean of [e_j.T*(p-m0)]^2 (denominator N) | m^2 |
| baseline_ellipse_mass | mean indicator of (p-m0).T*P0_pp^-1*(p-m0) <= 5.991464547107979 | dimensionless |
| candidate_ellipse_mass | same indicator using fixed candidate position mean and marginal P_pp | dimensionless |

Use solves, not explicit matrix inverses. Both ellipse calculations use the
marginal 2x2 covariance, never a conditional/Schur covariance. If an ellipse
covariance is not strictly positive definite, record that metric's failure;
never repair it. At the checkpoint the candidate equals its input exactly.

Compute the candidate/baseline deterministic coordinate means, directional
variances and mixed covariance from each full mean/P. Their directional
second moment about m0 is e_j.T*P_pp*e_j + [e_j.T*(m_p-m0)]^2. Report full 2x2
matrices, vector mean differences, eigenvalues/directions and difference norms
as descriptive quantities, alongside the nine checked scalar reference metrics.
Raw saved terminal yaw is wrapped. Never compare its ordinary position/yaw
sample covariance with unwrapped candidate covariance, and never silently
unwrap it. Primary frozen comparisons are position-only; analytical tests
validate the candidate's yaw cross terms.

## Monte Carlo uncertainty and interpretation

For each saved distribution split samples into 16 contiguous equal batches
and recompute each scalar metric separately in each batch. The point estimate
uses the entire saved distribution, not the average batch covariance. MCSE is
the standard deviation of the 16 batch estimates (ddof=1) divided by sqrt(16).
For each count pool the three independent replicate point estimates equally;
pooled SE is sqrt(sum(SE_r^2))/3. Pool full own-mean covariances by the same
average without adding between-replicate mean variation. Also report replicate
minimum, maximum and sample standard deviation.

For every metric require |pooled_8192-pooled_2048| <=
4*hypot(SE_8192,SE_2048)+1e-12 in that metric's units. At the largest count,
require 3*pooled_SE no greater than:

- Means: 0.05 times the original EKF x/y marginal standard deviation, in m.
- Directional variances and displaced second moments: 0.10 times
  max(abs(pooled reference metric), 1e-12 m^2).
- Mixed covariance: 0.10 times max(sqrt(pooled reference variance_narrow times
  pooled reference variance_wide), 1e-12 m^2).
- Either ellipse mass: 0.02, dimensionless (two percentage points).

A metric is resolved only if both stability and precision pass with all
required inputs present. The previous 70 checks do not establish resolution
for mixed covariance or candidate ellipse mass. Report all 90 new outcomes;
missing/failed inputs are inconclusive, not dropped.

For the seven moment metrics, compare each deterministic method's value with
the pooled high-count reference using 4*pooled_SE+1e-12 as a descriptive
agreement band in that metric's units. For either ellipse mass, report its
difference from nominal 0.95 and the same band. These are per-metric numerical
diagnostics, not simultaneous statistical tests, implementation acceptance
criteria or navigation calibration. An accurately resolved mismatch is an
unfavorable approximation result; failed reference resolution is inconclusive.
Report signed errors and baseline/candidate changes without selecting favorable
endpoints. No minimum improvement threshold or tuning loop is authorized.

## Launch guards, budget and artifacts

The runner must pin the SHA-256 of the reviewed committed JSON before launch;
the JSON also pins this document's SHA-256. Require a clean committed checkout
and exact on-disk/committed protocol hashes. Verify the preserved baseline
sources listed in the JSON, frozen configuration, both original manifest
digests, artifact counts and every artifact hash before reading evaluation
values. Audit manifest source hashes against their historical commits. Record
actual input roots as metadata; use explicit path arguments rather than
embedding machine-specific absolute paths in the protocol. Output must be distinct from and outside both frozen input directories,
and must not overwrite any source or protocol path. Refuse an existing output
directory even if empty.
A dirty checkout, changed protocol/input, missing input or invalid output path
refuses launch before creating output; record the refusal in command/test logs.

Freeze one comparison launch at 600 s wall time and 512 MiB peak RSS, including
input audit, candidate replay, metric computation, plots and output work.
Start timing at runner entry and check the budget throughout; no automatic
increase, alternate method, resampling or silent retry. Preflight refusals are
not comparison runs. Once output is created, initialize the complete expected
matrix and retain every case/time/count/replicate and every metric. Statuses
are complete, failed or not_started; partial rows retain successful metric
values and mark unavailable metrics explicitly. A row is complete only when
all nine metrics are complete. On a budget stop, mark the interrupted work
failed and untouched work not_started, retain available artifacts, and exit
nonzero. Other computation failures retain reasons and continue unaffected
work within budget; any failed/unstarted required result makes the run incomplete.

Save protocol copies and hashes, command, implementation commit/branch/dirty
state, versions, actual cost, input/source/output SHA-256 values, candidate
snapshots, comparator snapshots, all 90 metric rows and all 90 resolution
outcomes. Include the complete planned matrix and failure accounting even if
execution stops. Save plots in the fresh ignored results directory. At the
end recheck original artifacts and preserved sources; record any mismatch as
a failure. Record hashes for completed output files without self-hashing the
manifest. Keep the original baseline/reference folders read-only throughout.

Inspect all 10 case/time entries and equal-spatial-axis sample/ellipse plots,
using the saved highest-count seed-91001 samples with full ranges retained.
Plot means, directional and mixed covariance errors, ellipse mass and numerical
resolution. Do not add truth overlays or navigation-performance claims.

Independent review must inspect the final diff and artifacts, including any
regression or non-Gaussian shape remaining after moment changes. Update issues
#1--#3 with actual tests, metrics, limitations, artifact paths and local commits.
Local completion is separate from merge or release; no push is authorized.
Stop for Ryan's review. The sole next decision is whether this fixed candidate
warrants a separately scoped navigation experiment. Correct first/second
moments, PSD, or nominal ellipse mass do not establish correct checkpoint
uncertainty, full distribution shape, tail accuracy, calibration or recovery.
