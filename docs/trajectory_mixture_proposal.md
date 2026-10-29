# Synthetic trajectory mixture study proposal

Ryan approved this proposal document and its independent review on 2026-10-18
in [issue #6](https://github.com/rzcheng/Navtrace/issues/6). Approval includes
choosing proposed constants, not evaluating them. The starting commit is
`7e791380508dd1eefbbf25a8bce79b3dd9e80b44` on the isolated
`codex/trajectory-mixture-design` branch. The reviewed
[design](trajectory_mixture_design.md), SHA-256
`644809e2f91ad5df3c3876b60625595981d87633ba2ddf0475be483daa7f6596`,
is preserved unchanged.

**Proposed next authorization:** implement the one existing mixture variant,
complete the bounded checks below, and attempt exactly one fresh synthetic
study with M=2048 and four independent seed groups for each stochastic case.
The study would have a 120 s wall-time, 512 MiB peak-RSS and 64 MiB retained-output
cap. This is a request for later approval, not permission to start that work.
None of these proposed counts, inputs, thresholds or budgets has been evaluated.

## One decision and unchanged variant

The decision is whether this fixed prior-proposal mixture can meet a modest,
predeclared synthetic accuracy and weight-concentration requirement at this
budget, or whether it fails or remains inconclusive. A favorable result would
support another scope review only. It would not establish navigation performance,
long-outage feasibility, nonlinear position calibration in general, or production
readiness.

Keep persistent angular paths a=(unwrapped yaw, yaw rate, bias), conditional
Gaussian ell=(px,py,speed) with full 3x3 covariance, fixed M, and log likelihood
weights. Use prior angular transitions, no resampling, merging, rejuvenation,
adaptive count, inflation, alternative proposal, or moment matching. The full
six-state mixture moments are reports, never the continuing filtering state.
Keep old-state Euler position and heading, independent end-step diffusion,
integer timestamps, gyro-before-wheel ordering, positive measurement variance,
and all correlations. No GPS, truth, actual bias, fault label, reference sample,
or future observation is supplied to the candidate.

This is one small fixed matrix, not a pilot followed by a sweep. There is no
sample-count escalation, input adjustment, extra replicate, seed replacement,
threshold change, or selective retry after an outcome. Existing production,
configuration, tests, protocols, prior results, consumed-seed labels and saved
artifacts remain unchanged. The new synthetic random seeds below are algorithm
randomization keys, not fresh navigation recordings.

## Frozen inputs

Use x=(px,py,theta,v,omega,b), a=(theta,omega,b), ell=(px,py,v), with the units
in [model.md](model.md). All unspecified checkpoint means, covariance entries,
and diffusion coordinates below are exactly zero. The factored definitions
specify full covariance matrices; they are not an invitation to discard cross
terms. The candidate receives the assembled full six-state mean and covariance,
not the factors, analytical conditional gain, oracle values, or case labels.
Case names are runner/evaluation metadata only.

For the two runner cases, the checkpoint is timestamp 0 us, max_step=0.01 s,
max_gap=2 s, and no observation at the checkpoint is reused. Inputs are already
ordered. Reports are only at the checkpoint or an actual received observation
timestamp, after all same-time gyro/wheel events. A report request never adds a
step. Scalar sensor variances in the tables are in their sensor units squared.
Diffusion intensities are ordered (qv,qomega,qb), in m²/s³, rad²/s³, rad²/s³.

### L: deterministic angle, linear position and speed

Use

\[
\mu_x=(0.5,-0.25,0.2,1.5,0,0)^T,\qquad
P_{aa}=0,\quad P_{\ell a}=0,\quad P_{\ell\ell}=J J^T,
\]

\[
J=\begin{pmatrix}0.2&0&0\\0.05&0.15&0\\0.04&-0.02&0.25\end{pmatrix}.
\]

Each row of J has the unit of its ell coordinate; its driving standard normal
coordinates are dimensionless. Diffusion is (0.04,0,0). The heading is fixed,
so the entire six-state model has an exact linear Gaussian solution on the
three-coordinate ell support. Every component has the same angular state and
conditional Gaussian; no angular random draw is needed.

| timestamp (us) | gyro value | gyro variance | wheel value | wheel variance |
|---:|---:|---:|---:|---:|
| 100000 | 0.03 | 0.02² | 1.4 | 0.1² |
| 200000 | -0.02 | 0.02² | 1.6 | 0.1² |
| 300000 | 0.01 | 0.02² | 1.45 | 0.1² |

Both listed observations are received in gyro-then-wheel order at each row.
Reports are at 0, 100000, 200000 and 300000 us. This case has one candidate run
with M=2048, not four duplicates counted as independent evidence.

### H: the existing shared-heading analytical case

Use the existing diagnostic's positive-speed, positive-heading-variance case:
initial position zero, deterministic v=2 m/s, omega=b=0,
\(\theta\sim\mathcal N(0,0.4)\), and zero diffusion, with no observations.
The full checkpoint covariance has only Ptheta,theta=0.4. The horizon is T=0.1 s.

This is a direct prediction-primitive check with a prescribed schedule, not an
event-runner case. For each of the four independent seed groups, draw the
initial angular ensemble once and retain two copies. Branch H1 makes one
Euler step h=T; branch H8 makes eight steps h=T/8. Those are the two already
studied partitions, not a step-size sweep. Inspect the checkpoint and endpoint
only. There is no report-triggered propagation or pseudo-observation. All
components and initial weights are identical between the two branches of a
seed group; no noise or additional random draw occurs on either branch.

Each branch retains its own completion/failure and cost record. H1 and H8 are
paired checks of the same finite integration, not eight independent replicates.
Finite-M statistical summaries use four seed groups, and use H1 alone for the
accuracy estimator. H8 is the required paired partition check; missing H8 is
not excused by a successful H1.

### O: one correlated, observed, diffusive sequence

Use

\[
\mu_a=(0.2,0.1,-0.02)^T,\qquad \mu_\ell=(0.5,-0.25,1.5)^T,
\]

\[
L_a=\begin{pmatrix}0.3&0&0\\0.02&0.08&0\\-0.01&-0.03&0.04\end{pmatrix},
\qquad
K=\begin{pmatrix}0.4&0.2&-0.3\\-0.2&0.5&0.1\\0.3&-0.4&0.2\end{pmatrix}.
\]

The angular factor's row units are rad, rad/s and rad/s. K has the unit of
its ell row divided by the unit of its angular column. J is the factor defined
in L. The complete correlated checkpoint in block order (a,ell) is

\[
A=L_aL_a^T,\qquad B=KA,\qquad D=KAK^T+JJ^T,
\qquad P=\begin{pmatrix}A&B^T\\B&D\end{pmatrix}.
\]

This definition freezes all within-block and cross-block correlations without
numerically evaluating the matrix in this proposal. Diffusion is
(0.04,0.0004,0.0001). Gyro variance is 0.02² and wheel variance is 0.1² throughout.
The received observation sequence is exactly:

- At 10000 us: gyro value 0.09, then wheel value 1.4.
- At timestamp 50000*j us for each integer j=1,...,20: gyro value
  \(0.08+0.01(-1)^j\).
- At timestamp 100000*k us for each integer k=1,...,10: wheel value
  \(1.5+0.1(-1)^k\), after the same-time gyro.

These are fixed sensor values, not measurements simulated from a hidden truth.
There is no latent truth path to pass to the candidate. Reports are at
0, 10000, 250000, 500000, 750000 and 1000000 us. The first nonzero report is
exactly one Euler step followed by gyro and wheel; the later reports involve
multiple steps. This single sequence supplies a bounded weight-concentration
assessment, not a claim to represent navigation operating conditions.

## Randomization, pairing and planned inventory

Fix M=2048 for every candidate run. Use independent NumPy PCG64 streams derived
with SeedSequence entropy words (20261018, case_id, replicate_seed, phase).
Case identifiers are L=1, H=2, O=3; phase 0 is initialization and phase 1 is
angular transition noise. Replicate seeds, in execution order, are 95001,
95002, 95003 and 95004. L uses metadata replicate_seed=0 and consumes no angular
randomness. Development-only deterministic-control checks reserve seed 94999;
none of the four study seeds is used before the single study launch.

Initialization draws a standard-normal array in component-major order with
active angular coordinates ordered theta,omega,b. The active covariance's
positive-diagonal lower Cholesky factor defines its transform. At each substep,
transition draws are component-major with nonzero diffusion coordinates ordered
omega,b; deterministic coordinates consume no random numbers. Speed diffusion
is integrated in the conditional covariance and is never sampled. No antithetic
pairing, randomized moment matching or low-discrepancy replacement is allowed.
The full NumPy/Python versions and stream state conventions must be recorded.

For H, both branches use the same initialization, not two streams bearing
similar seeds. For O, the next prior proposal depends only on the existing path,
step and its stream. Reports consume no random draws. Invalid events must be
validated before random-state consumption or otherwise restore it transactionally.
Changing which prescribed reports are retained cannot change paths, final
weights, conditional states, or random-state advancement.

Before launch, create the complete status inventory, including all requested
reports and comparison quantities:

| case group | independent groups | branch rows per group | prescribed rows |
|---|---:|---:|---|
| L | one deterministic group | one | L/0 |
| H | four seed groups | two paired | H/95001/H1, H/95001/H8; likewise 95002, 95003, 95004 |
| O | four seed groups | one | O/95001, O/95002, O/95003, O/95004 |

There are thirteen candidate branch/run rows. Execute L first, then H in seed
order with H1 before H8 for each seed, then O in seed order. Preallocate their
report and metric entries as not-started; never create only successful rows.
H group-level uncertainty has four independent groups, never eight branches.
Report every individual run as well as the prescribed aggregate. Repetitions
vary proposal randomness for the same fixed sensor sequence; they are not
independent navigation trials or fresh observation realizations.

## Restricted support and numerical policy

This bounded prototype does not claim a general singular-Gaussian solver.
Its accepted checkpoint supports are coordinate-deterministic blocks plus
strictly positive-definite active blocks. In covariance dimensionless units,
use fixed scales (1 m,1 m,1 rad,1 m/s,1 rad/s,1 rad/s) in original state order;
use the corresponding subscales for blocks. Scaling is for numerical checks,
not a change to the model or reported units.

At the checkpoint, an exactly zero diagonal entry declares a deterministic
coordinate only when its complete corresponding row and column are exactly
zero. Nonzero covariance with a deterministic coordinate is invalid. Positive
diagonals define the active coordinates; they are not dropped by a small-eigenvalue
rule. The active angular block must be SPD with

\[
\lambda_{\min}>10^{-12}\max(1,\lambda_{\max})
\]

in those fixed dimensionless units. Factor and solve only that block. Derive the
conditional mean and covariance from the full input P by Gaussian conditioning,
not from the fixture's K or J. For a rank-zero angular block, retain the entire
ell Gaussian. Check the derived conditional ell support in the same way. L and
O have active ell coordinates (px,py,v); H has no active ell coordinate.

For L/O, their conditional covariances must continue to satisfy the same active
SPD resolution condition at every step/update. H must retain exactly zero
conditional covariance and deterministic rate/bias/speed support. A rotated
singular active block, or a positive but insufficiently resolved active
covariance, is unsupported/unresolved; report it and do not clip it to a lower
rank. This conservative restriction can cause an inconclusive result even for
a mathematically valid Gaussian. It is an explicit limitation, not a failed
proof that all singular cases work.

Check finite values, symmetry and PSD throughout, using the existing
scale-aware tolerance 1e-10*max(1,max(abs(P))) in dimensionless units for
roundoff checks. Only roundoff symmetrization is permitted. A materially negative
covariance, invalid support, nonpositive observation variance or nonfinite input
is invalid; no jitter, clipping, inflation, additional noise, or relaxed
tolerance is allowed. Reconstructed mixture covariance need only be PSD, not
SPD: finite angular support and deterministic limits can make it singular.

Use scalar likelihoods and log normalization as in the reviewed design. Retain
every component's finite log weight, angular state, full conditional Gaussian
and identity even when a reported exponential weight is zero. Any nonfinite
retained log weight, invalid conditional state, or failed normalizer is a
numerical failure; dropping the component is prohibited. For study outcomes,
any exponential underflow reported during normalization or moment reconstruction
makes numerical resolution inconclusive. It is still recorded and all finite
components continue; there is no hidden truncation or weight reset.

## Independent comparisons and their limits

The evaluator must not import candidate moment, conditioning, likelihood,
normalization or covariance helpers. It may call the public candidate interface
to obtain outputs. It also must not import the stopped Gaussian-moment candidate,
conditional reference sampler, or their numerical helpers. The following three
bounded analytical constructions are the only independent inference comparisons.
None of the new study comparisons uses quadrature, a backward path sampler,
numerical endpoint integration, or a general multi-step nonlinear position
oracle. The unchanged regression suite may still execute its existing synthetic
quadrature and sampler tests; those are validation of preserved code, not new
study comparisons or use of saved reference-study arrays.

### L: exact full-state linear Gaussian conditioning

For fixed theta=0.2, every ell substep is linear. Build the joint Gaussian of
the initial ell, independent scalar end-step speed noises and the received
wheel noises under the prescribed finite schedule. Propagate their linear
coefficients to each report and condition the joint vector directly on the
wheel prefix. Gyro observations have a known deterministic mean and add only
their scalar likelihoods. This independent batch conditioning is confined to
L and its prefixes; it does not reuse the candidate's recursive Joseph update.

Compare every full six-state mean and covariance entry, including position/speed
cross terms, with that exact law. Whiten the active ell covariance using the
oracle's lower Cholesky factor, and compare to identity. Compare all conditional
component means/covariances, equal weights, observation counts, and accumulated
log evidence too. The deterministic angular rows/columns and mean values are
checked separately. Equality is an implementation requirement, not a finite-M
accuracy test; duplicating an identical angular component introduces no sampling
error in this case.

### H: exact shared-angle moments and paired finite integration

The exact endpoint is
\((Tv\cos\theta,Tv\sin\theta,\theta,v,0,0)\). Use the already documented
Gaussian characteristic identities from [prediction_closure.md](prediction_closure.md)
for its full mean/covariance: exact x/y variances, signed y/theta covariance,
zero x/theta and x/y covariance, and deterministic remaining coordinates.
The active oracle moment covariance is on (px,py,theta); checkpoint support is
theta only. Full means and all covariance entries must be retained.

For each seed group, independently form the empirical endpoint of its exact
same initial angles using that explicit endpoint map and centered scalar sums.
Compare each branch's full represented moments and per-component state with
that empirical map. This is a deterministic implementation/partition check
conditional on the drawn angles. Separately compare the four H1 empirical
mixtures with the continuous analytical moments under the finite-M rules below.
The empirical comparison cannot substitute for the continuous one. In particular,
a finite ensemble need not have zero x/theta covariance or exact heading variance.

### O: full-six moments for the first step only

The checkpoint has exact full-six Gaussian moments. For the 10000 us report,
let W consist of x0 and the three independent end-step noises
(eta_v,eta_omega,eta_b), with their diagonal covariance (qv*h,qomega*h,qb*h),
h=0.01 s. W is a nine-dimensional Gaussian. Condition it jointly on the two
linear observations

\[
z_g=\omega_0+b_0+\eta_\omega+\eta_b+\epsilon_g,
\qquad z_w=v_0+\eta_v+\epsilon_w,
\]

using the exact Gaussian conditioning formula and independent sensor covariance
\(\operatorname{diag}(R_g,R_w)\). This retains observation-induced correlations
with the process noises. Do not first project the moved six-state law to a
Gaussian and then condition it.

In that conditioned Gaussian, write \(Y=F_0W\) for the linear endpoint part
\((p_{x,0},p_{y,0},\theta_0+h\omega_0,v_0+\eta_v,
\omega_0+\eta_\omega,b_0+\eta_b)\), and D for
\((v_0\cos\theta_0,v_0\sin\theta_0)\). The final state is
\(X^+=Y+h E_pD\), where Ep inserts a two-vector into the position coordinates.
For conditioned means/covariances, set

\[
\psi_k=\exp(ik\mu_\theta-k^2\sigma_\theta^2/2),\qquad
c=\operatorname{Cov}(v_0,\theta_0).
\]

Derivatives of the Gaussian characteristic function give

\[
E[v_0e^{ik\theta_0}]=(\mu_v+ikc)\psi_k,
\]

\[
E[v_0^2e^{ik\theta_0}]=
\left[\sigma_v^2+(\mu_v+ikc)^2\right]\psi_k,
\]

\[
E[Y_jv_0e^{ik\theta_0}]=
\left[\operatorname{Cov}(Y_j,v_0)+
(\mu_{Y_j}+ik\operatorname{Cov}(Y_j,\theta_0))
(\mu_v+ikc)\right]\psi_k.
\]

At k=1, real/imaginary parts give E[D] and E[Y D^T]. At k=2,

\[
E[D_x^2]=\tfrac12(E[v_0^2]+\Re E[v_0^2e^{2i\theta_0}]),\quad
E[D_y^2]=\tfrac12(E[v_0^2]-\Re E[v_0^2e^{2i\theta_0}]),\quad
E[D_xD_y]=\tfrac12\Im E[v_0^2e^{2i\theta_0}].
\]

Center these products, then use

\[
E[X^+]=E[Y]+hE_pE[D],
\]

\[
\operatorname{Cov}(X^+)=\operatorname{Cov}(Y)
+h\operatorname{Cov}(Y,D)E_p^T
+hE_p\operatorname{Cov}(D,Y)
+h^2E_p\operatorname{Cov}(D)E_p^T.
\]

This is an independently authored, fixed one-step full-six oracle, including
position/angular, position/speed, and position/rate/bias correlations. It is
not extended to subsequent nonlinear paths. Its mathematical derivation and
implementation must be reviewed independently before study launch.

### O: later reports have a linear-marginal oracle only

For all O prefixes, independently filter u=(theta,v,omega,b) with its exact
linear Gaussian dynamics and gyro/wheel observations. The evaluator uses the
full checkpoint marginal, the same old-state transition and independent
end-step diffusion, and the actual prefix. This comparison includes all four
means, ten unique covariance entries, correlations, and exact log likelihood.
At the first report, this marginal must agree with the nine-dimensional batch
conditioning result within the deterministic tolerance; otherwise the oracle
check fails and the study does not launch.

At 250000, 500000, 750000 and 1000000 us, position means, the full position
covariance, and every position/u cross-covariance have **no independent oracle
in this study**. Retain them and their across-replicate variation as descriptive
outputs only. A correct latent marginal, stable repeated position summaries,
PSD, or a favorable weight diagnostic does not validate those nonlinear position
quantities. No full nonlinear position-accuracy or coverage conclusion is allowed
from these later reports.

## Bounded implementation checks before the study

These are future checks, not tests or fixtures created in the present task.
Use only the declared case parameters and the explicit diagnostic variations
below. They do not consume the study seeds. Check all relevant full matrices,
not just diagonal variances. Independent analytical checks satisfy the repository's
math-validation requirement; no new finite-difference motion approximation is
needed for these linear conditional maps.

1. **Checkpoint split and support.** On O's full assembled P, condition at
   a=mu_a and at a=mu_a+La*(1,-1,1)^T. Compare the derived mean and covariance
   to mu_ell+K*(a-mu_a) and JJ^T, and reconstruct all checkpoint blocks
   algebraically. Check H's one-coordinate support and L's deterministic a.
   Reject an L checkpoint with Ppx,theta=Ptheta,px=1e-6 while Ptheta,theta=0.
   Also reject as unsupported/unresolved an angular block uu^T with
   u=(0.3,0.08,0)^T, B=0 and D=JJ^T, and a diagonal active angular block
   diag(0.3²,0.08²,1e-16), B=0, D=JJ^T. Do not repair these matrices.
2. **Old-state step and diffusion timing.** Use O's conditional Gaussian at
   a=mu_a. With h=0.01, prescribe the first angular innovation
   (0,0.03,-0.01) and the second innovation zero. Independently expand the
   two-step conditional position as a linear function of initial ell and the
   two independent speed noises. Check full conditional means/covariances and
   angular endpoints: the first rate innovation leaves theta1 unchanged and
   enters theta2 at the end of the second step. Position uses the old angle,
   so that rate innovation first affects position in the third step. Speed
   noise does not affect the position increment in its own step. A zero-duration step is a no-op.
   The prescribed innovations are a deterministic conditional-map check, not
   alternative proposals in the study.
3. **Both likelihoods, conditioning and reconstruction.** Use two manually
   declared angular points mu_a+La*(1,-1,1)^T and mu_a-La*(1,-1,1)^T, their
   analytical conditional Gaussians, and equal initial weights. Apply gyro
   0.09 with variance 0.02², then wheel 1.4 with variance 0.1², without motion.
   Compare likelihoods to independent scalar Gaussian formulas, conditional
   updates to batch Gaussian conditioning, and every full-six moment to raw
   second moments and the closed two-component between-mean covariance.
   Gyro leaves each conditional Gaussian unchanged but changes mixture weights;
   wheel uses predictive speed variance and corrects all conditional cross
   terms. This two-component algebra check is not a second study variant.
4. **Log weights and transactional failures.** A normalization-primitive input
   with equal initial weights and successive log-likelihood pairs (0,-1000)
   and (-1000,0) must preserve both states and recover equal relative weights.
   Record any reporting underflow. Reject retained NaN, positive infinity or
   negative infinity, a zero sensor variance, and a nonfinite observation.
   An invalid event preserves the prior time, all components and random state.
5. **Causality and runner schedule.** With O and control seed 94999, compare
   prescribed-report requests with a final-report-only request; the final
   mixture and random state must agree. Change only the final O gyro/wheel
   values to 9 and -9; earlier stored prefix snapshots must remain unchanged.
   A same-time wheel-before-gyro input and a duplicate gyro event are rejected
   without proposal consumption. H uses the primitive interface, never an
   event-runner report as a hidden prediction request.
6. **Bounded oracle and record checks.** Independently review the L batch
   construction, H analytical formulas and O one-step characteristic formulas.
   Check full symmetry/PSD, oracle conditioning identities and the O one-step
   versus latent-filter marginal equality. Check that an injected resource
   stop retains the complete preallocated run/report inventory, including
   paired H branches and all not-started entries.

For deterministic numeric comparisons, use absolute tolerance 1e-10 after
fixed physical scaling for means and covariance entries, and 1e-10 for log
likelihoods and normalized weights. Exact deterministic zero support stays
exactly zero; equality of random state and unmodified prior snapshots is exact.
For L's active covariance, also require maximum absolute entry of the
oracle-whitened covariance minus identity <=1e-10. For H's two branches,
compare per-component endpoints and full moments to the same empirical endpoint
map at the deterministic tolerance. These tolerances are separate from the
existing PSD check and from stochastic accuracy criteria; none may be loosened.

In isolated in-memory checks, respectively remove checkpoint cross-block B,
use the new angle for displacement, and omit the wheel likelihood while keeping
its conditional update. The corresponding correlation, timing and joint-mixture
checks must detect all three mutations. Restore automatically; no persistent
source or artifact mutation is permitted. A failed exact check or oracle review
blocks the study. No favorable stochastic outcome can waive it.

## Finite-M accuracy, precision and concentration rules

Use normalized mixture covariance, including all within- and between-component
terms, with no M-1 sample correction. Every full-six mean and covariance is
saved for every report. Analytical accuracy applies to H at its checkpoint and
endpoint, O at its checkpoint/10000 us report, and O's four-state marginal at
later reports. L uses the deterministic rules above.

For an oracle-supported stochastic vector, let mu* and P* be its exact moments
on its known positive support. Use the oracle covariance's lower Cholesky G,
not directions chosen from a candidate run. For each replicate r retain:

\[
d_{m,r}=G^{-1}(\widehat\mu_r-\mu_*),\qquad
D_{P,r}=G^{-1}\widehat P_rG^{-T}-I.
\]

Evaluate every coordinate of dm and every unique entry of DP. Supports are
H checkpoint (theta), H endpoint (px,py,theta), O checkpoint/first report
(all six states), and later O reports (theta,v,omega,b). Deterministic coordinates
outside the H support use the deterministic rules instead; no denominator floor
or pseudoinverse is introduced. Original-coordinate means and all covariance
entries remain in the report alongside these normalized metrics.

In H, also evaluate the relative x and y variance errors
\(\widehat P_{xx}/P^*_{xx}-1\) and
\(\widehat P_{yy}/P^*_{yy}-1\). The known positive analytical variances are the
fixed denominators. Oracle whitening and these predefined x/y directions guard
against hiding a narrow-direction deficit behind large marginal scales. No
post-hoc axis selection, variance floor or redefined target is allowed.

The proposed normalized error allowance epsilon is 0.10 for each mean quantity,
0.20 for each covariance/variance quantity, and 0.5 for O's cumulative log-evidence
error against its linear oracle. Evidence error is in natural-log units and
uses the same replicate rules. These are bounded engineering acceptance targets,
not established sampling confidence limits or calibration claims.

For each scalar error d with four independent replicate values, define

\[
\bar d=\tfrac14\sum_{r=1}^{4}d_r,\qquad
s_e=\sqrt{\frac{\sum_{r=1}^{4}(d_r-\bar d)^2}{4(4-1)}}.
\]

The descriptive precision requirement is 4*se <=epsilon/2. A quantity passes
only if that precision condition holds and |mean error|+4*se <=epsilon.
It fails the finite-M target if |mean error|-4*se >epsilon. Other cases are
inconclusive, including insufficient precision without a clear failure.
Additionally, any individual |d_r|>2*epsilon fails that quantity's feasibility
target regardless of the average. Preserve and report all individual values;
never remove a replicate or declare success by pooling away a bad run.

For each oracle-supported covariance, also require every individual replicate's
oracle-whitened covariance-error spectral norm <=0.5 and the spectral norm of
the average covariance error <=0.3. Exceeding either is a finite-M covariance
failure. These fixed guards supplement the entrywise assessment; they are not
hypothesis tests. No oracle covariance with unresolved numerical support may be
inverted to obtain a favorable score; such comparison is inconclusive.

Average each replicate's own mean and its own normalized-mixture covariance
separately. This covariance average excludes between-replicate mean spread.
Do not label it the covariance of a pooled mixture: pooling normalized mixtures
would add a between-replicate term and answer a different question. H1/H8
pairing contributes no additional independent sample to se. With only four
independent groups, the uncertainty estimate is coarse. The 4*se rules are
explicitly descriptive, not simultaneous confidence guarantees across correlated
moments/times, and not coverage probabilities.

For each O event phase, and for every replicate, require ESS >=0.10*M and
maximum normalized weight <=0.05. A violation is a concentration failure of
this frozen variant at this budget. Retain the minimum ESS, maximum weight,
all event values, finite log-weight range, underflow counts, likelihood
normalizers and stable component identities. L/H must retain equal weights
within the deterministic tolerance. An ancestry count of M does not override
poor ESS, and ESS passing does not establish accurate position moments.

Global decision precedence applies at the final pre-launch gate and during the
frozen study. Any actual accuracy, concentration,
implementation, invalid-input, material covariance, arithmetic or resource
failure makes the result no-go for this frozen variant/budget, even if another
row is incomplete or inconclusive. Otherwise, unsupported numerical resolution,
insufficient precision, underflow, a missing required replicate/comparison or
unresolved resource measurement makes the result defer/inconclusive. Only when
all exact checks, every independently supported accuracy/precision quantity,
all individual-run guards, concentration requirements and numerical/resource
rules pass, with every planned row complete, does the bounded target pass.
Earlier pre-launch check failures may be resolved only under the bounded repair
rule below; their original evidence remains, and an unresolved final gate still
blocks launch. Intentional rejection fixtures count as passed checks when their
specified rejection occurs. Neither no-go nor defer authorizes increasing M or
changing the proposal. Even a pass leaves later nonlinear position moments
unresolved and ends at review.

## Budget, status records and no retries

Run the study serially in one candidate/evaluator worker process after the
pre-launch gate, with a separate monitoring wrapper and no worker descendants
or parallel replicates. Before importing numerical libraries, set
OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
VECLIB_MAXIMUM_THREADS and NUMEXPR_NUM_THREADS to 1 for those processes.
The assessment caps are 120 seconds elapsed from wrapper start through final
artifact flush, 512 MiB peak RSS including wrapper and worker, and 64 MiB of
retained study output. These include oracle, reporting and plotting work.

The wrapper monitors elapsed time and live aggregate RSS at 100 ms intervals,
independently of candidate callbacks. Retain the sampled live peak and each
process's OS-reported lifetime high-water RSS separately. Assess memory against
the larger of the sampled aggregate peak and the conservative sum of wrapper
and worker high-water values. The latter is a bound, not an exact simultaneous
peak. Record the platform-specific RSS units and convert them explicitly.
Unavailable or incomplete required measurements are unresolved; never infer
compliance from a missing value. Polling detects overruns and does not promise
instantaneous prevention. Any observed or exit-detected overrun is a global
resource failure even when all numerical rows completed. Stop remaining work
when a limit is detected; retain the actual elapsed time including failure-record
flush if that extends beyond the deadline.

Before each artifact write, check the output allowance. Reserve 4 MiB within
the 64 MiB total for status, failure metadata and monitor logs; snapshots, tables
and figures may not consume that reserve. No compression or deleted output may
conceal a cap violation. Actual costs are measured only after a later approval;
no runtime or memory sufficiency claim is made now.

A resource stop fails the current unfinished row, retains partial evidence, and
marks later rows not-started. If detected after row completion, preserve those
row statuses and record a failed global resource assessment. One numerically
failed or unresolved row is not retried; remaining rows may proceed in the fixed
order only if the process is valid and the global budget remains. No post-hoc alternate seed, smaller report set, skipped difficult
case, third H partition, or second launch is allowed. Failure/inconclusive
results are valid completed study outcomes, not reasons to seek success quietly.

Use two explicit status fields so scientific disagreement is not mislabeled
as a software crash:

| execution status | meaning |
|---|---|
| not-started | planned row not entered because of a failed gate or exhausted budget |
| invalid | invalid supplied input, including nonfinite values, incompatible deterministic support, nonpositive sensor variance, or timestamp contract; retain the reason |
| unresolved | a valid input or inference state cannot be numerically resolved under the frozen support/measurement rules; stop that row and retain its last valid prefix |
| failed | exception, materially invalid covariance produced during inference, nonfinite retained inference, failed exact check, or resource stop |
| completed | all scheduled events/reports and required outputs for that row were produced |

Every row, including an incomplete row, also carries assessment pass, fail, or
inconclusive. Not-started and unresolved rows are inconclusive; invalid and
failed study rows fail. A completed row may pass, fail or be inconclusive: a
clear finite-M or ESS failure is a fail, while insufficient replicate precision,
reporting underflow or a missing usable comparison is inconclusive unless a
separate failure is present. A valid but insufficiently resolved active covariance
is unresolved, not an invalid input or material covariance failure. Preserve the
last-valid timestamp/phase and mark all subsequent outputs unavailable.

Intentionally invalid/unsupported inputs
in the fixed validation checks are successful rejection checks only when the
specified rejection occurs; they are not study rows to be omitted.

All reports and metrics have their own available/unavailable status and reason.
Report last-valid timestamp/observation phase for incomplete rows. Keep the
original planned denominator, including every branch in a partially completed
H pair. Earlier valid results remain preserved, but no incomplete run or seed
group is used in a substitute aggregate. Four-group aggregate entries remain
unavailable if any required replicate is missing.

## Future files, validation gate and artifacts

If Ryan approves this final proposal, limit future work to:

- `tools/trajectory_mixture.py`: the fixed candidate and its public primitive/
  runner-state interface; an implementation author owns it.
- `tests/test_trajectory_mixture.py`: the bounded deterministic candidate checks
  and mutations; the implementation author owns it, with independent review.
- `tools/run_trajectory_mixture_study.py`: the fixed inputs, independent bounded
  oracles, evaluation, status/budget handling and one-launch runner; the parent
  owns it independently of the candidate implementation.
- `tests/test_trajectory_mixture_study.py`: input/status/oracle/causality checks
  for that runner; the parent owns it, with read-only independent review.
- `docs/trajectory_mixture_results.md`: all eventual outcomes, limitations,
  preservation audit and decision; the parent owns it.

The existing design and this final proposal remain frozen. No production C++,
old source/tests, dependency, configuration, CI, gate, or prior protocol changes
are included. Any need to change the frozen method, inputs, support rules,
thresholds or ownership scope requires another review before execution.

Before the single study launch, independently review the implementation and
all bounded oracle equations/code; pass the exact checks and mutation detection;
configure/build Debug, run CTest, run full pytest with RuntimeWarning as an
error, and run the real C++ development demo with development seed 20260921 in
a fresh directory. Inspect its existing trajectory/error plots. Activate the
project-local environment and retain Eigen/GoogleTest and pinned dependencies.
No C++ changes are planned, so no new sanitizer run is required by this scope.
No held-out navigation seed, consumed recording replay or saved conditional-
reference study sample is used. Existing pytest synthetic quadrature and sampler
regressions remain part of the unchanged required suite. No hosted-CI success
is inferred.

These pre-launch checks are separate from the 120 s study cap. Reserve a
separate aggregate execution allowance of 900 s wall time, 2 GiB process-tree
peak RSS and 256 MiB retained validation/demo output for this approved milestone's
checks. Aggregate elapsed execution time and retained output over every invocation,
including failed ones and repairs; apply the 2 GiB memory limit to each active
invocation tree, including its monitor. Record live aggregate RSS at 100 ms,
retain available OS job high-water values separately, and assess the larger
observed value against the limit. Validation descendants can be short-lived:
these are observed resource limits, not proof of an exact continuous tree peak.
Polling granularity is a declared limitation, not by itself an unresolved gate.
An actual monitor/query failure or unsupported required OS accounting leaves
that resource gate unresolved and blocks study launch; an observed overrun
fails it. The monitor and its platform/unit accounting must be independently
reviewed before the first validation command. A failed or unresolved gate blocks
study launch. Repairing a demonstrated implementation bug
before launch may repeat the affected check within the same allowance and
unchanged protocol, with independent review; it is not permission to inspect
study-seed outcomes, tune the candidate, or retry the study. Exhausting the
allowance stops for review. After all gates pass, freeze the clean source commit
and hashes before opening any study RNG stream.

Use fresh ignored roots `results/trajectory-mixture-checks-v1/`,
`results/trajectory-mixture-demo-v1/`, and `results/trajectory-mixture-study-v1/`.
Before the first validation invocation, require all three roots to be absent;
otherwise stop without overwriting or silently choosing a retry suffix. Within
this one approved milestone, permitted pre-launch check repeats use new numbered
attempt subdirectories and log names inside the checks/demo roots. Preserve
every earlier attempt. The study root is created for exactly one launch and is
never reused. If a final pre-launch gate blocks launch, record the complete
planned study inventory as not-started with its reason in the checks root.
The study manifest must contain approved proposal/source hashes,
checkpoint factors and assembled inputs, exact schedules, M/seed/stream map,
software versions, complete planned status matrix, start/end times, measured
costs, and all output hashes. Preflight preservation hashes cover all existing
tracked sources and saved study artifacts; compare them again after execution.

The fresh study root retains the complete input JSON, run/report/metric status
matrix, independent oracle moments, individual and aggregate comparison tables,
and per-run event diagnostics. Save component a, full conditional m/C, finite
log weights and identities at every prescribed report, with H's shared initial
snapshot and each branch endpoint. This is bounded report storage, not a full
angular path archive. The fixed random streams, source/runtime hashes and step
schedule provide replay provenance; no saved sample array from an old study is
opened. Save two fixed study figures from these existing tables, within the
same study caps: H x/y variance ratios for every paired branch and seed, with
the exact ratio one marked; and O per-event ESS/maximum weight plus per-report
supported accuracy summaries for every replicate, with full-six versus latent-
four coverage labeled. Show failed, unresolved and missing entries explicitly;
no selected best seed, later-position accuracy implication or new metric is
introduced by the plots. The parent inspects both figures with the manifest,
tables, snapshots and final diff. Validation logs, mutation outcomes, resource-
monitor logs and the C++ demo live in their separate capped roots. No failure
file or unfavorable row is deleted to meet an output limit.

## Proposed decision and stop boundary

Approval would authorize only those implementation files, pre-launch checks,
and one fresh frozen synthetic study. It would not approve another method,
recording/reference comparison, new recording, recovery policy, GPS return,
retuning, production integration, push, merge or release.

The proposal deliberately allows no-go and inconclusive outcomes. Prior
transition proposals may concentrate badly; four repetitions may not establish
the requested precision; the chosen component count or resource caps may be
inadequate; conservative support handling may refuse valid nearly singular
inputs. Passing all supported comparisons would still leave multi-step nonlinear
position accuracy without an independent oracle. Those are limits to report,
not hidden work items or automatic reasons to enlarge the study.

## Proposal work and independent review

The author read issue #6, the reviewed design, instructions and existing model
context, and used symbolic/manual algebra to specify this proposal. No proposed
constant was evaluated. No implementation, executable pseudocode, new test or
fixture file, numerical calculation, quadrature, sampling, simulation, replay,
benchmark, build, CTest, pytest or demo was run. File and Git reads and byte-hash/
diff checks are preservation work only. Only this new proposal note is authored.

### Review disposition and reconciliation

The design author specified the three case groups, correlated inputs, one-step
oracle, random streams, targets and failure rules. An independent mathematical/
protocol reviewer checked the complete draft against the reviewed design and
existing model. The parent also read every section and manually checked the
conditional nine-dimensional oracle, process-noise correlations, full moment
reconstruction, shared-heading pairing and claimed comparison coverage.

Review added direct narrow-variance/whitened covariance checks, per-replicate
guards, explicit four-group uncertainty accounting and a clear separation of
later position outputs from independently supported comparisons. The reviewer
confirmed the one-step characteristic identities and the bounded interpretation
of the four-replicate descriptive precision rule.

The main disagreement was an incomplete status contract: a valid but unresolved
covariance could stop a row without a matching execution status. The parent
added unresolved execution, assessments for incomplete rows, preserved-prefix
accounting, and failure-before-defer-before-pass precedence. Further corrections
made rate-noise timing explicit, bounded the two figures, distinguished existing
regression sampling from new study comparisons, and specified resource
measurement limitations and retention of every validation attempt. None changed
the case inputs, M, study seeds, stochastic targets or selected candidate.

Disposition: the concrete proposal is ready for Ryan's implementation/execution
review. Its engineering targets and resource limits are proposed constraints,
not observed feasibility. Numerical support restrictions, finite-M error,
concentration, coarse replicate precision and later nonlinear position accuracy
remain limitations. A review acceptance cannot turn them into passed experiments.

### Preservation and work performed

The parent checked all 88 files tracked at the starting commit, including the
accepted design note; their hashes are unchanged. All 541 prior-study files and
their file lists still match the saved preservation record. The 12 later
diagnostic evidence artifacts, 24 development-demo artifacts and saved manifest
checks also match. Only this new proposal is added, and the parent inspected its
complete diff and whitespace. File hashing did not decode or analyze saved
sample arrays.

The original checkout is clean at c834bf3 and the integration branch is clean at
53be2f7. No code, test, configuration, prior protocol, recorded result or saved
artifact changed. No numerical calculation or experimental validation was run
for this proposal; no build, CTest, pytest, demo, sanitizer or hosted CI success
is claimed. No push, merge, release or remote change was made.

### Exact approval requested

Approve only the five future files listed above, their independent review and
bounded pre-launch validation (900 s aggregate command execution, 2 GiB observed
per-invocation tree memory, 256 MiB retained validation/demo output), followed
only after passing the final gates by one fresh fixed synthetic study. That
study has M=2048, the thirteen prescribed branch/run rows, the four paired or
independent seed groups as specified, and limits of 120 s, 512 MiB assessed
wrapper/worker peak memory and 64 MiB total output. All inputs, criteria,
status rules and no-retry conditions in this document would remain fixed.

Afterward report every result and limitation, update the relevant issue, make
local commits, and stop for review. This proposal does not authorize its own
implementation or execution. Work stops here pending Ryan's approval.
