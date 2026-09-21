# M3: predefined diagnosis and consistency study

Starting revision: `32aaf85`, M2 preserved. This protocol is frozen before new
evaluation. Original filter/prior file SHA-256:
`23b03b85379e44aac9caf702e2c873789533ce93bc2a4b2b2db874eaa41c54b4`.
No tuning of Q, R, P0, or the gate is authorized by this experiment.

## Competing explanations before changing behavior

| Hypothesis | Expected evidence | Small discriminating check | Evidence against |
|---|---|---|---|
| A: weak yaw/rate/bias information before loss | Weak scaled sensitivity; coupled rate/bias errors | 2 versus 20 s horizons; stationary versus moving | Strong directions and small pre-loss errors |
| B: restrictive/inaccurate prior | Errors already implausible before loss | Record errors/P at 1.99 s; compare sampled-prior sanity case | Early uncertainty describes errors |
| C: process-model mismatch | Nominal stress NEES differs from matched sanity | Independent Euler–Maruyama truth with sampled prior | Both studies show comparable behavior (not conclusive) |
| D: misleading covariance/cross-covariance | Error lies in small S direction; small ellipse misses truth | Full eigen-direction innovation and marginal position NEES | Residual lies in well-supported large-variance directions |
| E: large-error linearization | Large yaw errors and local Gaussian diagnostic failure | Record wrapped error/std and nonlinear motion sensitivity | Failure starts while local errors remain small |
| F: bad fix changes other states through correlations | Large accepted correction in yaw/rate/bias | Independent K r and Joseph recomputation | No cross-state corrections; component error actually improves |
| G: self-reinforcing rejection | Clean fixes rejected; GPS correction exactly zero | Compare accepted/rejected histories and counterfactual ungated run | Continuing GPS corrections explain divergence |
| H: numerical/timestamp/logging bug | Independent calculations disagree | Reproduce raw outputs, finite differences, pre/post snapshots | Reproduction and independent identities agree |

These mechanisms may coexist. No single plot proves causation; PSD, low RMSE,
local observability, consistency, and recovery are different properties.

## Reproduction and event diagnosis

Reproduce nominal and early-dropout/offset M2 seed 70002 with both EKFs into
`results/m3-reproduction`. Require exact old CSV equality before modifications.
After diagnostic instrumentation, require the original trajectory CSVs and
all original GPS columns remain identical. Extend logs only; preserve equations.
Inspect all M2 seeds as consumed diagnosis data, with table times 1.6,1.8,
40, first accepted offset,44.8,45,46,50,60 s. Use full pre/post x,P; z,R;r,S;
eigenvalues/vectors/contributions; elapsed GPS availability/acceptance and
rejection streak; wrapped state corrections. Independent Python reconstructs
K r and Joseph P. Other-sensor corrections remain visible in trajectory rows.

## New evaluation, maximum 170 replays, one worker

Settings are in `scenarios/consistency.json`. Development uses only consumed
seed 20260921 and M2 70001–70003. Stress seeds **81001–81010** are new independent
sensor realizations, paired across all seven conditions and two unchanged EKFs
(140 replays). DR is unchanged and does not address covariance/observability;
M2 DR results are preserved, not rerun as a new estimator comparison.

- Original analytical trajectory: nominal; early dropout [2,22) s with clean
  or +(6,-4) m GPS return during [22,27); later dropout [20,40) with clean or
  offset return during [40,45). Both outages last 20 s. Return motion phase
  still differs; do not attribute every early/later difference solely to onset.
- New weak-start family: same first 0.2 s and initial truth as original, then
  stationary through 20 s; from 20 s onward use original speed/rate commands,
  integrated from its own pose. Test the same late clean and offset returns.
  The prior is unchanged. The stop/restart are deliberate process mismatch;
  wheel measurements observe them. This tests limited pre-loss motion versus
  the original moving/turning first 20 s, not a perfectly isolated motion mode.
- All three sensors keep 100/20/5 Hz and original Gaussian noise; constant
  true gyro bias +0.015 rad/s in stress. Same sensor random draws across families;
  remove/offset GPS from nominal rows without changing R or other sensors.

**Model-matched sanity:** 30 independent seeds **82001–82030**, 2 s, basic EKF
only. Independently implement discrete Euler–Maruyama truth in Python at 10 ms,
with v/rate/bias diffusion exactly the original q*h. Sample initial truth from
N(original mean,P0), wrap yaw, then generate independent noisy sensors. This
matches the discrete stochastic model, not an exact nonlinear Gaussian filter.
Use separate random streams for initial state, process, gyro, wheel, GPS.
The main analytical stress scenarios remain unchanged and intentionally mismatched.

Smoke one development seed first; record runtime and byte size, estimate the
full study, then run sequentially. Compress raw CSVs losslessly after evaluation
to keep storage modest; hash retained bytes. No parameter sweep or repeated
evaluation-seed tuning. Failures are counted explicitly, retained with stderr,
and excluded only from statistics requiring missing quantities (denominator shown).

## Statistics fixed before evaluation

Evaluation uses every 100 Hz timestamp after all simultaneous observations.
Compute wrapped full-state NEES with a Cholesky solve; position NEES uses the
marginal 2x2 P block. No inverse-block substitution, pseudoinverse, jitter, or
eigenvalue clipping. Undefined statistics carry explicit counts/nulls.
Record yaw/bias error and std, rate/bias correlation, 95% nominal position
ellipse coverage, all available pre-decision NIS, and rejection history.

Fixed stress timestamps (seconds): 0,1.8,1.99,19.99,21.99,22,27,39.99,40,45,60.
Matched timestamps: 0,.5,1,2. Aggregate across independent runs at the SAME
timestamp (N=10 stress, N=30 matched). Mean-NEES reference bounds use SciPy's
chi-square quantiles with N*d degrees of freedom divided by N, alpha=.05.
These are approximate references for the nonlinear matched EKF, descriptive
only for deterministic-prior stress/fault/gated cases. Do not use accepted-only
NIS as unbiased calibration evidence or correlated time samples as independent N.

Save all per-run RMSE, final/max error, actual outage-end error, clean/fault GPS
counts, longest clean rejection span, and settling: error <=1 m over >=2 s of
elapsed timestamps, max supported sample gap 10 ms. A gap breaks the dwell;
201 regular samples still span 2 s. No interpolation over missing observations.
First accepted clean fix and position settling are separate. Unobserved times
are null and observation horizons shown. Summaries retain all losses/failures.

## Offline local sensitivity

Use actual C++ discrete F evaluated on independently specified nominal paths;
Phi(k)=F(k) Phi(k-1), Phi(0)=I. Stack R^(-1/2) H Phi D, never Kalman correction
matrices. D=diag(1 m,1 m,.1 rad,1 m/s,.1 rad/s,.03 rad/s). Stationary, straight
v=1.5, and constant-turn v=1.5,rate=.12; horizons 2 and 20 s; compare all available
sensors with GPS absent. SVD rank tolerance 1e-9*s_max. Save weak normalized and
physical directions. Gyro-only ambiguity is a regression test. This deterministic
local calculation ignores process-noise estimability and is not global observability.
Straight motion can inform heading and rate through position; turning is not required.

Illustrative stress plots always use the lowest new seed, never the worst/best
chosen afterward; retain full-scale axes. M2 event plots are explicitly diagnostic.
Freeze tested implementation before evaluation and record source/binary/input hashes.

## Decision checkpoint

Write confirmed mechanisms, contributors, unsupported alternatives, and remaining
uncertainty. Choose one of defect correction, justified model/prior change,
separate recovery policy, or M3-only limitation. Smooth offset GPS cannot prove
absolute trustworthiness. Do not implement M4 without an explicit causal trust
assumption and separately committed design/protocol/fresh seeds.
