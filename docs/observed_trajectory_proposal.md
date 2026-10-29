# Proposed observed-path check

This is the next decision proposed by [issue #9](https://github.com/rzcheng/Navtrace/issues/9),
not authorization to execute it. The current milestone permits design and
fixed-coordinate prototype validation only. Seeds 96001–96004 below are reserved
and must remain unopened until Ryan approves this exact experiment.

## Question and one candidate

Does the frozen received-prefix angular proposal give adequately resolved
finite-M moments on the already published O synthetic sensor sequence, within
the existing study resource caps? Use only the proposal in
[observed_trajectory_design.md](observed_trajectory_design.md), with M=2048.
There is no count increase, alternate proposal, resampling, tuning, recovery or
production integration. The candidate rebuilds a full received-prefix posterior;
it is not the previous persistent forward filter.

Uniform weights follow from the exact Gaussian proposal algebra. They are an
implementation check, not empirical proof that the old collapse problem or
nonlinear position approximation is solved. This study asks about approximation
and cost after that algebraic change. It is not a paired performance comparison
against the consumed prior-proposal seeds, and it cannot overturn the earlier
H precision result or establish later nonlinear-position accuracy.

## Fixed inputs, prefixes and randomization

Use the unchanged O checkpoint, diffusion, sensor variances and exact 32-event
sequence in `docs/trajectory_mixture_proposal.md`, section O. Its frozen SHA-256
is `a9c23b2b40aaa220c030450e35431fd8a29c08de2f2b6b8bf2bde47b014b70ae`.
Use the full correlated checkpoint in physical order `(px,py,yaw,speed,rate,bias)`,
checkpoint timestamp 0, max step 0.01 s and max gap 2 s. Do not copy a rounded
covariance from a results table. Assemble the declared factors and retain the
assembled input in the new manifest. The sequence contains fixed sensor values,
not simulated truth. No recording or saved sample array is opened.

The six prefix endpoints, in order, are 0, 10000, 250000, 500000, 750000 and
1000000 us. Endpoint zero has no events. At each later endpoint include exactly
the received events through it, preserving gyro before wheel. In particular,
10000, 500000 and 1000000 us include the same-time wheel; 250000 and 750000 us
end after gyro. No report inserts a propagation step. Each prefix gets an
independent batch call from the same checkpoint; earlier reports are immutable.
The final prefix uses 100 positive substeps and 32 observations, inside the
prototype's 128-step/64-event limits.

Four group keys are 96001, 96002, 96003 and 96004, in that order. Within each
group run prefixes in the above order. For prefix index j=0,...,5, construct
NumPy `Generator(PCG64(SeedSequence([group_key, 9, j])))`. Draw one row-major
`standard_normal((2048, dU))` array of float64 posterior-standard coordinates,
then call the candidate. No independent Z draws are made: integrate the full
conditional Z Gaussian. dU and coordinate order come from the frozen prototype
layout and must be saved. There is no redraw, count sweep or optional stopping.

These are 24 planned prefix rows and four independent algorithm groups at each
prefix, not 24 independent navigation trials. Prefix streams are independently
keyed; times are not pooled as extra replicates. Seeds 95001–95004 remain consumed
historical study keys; 94999 and the existing development/demo seeds remain
validation keys. No proposed-study key may be used in tooling smoke tests.

## Independent comparisons

Use the previously reviewed frozen O analytical targets, without changing their
inputs or definitions: full six-state checkpoint moments, the exact one-step
conditioned-nine-dimensional construction at 10000 us, and the exact linear
four-state latent Gaussian filter for `(yaw,speed,rate,bias)` thereafter.
The implementations in `tools/run_trajectory_mixture_study.py` remain read-only.
Re-review the new runner's use of these independent functions before launch;
prototype posterior matrices or maps cannot supply their expected values.

Retain every full six-state mean/covariance, but label position and position/
latent correlations at later times **descriptive, without an independent oracle**.
No new nonlinear multi-step oracle or empirical calibration claim is included.
Use unwrapped yaw consistently. Check the exact log evidence at every prefix
against the latent linear filter, including zero at the empty prefix.

For oracle mean mu and covariance P=GG^T on its stated positive support, use
`dm=solve(G, mean-mu)` and `DP=G^-1 covariance G^-T-I`, using two triangular
solves rather than explicit inverses. Evaluate every mean coordinate and
unique covariance entry. Do not choose axes after seeing candidate outputs,
floor denominators or use a pseudoinverse to score unresolved oracle support.
Retain original-unit values alongside the normalized errors.

The error targets remain epsilon=0.10 for means and 0.20 for covariance entries.
For each scalar across four groups, define average d and
`se=sqrt(sum((d_r-average_d)^2)/(4*3))`. Require `4*se <= epsilon/2` and
`abs(average_d)+4*se <= epsilon` for pass. A quantity fails if
`abs(average_d)-4*se > epsilon`, or any individual `abs(d_r)>2*epsilon`.
Otherwise it is inconclusive. Require each individual whitened covariance-error
spectral norm <=0.5 and the spectral norm of its four-group mean <=0.3.
Average each replicate's own covariance separately from its mean; do not add a
between-replicate spread and call that the same statistic.

These four-SE rules are descriptive engineering limits with only four groups,
not calibrated confidence intervals, simultaneous guarantees or ellipse coverage.
Both complete failure and inconclusive outcomes are legitimate terminal results.

## Exact checks, retention and decisions

At every row require finite coordinates/states/log density terms and symmetric
PSD covariances under the unchanged policy. Check every supplied point's
`log prior + log marginalized likelihood - log proposal - log evidence` against
zero with absolute tolerance 1e-10. Require absolute log-evidence error <=1e-10.
Check normalized weights against 1/M at the unchanged 1e-10 numerical tolerance;
record ESS and maximum weight as structural diagnostics, not a new concentration
success criterion. Preserve all support refusals and failed identities.

Save all 24 planned status rows before the first draw. Each row records group,
prefix, execution status, assessment, all supported errors, exact-check maxima,
full moments, dimensions, time/memory/output measurements and failure reason.
A failed row is not replaced or removed. Missing required rows/replicates cannot
produce a pass. Track invalid, failed, unresolved, completed and not-started
execution distinctly from pass/fail/inconclusive assessment.

For every completed row retain endpoint angular values, conditional ell means
and full 3x3 covariances, normalized log weights and represented full moments in
an NPZ snapshot. Retain input factors/assembled matrices, sensor sequence,
per-prefix coordinate layout, seed/bit-generator identity and initial/final RNG
state, source/configuration/runtime hashes and coordinate-array SHA-256 over
C-order little-endian IEEE-754 float64 bytes. Also retain every point's log prior,
marginalized log likelihood, log proposal and importance-ratio residual vectors,
so maxima can be audited without reopening streams. Dense
prefix posterior matrices, full angular histories and full supplied coordinate
arrays are deliberately not saved; this storage choice is fixed before launch.
They may be regenerated in a separately authorized reproduction, not silently
recomputed after this experiment. An endpoint audit reconstructs mixture moments
from the retained endpoint snapshots without reopening RNG streams.

Produce two fixed figures from these tables: supported mean/covariance error
summaries for all four groups at all six correctly keyed times, clearly separating
full-six and latent-four support; and elapsed time/dimension plus structural
weight diagnostics at every prefix. Show unavailable, failed and inconclusive
entries. Do not imply a later position-accuracy curve or select the best group.
Parent and independent reviewer inspect both figures and every row/status.

Any actual exact-identity, accuracy, covariance, arithmetic or resource failure
makes the experiment no-go. Otherwise unresolved support, insufficient precision,
missing comparisons/replicates or unresolved resource accounting makes it defer.
Only all completed rows with every required comparison passing can pass this
bounded synthetic question. A pass still leaves later position accuracy, H
precision, long-prefix scaling and navigation feasibility unanswered.

## Frozen source for a later approval

The deterministic prototype passed its reviewed checks and is frozen locally at
`1a3ebcf9982688c71c63e1d36112be5ffc5d9618`. A later runner must verify these SHA-256 values
before launch; this proposal grants no permission to change them:

| file | SHA-256 |
|---|---|
| `docs/observed_trajectory_design.md` | `7ca53abdf9b7618f56359eaf865f3682b15960d6d7b7e6a2c9c333ee06000ef8` |
| `tools/observed_trajectory.py` | `1c8d4700b49ba81dfa935127c893b57acdf600d53018eea6a270be33d1ad7899` |
| `tests/test_observed_trajectory.py` | `35f928b9ed7106b9a100fcd6c8237633c719da8f0ca9d58652a19d0a44fc425a` |
| `tools/run_trajectory_mixture_study.py` | `60538d532b3bd01b416c5d7f4678d5d18ed4fdae6e89320f27da2c5f146dc674` |

The last file supplies the preserved independent oracles and command monitor,
not the new candidate. The future runner and its tests need their own independent
review and source freeze. The current deterministic checks used at most two
physical substeps and three supplied coordinate rows; feasibility at the proposed
100-substep/2048-point scale is deliberately unmeasured until that approved run.

## Budget, implementation gate and stop

A later approval would authorize one new runner, its deterministic tests and a
results note: `tools/run_observed_trajectory_study.py`,
`tests/test_observed_trajectory_study.py` and `docs/observed_trajectory_results.md`.
It would not authorize changing the now-tested prototype or this design to fit
study outcomes. The parent owns runner/results/integration/Git; an independent
reviewer owns no files. Source must be clean, committed, independently reviewed
and hashed before opening any reserved stream.

Pre-launch checks would use fresh
`results/observed-trajectory-study-checks-v1/` and
`results/observed-trajectory-study-demo-v1/`, with 900 s aggregate command time,
2 GiB assessed per-invocation tree memory and 256 MiB retained validation/demo
output, including failures. Create these roots once and preserve unique numbered
attempt subdirectories for allowed demonstrated implementation-bug repairs within
the same budget. Such repairs may repeat affected checks after independent review;
they may not alter fixtures, tolerances, the candidate or this protocol. Run
affected checks, full pytest, CTest and the
unchanged development demo; inspect artifacts. Use the reviewed monitor and
retain its 100 ms sampling/short-lived-descendant caveat. No C++ changes are
planned; no hosted CI or sanitizer success is inferred.

The single study would use fresh `results/observed-trajectory-study-v1/`, capped
at **120 s through final flush, 512 MiB assessed wrapper-plus-worker peak memory,
and 64 MiB retained output**. Reserve 4 MiB for statuses/failure metadata before
artifact writes. Require all three roots absent at milestone initialization,
then permit only the numbered validation attempts described above. The study root
is created for exactly one launch. Do not overwrite or silently choose a retry
suffix. Measure preparation, conditioning, materialization, checks, plots
and finalization within the study cap. No child processes are permitted in the
study worker. Record full planned inventory and not-started reasons if a gate
or cap prevents further rows. Retain all failures; no retry or larger budget.

After the single attempt, inspect numerical tables, endpoint snapshots, plots,
hashes and resource logs; update the issue with actual evidence; commit locally
and stop. Do not push, merge, release or alter production. This proposal alone
does not authorize any of those future files, validations or study draws.
