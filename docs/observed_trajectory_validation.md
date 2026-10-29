# Observed-path prototype checks

The staged local milestone in [issue #9](https://github.com/rzcheng/Navtrace/issues/9)
is complete. Independent mathematical review accepted one received-prefix
Gaussian angular-path proposal, and its deterministic prototype passed the
required checks. No stochastic candidate experiment ran. The previous
prior-proposal study remains **no-go**.

The next decision is whether to approve the single bounded experiment in
[observed_trajectory_proposal.md](observed_trajectory_proposal.md). It reserves
four unopened seed groups for 24 synthetic prefix rows at M=2048, with a single
120 s / 512 MiB / 64 MiB attempt. It is not an automatic continuation of this
prototype milestone.

## What changed and why

The new prototype conditions the entire received gyro/wheel prefix in active
Gaussian primitive coordinates. It then describes position and speed as a full
conditional Gaussian for each supplied angular path. Full checkpoint and
observation-induced correlations remain. The physical Euler model, noise timing,
sensor order and production estimator are unchanged.

The design review rejected a local new-noise adaptation as the milestone's
candidate: with the past angular path fixed, wheel speed cannot inform the new
independent angular noise. Rebuilding the received-prefix posterior retains
wheel information through checkpoint correlations and the dependencies induced
by earlier gyro observations. It can revise inferred past paths within the new
prefix calculation, but cannot alter a previously returned report or use an
unreceived same-time wheel observation.

The importance identity is exact: prior density times marginalized observation
likelihood divided by the angular proposal equals the Gaussian evidence. The
prototype exposes all density terms and their normalization constants. Its
conditional position/speed law uses old-angle, old-speed propagation with
end-step diffusion and full covariance. No likelihood is applied a second time.

This is a bounded batch diagnostic with increasing dense-matrix cost. It is not
a streaming filter or a production replacement. Uniform theoretical weights do
not establish empirical ESS improvement, accurate finite-M position moments or
calibration. GPS is excluded because its nonlinear position observation would
invalidate the simple Gaussian-prefix conditioning used here.

Exactly five files were added across this milestone:

- `docs/observed_trajectory_design.md` records the reviewed equations and gates.
- `tools/observed_trajectory.py` implements the immutable, bounded prefix API.
- `tests/test_observed_trajectory.py` supplies independent deterministic checks.
- `docs/observed_trajectory_proposal.md` freezes the unexecuted next proposal.
- This validation note records actual evidence and the stop boundary.

## Independent checks and outcomes

The implementation agent derived the candidate and owned only its two new
Python files. It performed no numerical execution. The independent read-only
reviewer separately derived the proposal and reviewed the final design, complete
source/tests, monitor reuse, logs, demo figures and next proposal. The parent
reconciled the derivations, inspected the source and every validation result,
ran commands centrally, checked preservation and owned Git operations.

Before implementation, review clarified active checkpoint factors, exactly
which SPD matrices receive the existing resolution test, empty-coordinate laws,
full Gaussian normalization and physical-oracle independence. Before the first
test run, parent review also found that coordinate width needed checking before
numeric coercion; the agent added that guard and conversion-trap tests. These
were review corrections, not failed numerical attempts. No equations, fixtures
or tolerances changed after execution began.

The new tests build an independent physical 9D one-step or 12D two-step Gaussian,
condition it directly, and compare it with the prototype. Expected observation
and endpoint maps do not reuse the prototype's maps, layout or old study oracles.
The checks cover every gyro/wheel prefix of the fixed two-step fixture, full
cross-covariances, both marginalized and full-joint importance identities, scalar
wheel conditioning, the remaining gyro rate/bias ambiguity and causal prefixes.

All analytical Gaussian, endpoint-map and density checks passed their unchanged
absolute tolerance of `1e-10`. Finite-difference checks used `1e-6` perturbations
and `1e-7` absolute/relative derivative tolerance. Symmetry/PSD checks used the
existing scale-aware tolerance without clipping. The empty-support and singular/
unresolved rejection cases passed. Earlier outputs were immutable and repeatable.

Seven temporary in-memory mutations were detected and restored: omitted
likelihood, doubled likelihood, omitted prior normalizer, omitted proposal
normalizer, discarded checkpoint cross-correlation, discarded conditional
cross-covariance and use of the new angle in displacement. Their expected
assertion failures are regression evidence, not failed validation attempts.

Actual metered commands:

| attempt | check | result |
|---:|---|---|
| 1 | Debug configure with existing pinned dependencies | passed |
| 2 | Debug build, parallelism 2 | passed |
| 3 | CTest | 23 passed |
| 4 | unchanged real C++ development demo, seed 20260921 | passed; both figures inspected |
| 5 | new deterministic test file, RuntimeWarning fatal | 35 passed in 0.13 s |
| 6 | full project pytest, RuntimeWarning fatal | 278 passed in 19.95 s |

There were no failed or unresolved command attempts. New prototype tests used
at most two physical substeps and three fixed coordinate rows. They drew no
random samples. The full regression suite includes existing fixed-seed sampler/
quadrature tests, and the unchanged demo generates its normal development data;
those were authorized regression validation, not new candidate evaluations.
Reserved study seeds 96001–96004 were not opened. No saved reference array was
used to evaluate this prototype, and no consumed recording was replayed for it.

The C++ demo's gated position RMSE was 0.132119 m nominal and 0.174847 m dropout;
dropout end-window error was 0.374852 m. These reproduce the unchanged development
behavior and say nothing about this prototype's navigation performance. The
parent and reviewer inspected full trajectories and detailed error plots,
including the correct reference-window/outage labels. No C++ source changed,
so no new sanitizer run was required. No hosted CI run is claimed.

## Resources and preservation

Six validation invocations charged **35.833361 s of 900 s**, including the
monitor's conservative finalization allowance. The largest assessed per-invocation
peak was **296,452,096 bytes of 2 GiB**. The final full-suite receipt counted
**206,311,337 bytes of 256 MiB** retained output; its persisted pre-final-flush
resource row counted 206,311,115 bytes. These differ by final metadata writes,
not by discarded evidence. Final report/manifest metadata remains subject to
the same output cap and is checked separately at finalization.

The unchanged monitor samples process-tree RSS every 100 ms and retains OS
high-water values. Short-lived descendant peaks can escape polling; the numbers
are the larger observed assessment, not a proof of exact continuous peak memory.
No unsupported resource measurement or observed overrun occurred. These costs
cover validation, including preserved C++ regressions, and are not a benchmark
of the proposed 100-step, 2048-point experiment.

All **95 baseline tracked files**, the complete inventory and bytes of **1,184
saved worktree artifacts**, and **579 older artifact hashes** remained unchanged.
All 24 manifest-listed new demo artifacts and its source hashes verified. The
original checkout stayed clean at `c834bf3`, and the earlier position-moment
integration checkout stayed clean at `53be2f7`. No production/configuration,
prior protocol, consumed-seed label, old result or recorded failure changed.

Fresh evidence is in `results/observed-trajectory-checks-v1/`: the validation
ledger, six numbered command logs/resource records, new test caches, copied font
cache, preservation records and final manifest. The demo is in
`results/observed-trajectory-demo-v1/`. Generated artifacts remain outside Git.
The ledger retains exact commands and source/runtime provenance; no attempt
was deleted or replaced.

## Source freeze and remaining decision

The reviewed design was frozen before implementation at `e12098e`, SHA-256
`7ca53abdf9b7618f56359eaf865f3682b15960d6d7b7e6a2c9c333ee06000ef8`.
The tested prototype and tests were committed locally at
`1a3ebcf9982688c71c63e1d36112be5ffc5d9618` on
`codex/trajectory-mixture-design`:

| file | SHA-256 |
|---|---|
| prototype | `1c8d4700b49ba81dfa935127c893b57acdf600d53018eea6a270be33d1ad7899` |
| tests | `35f928b9ed7106b9a100fcd6c8237633c719da8f0ca9d58652a19d0a44fc425a` |

The new deterministic prototype is mathematically supported within its stated
contract and has passed these fixed checks. Finite-M accuracy, cost and numerical
resolution at the proposed longer prefixes remain unmeasured. Later nonlinear
position accuracy still lacks an independent oracle. The old ESS failures,
individual mean-error failure, H precision limitation and underflow evidence
remain valid historical findings.

Recommendation: approve only the separately frozen synthetic proposal if those
remaining feasibility questions are worth measuring. No further experiment,
production change, push, merge or release was performed or authorized by this
completion. Work stops here for that decision.
