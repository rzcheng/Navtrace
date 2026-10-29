# Synthetic trajectory mixture results

The single approved study in [issue #7](https://github.com/rzcheng/Navtrace/issues/7)
finished on 2026-10-18 with a **no-go** decision. All thirteen prescribed runs and
all 44 reports completed. The fixed M=2048 prior-proposal mixture failed the O
weight-concentration requirement in every seed group, and one individual latent
mean error exceeded its frozen ceiling. H's narrow position-variance comparison
remained inconclusive. Completing the computation within budget did not establish
accuracy, calibration, or navigation feasibility.

There is also an unresolved reporting defect: the lower two O accuracy panels
place saved errors at the wrong timestamps. The keyed numerical tables and
upper ESS/maximum-weight panels are correct. The original source, figure and
results remain unchanged; there was no post-launch repair, regeneration or retry.
A proposed reporting-only follow-up is tracked in
[issue #8](https://github.com/rzcheng/Navtrace/issues/8) and needs separate approval.

## Frozen scope and source

The [proposal](trajectory_mixture_proposal.md) was frozen at `7e44d76`, SHA-256
`a9c23b2b40aaa220c030450e35431fd8a29c08de2f2b6b8bf2bde47b014b70ae`.
The [design](trajectory_mixture_design.md) retained SHA-256
`644809e2f91ad5df3c3876b60625595981d87633ba2ddf0475be483daa7f6596`.

The clean implementation commit before opening study streams was
`036e14f7a547ff0cfab1666c5bcd31f6f8f357b7`, on
`codex/trajectory-mixture-design`. The source freeze added only:

- `tools/trajectory_mixture.py` and `tests/test_trajectory_mixture.py`;
- `tools/run_trajectory_mixture_study.py` and
  `tests/test_trajectory_mixture_study.py`.

This report is the fifth and final authorized tracked file. Existing code,
configuration, dependencies, protocols, tests and prior results were preserved.
No C++ estimator, Q/R, prior, gate, recovery rule or sensor ordering changed.
No saved reference distribution was evaluated, no new navigation recording was
created, and no alternative candidate was tried. Study seeds 95001–95004 were
opened only in this one launch; 94999 was the prelaunch control key.

The candidate keeps persistent unwrapped angular paths and a full conditional
Gaussian for position/speed. It uses the frozen prior transition, old-state Euler
motion, independent end-step diffusion, scalar observation likelihoods, fixed
component identities and finite log weights. It does not resample or replace the
continuing distribution with its reported Gaussian moments.

## Results

### L: exact linear limit

L completed all four reports and passed all exact checks. The largest absolute
error across its mean, covariance, whitened covariance, component, evidence,
weight and counter checks was `3.4528e-14`, below `1e-10`. All 2048 components
remained equally weighted. This confirms the declared deterministic-angle
linear limit, not a general nonlinear accuracy claim.

Across L and the paired H runs, all 180 recorded exact checks passed. Their
largest recorded absolute error was also `3.4528e-14`.

### H: shared heading and paired partitions

Both H1 and H8 completed for every seed. Their components and full moments
agreed with the same finite-ensemble analytical endpoint and with each other
within the fixed tolerance. The partitions are paired evidence, not eight
independent replicates. Four H1 seed groups supply the finite-M summaries.

| seed | endpoint x variance / exact | endpoint y variance / exact | paired H1/H8 execution | overall H assessment |
|---:|---:|---:|---|---|
| 95001 | 1.038579 | 0.991449 | completed | inconclusive |
| 95002 | 0.959820 | 0.986513 | completed | inconclusive |
| 95003 | 1.011852 | 1.004671 | completed | inconclusive |
| 95004 | 0.896450 | 0.971899 | completed | inconclusive |

The x-variance relative error averaged `-0.0233247`, with four-SE `0.125378`.
That exceeds the frozen precision ceiling `0.1` for covariance/variance errors,
so the comparison is inconclusive even though its observed average is small.
The y-variance error averaged `-0.0113666`, with four-SE `0.0270720`, and passed.
Five of the eleven endpoint scalar criteria were inconclusive; six passed.
The checkpoint's two scalar criteria passed. Both covariance spectral guards
passed; the endpoint norm of the average whitened covariance error was `0.0314843`.

These four-SE quantities are descriptive engineering rules. They are not
calibrated confidence intervals or simultaneous coverage guarantees.

### O: correlated observed sequence

All four runs completed all 32 observation phases and six reports. All failed
the requirement ESS >= 204.8 at every phase. The separate maximum-weight limit
of 0.05 passed in every run.

| seed | first ESS failure, time/phase | minimum ESS | maximum weight | final cumulative underflow count | execution / assessment |
|---:|---|---:|---:|---:|---|
| 95001 | 0.60 s, gyro | 102.2160 | 0.034015 | 359 | completed / fail |
| 95002 | 0.60 s, gyro | 104.3583 | 0.024397 | 360 | completed / fail |
| 95003 | 0.55 s, gyro | 103.7259 | 0.028941 | 322 | completed / fail |
| 95004 | 0.50 s, gyro | 95.4530 | 0.029370 | 342 | completed / fail |

The underflow values are cumulative diagnostic counts across operations, not
counts of distinct discarded components. Every finite log weight and component
was retained. Underflow makes numerical-resolution claims inconclusive; it does
not erase the independent ESS and accuracy failures.

At 1 s, the yaw-rate coordinate of the Cholesky-whitened latent mean error was:

| seed | whitened mean error, coordinate 4 |
|---:|---:|
| 95001 | -0.0589455 |
| 95002 | 0.133351 |
| 95003 | -0.150754 |
| 95004 | 0.236997 |

Seed 95004 exceeds the individual ceiling `2*epsilon = 0.2`. The four-group mean
was `0.0401622` with four-SE `0.353458`; averaging does not rescue that individual
failure. These are dimensionless errors whitened with the full correlated oracle
covariance, not raw yaw-rate errors in rad/s.

The exact full six-state oracle applies only at the checkpoint and after the
first 0.01 s step. The first-step yaw-rate whitened-mean comparison was
inconclusive because four-SE `0.0680104` exceeded its `0.05` precision ceiling.
Later comparisons use only the exact four-state latent Gaussian marginal.
Later nonlinear position and position/latent cross-covariance remain descriptive.

| O report time (s) | scalar criteria passed | inconclusive | failed | norm of average covariance error |
|---:|---:|---:|---:|---:|
| 0 | 28 | 0 | 0 | 0.0208163 |
| 0.01 | 27 | 1 | 0 | 0.0426937 |
| 0.25 | 8 | 7 | 0 | 0.0941145 |
| 0.50 | 11 | 4 | 0 | 0.0919984 |
| 0.75 | 7 | 8 | 0 | 0.0585678 |
| 1.00 | 8 | 6 | 1 | 0.108320 |

All individual covariance spectral norms were below 0.5; all average norms were
below 0.3. Passing those guards is not calibration. The H/O aggregate tables
contain 129 scalar criteria in total: 97 pass, 31 inconclusive, one fail.
These counts do not include the separately failed O ESS requirement or the
underflow resolution limitation. Failure takes precedence over inconclusive.

## Reporting defect retained

`inventory.json` is serialized with sorted string keys, then loaded before the
worker runs. O report-key order becomes
`0, 10000, 1000000, 250000, 500000, 750000`. The plotting code iterates those
values but pairs them with the chronological report-time array. Consequently:

- the 1 s errors appear at 0.25 s;
- the 0.25 s errors appear at 0.50 s;
- the 0.50 s errors appear at 0.75 s;
- the 0.75 s errors appear at 1 s.

The first two times are unaffected. The upper panels use explicit event
timestamps and are unaffected. The H figure is unaffected. The parent and
independent reviewer both found this mismatch during visual/numerical inspection.
The lower O traces must not be used to infer when an accuracy error occurred.
Use the explicit report keys in the saved tables and the table above.

The study's required presentation is therefore not fully correct, despite all
runs completing. This additional reporting failure reinforces the terminal
no-go; it is not presented as a successful plotting check. No implementation source or study output
was changed after discovering it. Issue #8 proposes a separate correction from
saved tables only, preserving the original figure and no-go decision.

## Validation and independent review

The implementation subagent wrote the candidate and its tests, then stopped
writes. It executed no numerical work. The parent inspected its full-covariance
split, old-state transition, Joseph wheel update, finite-log handling and
transactional RNG restoration, and ran all checks through the reviewed monitor.
The parent separately implemented the resource wrapper, independent oracles,
assessment logic and runner tests.

The independent review subagent was read-only throughout. It checked the Darwin
RSS API/layout/units before the first command; then checked the conditional maps,
likelihoods, within/between mixture covariance and all three mutation tests.
It independently derived the L batch linear oracle, H analytical/empirical
moments, conditioned nine-dimensional first-step O moments, and later latent KF.
It reviewed final source hashes, command evidence, demo plots, study artifacts
and the reporting defect. Its claims were checked against code and artifacts,
not accepted as substitutes for execution.

Actual commands and results:

| attempt | purpose | result |
|---:|---|---|
| 1 | initial focused tests, RuntimeWarning fatal | 37 passed, 2 failed |
| 2 | declared L matrix-product diagnostic | finite result; exact agreement with direct contraction |
| 3 | repaired focused suite | 40 passed |
| 4 | Debug configure, existing pinned Eigen/GoogleTest sources | passed |
| 5 | Debug build, parallelism 2 | passed |
| 6 | CTest | 23 passed |
| 7 | full pytest, RuntimeWarning fatal | 230 passed |
| 8 | real C++ development demo, seed 20260921 | passed; both plots inspected |
| 9 | final candidate/evaluator checks after reporting/gate edits | 43 passed |
| 10 | evaluator checks after repeat-launch preservation fix | 18 passed |
| 11 | independent audit of all saved study snapshots and hashes | passed |

The first failure was in the L oracle's 33-dimensional matrix product. The
backend emitted divide-by-zero, overflow and invalid warnings while returning a
finite result exactly equal to direct `einsum` contraction. The diagnostic and
failed attempt are retained. The oracle now uses direct contraction; no warning
was suppressed and no tolerance, fixture or numerical target changed. Later
focused and full tests passed with RuntimeWarning still fatal.

Other review repairs covered incomplete-group aggregation, failure precedence,
oracle support before inversion, resource failure recording and finalization,
plotting-cache confinement, group assessment visibility, and preserving existing
study evidence when a second launch is rejected. The final small gate/reporting
changes received affected-test reruns; the full 230-test result is the actual
full run before those final changes, not a claim that a larger suite was rerun.
The three required in-memory mutations were detected and automatically restored.

The development demo used unchanged settings. Nominal gated position RMSE was
0.132 m; dropout gated RMSE was 0.175 m and its end-window error was 0.375 m.
These are a regression/demo check, not trajectory-mixture validation.
No C++ source changed, so the scope required no new sanitizer run. No hosted CI
run is claimed.

The artifact audit checked all 44 snapshots, stable IDs, finite states and
90,112 saved finite log weights, normalized represented weights, covariance
symmetry/PSD and independent full-moment reconstruction. Maximum reconstruction
difference was `1.6324e-15`. Twelve reports carried underflow diagnostics.
All 53 manifest-listed artifacts passed their recorded SHA-256 checks. The
independent reviewer also checked the gate, inventory, resources and frozen
validation-ledger hashes.

## Resource accounting and preservation

Validation used eleven retained command attempts. The conservative total charge
was **74.5582 s / 900 s**, including one second reserved for each attempt's final
metadata flush. Recorded pre-final-flush execution totals were 63.5582 s; they
are kept separate from that charge. Maximum assessed invocation memory was
**1,054,638,080 bytes / 2 GiB**. Counted retained validation/build/demo/source-cache
output at the end of the last metered command was **203,113,467 bytes / 256 MiB**.

The initial output monitor counted checks/demo roots but omitted build products
and caches. This was found and reconciled before the build; prior outputs had a
conservative 2 MiB bound. Subsequent monitoring includes build products, caches,
and whole tools/tests trees conservatively. Two 5-byte disposable test fixtures
from the first focused runs were already absent after shared pytest temporary
cleanup; their loss is disclosed in `output-accounting-reconciliation.json`.
This task did not delete them. All command logs, numerical results and resource
records remain. Later pytest used unique retained temporary directories inside
the checks root. Do not read this as a claim that every temporary file survived.

The single study used **3.40235 s / 120 s** through final flush, **138,936,320 bytes
(132.5 MiB) / 512 MiB** assessed memory, and **13,199,081 bytes / 64 MiB** retained
output. Sampled live aggregate peak was 134,299,648 bytes. Wrapper and worker
lifetime high waters were 37,879,808 and 101,056,512 bytes; their conservative sum
was the assessed value. All required measurements were available, with zero
reported process-exit sampling races; no forbidden worker descendants were observed.

RSS was sampled every 100 ms. Validation's available OS job/self high waters
and sampled aggregate are observed bounds, not proof of an exact continuous
process-tree peak. The study's lifetime-peak sum is conservative, not an exact
simultaneous peak. Resources record elapsed through primary metadata flush;
the wrapper's captured stdout, retained in `study-launch-receipt.json` under the
checks root, records the final 3.40235 s observation.
The small final metadata size increase is included in the final byte count above.

The preservation audit verified all 89 pre-existing tracked files and 579 saved
artifacts unchanged before and after execution. The original checkout remained
clean at `c834bf3e5b9b421564c2e0050092895abe12dd43`; the position-moment integration
checkout remained clean at `53be2f72d15f8d485640eaa09868210231b3e886`.
The only tracked additions are the five authorized files. Nothing was pushed,
merged or released; remotes and published history were unchanged.

## Artifacts and boundary

All paths below are relative to this worktree:

- `results/trajectory-mixture-checks-v1/`: all numbered logs/resource records,
  preservation records, output reconciliation, frozen validation snapshot and
  `gate.json` (including all 93 reviewed source hashes and the font-cache hash).
- `results/trajectory-mixture-demo-v1/attempt-01/`: real C++ demo manifest,
  metrics, trajectories and error plots.
- `results/trajectory-mixture-study-v1/`: input/factor JSON, oracle moments,
  13-row/44-report inventory and aggregate tables, event diagnostics, 44 component
  snapshots, both original figures, resource record and manifest.

The study manifest identifies the clean source commit and hashes all 53 listed
artifacts, including the copied font cache and excluding the manifest itself. The frozen validation snapshot preserves the
prelaunch ledger even though the post-study artifact audit added a later attempt.
Generated artifacts remain ignored, not committed.

This milestone ends with no-go and the original reporting defect documented.
It does not justify increasing M, changing the proposal, retuning, testing
recordings, adding recovery, or claiming later position calibration. Any
estimator follow-up requires a separately reviewed scope. The single next
recommended decision is whether to approve issue #8's reporting-only correction
from the existing saved tables, with no estimator execution or study retry.
