# Observed-path experiment results

The single approved experiment in [issue #10](https://github.com/rzcheng/Navtrace/issues/10)
finished on 2026-10-18 with a **defer** decision. All 24 planned prefix rows
completed at M=2048. Of 110 scalar comparisons, 106 passed and four were
inconclusive under the frozen precision rule; none failed. All exact identities,
covariance checks and spectral guards passed. The attempt stayed within its
resource limits. These results do not satisfy the protocol's all-comparisons-pass
condition.

The observation-conditioned proposal is computationally feasible for this short
synthetic sequence. Later nonlinear position accuracy, the earlier H precision
limitation, long-prefix scaling and navigation behavior remain unresolved.
Uniform weights are an algebraic property of this proposal, not empirical
calibration or proof of accurate position distributions. The previous
prior-proposal study's no-go remains unchanged.

## Scope and source freeze

Ryan approved exactly the experiment in
[observed_trajectory_proposal.md](observed_trajectory_proposal.md), frozen at
`9a0e6c3`, SHA-256
`37c607341e4b4ceff7bbe550a5bd9b4778c31904985f248baaf934ae91c52491`.
The clean reviewed implementation commit before the first draw was
`075cd1271f76186f33337d64ec2c193f90fe3c40` on
`codex/trajectory-mixture-design`.

Only `tools/run_observed_trajectory_study.py`,
`tests/test_observed_trajectory_study.py` and this results note were added.
The prototype, old oracles, complete correlated O checkpoint, 32 fixed sensor
observations, diffusion, sensor variances, step partition, criteria and numerical
tolerances stayed fixed. Production C++, tuning, GPS gate and recovery behavior
were unchanged. No recording or saved conditional-reference samples were opened
for candidate evaluation. No navigation trial was run.

Groups 96001–96004 are now **consumed**. Each group used six independently keyed
prefix streams, `PCG64(SeedSequence([group, 9, prefix_index]))`, with one
row-major float64 `(2048,dU)` standard-normal draw per row. Conditional linear
primitives were integrated, not drawn. These are four algorithm groups at each
time, not 24 independent navigation trials. There was one launch, no retry,
count increase, seed substitution or post-launch code/plot repair.

| new source | SHA-256 |
|---|---|
| runner | `f5ac9b596a74db580fdb3c62a5fe432c4a1026f63207dd6a4996f2ca6430f2ca` |
| tests | `ccfe23100737e8bec1244b7ad8c51c810087135143be4991b2c39054f1278969` |

The prelaunch gate hashes all 102 source files and binds the initialization,
validation ledger, preserved-file audit and font cache. Its SHA-256 is
`bc59c48a180b04dbf9e358aecd0f60e91f192876a79818bba90289fa64fe93ba`.
All three output roots were absent at initialization; the study root was still
absent at the independent final launch review.

## Numerical results

The checkpoint and first-step comparisons use independent full six-state
oracles. Later comparisons use only the exact four-state latent marginal
`(yaw,speed,yaw_rate,gyro_bias)`. Full position moments and cross-covariances
are saved at every prefix, but later position values are descriptive without an
independent oracle. Comparisons whiten with the complete oracle covariance;
coordinate labels below identify positions in that correlated transformation,
not raw errors in meters or radians.

| prefix (s) | support | scalar pass | inconclusive | failed | norm of average covariance error |
|---:|---|---:|---:|---:|---:|
| 0 | full six | 27 | 0 | 0 | 0.0381245 |
| 0.01 | full six | 26 | 1 | 0 | 0.0283890 |
| 0.25 | latent four | 14 | 0 | 0 | 0.0369031 |
| 0.50 | latent four | 13 | 1 | 0 | 0.0154554 |
| 0.75 | latent four | 14 | 0 | 0 | 0.0237140 |
| 1.00 | latent four | 12 | 2 | 0 | 0.0247323 |

All four inconclusive quantities are whitened mean coordinates. Their four-SE
values exceed the frozen precision ceiling of 0.05. Small observed means do not
remove that requirement.

| prefix (s) | mean coordinate | four-group mean error | four-SE | precision ceiling |
|---:|---|---:|---:|---:|
| 0.01 | 2, yaw slot | -0.00955081 | 0.0512898 | 0.05 |
| 0.50 | 4, yaw-rate slot | -0.00508292 | 0.0522316 | 0.05 |
| 1.00 | 2, yaw slot | 0.00275793 | 0.0623056 | 0.05 |
| 1.00 | 5, gyro-bias slot | -0.0113611 | 0.0634477 | 0.05 |

The largest individual absolute mean error was 0.0486941, and the largest
individual absolute covariance-entry error was 0.0621850. No individual
accuracy ceiling was exceeded. The largest individual covariance-error spectral
norm was 0.0891039, below 0.5; the largest norm of a four-group average was
0.0381245, below 0.3. All six aggregate spectral comparisons passed.
The four-SE rules are descriptive engineering limits with only four groups,
not calibrated confidence intervals or simultaneous coverage guarantees.

Every row completed; there were no invalid, failed, unresolved-execution or
not-started rows. Group-level precision assessment marks 12 rows pass and 12
inconclusive. This row count does not add independent replicates. The 24
snapshots retain full endpoint mixtures and every point's prior, marginalized
likelihood, proposal and importance-residual vectors.

| exact or audit quantity | largest absolute discrepancy |
|---|---:|
| pointwise importance identity | 2.27374e-13 |
| log evidence versus independent oracle | 4.26326e-14 |
| normalized weight versus 1/2048 | 5.42101e-20 |
| sum of normalized weights versus 1 | 1.11022e-16 |
| independent saved-endpoint mean reconstruction | 4.88498e-15 |
| independent saved-endpoint covariance reconstruction | 1.47105e-15 |

The endpoint-audit values are the maxima across the parent's independent
reconstruction and the reviewer's separate component summation. All are below
the unchanged 1e-10 exact-check tolerance. Every retained
conditional 3x3 covariance and full 6x6 covariance passed the unchanged
scale-aware symmetry/PSD policy. No eigenvalue clipping, jitter or inflation
was used. Positive covariance does not establish calibration. Structural ESS
was 2048 to rounding, and maximum weight was 0.00048828125 throughout; these
are checks of the exact proposal identity, not a new empirical concentration
success criterion or a paired comparison with consumed groups 95001–95004.

## Validation, repair and independent review

The implementation subagent owned only the new runner and tests. It performed
static syntax checks, then stopped writes; it ran no numerical validation or
study stream. The read-only reviewer independently checked the mathematics,
old oracle use, resource monitor, complete source, deterministic tests and launch
gate. The parent inspected the source, ran all commands centrally, verified
preservation and owned integration/Git operations.

Before execution, review required durable planned statuses and stream/snapshot
provenance, handling of early setup failures, checks after final metadata writes,
and timestamp-keyed figures with a readable status table. These are runner
safeguards; they did not change the frozen estimator or mathematical criteria.

The first focused attempt exposed a runner adapter bug that static review had
missed: frozen fixture tuples were sent to a prototype API requiring Observation
objects. One test failed and 51 passed. That attempt's complete log/resources
and before/after source hashes remain saved. The approved prelaunch repair
converted the selected tuples without changing their order, values or variances,
and added interface assertions to the mocked worker test. Independent review
accepted that bounded repair before the next command. No tolerance or fixture
was weakened.

| attempt | command/check | actual result |
|---:|---|---|
| 1 | Debug configure | passed |
| 2 | Debug build, parallelism 2 | passed |
| 3 | CTest | 23 passed |
| 4 | unchanged real C++ development demo | passed; both figures inspected |
| 5 | focused runner tests | 1 failed, 51 passed; adapter bug retained |
| 6 | focused tests after reviewed repair | 52 passed in 0.56 s |
| 7 | full pytest | 330 passed in 19.48 s |
| 8 | source/artifact/checkout preservation audit | passed |

Python checks treated RuntimeWarning as an error. New real-candidate tests used
at most one step and two fixed coordinate rows; the 24-row worker plumbing test
mocked all RNG constructors and the candidate. Reserved streams were opened
only during the committed study. Existing regression tests and the unchanged
development demo retain their existing fixed-seed uses. The demo reproduced
gated RMSE 0.132119 m nominal and 0.174847 m dropout, with dropout end-window
error 0.374852 m. These are development regression figures, not candidate
navigation performance. No C++ changes required a new sanitizer run; no hosted
CI result is claimed.

After launch, the parent inspected all rows and both figures, checked every
manifest hash, and independently reconstructed all 24 full endpoint moments,
density identities, normalized comparisons and four-group assessments from saved
artifacts. That inspection reopened no RNG stream and reran no candidate.
The independent reviewer separately accepted all 24 saved mixtures, density
vectors, normalized comparisons, symmetry/PSD checks, stream metadata, both
figures and this report. Its component-sum reconstruction had maximum mean
discrepancy 9.15934e-16 and covariance discrepancy 1.47105e-15. Both reviewers
verified all 34 study-artifact and 102 source hashes. No reporting defect or
post-launch repair was found.

## Resources and retained evidence

Eight prelaunch commands charged **39.516795 s of 900 s**, including the failed
attempt. Highest assessed per-invocation memory was **313,901,056 bytes of 2 GiB**.
The last command's final-flush receipt counted **209,269,607 bytes of 256 MiB**
retained validation/demo output; later review metadata is also included in the
final evidence-manifest accounting. No failed output was deleted.

The one study's terminal receipt recorded **4.899933 s through final flush**,
a conservative **5.832616 s** charge, **229,294,080 bytes (218.672 MiB)** assessed
wrapper-plus-worker peak, and **9,490,311 bytes** retained output. Limits were
120 s, 512 MiB and 64 MiB, with 4 MiB reserved for metadata. The earlier persisted
resource record counted 227,655,680 bytes assessed peak and 9,490,310 output
bytes; the terminal receipt includes the last metadata allocation/write and is
the final assessment. Both records are preserved, not silently made equal.

The monitor samples process-tree RSS every 100 ms and retains OS high-water
measurements. Short-lived descendant peaks can escape sampling. The study worker
created no observed descendants; the final assessment uses the larger of live
aggregate RSS and the sum of wrapper/worker lifetime high waters. This is the
reviewed resource assessment, not a continuous exact RSS measurement.

The largest row elapsed time was 0.486811 s. At the final prefix the batch had
100 substeps, 32 observations, 203 angular primitive dimensions and 103 residual
dimensions. Runtime was Python 3.11.9 / NumPy 2.2.6 on Darwin arm64 with the five
thread controls set to one. These measurements cover this single short
synthetic workload and do not establish long-outage or real-time feasibility.

Evidence roots:

- `results/observed-trajectory-study-checks-v1/`: initialization, eight retained
  attempts, repair record, gate, preservation and independent saved-artifact
  audits, terminal launch receipt and final evidence manifest.
- `results/observed-trajectory-study-demo-v1/`: unchanged development replay
  metrics, 24 manifest-listed artifacts and two inspected figures.
- `results/observed-trajectory-study-v1/`: all 24 snapshots, complete inventory
  and comparisons, inputs/factors, analytical targets, runtime/stream provenance,
  resource record, both figures and study manifest.

The study manifest SHA-256 is
`3c4299251bfbc6080264a5d1861680b980a2e29105b0aabb2f58957b5fc05e2d`.
It verifies 34 artifacts and 102 source files; the manifest itself is the 35th
study file. Full supplied coordinates, dense prefix posteriors and angular
histories were omitted by the frozen storage policy. Their absence is not
permission to regenerate them. Generated artifacts remain outside Git.

All 100 baseline tracked files, 2,288 prior worktree artifacts and 579 older
artifact hashes remained unchanged. The original checkout remains clean at
`c834bf3`, and the prior position-moment integration checkout remains clean at
`53be2f7`. The final preservation audit records these checks separately from
the new study evidence.

## Stop and next decision

The implementation and bounded experiment are complete locally, with a defer
result. Preserve the four precision limitations, old no-go findings and all
consumed groups. There is no approval for another draw, larger M, relaxed rule,
production integration or recovery policy. Nothing was pushed, merged or released.

The next decision is whether to authorize a **design-only review of an independent
multi-step position oracle for this same synthetic model**. I recommend that
bounded mathematical review before considering another numerical study, because
later position accuracy is still unmeasured here. It would produce a reviewed
specification or a concrete analytical blocker; it would not open streams or
rerun this experiment. That next scope has not started.
