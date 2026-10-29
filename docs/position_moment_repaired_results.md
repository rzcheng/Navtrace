# Position moment comparison after the metric repair

The approved replacement run completed on 2026-10-09. The fixed candidate
improves conditional position means but retains severe narrow-direction
covariance error. In the selected lockout endpoint its nominal 95% ellipse
contains only 22.37% of the reference samples, down from the baseline's 45.85%.
The recommendation is **not to advance this unchanged candidate to a navigation
experiment now**. This recommendation is based on the completed bounded study;
no additional estimator, tuning or recovery policy was implemented.

This report covers the one replacement launch Ryan approved after the first
run failed in metric tooling. The original [failure report](position_moment_results.md)
and every first-run artifact are unchanged. The approved scope, repaired
operations and pre-launch checks are in
[position_moment_repair_validation.md](position_moment_repair_validation.md).

## Execution and completeness

The replacement used clean local commit
`89077fcfb0236b2cd27f376dcb3af24037603f8a` (`fix position metric projections`)
on `codex/position-moments`. Only the two N-by-2 sample-to-axis products changed
from matrix multiplication to equivalent direct contraction with `optimize=False`.
Candidate, runner, plots, original protocols, samples, thresholds, priors,
Q/R, production C++, gate and recovery behavior stayed fixed.

```
python -W error::RuntimeWarning tools/run_position_moment_comparison.py \
  --long-outage <original-checkout>/results/long-outage-study \
  --reference <original-checkout>/results/conditional-reference-study \
  --output results/position-moment-comparison-repaired
```

The manifest records the exact arguments, absolute input roots, clean source
state and environment. Python 3.11.9, NumPy 2.2.6, SciPy 1.15.3 and Matplotlib
3.10.3 were used from the existing project environment. The frozen JSON SHA-256
remains `261309e0338ff048c5dc1626a7b24569930a390f6b51071e4eaaa09ab02a20e5`.

| Attempt | Status | Complete distributions / metrics | Resolved numerical checks | Wall time | Peak RSS |
|---|---|---|---|---:|---:|
| Original, `55bb5b1` | Failed in metric tooling | 30/90, 270/810 | 0/90 available | 4.708876 s | 283.859375 MiB |
| Approved replacement, `89077fc` | Complete | 90/90, 810/810 | 90/90 | 4.639067 s | 296.234375 MiB |

Both attempts fit the unchanged 600 s / 512 MiB bounds. The replacement has
zero failed or unstarted rows, zero failure reasons, all ten case snapshots,
all 30 count-pooled summaries, and all four complete plots. No third launch
occurred. The reference is numerically resolved under all frozen stability and
precision rules; this does not mean the estimator agrees with that reference.
No resolution outcome is inconclusive in the replacement.

The two recordings remain consumed seeds 83001 and 83021. Reports use the full
1.99 s checkpoint and causal gyro/wheel prefixes at 5, 10, 15 and 22 s. At 22 s
the run includes gyro/wheel and stops before returning GPS. Conditional samples
are not independent navigation trials or fresh validation recordings.

## All endpoints

Numbers below use equal pooling of the three saved N=8192 replicates. The
mean-vector and covariance Frobenius norms are descriptive summaries. All
coordinate, directional and mixed metrics have their separately frozen MCSE
and agreement bands in `resolution.json`; full means, matrices, eigenvalues
and directions are retained in the JSON artifacts.

| Recording | Time (s) | Mean error norm, baseline / candidate (m) | Covariance error Frobenius norm, baseline / candidate (m²) | Ellipse mass, baseline / candidate |
|---|---:|---:|---:|---:|
| 83001 | 1.99 | 0.001791 / 0.001791 | 0.000381 / 0.000381 | 0.952840 / 0.952840 |
| 83001 | 5 | 0.022717 / 0.000480 | 0.005856 / 0.003107 | 0.942993 / 0.946045 |
| 83001 | 10 | 0.071471 / 0.002489 | 0.022641 / 0.009507 | 0.928345 / 0.935384 |
| 83001 | 15 | 0.133565 / 0.009045 | 0.080422 / 0.039256 | 0.916260 / 0.928630 |
| 83001 | 22 | 0.265739 / 0.008917 | 0.304308 / 0.131465 | 0.849243 / 0.886475 |
| 83021 | 1.99 | 0.000172 / 0.000172 | 0.000608 / 0.000608 | 0.948690 / 0.948690 |
| 83021 | 5 | 0.053550 / 0.008059 | 0.005735 / 0.015655 | 0.911174 / 0.926676 |
| 83021 | 10 | 0.227185 / 0.021388 | 0.303201 / 0.110294 | 0.744019 / 0.799235 |
| 83021 | 15 | 0.222803 / 0.023241 | 0.294526 / 0.108846 | 0.758341 / 0.806274 |
| 83021 | 22 | 1.096132 / 0.033936 | 6.429976 / 2.042698 | 0.458537 / 0.223674 |

The candidate's mean discrepancy decreases at every post-checkpoint endpoint.
Its covariance-norm discrepancy improves at seven of those eight endpoints,
but worsens at 83021/5 s from 0.005735 to 0.015655 m². Ellipse mass improves
at seven endpoints and worsens sharply at 83021/22 s. Checkpoints are identical
by design. These counts do not represent independent trials or a general
performance rate.

## Moment agreement and remaining error

For each of the seven scalar moment quantities, the frozen descriptive band is
four pooled MCSE plus 1e-12 in the metric's units. These are separate numerical
diagnostics, not simultaneous statistical tests or calibration criteria.

| Moment group | Baseline within band | Candidate within band |
|---|---:|---:|
| x/y coordinate means | 10/20 | 20/20 |
| Narrow own-mean variance | 2/10 | 2/10 |
| Wide own-mean variance | 5/10 | 10/10 |
| Mixed covariance | 6/10 | 8/10 |
| Narrow second moment about baseline mean | 2/10 | 2/10 |
| Wide second moment about baseline mean | 5/10 | 10/10 |
| Total | 30/70 | 52/70 |

Both successful narrow-moment comparisons are the starting checkpoints.
**Every post-checkpoint narrow variance remains below the reference band.**
Candidate mixed covariance also remains outside its band at 83001/15 s and
83001/22 s. Across all 70 scalar discrepancies, 45 absolute errors shrink,
11 grow and 14 checkpoint values are unchanged; every result remains recorded.
The counts themselves are descriptive, not an implementation acceptance target.

At 22 s, in the original EKF principal basis:

| Recording | Reference narrow variance (m²) | Four-MCSE half-width (m²) | Baseline variance (m²) | Candidate variance (m²) | Reference / candidate |
|---|---:|---:|---:|---:|---:|
| 83001 | 0.227596 | 0.010903 | 0.115196 | 0.113200 | 2.011 |
| 83021 | 2.083940 | 0.149656 | 0.043274 | 0.048159 | 43.272 |

The candidate's wide variance at the 83021 endpoint is 29.110528 m² versus
reference 28.971748 m² and baseline 35.016747 m². Improvement in this large
component helps reduce the overall covariance-norm error from 6.429976 to
2.042698 m² while leaving the narrow component severely underestimated.
Thus the result includes a covariance approximation failure, not just curved
shape or tail error after successful moment correction.

The 83021 endpoint candidate ellipse mass is 0.223674 with four-MCSE half-width
0.010923; baseline mass is 0.458537 with half-width 0.011079. The candidate mean
moves close to the reference mean while its ellipse stays too thin. Both
ellipse masses are far from nominal 0.95, and the candidate is worse. At 83001,
ellipse mass increases from the baseline's 0.849243 to the candidate's 0.886475,
remaining below nominal 0.95; its four-MCSE half-width is 0.007641. These are conditional
sample masses, not navigation-trial coverage estimates.

## Plot and artifact review

The coordinator and independent reviewers inspected all four replacement plots:

- `position-shape.png`: all ten prescribed N=8192, seed-91001 panels retain
  full sample ranges and equal spatial axes. Curved distributions and thin
  candidate ellipses remain visible, especially at 83021/22 s.
- `moment-errors.png`: all seven moment quantities at all endpoints are now
  populated, including narrow-variance and mixed-covariance discrepancies.
- `ellipse-mass.png`: all three counts are present for both methods and cases.
  The late 83021 deterioration is visible at every saved count.
- `resolution.png`: all 90 cells resolve. Green cells describe numerical
  resolution of the reference, not candidate agreement or success.

Independent audit verified all ten output hashes, all 83 launch-source hashes
against disk and commit `89077fc`, both original input manifests, and all 517
original artifact hashes. All eleven first-run files match the pre-repair audit.
The replacement `case-snapshots.json` is byte-identical to the failed run, so the
candidate outputs did not change. Every N=512 value, batch value and MCSE is
also identical to the first run (maximum difference zero).

The reviewer independently recomputed all 90 resolution decisions from the
saved metric estimates. Largest stability-limit fraction was 0.55850 and
largest precision-limit fraction 0.76898. Independent scalar spotchecks at
83001/22 s/N=2048/seed=91003 and 83021/22 s/N=8192/seed=91002 agreed on all nine
metrics and batch statistics: maximum value difference 7.11e-15, MCSE difference
2.78e-16, and both ellipse masses exact. No new comparison or candidate replay
was used for this audit.

All ten candidate covariance snapshots are symmetric and positive. Latent
mean/covariance differences from the saved C++ snapshots remain at most
9.77e-15 / 1.72e-15, below 1e-9. Observation counts remain exactly
0/362/962/1562/2402. These code/model checks do not repair or validate the
remaining conditional position error.

## Implementation, validation and local state

The evaluation subagent implemented the two-operation repair and independent
full-size regressions. A separate evidence reviewer checked equivalence,
reran the focused tests and repeated the in-memory original-expression failure.
The mathematical reviewer independently interpreted the complete results and
recommended against advancing this fixed candidate. The coordinator inspected
the source diff, integrated centrally, ran checks and the replacement, viewed
all plots, audited preserved outputs and reconciled the reports.

Before launch: 30 focused evaluator tests, 167 full pytest tests with
RuntimeWarning as error, and 23 CTest cases passed. The real C++ development
demo completed in a fresh directory and both figures were inspected. Existing
260 third-party Python deprecation warnings remain. No production C++ changes,
sanitizer run, hosted CI run, push, merge or release is claimed.

Local artifacts:

- Replacement: `results/position-moment-comparison-repaired/`.
- Original failure, preserved: `results/position-moment-comparison/`.
- Logs and before/after preservation audits: `results/position-moment-repair-checks/`.
- Fresh development demo: `results/position-moment-repair-demo/`.
- Replacement manifest SHA-256:
  `5149f1c4cb8f6d20c01cc06e17a2174214f92d1c3aeeb216e57df2b126c4891e`.
- Original manifest SHA-256:
  `1052d424ea7a75b7303f34508c73aa64b0577000e513215d003d9c7a689422e8`.
- Repair/launch commit: `89077fc`; earlier protocol `4c5f843`, implementation
  `55bb5b1`, and failure report `258f05c` remain in history.
- Integration branch: `codex/position-moments`. Original checkout remains clean
  at `c834bf3` on `feat/conditional-reference`; all work remains local.

The authorized milestone is complete locally, including its unfavorable
results. Issues #1–#3 retain the evidence and await review separately from any
merge or release. This turn did not restart the previously stopped heartbeat.

## Decision boundary

The single next decision is whether this unchanged candidate warrants a
separately scoped navigation experiment. The recommendation is **no**: the
mean improvement does not resolve the defining narrow-direction uncertainty
error, and the selected lockout endpoint's ellipse mass substantially worsens.

This study did not test returning GPS, reacquisition, automatic recovery,
navigation calibration, different checkpoints or independent new recordings.
It neither proves a universally inferior method nor authorizes production
changes. Any different candidate or further experiment requires a new reviewed
scope. Work stops at this boundary with both attempts preserved.
