# Position moment comparison: incomplete

The one approved saved-reference comparison was launched on 2026-10-09 from
clean local commit `55bb5b1f2747b2146f50e0a7445132f7ef69c05a`, branch
`codex/position-moments`. It **failed in evaluation tooling**. This is not a
completed numerical comparison or a resolved negative estimator result.
No second launch, candidate adjustment, new sample or new recording followed.

The candidate passed all ten snapshot checks, but metric evaluation failed
for every 2048- and 8192-sample distribution. All 90 resolution outcomes are
unavailable. The question of whether this candidate warrants a navigation
experiment remains unresolved. The next decision is whether to authorize the
specific tooling repair and one replacement comparison described below.

## Run and provenance

The frozen protocol is `docs/position_moment_protocol.md` at `4c5f843`;
JSON SHA-256 is
`261309e0338ff048c5dc1626a7b24569930a390f6b51071e4eaaa09ab02a20e5`.
The implementation and independent review are recorded in
[position_moment_validation.md](position_moment_validation.md).

The launch used the original long-outage and conditional-reference directories
as explicit absolute inputs and a fresh ignored output directory:

```
python -W error::RuntimeWarning tools/run_position_moment_comparison.py \
  --long-outage <original-checkout>/results/long-outage-study \
  --reference <original-checkout>/results/conditional-reference-study \
  --output results/position-moment-comparison
```

The exact command, absolute input roots and environment are in the saved
manifest; the interpreter was the original project-local Python environment.
The run exited 1 in **4.708876 s**, using **283.859375 MiB** peak RSS.
It did not reach the frozen 600 s / 512 MiB limits.

| Required output | Actual result |
|---|---|
| Ten case/time snapshots | All ten complete |
| 90 sample-distribution rows | 30 complete, 60 failed, zero not_started |
| 810 metric entries | 270 complete, 540 failed |
| 90 medium/high resolution outcomes | All 90 failed/unresolved |
| Four plots | Files produced; important missing data noted below |
| Initial and final artifact audit | All 517 original artifact hashes match |
| Output hashes | All ten non-manifest output hashes match |

All 30 completed rows use N=512. Every N=2048/8192 row failed with
`divide by zero encountered in matmul`. All rows and reasons remain in the
matrix. No failure was replaced with zero or dropped. The independent artifact
review also checked both original manifest digests and all 81 launch-source
hashes against disk and the implementation commit.

Candidate observation counts are exactly 0, 362, 962, 1562 and 2402 at the
five reports in both recordings. Maximum saved latent mean/covariance error
against the original C++ snapshots is 9.77e-15 / 1.72e-15, below the frozen
1e-9 limit. Candidate checkpoints equal the inputs exactly. Full snapshot
covariances are symmetric and positive; this is an implementation check,
not evidence of calibration or correct conditional position uncertainty.

## Retained partial measurements

These are descriptive **N=512-only** values, equally pooled over the three
reference replicates. None meets the frozen resolution requirements because
the required medium/high outputs failed. Norms are descriptive even in a
complete run. They are retained to show both improvement and worsening,
not used to select a candidate or claim validation.

| Recording | Time (s) | Mean error norm, baseline / candidate (m) | Covariance error Frobenius norm, baseline / candidate (m²) | Ellipse mass, baseline / candidate |
|---|---:|---:|---:|---:|
| 83001 | 1.99 | 0.012519 / 0.012519 | 0.001725 / 0.001725 | 0.953776 / 0.953776 |
| 83001 | 5 | 0.021375 / 0.006164 | 0.005631 / 0.007882 | 0.940755 / 0.944661 |
| 83001 | 10 | 0.083978 / 0.013081 | 0.018771 / 0.018608 | 0.914714 / 0.928385 |
| 83001 | 15 | 0.125560 / 0.016084 | 0.105002 / 0.037677 | 0.919922 / 0.932943 |
| 83001 | 22 | 0.253659 / 0.034955 | 0.375209 / 0.132693 | 0.856120 / 0.882161 |
| 83021 | 1.99 | 0.003888 / 0.003888 | 0.002489 / 0.002489 | 0.947917 / 0.947917 |
| 83021 | 5 | 0.053769 / 0.004632 | 0.028508 / 0.040836 | 0.903646 / 0.925781 |
| 83021 | 10 | 0.237902 / 0.053218 | 0.238896 / 0.156575 | 0.747396 / 0.801432 |
| 83021 | 15 | 0.229355 / 0.019962 | 0.465499 / 0.244412 | 0.755859 / 0.809896 |
| 83021 | 22 | 1.059387 / 0.079767 | 7.070538 / 2.074741 | 0.466797 / 0.227214 |

At 22 s in recording 83021, the candidate reduces the descriptive mean and
covariance-norm discrepancies but has much lower ellipse mass: 0.227214 versus
0.466797. The four-MCSE widths for those low-count masses are 0.045000 and
0.054022, respectively; these do not substitute for failed resolution checks.
The original-EKF narrow-axis reference variance is 1.933985 m², compared with
0.048159 m² for the candidate and 0.043274 m² for the baseline. Severe narrow
covariance error remains. A lower matrix norm does not establish corrected
moments; it can be dominated by the wide direction.

Both the coordinator and independent reviewers inspected all four saved plots:

- `position-shape.png` contains all ten equal-axis, full-range panels using the
  prescribed saved N=8192, seed-91001 samples. Curved position shape and narrow
  ellipse mismatch remain visible. This descriptive rendering is separate from
  the failed high-count metric computation.
- `moment-errors.png` is empty because no high-count resolution output is
  available. Its empty axes mean unavailable data, not zero error.
- `ellipse-mass.png` contains only the N=512 traces, although its frozen legend
  lists all three counts. The medium/high traces are unavailable.
- `resolution.png` explicitly marks all 90 entries failed.

The outputs remain byte-for-byte as generated, including these incomplete
figures. The original conditional-reference results and recorded failures
remain unchanged. Conditional samples are not independent navigation trials.

## Diagnosis and proposed next scope

Read-only diagnosis isolated the failure to the two large, thin sample-to-axis
products in `tools/position_moment_metrics.py` (`centered` and `displaced`).
The worker and coordinator separately reproduced the local NumPy 2.2.6 behavior
with bounded synthetic data, without reading saved samples or replaying the
candidate:

```
a = np.arange(2048 * 2, dtype=float).reshape(2048, 2) / 2048
with np.errstate(over='raise', invalid='raise', divide='raise'):
    a @ np.eye(2)
```

Identity multiplication raises the same divide-by-zero error at N=2048 and
8192, but succeeds at N=512. NumPy reports the Accelerate backend. This isolates
a local matrix-product path; the deeper backend cause is not established here.
The existing `conditional_reference.multiply_samples` helper already documents
and avoids this issue with direct contraction. The new tooling tests used
small synthetic counts and missed the size threshold.

A synthetic-only in-memory substitution using `np.einsum(..., optimize=False)`
completed all nine metrics at the actual three counts under unchanged strict
error handling. Independent closed-form moment errors were at most 8.89e-16;
N=512 metric outputs were identical. The coordinator independently confirmed
exact identity results for direct contraction at all three sizes. No source
repair has been applied after the failed launch.

Recommended approval scope for issue #3:

1. Replace only the two sample-to-axis products with direct contraction, keeping
   all equations, candidate code, inputs, tolerances, thresholds and budgets fixed.
2. Add deterministic synthetic tests at N=512/2048/8192 with independent scalar
   arithmetic and strict floating-point errors. Independently review the diff,
   run regression checks and commit the repaired tooling before evaluation.
3. Authorize exactly one replacement launch in a new directory under the same
   600 s / 512 MiB budget. Preserve this failed run permanently and report both
   attempts. No new samples, recordings, estimator choices or tuning.
4. Inspect every result and stop for review again. Navigation, production
   changes and recovery remain outside scope.

Until that approval, the saved comparison remains incomplete. No issue is
closed as though the full milestone succeeded.

## Artifacts and repository state

- Run: `results/position-moment-comparison/`, including manifest, comparisons,
  snapshots, pooled metrics, resolution outcomes, protocol copies and figures.
- Launch log: `results/position-moment-integration-checks/comparison-launch.txt`.
- Manifest SHA-256:
  `1052d424ea7a75b7303f34508c73aa64b0577000e513215d003d9c7a689422e8`.
- Protocol commit: `4c5f843`; reviewed implementation/validation: `55bb5b1`.
- Integration branch: `codex/position-moments`; commits are local only.
- Original checkout remains clean at `c834bf3` on `feat/conditional-reference`.

The candidate and evaluation workers used separate worktrees. Independent math
and evidence reviewers were read-only. The coordinator inspected their files,
ran integration checks and the actual comparison, verified artifacts and owns
Git operations. No push, merge, release, production retuning or hosted CI claim
is part of this report.
