# Exact moment target checks

The bounded validation-proof repair passes the fixed deterministic checks. It
addresses [#14](https://github.com/rzcheng/Navtrace/issues/14), with CI dependency
setup in [#15](https://github.com/rzcheng/Navtrace/issues/15). This result does
not authorize Stage B, a new saved-reference comparison, a sampling campaign,
or a production estimator change.

The [repair protocol](position_moment_proof_repair_protocol.md) was reviewed and
frozen before numerical execution. Source commit
`8c3e1f8a140dc7adb10a5bf8680ed8b8847cdb1d` was clean for every numerical command.
The implementation and test files were not changed after execution began.
The earlier failed Stage A report and artifacts at `474373d` remain intact:
69 passed and 23 failed in the focused suite, and 451 passed and 23 failed in
the full suite. The earlier no-go and defer decisions remain unchanged.

## What changed

The old tests compared two independently rounded 256-bit balls as if one had
to contain the other. The repaired tests instead build an independent enclosure
of the exact physical target using Python integers and Fraction arithmetic.
Every reported inclusion proves that the entire independent target enclosure
fits inside the unchanged oracle enclosure. Overlap alone still fails.

The test helper uses the fixed outward grid 2^-640, fixed series and
proved tails from the protocol. Input-based exact identities remain exact.
Complete norm and scale checks use rational square inequalities, including the
full returned bound. The oracle remains at 256 bits, with unchanged source,
physical fixtures, tolerances, mutation definitions and finite-difference steps.
The test reference has changed; this is not a claim that its precision is still
256 bits. No covariance clipping, output padding or adaptive retry was added.

Two agents implemented disjoint files: one wrote the independent helper and CI
setup, and one adapted the original target assertions. An independent reviewer
checked the derivations, outward rounding, tails, fixture and threshold
preservation, whole-bound inequalities and publication scope before execution.
The parent inspected their diffs, ran the commands, and independently checked
the retained exact receipts. The reviewer also checked those receipts, the test
logs and both demo figures. No numerical result was accepted solely on an
implementation agent's claim.

## Checks actually run

| Check | Result |
| --- | --- |
| Independent proof helper | 26 passed in 0.46 s |
| Focused oracle suites | 92 passed in 112.24 s |
| Debug CMake configure and build | passed |
| CTest | 23 of 23 passed in 0.74 s |
| Full pytest, RuntimeWarning fatal | 499 passed, 1 skipped in 140.57 s |
| Existing saved-plot regression with preserved local inventory | 1 passed in 0.08 s |
| Unchanged real C++ development demo | completed; 24 artifact hashes verified |

The full suite's single skip was the existing saved seed 95004 plot check:
its ignored historical inventory is absent from a fresh checkout. The separate
check read that preserved inventory without rerunning any study or sampler.
No oracle test was skipped. All 260 warnings were Matplotlib/Pyparsing
deprecations; RuntimeWarning remained fatal. No C++ source changed, so there
was no extra local sanitizer build. The hosted workflow retains its sanitizer
builds and all existing development smoke paths.

The demo used the unchanged development seed 20260921. Its metrics CSV is
byte-for-byte equal to the preserved earlier demo. Gated position RMSE is
0.13211917187409775 m nominal and 0.17484668873395803 m with the dropout; the
outage-end window error is 0.3748515225300933 m. The parent and reviewer inspected
`trajectories.png` and `errors.png`, including the worse dropout-end behavior.
This is regression evidence, not a new navigation-performance result.

## Numerical evidence and remaining failures

The focused log contains 232 structured records and 92 proof contexts. The
parent and reviewer independently rechecked all 1,916 exact membership
receipts, including their rational margins and widths. Every membership was
proved. The largest context used 503,904 operations, below the fixed 10,000,000
limit. The 38 complete six-state moment checks had maximum width
4.703987885084317e-73. All eight shared-heading controls reached their gates;
the largest of 12 positive-axis relative widths was 2.3517363834710095e-49,
below the unchanged 1e-10 criterion.

The complete one-step certificate was eligible, with mean bound
5.986045271624095e-16, covariance bound 2.2349362287357907e-15, and evidence bound
2.3056642348055324e-16. Maximum finite-difference errors were
2.1141271483959512e-11 for first derivatives and 1.496163670808812e-11 for second
derivatives, below the original 1e-7 tolerance with step 1e-6.

All six prescribed mutations were detected above the unchanged 1e-10 threshold
and then restored:

| Mutation | Certified error shown as a decimal |
| --- | ---: |
| Drop different-step terms | 0.0012793715195782694 |
| Wrong minus-phase conjugation | 1.48 |
| Omit initial-position cross terms | 0.0005399174546980368 |
| Omit final-latent cross terms | 0.0008131156867752434 |
| Condition twice | 5.352105698372682 |
| Use new-state displacement | 0.0000741039580028727 |

The historical narrow-control limitation remains. At V=1e-8 its signed relative
error is -6.224841179794171e-9 and it resolves under the unchanged 1e-7 criterion.
At V=1e-12 it is 4.843588053936489e-5 and remains unresolved. The earlier report's
4.843588053953116e-5 value is preserved; the new exact-target display has a tiny
rounding difference and does not replace that historical measurement.

Other expected unfavorable outcomes remain: the factor-law bridge withholds a
bound of 1.9073495423072018e-6, the hidden narrow-direction error withholds
1/16, the scale fixtures split two eligible and two withheld, and the norm
fixture stays inconclusive. The fixed group controls remain pass, fail and
inconclusive. These are successful detection checks, not evidence that the
unfavorable cases have been corrected. Passing these deterministic fixtures
does not establish calibration or general saved-study performance.

## Artifacts, resources and preservation

New ignored evidence is under `results/position-moment-proof-repair-checks-v1/`:
`source-review.json`, numbered attempt logs and resource receipts,
`numerical-summary.json`, `demo-review.json`, preservation records and the final
binding. The demo is in `results/position-moment-proof-repair-demo-v1/run/`.
The command ledger includes all failed and completed attempts. Exact rational
receipts are in the focused log; decimal tables above are only readable summaries.

One metadata-only bootstrap failed because the parent had not initialized the
ledger before invoking its wrapper. The audit child and numerical work never
started. That failed log is retained, charged 1.062764291 s including the
conservative flush allowance, and marked with unassessed memory. No memory
bound is claimed for that attempt. All subsequent monitored commands completed
within the frozen limits. The largest assessed memory was 1,078,820,864 bytes;
100 ms sampling can miss short-lived descendant peaks and is not an exact
simultaneous-tree high-water measurement. Final charged time and retained size
are recorded in the final receipt against 900 s and 268,435,456 bytes. Reused
read-only dependencies are distinguished from new build archives and outputs.

The preservation audit checks both old final bindings, 114 old source files,
1,351 old Stage A artifacts, 5,007 prior artifacts, 579 older artifacts and
11,370 original environment files. It also checks for additions to the reused
backend and environment, and for changes to the three protected checkouts.
The new repair changes only the protocol's explicit file list.

The unchanged oracle hashes are:

- `tools/position_moment_enclosures.py`: `ef639c629c53fb954b5979e66159268a3d91dc74b59595483c8937156e338ef2`
- `tools/moment_precision.py`: `a48606058f3a00a645f0574125ac8e8c5e51405dabd92f3152087a82a1941c5e`

## Publication and boundary

The original unpublished research history is preserved locally. Publication
uses a new branch from remote main `06e05f2` containing the reviewed final tree,
without publishing private intermediate history. The final source-to-public
commit mapping and actual hosted CI results belong in the PR and issue evidence.
A fresh public clone can build and test the implementation; it does not contain
the ignored saved inputs or all historical local commits named in the reports.
Strict historical runner checks were not relaxed.

CI installs hash-checked python-flint 0.9.0 binary wheels into an isolated target
on Python 3.11, ubuntu-24.04 x64 and macos-15 arm64. There is no source-build
fallback and no oracle-test skip. Hosted results must be inspected on the exact
published head before merge; local passes do not stand in for hosted CI.

The bounded repair answers its local validation question affirmatively. The
next scope decision is whether to review a separate Stage B proposal. No new
comparison, experiment, full O sequence or recovery policy was run here.
