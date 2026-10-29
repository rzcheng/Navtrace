# Position moment enclosure checks

The Stage A implementation in [issue #14](https://github.com/rzcheng/Navtrace/issues/14)
is present locally, but its validation is **incomplete**. The first focused run
had **69 passed and 23 failed** tests. The full suite had **451 passed and the same
23 failures**. We are stopping at this report with the issue open. This is not a
numerical eligibility pass or approval for Stage B.

The failures exposed incorrect validation assertions. They do not establish a
physical-model defect, but they also do not certify the implementation. No tests,
source, precision, fixtures or tolerances were changed after execution. Both
failed attempts remain in the resource ledger.

## Scope and source freeze

Ryan approved Stage A, including its isolated pinned dependency, on 2026-10-20.
The scope is the frozen [saved moment design](saved_moment_comparison_design.md)
at `b8b8b39e51b78158fb5893d7c7561e3df4a4c987`, SHA-256
`6211ec548ca7d2bfc618d0aac0e2b4adfe74b80b580d0637ac2351cab33120dd`.

Only six files were added: the optional dependency lock, two Python modules,
two test modules and this report. The new code independently constructs physical
Gaussian marginals and raw position moments under the canonical and declared
factor laws. It retains full covariance correlations, exact input and process
semantics, arithmetic/law error contributions, directional precision bounds,
support refusals, and scalar/group/norm decision helpers. It contains no artifact
reader, sampler, candidate call or comparison runner.

Parent and independent reviewer accepted the source and fixed tests before
execution. The clean local freeze is `8ca32563fe4fe1e0c02871bd18eef2f6543d2e20`
(`bound the position moments`). These five files have remained unchanged:

| file | SHA-256 |
|---|---|
| requirements-oracle-lock.txt | `38c0406ddf59d4d3b4e8c1a38acf638ce7285276e9b0190c27ad89b33ad6500e` |
| tools/position_moment_enclosures.py | `ef639c629c53fb954b5979e66159268a3d91dc74b59595483c8937156e338ef2` |
| tools/moment_precision.py | `a48606058f3a00a645f0574125ac8e8c5e51405dabd92f3152087a82a1941c5e` |
| tests/test_position_moment_enclosures.py | `3a220b986399b62d633fd251c31135e3662f0ff10c270e712d045f3ba73576d1` |
| tests/test_moment_precision.py | `7ead0a1395a9626918d92f9c0316db2d8b6e087e302481bb39b91d053d89c221` |

The single python-flint 0.9.0 wheel matched the frozen SHA-256
`025fd4e77f2cbbf40f63b9cd7571aad3803deb57aecd72e7c397480fe84921a0`.
It installed into the separate dependency directory without importing the backend
or changing the original virtualenv. Arithmetic then used the reviewed fixed
256-bit context and one thread. Exact dyadic serialization refuses exponents
outside the fixed absolute bound 1,000,000 before allocating a large Fraction;
this reports unresolved arithmetic rather than replacing a value with zero.

## Checks actually run

| check | result |
|---|---|
| Debug configure and build | passed; C++ sources unchanged |
| CTest | 23 passed, zero failed |
| Pre-execution preservation | passed at clean source freeze |
| First focused tests, RuntimeWarning fatal | 69 passed, 23 failed in 79.26 s |
| Full pytest, RuntimeWarning fatal | 451 passed, same 23 failed in 112.16 s |
| Unchanged real C++ development demo | passed; both figures inspected |
| Retained-log and demo hash audit | passed; 65 focused evidence records, 24 demo artifacts |
| Final preservation | passed; all protected bytes and checkout states unchanged |

All 382 pre-existing Python tests passed in the full suite. Its existing unit
fixtures include sampler/candidate smoke paths. The new checks themselves use
fixed deterministic inputs and no RNG. No new full O history, saved-output
comparison, recording study, navigation experiment or consumed stream was run.
No C++ changes were made; no new sanitizer or hosted CI result is claimed.

## Why the validation failed

Twenty-two checks require the implementation ball A to contain the entire
independent target ball E. Both are computed at 256 bits. Even when an exact
value x belongs to A and E, E need not be a subset of A. Thirteen failures are
on log evidence and nine on moments or raw kernels. Typical printed differences
are at rounding scales around 1e-77; matching printed digits is not a proof.

The remaining failure requires a returned lower-bound interval's upper endpoint
to be below an independently rounded lower endpoint for sqrt(41)/16. The norm
helper reached the expected `inconclusive` classification, but this extra
ordering assertion does not test the intended bound correctly. A future exact
rational check could prove a nonnegative L satisfies L² <= 41/256, while retaining
the existing tightness requirement. It was not substituted after seeing results.

Parent, test author and independent reviewer missed these predicate mismatches
before execution. Static review was not a numerical pass. The design forbids
changing fixtures, operation orders, precision or criteria after inspection;
we did not replace containment with overlap, midpoint agreement, padding or an
absolute-error check. No concrete physical-moment defect was established by
these traces. Correctness remains unvalidated at the failed gate.

Several later assertions were not reached. All eight shared-heading controls
passed their moment containment checks before failing on evidence, so their
relative-width and frozen-oracle checks did not execute. The real one-step
certificate was not reached. Five prescribed mutations were detected and
restored; the new-state-displacement mutation stopped at its unmodified baseline.
Source coverage of twelve fixture families and six mutations must not be read
as successful validation of them all.

## Partial evidence that was reached

These results concern the fixed synthetic helper examples only. Rounded numbers
below are display values; exact endpoint evidence is retained in the logs.

| fixed check | observed result |
|---|---|
| off-diagonal scale, c = ±0.5e-7 | bound about 7.07107e-8; eligible |
| off-diagonal scale, c = ±2e-7 | bound about 2.82843e-7; withheld |
| declared-factor law bridge | relative bound about 1.90735e-6; withheld |
| hidden narrow covariance error | exact directional error 1/16; withheld despite matching diagonals |
| four zero mean-error groups | pass |
| four 0.3 mean-error groups | fail |
| alternating -0.1, 0.1 groups | inconclusive |
| diagonal norm 0.25 / 0.6 / 0.5 | pass / fail / pass |

Detected mutations omitted cross-step pairs, reversed the conjugation sign,
omitted initial-position or final-latent cross terms, or conditioned twice.
Their certified differences exceeded the unchanged 1e-10 detection limit.
These are intentional fault detections, not additional ordinary failures.

The old binary64 oracle's V=1e-12 result remains unresolved at relative error
`4.843588053953116e-5` against its 1e-7 criterion. This run gives no new resolution
of that limit. Issue #10 remains **defer**, the earlier **no-go** is unchanged,
and this report makes no calibration or estimator-performance claim.

## Review, resources and preservation

The implementation agent owned the two tools modules. The validation agent
owned the two test files and independently derived rational conditioning,
scalar characteristic and explicit physical targets. The reviewer inspected
math, interfaces, tests, dependency setup and monitoring before execution, then
read the failures, checked hashes and inspected both actual demo figures.
Agents did not execute numerical work or perform Git operations.

The parent inspected the complete source and target constructions, ran every
monitored command, read the retained evidence, inspected the figures, reconciled
the diagnosis and preserved the frozen code. All three agents agreed that a new
validation-proof review is needed. The possible norm-only assertion repair was
not applied because it would leave the main containment gate unresolved.

Setup used 1.781253 s through final flush (2.769083 s charged), 112,312,320 assessed
memory bytes and 28,598,426 retained bytes, within its separate 120 s / 2 GiB /
128 MiB limits. Across all nine monitored validation commands, including both failed test runs
and the final preservation audit, charged time was 240.399202 s.
Retained validation output was about 201.4 MiB including the final receipts;
the final manifest records its exact byte inventory.
The largest validation memory assessment was 343,719,936 bytes. All commands,
including failures, stayed within the 900 s / 2 GiB / 256 MiB validation caps.
Accounting includes the build and tool/test/cache trees, not only new results.
The monitor retains its 100 ms sampling and short-lived-descendant caveat;
assessed memory is not an exact instantaneous peak proof.

The preservation audit covers 108 prior tracked files, 5,007 prior worktree
artifacts, 579 older artifacts, all 11,370 original virtualenv files, 115 installed
backend files and the pinned wheel. Protected checkouts remain clean: the original
at `c834bf3e5b9b421564c2e0050092895abe12dd43` and earlier integration at
`53be2f72d15f8d485640eaa09868210231b3e886`. The frozen design, production code,
parameters, historical outputs and protocols are unchanged.

Artifacts are retained under:

- `results/position-moment-enclosure-checks-v1/`: source review, numbered attempts,
  terminal receipts, failure review, exact evidence, summary and preservation.
- `results/position-moment-enclosure-dependency-v1/`: pinned wheel, isolated site,
  installer, setup review and separate resource ledger.
- `results/position-moment-enclosure-demo-v1/run/`: unchanged demo inputs, replay,
  metrics, manifest and both figures.

Demo metrics exactly match the prior development demo: gated position RMSE is
0.13211917187409775 m nominal and 0.17484668873395803 m dropout; dropout end-window
error is 0.3748515225300933 m. These baseline results do not validate the oracle.

## Stop and next decision

Issue #14 stays open because the required exact-target validation is incomplete.
No new focused attempt or source change is authorized by this report. Nothing
was pushed, merged or released. Work stops on `codex/trajectory-mixture-design`.

The recommendation is a **design-only revision of the validation proof** for the
same fixed fixtures: specify exact rational endpoint checks and the norm-square
proof, then derive a sound exact-target check for transcendental expressions.
It must keep the 256-bit reference, existing accuracy limits, all failures and
model inputs fixed, and explicitly establish what constitutes a proof or refusal.
Stop at that reviewed specification before any test change or execution. Stage B,
full O execution and saved-output scoring still require separate approval.
