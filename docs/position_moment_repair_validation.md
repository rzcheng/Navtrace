# Position metric repair and replacement run

Ryan approved the concrete scope in `position_moment_results.md` on 2026-10-09.
This is an authorization addendum to the unchanged frozen protocol, not a change
to its equations, candidate, measurements, uncertainty rules or resource limits.
The first comparison remains a failed run at `55bb5b1`, documented at `258f05c`.

## Approved scope and acceptance

- Replace only the two N-by-2 sample-to-axis products (`centered` and `displaced`)
  in `tools/position_moment_metrics.py` with equivalent direct contraction using
  `optimize=False`. Keep all floating-point error checks and tolerances intact.
- Add deterministic synthetic regressions at N=512, 2048 and 8192 in
  `tests/test_position_moment_comparison.py`. Check the nine metrics and batch
  uncertainty against independent scalar calculations with rotated axes.
- Independently review the diff and regression evidence, run applicable tests,
  CTest and a fresh real C++ development demo. Commit clean reviewed code before
  evaluation. Parent owns integration and all Git operations.
- Execute exactly one replacement comparison under the same 600 s wall / 512 MiB
  peak RSS limits, writing to `results/position-moment-comparison-repaired/`.
  No second replacement attempt or automatic budget increase is authorized.
- Preserve every first-run artifact and its manifest digest
  `1052d424ea7a75b7303f34508c73aa64b0577000e513215d003d9c7a689422e8`.
  Keep all original reference/input artifacts read-only. Verify them after work.
- Inspect all results and plots, retain unfavorable or inconclusive findings,
  update issues #1–#3, and stop for review. Completion does not require an
  improvement in approximation quality.

The unchanged JSON SHA-256 is
`261309e0338ff048c5dc1626a7b24569930a390f6b51071e4eaaa09ab02a20e5`.
No candidate, sampler, runner, plotting source, prior, Q/R, gate, recovery policy,
production C++, dependency or old protocol is part of this repair. All consumed
seed labels and the separation between estimator inputs and evaluation data
remain unchanged. No new recording or sampling study is authorized; the existing
development demo remains an integration check.

## Validation record

The evaluation worker changed exactly the two projections plus one explanatory
comment. The new parametrized test checks all nine metric values, all 16 batch
values and their MCSE at each actual count using scalar sums, known rotated
axes and scalar elimination for marginal ellipse distances. The mathematical
definitions and uncertainty rules are unchanged. No saved sample was used to
choose a test, tolerance or implementation.

The separate evidence reviewer inspected the final diff against `258f05c` and
independently ran the focused suite: 30 passed, with RuntimeWarning treated as
an error. Both worker and reviewer performed an in-memory mutation restoring
the old products: N=512 passed; N=2048/8192 failed with the original matmul
error. Restoring the repair passed all three tests with source bytes unchanged.
No mutation was committed or written into the comparison implementation.

The coordinator inspected the final code/test diff and verified exact handoff
hashes before integration:

- Metrics: `841c5187afd23d9a02185183f3975cacb713087b7b2f3c83c04897308d2b623b`.
- Tests: `8455deec452221fc2005857c466cd83eb25cbaa3d591e186519a1cd02152bec8`.

Central checks passed: 167/167 pytest tests in 15.40 s with RuntimeWarning as
error, and 23/23 CTest cases. The existing 260 third-party Python deprecation
warnings remain. The fresh real C++ development demo completed all six runs;
both figures were inspected. Gated nominal/dropout position RMSE remained
0.132119 / 0.174847 m. These are unchanged production-path checks, not candidate
performance. No sanitizer or hosted CI run is claimed; no C++ source changed.

Logs are under `results/position-moment-repair-checks/`, and demo artifacts under
`results/position-moment-repair-demo/`. The pre-repair audit verified all 66
preserved baseline sources, all 11 first-run files, and seven fixed candidate,
runner, plot, protocol and historical report files. Their hashes are saved in
`before-repair-audit.json`. The first failed output is retained without edits.
The original mathematical implementation checks remain in
`position_moment_validation.md`.

## Replacement outcome

The one approved replacement launched at clean commit `89077fc` and completed
all 90 distributions, 810 metrics and 90 numerical-resolution checks in
4.639067 s at 296.234375 MiB. The candidate outputs and first failed run remain
unchanged. Read [position_moment_repaired_results.md](position_moment_repaired_results.md)
for all endpoints, unfavorable uncertainty/ellipse results, independent audit
and the next review decision. No third launch occurred.
