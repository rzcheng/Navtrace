# Saved mixture plot correction

The reporting-only correction in [issue #8](https://github.com/rzcheng/Navtrace/issues/8)
finished on 2026-10-18. The lower O panels now pair each saved error with its
explicit report timestamp. Seed 95004's maximum absolute whitened mean error
`0.23699700030170529` appears at **1 s**; its value at 0.25 s is
`0.09798650703049079`. The original study's **no-go** decision is unchanged.

Ryan approved this correction before edits and requested implementation and
review agents. The parent made one render from the existing `inventory.json`
into a fresh directory. There was no study rerun, candidate/oracle computation,
new sampling, new metric, tuning or production change.

## Change and preserved evidence

The old plot paired chronological times with dictionary iteration order.
Sorted JSON puts `1000000` before `250000`, so several errors appeared at the
wrong times. The plotting helper now looks up every prescribed timestamp
explicitly. An optional correction destination must be fresh and outside the
original study; existing destinations, descendants and aliases are rejected.
The normal study output path remains supported.

Only the runner, its reporting tests and this correction note changed. The
original [results report](trajectory_mixture_results.md), protocols, candidate,
metrics, samples and plots remain untouched. Its description of an unresolved
reporting defect is the historical record; this separate note records the
approved repair. It does not revise the original numerical findings.

Fresh artifacts:

- `results/trajectory-mixture-plot-correction-v1/observed-diagnostics.png` is the
  corrected O figure. All four seeds retain their `completed / fail` labels.
- `results/trajectory-mixture-plot-correction-v1/heading-variance.png` is
  byte-identical to the original H figure. Every branch/seed remains inconclusive.
- `results/trajectory-mixture-plot-correction-v1/manifest.json` records source,
  input, figure and validation-evidence hashes.
- `results/trajectory-mixture-plot-checks-v1/` contains the exact commands, logs,
  render driver, captured figure coordinates, copied font cache and preservation
  records. Generated artifacts remain outside Git.

The preservation audit verified all 54 files in the original study directory,
all 1,069 original check files, all 25 original demo files and 579 older artifact
hashes. The original study directory has the same file inventory and bytes.
Of 94 tracked files in the pre-correction commit, only the two approved source
files changed. The original checkout remains clean at `c834bf3`; the earlier
position-moment integration checkout remains clean at `53be2f7`.

## Checks and independent review

The implementation agent owned the runner and its tests, then stopped writes.
It executed no tests or plots. A separate read-only reviewer inspected the full
diff, selected test scope, command logs, render driver, both corrected PNGs and
saved plot coordinates. The parent inspected the same code and figures and
ran all validation centrally.

Actual checks for this correction:

- **13 selected Python cases passed**, with RuntimeWarning fatal and no skips:
  ten cases across five new reporting test functions, plus the existing
  output-reserve, plotting-cache and repeat-launch preservation checks.
- The sorted-JSON regression detected an in-memory restoration of the original
  wrong ordering. Plot spies also verified the actual lower-panel line arguments.
  Other checks covered unavailable values, saved seed 95004 and destination
  protection, including a symlink alias.
- **23 CTest cases passed** using the existing Debug build. These are deterministic
  fixed-input C++ regressions that exercise the unchanged EKF/replay methods;
  no C++ source changed.
- The single saved-table render passed. Captured real figure objects matched all
  **48 lower-panel timestamp/value pairs**, all four O labels and all eight H
  labels. The render did not import the candidate module. The H figure's hash
  matched the original exactly.
- All preservation comparisons passed. The original font cache was copied into
  the fresh checks directory before rendering; original caches were not reused
  as writable output.

The full Python suite, sampling-based tests, C++ demo, study, oracles and candidate
were not run for this correction. Those would exceed its reporting-only scope.
No new build, sanitizer run or hosted CI run is claimed. The earlier study's
validation record is historical, not a new result from this correction.

## Hashes and reproducibility

The baseline commit is `bb891566df8c031b89d2288eb051743a0669c7f3` on
`codex/trajectory-mixture-design`. The source used for the original study remains
available at `036e14f7a547ff0cfab1666c5bcd31f6f8f357b7`.

| file | SHA-256 |
|---|---|
| corrected runner | `60538d532b3bd01b416c5d7f4678d5d18ed4fdae6e89320f27da2c5f146dc674` |
| corrected runner tests | `17b50d5022db19eea5005569e19c2c5203fa3577516b634b2c4e642458610d50` |
| saved inventory | `b94de595703e6a2c555fb84c42093f3ce7a5a8c372347b0bbd8b8ebfa73f6cfc` |
| original study manifest | `8b629c55fc5fddc8596af7e0e5322d888ffe76dd0cd477fc5984275eeffeb14d` |
| corrected O figure | `69ffff83add6e6a201e257b90b74e3194b732e35ee72db15686948fc26016bbf` |
| unchanged H figure | `64c1912d35842388261b252725374f95bd66b668d8670064ae3c000a99b40bd8` |

`render-command.json` records the Python command; `render_saved_tables.py` is
its retained driver. It reads the saved inventory, calls the plotting function
with the fresh correction destination, captures the rendered line data and
checks the unchanged labels. It refuses to overwrite the completed correction.
`pytest-command.json` and `ctest-command.json` give the exact test selections.
`preservation-before.json` and `preservation-after.json` record the byte audit.

This fixes presentation only. All O ESS failures, seed 95004's individual error
failure, H precision limits and underflow limitations remain. No calibration or
navigation improvement follows from the repaired figure. Work stops here for
review; another estimator change or experiment requires its own approved scope.
