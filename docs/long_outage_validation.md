# Long-outage tooling validation

Before running evaluation seeds, the Debug build passed 23/23 CTest cases and
56/56 pytest tests (12.11 s for pytest). The AddressSanitizer/UndefinedBehaviorSanitizer
build passed the same 23 C++ tests. Warnings remain enabled with `-Werror`.
Python reported 260 third-party matplotlib/pyparsing deprecation warnings;
none were suppressed and no tests were skipped.

Eight new tests cover biased errors versus centered sample covariance, rotation
invariance, the distinction between aggregate covariance ratios and per-run
NEES, invalid covariance/failed sample counts, pre-GPS versus post-GPS state,
fixed protocol/seed roles, exact paired observation removal, a real four-replay
integration, and injected replay failures. Existing C++ tests still cover the
estimator math, gate, event ordering and input validation.

Mutation check: temporarily replace `errors.T @ errors / N` with
`centered.T @ centered / N` in `position_geometry`. The hand-example regression
failed (expected second moment 5, mutated value 1). Restore the implementation:
the same test passes. This catches loss of systematic error through inappropriate
centering. Mutation logs and restored source hash are saved under
`results/long-outage-validation/`; the mutation is absent from committed code.

The four-replay development smoke using the sanitizer binary completed without
failures in 7.928 s, retaining 6.457 MB. It used seed 20260921 only. Scaling this
conservatively suggests roughly four minutes and 194 MB for 120 replays; the
full study uses the Debug binary and records its actual cost separately.
The development geometry figure was visually inspected before evaluation.

```bash
.venv/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
.venv/bin/cmake --build build --parallel 2
.venv/bin/ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest -q
.venv/bin/cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DNAVTRACE_SANITIZE=ON
.venv/bin/cmake --build build-sanitize --parallel 2
UBSAN_OPTIONS=halt_on_error=1 .venv/bin/ctest --test-dir build-sanitize --output-on-failure
UBSAN_OPTIONS=halt_on_error=1 .venv/bin/python tools/run_long_outage.py \
  --binary build-sanitize/navtrace_replay --output results/long-outage-development --development-smoke
```

Run logs preserve exact output. Use a new output directory for another smoke.
The macOS/Linux CI workflow includes the development smoke; hosted CI execution
was not verified during this task. No new dependency was needed.

The subsequent clean-commit evaluation completed all 120 replays in 118.677 s
with zero numerical/output failures. See [measured results](long_outage_results.md).
Afterward all 409 artifact hashes and all 240 uncompressed output hashes were
verified, and exact nominal/dropout pairing was independently checked for all
30 seeds. Every fixed-time sample remained defined. All four actual plots were
visually inspected. Only documentation changed after the evaluation commit.
