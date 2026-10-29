# Position oracle validation

The bounded implementation in [issue #12](https://github.com/rzcheng/Navtrace/issues/12)
is complete locally on 2026-10-19. All ordinary deterministic checks passed,
including the six deliberately faulty computations and required regressions.
One narrow-position control remains **unresolved** under the frozen relative
precision criterion. Passing tests does not turn that result into a precision
pass or establish accurate moments in every direction.

No new comparison or sampling experiment was run. The complete published O
sequence was not executed by this oracle, and no saved candidate mixtures or
navigation recordings were evaluated with it. Issue #10 remains **defer**;
the earlier **no-go** and consumed groups 96001–96004 remain unchanged.

## Approved scope and source

Ryan approved implementation and deterministic validation of the reviewed
[specification](multistep_position_oracle.md), frozen at
`8be596d9e4fe708218c377cb144f0882d4a34242`, SHA-256
`411b448230d93787fd450f161f52ea4d424123c0b80e247c80d8854506344d48`.
Only these three files were added:

- `tools/multistep_position_oracle.py`: independent physical Gaussian history,
  one received-prefix conditioning, full endpoint moments and diagnostics.
- `tests/test_multistep_position_oracle.py`: independent fixed expected values,
  numerical checks, refusal cases and restored fault mutations.
- This validation note.

The code and tests were independently reviewed and committed cleanly before
the first oracle execution at `b445baff300709fabc705a1288990cdb23ff985c`
(`add independent position moments`). Neither changed after that execution.

| file | SHA-256 |
|---|---|
| oracle | `8a315d6d6b5ff46567e5e60d5b308b0414a6a3bce8049e45f08e507f3aca3cf5` |
| tests | `feae2e7c8217fb4e1c0c7c349cf62a6b4e9effabfb2d49f2d2f41052041fd8be` |
| preserved monitor | `60538d532b3bd01b416c5d7f4678d5d18ed4fdae6e89320f27da2c5f146dc674` |

The oracle uses the declared checkpoint factor as its law and records its
residual against the supplied covariance. It retains every physical history
correlation, conditioned initial position, and position/final-latent cross
block. Heading is the unwrapped lift used by the synthetic model. There is no
candidate inference import, RNG, artifact reader or comparison CLI.

The hard interface bounds remain 100 positive substeps, 32 received observations,
306 physical coordinates and six checkpoint-factor columns, with max step
0.01 s and max gap 2 s. Inputs remain causal and ordered, with gyro before wheel
at equal timestamps and end-step diffusion. Output arrays and nested diagnostics
are immutable. Invalid input, numerical failure and unresolved inference remain
separate outcomes; completed moments never imply certified directional precision.

## Validation actually run

The existing project environment was Python 3.11.9, NumPy 2.2.6, pytest 8.3.5
and Matplotlib 3.10.3 on Darwin arm64. The preserved monitor set its five
thread-count environment variables to one before numerical imports.
`RuntimeWarning` was fatal in focused and full Python testing.

| check | observed result |
|---|---|
| Debug configure and build | passed; production sources unchanged |
| CTest | 23 passed, zero failed |
| Unchanged C++ development demo | passed; both figures inspected |
| Pre-execution preservation | passed |
| Fixed oracle tests, first execution | 52 passed in 3.18 s; 43 JSON evidence records |
| Full Python suite | 382 passed in 36.88 s |
| Retained-log summary | passed; no oracle re-execution |
| Final preservation audit | passed; all protected bytes and checkout states preserved |

The focused suite covers the checkpoint and deterministic support, the preserved
nine-dimensional first-step O oracle, explicit linear-motion and two-step-noise
laws, scalar-conditioned rank-one histories at every prescribed prefix, literal
phase signs, pair limits, characteristic finite differences, shared-heading
100-step controls, partition invariance, signed and zero speed, unwrapped heading,
causal immutability, covariance checks, bounds and failure statuses.

The full historical suite and development demo use their existing regression
fixtures, including their existing synthetic sampler/candidate smoke tests.
They are not a new candidate comparison or a claim that no random generation
occurred anywhere in regression validation. The new oracle checks themselves
are deterministic and RNG-free; no reserved study streams were opened.

No C++ changes were made, so no new sanitizer run was required or claimed.
No hosted CI result is claimed.

## Measured accuracy and unresolved precision

These are maxima from the ordinary fixed checks, excluding intentional mutations.
They describe these fixtures, not an error bound over the full interface.

| quantity | largest observed absolute error |
|---|---:|
| full six-state endpoint mean | 2.220446049250313e-16 |
| full six-state endpoint covariance entry | 4.614364446098307e-16 |
| log evidence | 8.881784197001252e-16 |
| centered pair kernel | 1.0192386199113823e-15 |
| characteristic finite-difference check | 2.1141262755206376e-11 |

Ordinary exact checks used the unchanged absolute 1e-10 limit. Finite differences
used step 1e-6 and absolute/relative tolerances 1e-7, including the reviewed first
difference of an independent analytic derivative for the second moment.
The independent O covariance-block construction differed from its declared
factor Gram by 6.938893903907228e-18; that residual was retained.

Reported endpoint asymmetry was zero. The lowest logged ordinary endpoint
eigenvalue was -1.5702692935554511e-18 in the first rank-one prefix, within the
unchanged roundoff tolerance. It was not clipped. Covariance validity under
that tolerance is not evidence of resolved narrow-direction accuracy.

For the shared-heading control the distance is 2 m, so the narrow x variance is
`2 [1-exp(-V)]^2` m². The independent target uses `expm1` to retain this small
quantity. Both the assembled-matrix projection and direct projected-term sum
must meet the fixed 1e-7 relative criterion to label this fixture resolved.

| heading variance V (rad²) | expected x variance (m²) | returned x variance (m²) | relative error | result |
|---:|---:|---:|---:|---|
| 1e-8 | 1.9999999800000006e-16 | 1.999999967550318e-16 | 6.224841314703393e-9 | resolved for this fixture |
| 1e-12 | 1.9999999999979997e-24 | 2.0000968717590786e-24 | 4.843588053953116e-5 | **unresolved** |

The two projection routes agree in both controls, but share the pair-kernel
moments. Their agreement cannot detect cancellation already present there.
The V=1e-12 endpoint also retained a small negative eigenvalue,
-3.599235047677823e-29, inside the covariance tolerance. Neither positivity
checks nor small absolute error can replace the failed relative precision check.
No special-case correction, precision increase, clipping, new fixture or
threshold change was added after seeing this result.

## Fault sensitivity and retained failures

All six prescribed faulty computations were detected. Each mutation was local
to a test context, restored afterward and absent from the committed source.

| deliberate fault | actual detection |
|---|---|
| omit cross-step pairs | materially indefinite endpoint covariance |
| use plus phase for the conjugated minus phase | literal sign error 0.020000000000000004 |
| omit initial-position cross terms | materially indefinite endpoint covariance |
| omit final-latent cross terms | materially indefinite endpoint covariance |
| use new-state displacement | mean error 7.410395800287045e-5; covariance error 7.407392576673515e-6 |
| condition twice | mean error 0.003019548872180422; covariance error 0.007730649022556374; evidence error 5.352105698372681 |

These are expected detections, not ordinary oracle failures. Refusal tests also
confirmed distinct invalid-input, unresolved-solve and arithmetic-failure states.
Every ordinary focused and full-suite test passed on its first execution;
there was no post-execution implementation repair or test retry.

One metadata-only setup inspection initially used system Python 3.14.2, which
lacked pytest package metadata. It failed before producing its environment
record. The inspection was repeated with the existing project virtualenv and
retained in `initialization-metadata-repair.json`. No numerical input, test,
monitor, source or resource limit changed. The review also resolved interface,
JSON serialization, pre-coercion bounds and overflow classification issues
before source freeze and execution.

## Review, resources and artifacts

The implementation agent owned only the oracle module. A separate validation
agent owned only the tests and independently derived scalar characteristic and
small affine physical targets. A third agent reviewed all source, mathematical
identities, expected values and resource handling before execution, then checked
the actual logs, mutations, numerical limitations and demo figures. Agents did
not run unmonitored numerical work or perform Git operations.

The parent inspected the complete diff and independent target constructions,
ran all monitored checks, read the 43 evidence records, inspected both demo
plots, audited preservation and owns this report and local commits. The
reviewer's and validation agent's independent read-only audits agree with the
reported maxima and unresolved outcome.

All monitored commands stayed within 900 s aggregate charged command time,
2 GiB assessed per-invocation process-tree memory and 256 MiB retained output.
Across all nine monitored commands, including the final preservation audit,
charged time was 85.09262191609014 s. The largest assessed memory was 294666240 bytes
(about 281.02 MiB), in full pytest. Retained output was about 199 MiB including final metadata; the exact
byte inventory is recorded in the final manifest. The complete command
accounting is retained in the ledger. The focused command
alone was charged 5.009426375036128 s and assessed at 84574208 bytes; this is
whole-suite validation cost, not an isolated 100-step oracle benchmark.

Retained-output accounting includes the build, tool/test trees and caches as
well as fresh validation/demo files. The monitor retains its 100 ms RSS sampling
and short-lived-descendant limitation; its assessed peak is not an exact proof
of every instantaneous process-tree peak. Failed or unfinished attempts would
remain in the ledger, and resource limits were not increased.

Fresh artifacts are under:

- `results/multistep-position-oracle-checks-v1/`: source/fixture review, numbered
  command logs and resources, `numerical-summary.json`, full validation ledger,
  preservation audits and final evidence manifest.
- `results/multistep-position-oracle-demo-v1/`: unchanged development inputs,
  replay outputs, metrics, manifest, `trajectories.png` and `errors.png`.

The demo metrics exactly match the preserved previous demo. Gated position RMSE
was 0.13211917187409775 m nominal and 0.17484668873395803 m dropout; the dropout
end-window error was 0.3748515225300933 m. The figures retain the unfavorable
outage growth and correctly distinguish the nominal reference window from the
actual dropout interval. These are production baseline results, not oracle
accuracy measurements.

Preservation checks cover all 104 prior tracked files, 3,776 prior worktree
artifacts and 579 older artifact hashes. The original checkout remains clean at
`c834bf3e5b9b421564c2e0050092895abe12dd43`, and the previous position-moment
integration remains clean at `53be2f72d15f8d485640eaa09868210231b3e886`.
No old protocol, result, failed artifact, candidate, production file or tuning
changed. No push, merge, release or published-history rewrite occurred.

## Remaining limit and next decision

This milestone validates the frozen deterministic cases. The 100-step control
has only one uncertain checkpoint direction and no process diffusion; it does
not establish runtime or numerical precision for a full 100-step, 32-observation
O prefix with all process coordinates active. No position-distribution,
calibration, ellipse-coverage or navigation-performance claim follows from
these moment checks. Unresolved directions must not silently become whitening
denominators in a later comparison.

Stop here. The next scope decision is whether to approve a separate design for
numerical eligibility and a comparison with the already saved #10 outputs.
That design would need explicit precision refusal rules and could not change
#10's frozen verdict or reopen its consumed streams. This report does not
implement or execute that next comparison or an additional experiment.
