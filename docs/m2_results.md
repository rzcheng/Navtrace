# M2 measured GPS fault results

The frozen [protocol](m2_protocol.md) was committed as `8f4d66c` before the
evaluation seeds were run. Tested study code was committed as `0a99149`;
the full experiment ran from that clean revision. The C++ implementation and
`scenarios/filter.conf` are byte-for-byte unchanged from M1 (`2279293`).
No fault sizes, times, noise parameters, priors, gate thresholds, or recovery
criteria were changed after inspecting development or held-out results.

The study executed **45 C++ replays**, covering five cases, three methods,
and seeds **70001, 70002, 70003**. Those seeds were held out from M1 development;
they have now been inspected and must not be described as an untouched test set
for future tuning. Every case retains the same underlying realization within
its seed. Truth and fault annotations were used only in Python evaluation.

Artifacts, generated locally and ignored by Git:

- [Individual metrics](../results/gps-fault-study/metrics.csv),
  [aggregate metrics](../results/gps-fault-study/summary.csv), and
  [paired differences](../results/gps-fault-study/paired-comparisons.csv).
- [Recovery episodes](../results/gps-fault-study/recovery.csv) and
  [manifest](../results/gps-fault-study/manifest.json).
- [RMSE plot](../results/gps-fault-study/position-rmse.png),
  [gate decisions](../results/gps-fault-study/gate-decisions.png), and error
  traces for [70001](../results/gps-fault-study/seed-70001-errors.png),
  [70002](../results/gps-fault-study/seed-70002-errors.png),
  [70003](../results/gps-fault-study/seed-70003-errors.png).

## Position error

Whole-run position RMSE in meters, **median [minimum, maximum] across three
seeds**. These ranges are observed spread, not confidence intervals.

| Case | Dead reckoning | Basic EKF | Gated EKF |
|---|---:|---:|---:|
| Nominal | 15.818 [15.663, 15.871] | 0.190 [0.159, 0.201] | 0.196 [0.160, 0.200] |
| Dropout [20,40) s | 15.818 [15.663, 15.871] | 0.378 [0.343, 0.789] | 0.378 [0.272, 0.789] |
| Isolated spikes | 15.818 [15.663, 15.871] | 0.354 [0.346, 0.396] | 0.201 [0.161, 0.205] |
| Sustained offset | 15.818 [15.663, 15.871] | 5.697 [5.658, 5.788] | 0.378 [0.272, 0.789] |
| Early dropout + offset return | 15.818 [15.663, 15.871] | 5.964 [4.508, 8.957] | 6.775 [5.821, 22.000] |

The gate reduced RMSE for all three seeds in the spike and sustained-offset
cases, but worsened it for all three seeds in the early-dropout/offset-return
case. In nominal, it was slightly worse in two of three seeds. All individual
results, reference-window errors, actual outage-end errors, final errors, and
wrapped heading metrics remain in `metrics.csv`; medians do not replace them.
Dead reckoning's evaluated poses were verified identical across all five
variants of each seed, as expected when only GPS changes.

## GPS decisions and return behavior

| Case | Injected fixes rejected by gate | Clean fixes rejected by gate |
|---|---:|---:|
| Nominal | Not applicable (0 injected) | 7 / 903 |
| Dropout | Not applicable (0 injected) | 4 / 603 |
| Spikes | 9 / 9 | 7 / 894 |
| Sustained offset | 300 / 300 | 4 / 603 |
| Early dropout + offset return | 26 / 75 | 228 / 258 |

Counts pool the three prescribed realizations. They are empirical counts for
these particular injected faults, not a general detection-rate claim. The
ungated EKF accepted all GPS; DR ignored it. In the sustained-offset case,
rejecting all 100 injected fixes per seed made the gated estimate exactly match
its paired dropout result.

Position settling is error <=1 m for a full 2 s on the 100 Hz grid, measured
after the defined clean-return reference. The start and its confirmation time
are distinct. Missing delays remain null.

- Normal dropout: both EKFs met the criterion on all three seeds. Two were
  within the limit at the return timestamp (confirmed 2 s later); seed 70002
  began its qualifying interval 0.4 s later (confirmed at delay 2.4 s).
- Spikes: the gated EKF was within the limit at all nine return references.
  Basic EKF was within the limit at five and recovered later at four, with
  qualifying-start delays 0.8 or 1.4 s. All were confirmed 2 s after their starts.
- Sustained offset: gated behavior matched dropout. Basic EKF never met the
  criterion during the 20 s follow-up on any seed, despite immediately accepting
  clean GPS; its final position errors were 2.688, 2.937, and 2.879 m.
- Early dropout + offset return: neither EKF met the criterion during the
  15 s clean-return follow-up on any seed. This is **not observed within the
  horizon**, not proof that recovery can never happen.
- DR did not meet the criterion in any of its 18 return windows. No window
  had insufficient follow-up in this fixed protocol.

## Inspect the failure, rather than tune it away

In early dropout, all three gated runs rejected all 76 clean fixes from 45–60 s.
Their paths into that failure differed:

| Seed | Offset fixes accepted / 25 | NIS at first clean fix, 45 s | Final gated position error m | Final basic EKF error m |
|---|---:|---:|---:|---:|
| 70001 | 0 | 75.484 | 48.044 | 3.328 |
| 70002 | 25 | 169.026 | 8.991 | 1.779 |
| 70003 | 24 | 128.648 | 6.984 | 2.221 |

The threshold remained 9.21034. For seed 70002, the offset fix at 40 s was
accepted with NIS 2.048 while position uncertainty was large. At 45 s, the
clean-fix residual was (-6.407, 4.758) m and S eigenvalues were approximately
(0.372, 0.379) m². The filter had become confident around the displaced position;
the good observation was rejected. Seed 70003 shows the same pattern.

Seed 70001 had already drifted 23.09 m by 39.99 s. At 1.99 s its estimated
bias was -0.004381 rad/s versus simulated +0.015 rad/s; only two seconds of GPS
had been available to constrain heading/bias. At the clean return S was highly
anisotropic, with eigenvalues about (1.081, 1502.964) m². Large variance in one
direction does not make every residual plausible. These observations support
an initialization/observability and covariance-model investigation; they do
not isolate all causal contributions or justify an automatic reset.

## Verification and limits

- 22/22 GoogleTest/CTest cases and 22/22 pytest cases passed; no skips.
  C++ tests and Python integration tests also passed with the ASan/UBSan binary.
- The complete 15-replay **development-seed** study ran with ASan/UBSan and
  exactly matched Debug metrics and recovery records. The 45 held-out replays
  used Debug. No sanitizer findings occurred. Hosted CI/Linux execution remains
  unverified; CI is configured to run development data only.
- A deliberate 201-to-200 sample mutation wrongly declared a 1.99 s interval
  recovered. The focused regression failed, then passed after restoring the
  code. The mutation is absent from the final source.
- Review added explicit checks for a complete seed/scenario/method matrix and
  all expected recovery episodes. A plotting import-order issue was corrected
  so matplotlib selects its writable cache before importing plotting classes.
- Fourteen existing third-party matplotlib/pyparsing deprecation warnings
  remain; they were not suppressed. No compiler or estimator changes were made.
- The audit accounted for 335,025 events and 54 recovery episodes. Manifest
  source/input/output/executable hashes were verified against the clean run.
  Independent NumPy solves reproduced every saved GPS NIS within 6.14e-13
  scaled relative error. Minimum recorded covariance eigenvalue was 4.28e-7,
  and recorded symmetry error was zero; this does not establish calibration.

Validation logs are in `results/m2-validation/`, including the mutation log
and `heldout-audit.json`. Environment and dependency versions remain those in
the manifest and `docs/dependencies.md`. The fixed protocol SHA-256 is
`ee46d1a6315b8417173ad457dbc18649effc24f08e51496ddbe30384d76f5294`.

M2 stops here. Three seeds, one trajectory, constant gyro bias, and ideal wheel
measurements apart from white noise are limited evidence. The next useful task
is a controlled study of bias observability and covariance consistency during
early GPS loss, using newly reserved seeds, before choosing a reacquisition
mechanism. No recovery algorithm, calibration study, or later milestone was
implemented in this session.
