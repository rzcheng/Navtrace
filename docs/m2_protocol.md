# M2 GPS fault experiment: predefined protocol

This protocol is committed before evaluating seeds 70001, 70002, and 70003.
It extends M1 only with reproducible GPS fault experiments and measurement of
return behavior. No estimator tuning, reset, covariance inflation, slip model,
or uncertainty calibration is part of M2. The filter configuration's SHA-256
is frozen in `scenarios/gps_faults.json` and checked by the runner.

Use M1's 60 s analytical trajectory and noise/bias settings. Generate one
nominal sensor realization per seed and derive all variants from it. The
generator may read faults; the C++ executable receives only the ordinary
seven-column observations and the same M1 prior/filter configuration.
Offsets change GPS values without increasing their reported covariance.
Annotations and truth go only to the evaluator. A clean fix means no injected
fault, not zero measurement noise.

| Case | Prescribed modification | Uncorrupted-GPS return references |
|---|---|---|
| Nominal | None | Not applicable |
| Dropout | Remove GPS in [20,40) s | 40 s |
| Spikes | At 12, 28, 44 s add (12,-8), (-10,12), (15,5) m respectively | 12.2, 28.2, 44.2 s |
| Sustained offset | Add (6,-4) m to GPS in [20,40) s | 40 s |
| Early dropout + offset return | Remove GPS in [2,40) s, then add (6,-4) m in [40,45) s | 45 s |

For each seed/case run dead reckoning, the ungated EKF, and the gated EKF:
45 compiled replays total. All share the same initial prior and sensor streams.
The last case stresses a return after little initial GPS conditioning; it does
not promise that a particular failure will occur. Use development seed 20260921
to debug software before viewing the evaluation seeds. Record every evaluation
run; after inspection these three seeds are no longer an untouched test set.

## Metrics fixed before evaluation

Evaluate exact 100 Hz gyro timestamps, using the last output after simultaneous
updates. Report whole-run position RMSE, maximum and final position error,
wrapped heading RMSE/final signed error, and [20,40) s reference-window metrics.
For actual outages additionally report RMSE over that outage and position error
at its last sample before GPS returns. Never use a corrected return-time pose
as the end-of-outage error.

Join GPS diagnostics to evaluator-only fault timestamps. Count accepted,
rejected, and ignored observations separately for injected and clean GPS.
Report empirical rejection fractions with their actual denominators; use null
when the denominator is zero. DR ignores GPS and has no gate rejection rate.
These fractions apply to this finite fault set, not a universal detection rate.

For each return reference, define **position settling** as the earliest start
on the evaluation grid at or after that reference for which Euclidean position
error is <= 1.0 m at every sample through start + 2.0 s, inclusive. Require a
uniform complete grid; no missing sample may be treated as evidence of recovery.
Report both start delay and confirmation delay. Two seconds requires 201 samples
on a 100 Hz grid; 200 samples span only 1.99 s. This is a grid-based empirical
criterion, not a continuous-time guarantee or uncertainty statement.

Statuses are `within_limit_at_return` (qualifying run starts at reference),
`recovered` (starts later), `not_observed` (adequate follow-up but no qualifying
run), and `insufficient_followup` (less than 2 s of observations). Missing
recovery delays remain null, never zero. A brief dip below 1 m is not recovery.
Later departure after a qualifying dwell is allowed by this definition; the
error traces and final error remain part of the report.

For spikes, observation windows end at the sample before the next injected
spike. Other windows end at 60 s. Also report the delay to the first accepted
**uncorrupted** GPS in the same window. This is separate from position settling:
accepting GPS alone does not show accurate localization. DR's acceptance delay
is not applicable. A no-acceptance result is bounded by available follow-up.

Aggregate per-case/per-method position RMSE as the median and min/max across
the three seeds, while retaining all individual results and paired comparisons.
Show every seed in plots. Count each settling status rather than hiding failures
inside an average of only successful recoveries. No recovery-threshold changes,
favorable seed selection, percentage-improvement headline, or generalization
from three seeds is permitted. Freeze and record protocol/configuration/source
and executable hashes. Preserve the full trajectories and diagnostics.
