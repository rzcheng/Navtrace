# Matched-model long-outage follow-up

Freeze this protocol before running seeds 83001–83030. This follows M3's Path D
decision: test long-horizon uncertainty before designing recovery logic.
The question is whether the EKF's marginal position covariance describes its
position errors through a clean 20-second GPS outage when prior and process
noise match the stated discrete model. No estimator changes or parameter tuning.

## Fixed experiment

- 30 independent seeds, 83001–83030; development seed 20260921 only for tooling.
  These evaluation seeds are disjoint from M1–M3. They become consumed on use.
- 32 seconds per recording. Nominal GPS throughout; paired dropout removes GPS
  in [2,22) seconds, exactly 100 fixes. All remaining observations and truth
  are identical. GPS returns clean at 22 seconds; follow-up ends at 32 seconds.
- Basic and gated EKF, same frozen `scenarios/filter.conf`, 120 C++ replays.
  Gating remains at squared Mahalanobis threshold 9.210340371976184 for 2D GPS.
  No dead-reckoning comparison: this question concerns filter covariance.
- Reuse the independently implemented `generate_matched` Euler–Maruyama truth:
  10 ms increments, initial truth sampled from the filter's Gaussian prior,
  speed/rate/bias diffusion matching Q, independent prior/process/sensor streams.
  Sensors: gyro 100 Hz, wheel 20 Hz, GPS 5 Hz; standard deviations .008 rad/s,
  .05 m/s, .6 m. The estimator receives only sensor observations and its prior.
- This random-motion model can reverse or turn sharply. It is a controlled
  numerical experiment, not a realistic rover route. Matching dynamics/noise
  does not remove nonlinear EKF approximation error or guarantee Gaussian error.

## Measurements and interpretation

Evaluate at the full 100 Hz grid after all events at each timestamp. Report
whole-run position RMSE, [2,22) outage RMSE and last-sample error at 21.99 s,
final/max position error, wrapped heading error and all GPS decisions. Nominal
outage fields are absent/null. Settling retains M3's <=1 m for 2 elapsed seconds
on the 10 ms grid, within [22,32]; also count runs that settle but finish >1 m.
No observed settling is a censored result, never zero delay.

At the JSON's eleven fixed times, keep one sample per independent seed. At
22 s also analyze C++ **pre-GPS** state/P (after same-time gyro/wheel), separately
from post-GPS state/P. Record every first-return NIS and decision, including
rejected updates; no accepted-only selection.

For position error e = estimate minus truth and marginal 2x2 P, report:

- Mean error, centered sample covariance (denominator N-1), uncentered second
  moment E[e e^T] (denominator N), and mean reported P. Keep these distinct:
  a bias can disappear when errors are centered. Use world coordinates.
- Eigenvalues of the second moment whitened by Cholesky(mean P), descriptive
  aggregate ratios only. These are not mean per-run NEES and averaging P can
  hide directional conditional errors. Compare both, never substitute one.
- Each run's narrow/wide covariance-direction squared error divided by that
  direction's eigenvalue. Their sum equals marginal position NEES. Eigenvector
  signs are arbitrary; squared contributions are invariant.
- Position and full-state NEES ensemble means, and nominal 95% position-ellipse
  coverage counts, at each fixed time. Chi-square ensemble reference intervals
  assume independent Gaussian errors with correct covariance. Their use here
  is approximate for the basic nonlinear EKF and descriptive for the gated EKF.
  Each time is a separate correlated comparison; no multiple-testing claim.

Plot fixed-time ensemble NEES, position-error scatter with empirical and mean-P
ellipses at 22 s pre-GPS, and trajectory/error traces for the lowest seed selected
in advance. Aggregate ellipses are geometric summaries, not a common conditional
95% confidence region. Retain full plot ranges, all seeds and failure rows.

## Execution and stop

First test geometry against hand examples, paired data deletion, timestamp phase,
and failure accounting. Demonstrate a meaningful failing mutation, then restore.
Run a four-replay development smoke before evaluation; record measured runtime
and storage. Commit tested tooling before fresh-seed execution. Use one worker;
retain compressed C++ outputs, all inputs/truth, commands, versions, hashes and
source/git manifest. Refuse overwriting any existing output directory.

Report all 120 planned runs, failures and undefined statistics. A missing run
must not shrink an ensemble denominator. If an evaluation bug appears, preserve
the original artifacts and label any correction; never silently rerun seeds as
untouched. No Q/R/P retuning, fault scenarios, recovery policy, or new milestone
in this task. Inspect the diff and stop for Ryan's review.
