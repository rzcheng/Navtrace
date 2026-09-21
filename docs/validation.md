# M0–M1 validation record

The first development demo used seed **20260921**, 60 seconds, the committed
`scenarios/filter.conf`, and analytical straight/circular truth. Gyro bias was
constant 0.015 rad/s; noise standard deviations were GPS 0.6 m, wheel 0.05 m/s,
gyro 0.008 rad/s. Process/prior/gate parameters were set before the first demo
and were not retuned from these results. Reserved seeds 70001–70003 remain unused.

## Actual results

Sources: [metrics JSON](../results/first-demo/metrics.json),
[trajectory figure](../results/first-demo/trajectories.png),
[error figure](../results/first-demo/errors.png), and
[run manifest](../results/first-demo/manifest.json).
All estimates were read from the compiled C++ executable's CSVs. Python only
generated sensors/truth, evaluated, and plotted. Results are ignored by Git;
the committed generator/configuration reproduce them.

Evaluate at 6,001 common gyro timestamps, 0 through 60 s inclusive, after
all simultaneous observations. There is no interpolation. Position RMSE is
`sqrt(mean((estimated_x-true_x)^2 + (estimated_y-true_y)^2))`, not per-axis RMSE.
The 2,000-sample outage window is `[20,40)` s; end-window error is at **39.99 s**,
before the returning GPS measurement. For nominal, this is a reference window
with GPS still present. GPS input counts are 301 nominal and 201 dropout.

| Recording | Method | Whole RMSE m | Window RMSE m | End-window error m | Final error m | GPS accepted/rejected/ignored |
|---|---|---:|---:|---:|---:|---|
| Nominal | DR | 15.938997 | 10.288852 | 16.701670 | 35.161352 | 0/0/301 |
| Nominal | Basic EKF | 0.114672 | 0.104327 | 0.067158 | 0.021048 | 301/0/0 |
| Nominal | Gated EKF | 0.132119 | 0.113181 | 0.068279 | 0.019037 | 300/1/0 |
| Dropout | DR | 15.938997 | 10.288852 | 16.701670 | 35.161352 | 0/0/201 |
| Dropout | Basic EKF | 0.130551 | 0.132625 | 0.220877 | 0.008695 | 201/0/0 |
| Dropout | Gated EKF | 0.174847 | 0.217233 | 0.374852 | 0.010035 | 200/1/0 |

Heading error is wrapped into `[-pi,pi)` before calculating RMSE:

| Recording | Method | Whole heading RMSE rad | Window heading RMSE rad | Final signed heading error rad |
|---|---|---:|---:|---:|
| Both | DR | 0.525300 | 0.464463 | 0.906908 |
| Nominal | Basic EKF | 0.018085 | 0.010799 | 0.001287 |
| Nominal | Gated EKF | 0.019684 | 0.011759 | 0.001105 |
| Dropout | Basic EKF | 0.018109 | 0.010264 | -0.000950 |
| Dropout | Gated EKF | 0.020892 | 0.015997 | -0.001548 |

All 100 window GPS observations were accepted by both EKFs in nominal; there
were zero GPS updates in that window for dropout. DR ignores all GPS. No
outlier fault injection or broad outlier/recovery study was run this milestone.

Gating **worsened** whole-run position RMSE in both recordings. At 12.6 s an
ordinary noisy GPS observation had residual `[-0.730427, -1.788899]` m, S
`[[0.366641,-0.005439],[-0.005439,0.405927]]` m², and NIS 9.436151 versus
threshold 9.210340. This single rejection occurs in both paired recordings.
At 19.99 s the basic bias estimate was 0.0157438 rad/s; gated was 0.0160612,
versus generator truth 0.015 (read only for this investigation). The changed
pre-outage heading/bias estimates are consistent with the greater gated drift.
This is not a controlled causal decomposition or a reason to tune this seed.

The unusually small EKF outage error is from a favorable model: constant bias,
no wheel slip, calibrated frames, simple motion, a good initial pose prior, and
20 seconds of GPS before loss. Dead reckoning cannot estimate the injected
bias. Do not generalize these numbers to real robots or arbitrary outages.

## Tests and checks

- C++: **22/22 GoogleTest/CTest tests** pass in Debug and with ASan/UBSan.
- Python: **10/10 pytest tests** pass against both binaries; no skips.
- Warnings: project C++ targets build under six warning flags including
  `-Werror`; no compiler warnings. Pytest reports 14 third-party matplotlib
  3.10.3 / pyparsing 3.3.3 deprecation warnings; they are not suppressed.
- Coverage includes wrapped angles, stationary/reverse-straight/analytical-turn
  prediction, finite-difference Jacobians, the 4/13 hand update, cross-covariance
  use, gyro rate+bias, PSD/symmetry, time-scaled noise, transactional invalid
  inputs, duplicate/delayed/equal-time events, gating and outage semantics,
  deterministic replay, independent metric examples, and actual CLI behavior.
- The analytical circular test uses h=0.001 s and an independently derived
  global Euler position error bound 0.0008 m; tolerance is 0.00081 m.
  PSD tests allow -1e-10 for roundoff. No eigenvalues are clipped and no
  tolerances were weakened to get a pass.
- Minimum recorded EKF covariance eigenvalue in the first demo was about
  4.30e-7; maximum recorded symmetry error was zero. This establishes numerical
  validity in this run, **not uncertainty calibration**.
- The headless sanitizer demo runs all six method/recording combinations.
  Hosted macOS/Linux CI has not run; Linux is not locally verified.

Local command output is saved under `results/validation/`. The mutation check
temporarily changed speed process noise from `qv*h` to `qv`; the elapsed-time
test failed with 20 instead of 0.2 and 40 instead of 0.4. The source was restored,
rebuilt, and the full suite passed. See `mutation.log` and `restored-ctest.log`.

Review also found a real output-path alias bug: `output.csv` and
`./output.csv` could be opened as separate output streams but address the same
file. Paths are now compared with `weakly_canonical` before opening. The new
pytest regression was run against the old comparison and failed because the
CLI wrongly returned success; it passes after restoration of the fix. Evidence
is in `output-alias-regression.log`. Both deliberate regressions are absent
from final code.

## Remaining boundary

Euler/diffusion discretization is first order; PSD does not establish statistical
consistency. Bias and heading are weakly observable in some motions and during
outages. The GPS gate can reject good returning fixes after drift; no recovery
logic exists. No held-out performance, recovery time, real-time guarantee,
latency benchmark, real-world robustness, or superiority claim is supported.
The next task is a predefined bad-GPS experiment on reserved seeds, including
accept/reject behavior and a recovery criterion decided before running it.
