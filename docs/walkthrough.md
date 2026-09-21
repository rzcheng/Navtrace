# Review the estimator, one operation at a time

Start with `include/navtrace/ekf.hpp`: the six entries describe world position,
heading, body-forward speed, true yaw rate, and additive gyro bias. GPS observes
the first two; wheel speed observes speed; a gyro observes **rate plus bias**.
They are noisy observations of state, not perfect controls. Read the three
measurement Jacobians in `src/ekf.cpp` and identify each nonzero column.

## One prediction

Take state `[0,0,0,2,0.1,0]` and dt=0.01 s. `propagate_mean` returns
`[0.02,0,0.001,2,0.1,0]`. `transition_jacobian` adds `F(px,v)=0.01`,
`F(py,yaw)=0.02`, and `F(yaw,omega)=0.01` to identity at this heading.
`Ekf::predict` calculates `F P F^T + Qh`. With the default diffusion,
Qh's speed/rate/bias diagonal entries are `0.0025`, `0.0001`, and `1e-9`.
This explains both elapsed-time scaling and the units: variance intensity
times seconds gives variance. More than 10 ms is split into bounded steps.

Euler uses heading at the start of each step; an exact circular arc bends
within the step. Our independent arc test bounds the resulting discretization
error. `docs/model.md` states the matching first-order covariance approximation.
Check this choice before changing either the mean map or F/Q.

## One GPS correction

For an independent x coordinate with prior mean 10 m and variance 4 m², a
GPS measurement 12 m with variance 9 m² gives residual 2 m and innovation
variance 13 m². Gain is `4/13`; the posterior mean is `10+8/13 = 10.6153846`
m, and posterior variance is `36/13 = 2.7692308` m². The x contribution to
NIS is `4/13`. `Update.IndependentHandComputedGps` checks these numbers.

Read `Ekf::update_gps`: form r and S, factor S with Eigen LLT, solve for NIS
and gain, then correct state. Joseph form updates P while retaining all
cross-covariances. This lets GPS affect heading and bias indirectly once
motion has coupled them to position. The test `CrossCovariancesArePreservedAndUsed`
checks a simple instance. No matrix inverse is computed.

GPS gating compares the squared distance NIS with 9.2103. A 2D Gaussian
innovation with correct covariance would nominally lie below this 99% cutoff;
that is an assumption, not a measured detection rate. A rejection returns
diagnostics and skips correction entirely, so covariance does not shrink.

## Read the event boundary

`Replay::process` in `src/replay.cpp` validates input and ordering before
copying/predicting a candidate. It commits state and time only after a valid
operation. An invalid covariance leaves time unchanged. A valid GPS rejection
commits prediction at its timestamp. Same-time gyro/wheel/GPS events each run
once, and dt=0 adds no process noise.

During `[20,40)` s, there simply are no GPS events in the dropout file. Wheel
and gyro still update v and omega+b. Pose and its uncertainty evolve through
prediction; the filter never reads the outage interval. In dead reckoning,
latest wheel/gyro values are held for future intervals and bias remains the
initial guess. This is why the constant simulated gyro bias accumulates there.

A confident but wrong pose has a small S and potentially large residual.
Even a good returning fix may exceed the gate. M1 deliberately provides no
automatic reset or covariance inflation. In this development run, the gate
instead discarded one noisy fix at 12.6 s and yielded worse whole-run RMSE.
Read its row in `results/first-demo/nominal-gated-gps.csv` and compare the
pre-outage bias estimates described in validation. Do not tune it away.

## What Ryan should verify personally

Review `predict`, `update_gps`, `update_scalar`, and `Replay::process`; derive
the Jacobians and hand update; explain why the truth generator and evaluator
are independent. Reproduce the results and the process-noise mutation failure.
The current evidence supports a narrow statement about an implemented and
tested simulation experiment. Before turning it into a resume claim, establish
personal understanding and reproducibility, then evaluate reserved seeds and
bad-GPS scenarios with predefined metrics. This document does not assert that
Ryan personally authored or already understands every generated component.
