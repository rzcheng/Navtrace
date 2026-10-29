# Conditional outage reference protocol

Question: starting from the same estimated distribution at GPS loss and the
same subsequent wheel/gyro observations, how closely does the EKF approximate
the resulting position distribution? This diagnoses approximation **after** a
checkpoint. It does not establish that the checkpoint distribution is correct.

## Frozen inputs and selection

Start from inspected commit `407e95a` on `feat/long-outage-study`; create the
diagnostic branch from that commit, retaining all four unpushed commits.
Use `results/long-outage-study`, originally produced by clean commit `0694ea9`.
Verify its hashes and leave all files unchanged. No production C++, Q/R/prior,
gate, or recovery criterion changes. No truth enters reference inference.

Selection rule, fixed before reference runs: among consumed dropout recordings,
choose the lowest seed whose gated EKF rejected all 51 returning GPS fixes, and
the lowest seed that accepted the first returning fix and met the existing dwell
criterion. Inspection selects **83021** (lockout) and **83001** (non-lockout).
These are two diagnostic examples, not held-out navigation performance evidence.

Take the gated EKF's **1.99 s post-event** full six-state mean/P. It includes
all observations through that timestamp: gyro at 1.99 s, last wheel at 1.95 s,
last GPS at 1.8 s. Neither 2.0 s observation has been used. Condition on every
subsequent wheel/gyro observation, in original timestamp/order, through each
report time. At **22 s include gyro then wheel, stop before returning GPS**;
compare with the stored pre-GPS snapshot. Other reports are post-event at
1.99, 5, 10 and 15 s. Never condition an earlier report on later observations.

## Independent numerical reference

With no GPS, u = [unwrapped yaw, speed, yaw rate, bias] is linear Gaussian:
`u_next = A u + B xi`, `A = I; A[0,2] = h`, `xi ~ N(0,I3)`;
`B[1:4,:] = diag(sqrt([qv,qomega,qbias]*h))`. Wheel H selects u[1]; gyro
H selects u[2]+u[3]. Match the C++ step partition `ceil(dt/max_step)`, using
old yaw/speed for the Euler position increment and end-step diffusion.

Represent initial yaw by a real Gaussian centered on the checkpoint's stored
angle; wrap only for trigonometry/display. This is the same local Gaussian
prior interpretation, wrapped onto the circle, not a claim about a globally
Gaussian heading distribution. The u subsystem is independent of position;
its linear Kalman filtering therefore targets its conditional distribution
without a nonlinear approximation.

For each requested endpoint T, sample u_T from its filtered Gaussian, then
sample the path backward using only filter quantities up to T. This is an
offline path-sampling device for the **filtering marginal at T**. Earlier
reported marginals are sampled separately with their own observation prefix;
they are not smoothed with data after their report time.

Avoid a singular backward covariance subtraction by conditioning process noise.
For filtered `(m,C)` at k, put `D=A C A^T`. Given next state x,

```
Lambda = I + B^T solve(D,B)
E[xi | x] = solve(Lambda, B^T solve(D, x-A m))
Cov(xi | x) = solve(Lambda,I)
u_k = solve(A, x-B xi)
```

These follow by completing the square in the joint Gaussian density of xi and
x. Cholesky/triangular solves sample the three-dimensional noise conditional;
the deterministic yaw relation is then satisfied algebraically. No eigenvalue
clipping, jitter, extra process noise, weighting or resampling is needed, even
when one/all diffusion intensities are zero. Repeated samples are independent
draws of conditional paths, not copies produced by resampling.

Finally sample initial position conditional on sampled u_0 from the **full**
checkpoint Gaussian: gain `P_pu solve(P_uu,I)`, mean `m_p+gain*(u_0-m_u)`,
covariance `P_pp-gain*P_up`. Add `sum(h*v_k*[cos(yaw_k),sin(yaw_k)])`.
This preserves checkpoint position/heading/rate/bias correlations. Memory is
linear in sample count plus the small Gaussian history; do not retain whole paths.

The standard Gaussian filtering/backward-factorization background is described
in Särkkä and Svensson, [Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
The specialized noise-conditional derivation and implementation here are independent;
no estimator/sampling code is imported. Finite output is a numerical reference,
not truth or an exact posterior certificate.

## Validation, budget and convergence

Before recordings: test full correlated initial sampling, an independently
conditioned joint linear Gaussian, an analytical no-observation position
moment, Gaussian log likelihood/normalization, circular statistics, future-data
causality, step timing, and invalid inputs. Stochastic tests use fixed seeds and
six-standard-error bounds derived from Gaussian moments, separate from C++
tolerances. Demonstrate a meaningful failing mutation and restore it.

Counts **512, 2048, 8192**, each repeated with independent reference seeds
**91001–91003**, on both recordings: 18 runs, five endpoint distributions each.
Recording seeds are not reference seeds. SeedSequence keys include recording,
count, reference seed and endpoint; separate streams handle terminal state,
backward innovations and initial position. Pilot: 83001, N=512, seed 20260922,
all five times. Measure wall time/peak RSS and estimate full cost before launch.
Hard study budget: 600 s, peak RSS 512 MiB. No escalation of counts if unstable.
Freeze the pilot-based launch decision and tested code before the full study.

For mean x/y, variance and second moment along each EKF principal direction,
and mass in the nominal 95% 2D EKF ellipse, estimate MC standard error using
16 independent sample batches. Pool the three replicate estimates using equal
weights; pooled SE is sqrt(sum(SE²))/3. Also report actual replicate spread.
Highest-two-count stability requires differences <=4 combined SE (+1e-12 for
roundoff). High-count precision requires 3 pooled SE <=5% of the corresponding
EKF x/y standard deviation for means, <=10% of each moment (floor 1e-12), and
<=0.02 absolute for mass. Mark each metric/time separately; a failed criterion
is inconclusive at this budget, not permission to silently increase samples.
These are predefined numerical-resolution checks, not simultaneous confidence
or navigation-performance claims. Retain all failures and planned run counts.

Report means, marginal covariance eigenvalues/vectors, own-mean directional
variance, second moments around the EKF mean, ellipse mass, circular heading
mean/resultant and bias mean/std. Log exact u-filter likelihoods and observation
counts; weights/ESS/ancestry are not applicable because sampling is unweighted
and independent. Compare the u moments against C++ as a separate agreement check.
Ellipse mass is conditional reference probability, not ground-truth coverage.

Plot all highest-count seed-91001 position samples at selected report times on
equal spatial axes, retaining curved shape and full ranges. Plot mean differences
and directional moments over time, and sample-count/replicate convergence.
Save all samples, comparisons, inputs, hashes, version/commit metadata and actual
cost in fresh folders. No truth overlay is needed. Run CTest and all pytest tests;
sanitizers are required if C++ changes (none planned). Inspect diff and stop after
results, with one evidence-based next action; no estimator fix or recovery policy.
