# M3 diagnostic conventions

`Replay::process` captures x/P after prediction and before GPS correction, then
captures the posterior. Prior simultaneous gyro/wheel updates have already run.
The first ten GPS CSV fields retain their original meanings and values. Extra
columns carry pre/post x/P, z/R, wrapped correction, and timing. These fields
contain no truth, scenario, fault annotation, or scheduled outage.

`since_available_s` and `since_accepted_s` refer to the previous GPS events,
measured immediately before the current decision; missing history is blank.
`rejection_duration_s` is current time minus the first consecutively rejected
valid GPS event (zero at the first rejection). Acceptance clears that history.
An invalid/duplicate event cannot change any timer. A gap during a rejection
streak contributes elapsed time, but absence alone never creates a rejection.
Dead reckoning leaves the extra fields blank rather than inventing a covariance.

`tools/event_diagnostics.py::audit_gps_events` independently reconstructs r, S,
NIS, K, mean correction, and Joseph covariance in Python. It checks rejected
GPS snapshots are exactly identical. The eigendecomposition S=U diag(lambda) Uᵀ
provides signed residual projections and squared contributions
`(Uᵀr)_i²/lambda_i`; eigenvector signs are arbitrary. A high NIS means prediction
and observation disagree, not that the GPS is known to be faulty.

`tools/consistency.py::state_statistics` uses Cholesky solves. Full-state yaw
error is wrapped; position NEES uses P[:2,:2], not a precision-matrix block.
Singular/invalid covariances produce an undefined statistic and explicit validity
flag, with no jitter or pseudoinverse. A valid marginal position block may still
be diagnosed when full-state covariance is singular; dimensions are separate.
Coverage is an indicator for the nominal 95% 2D position ellipse. Time-averaged
coverage is descriptive; ensembles use one value per independent run at a fixed
predeclared timestamp. Missing runs remain in planned denominators. Full-state
NEES is a local diagnostic when heading errors become large or multimodal.

`elapsed_settling` preserves <=1 m for >=2 seconds of elapsed supported samples.
An observation gap over 10 ms breaks the dwell. Regular 100 Hz still requires
201 inclusive samples. It reports the qualifying start, confirmation, available
follow-up, and null delays for non-recovery; acceptance and settling are distinct.
This sampled criterion makes no assertion between unobserved timestamps.

`navtrace_linearize` is a separate offline helper. The replay program never reads
truth. The helper evaluates production F at the previous nominal state. Python
stacks whitened H Phi D using only process F products, physical availability,
and fixed dimensional scales. No P, K, or gated selection enters sensitivity.
Stationary no-slip speed near zero is not an instruction to set true yaw rate zero.

SciPy 1.15.3 supplies established chi-square quantiles only in Python; it is pinned
in the local environment ([release](https://pypi.org/project/scipy/1.15.3/)). No
statistics dependency is added to C++. Expected small-sample variation and EKF
nonlinearity mean even a matched sanity reference is not a certification test.
