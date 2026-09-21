# NavTrace M1 model

The state is `[px, py, yaw, v, omega, b]` (indices 0–5). Position is in a
local Cartesian world frame, meters. Yaw is counterclockwise from world +x,
in radians, stored in `[-pi, pi)`. Signed forward speed is body-frame m/s;
omega and additive gyroscope bias b are rad/s. Flat terrain, calibrated,
co-located sensors, and no lateral velocity are assumed. GPS has already
been projected; wheel input is speed; gyro input is angular rate, not heading.
No accelerometer, 3D attitude, or latitude/longitude conversion is implemented.

## Discretization (decided before filter implementation)

Use explicit Euler / Euler–Maruyama on steps h <= 0.01 s by default:

```
px' = px + h v cos(yaw)        py' = py + h v sin(yaw)
yaw' = wrap(yaw + h omega)    v' = v; omega' = omega; b' = b
```

The discrete Jacobian F is identity with these extra entries:
`F(0,2)=-h v sin(yaw)`, `F(0,3)=h cos(yaw)`,
`F(1,2)=h v cos(yaw)`, `F(1,3)=h sin(yaw)`, `F(2,4)=h`.
All are evaluated at the start of each substep. Covariance advances as
`P' = F P F^T + Qh`, using the Jacobian of exactly this discrete mean map.
This is a first-order approximation, not exact circular-arc propagation.
Substeps bound its error; independent analytical arc tests check it.

Independent continuous Brownian disturbances drive v, omega, and b:
`dv=sqrt(qv)dWv`, `domega=sqrt(qomega)dWomega`, `db=sqrt(qb)dWb`.
Intensities qv have units m²/s³; qomega and qb have rad²/s³.
`Qh=diag(0,0,0,qv*h,qomega*h,qb*h)` is the first-order diffusion
discretization consistent with Euler–Maruyama. It omits higher-order
within-step integrated noise terms. Position/heading uncertainty and
cross-covariances develop through subsequent F multiplications. Reducing
h changes this approximation; no time-independent per-event Q is added.
Zero dt is a no-op. Negative/nonfinite dt and gaps above 2 s are errors;
the whole prediction is transactional. Large gaps require an explicit
configuration choice (bounded at 60 s), not silent timestamp clamping.

## Observations and updates

GPS: `h(s)=[px,py]`, H selects columns 0,1. Wheel: `h(s)=v`, H selects
column 3. Gyro: `h(s)=omega+b`, H has ones in columns 4,5. The same
observations are not prediction controls for the EKF.

`r=z-Hs`, `S=HPH^T+R`; LLT solves compute K and `NIS=r^T solve(S,r)`.
No explicit inverse is used. After correction use Joseph form
`P=(I-KH)P(I-KH)^T+KRK^T` and wrap yaw. Only roundoff symmetrization
`(P+P^T)/2` is allowed. Covariance checks allow eigenvalues down to
`-1e-10*max(1,max(abs(P)))` for numerical roundoff, and analogous symmetry
tolerance; materially invalid covariance is an error, never clipped.
Measurement R must be finite, symmetric, strictly positive definite.
Failed operations preserve state/covariance; replay also preserves time.

The optional GPS gate rejects NIS > 9.210340371976184 by default. This is
**squared** Mahalanobis distance for a 2-dimensional observation. Under
an unbiased Gaussian innovation with correct S it is a nominal 99% chi-square
threshold (2 degrees of freedom), not an empirical detection rate. Rejection
keeps the prediction and its covariance, with no measurement correction.
There are no resets, inflation, or automatic reacquisition mechanisms.

## Prior and identifiability

The demo assumes the initial pose is known approximately: mean (0,0,0),
position standard deviation 0.2 m and heading 5 degrees. All methods start
from the same mean including speed 1.5 m/s, omega=0 and bias=0. EKF initial
standard deviations for speed, omega, bias are 0.5 m/s, 0.1 rad/s, 0.03 rad/s.
The file prior is diagonal, while the public C++ API accepts full covariance.
The generator's actual bias is not passed to any estimator.

A gyro alone observes omega+b, not the two separately. Moving position
observations coupled with the nonholonomic speed model provide information
about heading and turning, which can help distinguish bias. Stationary
position data do not provide reliable heading or bias identification.
Reverse motion, low speed, weak excitation, wheel slip, and outages can
make this distinction poor. A numerical estimate is not observability proof.
The simple model permits overconfidence; a confident wrong estimate can
reject legitimate GPS after drift. M1 preserves this possible failure.

## Replay and baseline

Event time is signed integer microseconds relative to a common t=0. Input
must already be chronological, with gyro, wheel, GPS ordering at equal
timestamps. Each sensor can appear at most once per timestamp. Invalid
events do not advance time. Statistically rejected GPS does advance time.
No future interpolation or delayed-data sorting occurs in C++.

Dead reckoning uses causal zero-order holds of the latest wheel and gyro
measurements (subtracting only the fixed prior bias, zero in the demo),
integrating with the same bounded Euler step. A sample affects subsequent
motion, never preceding motion. It ignores GPS, retains no covariance, and
does not estimate bias. It receives exactly the same sensor file and prior.
