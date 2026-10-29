# Conditional outage reference results

The two selected cases support **both conditional-mean approximation error and
position covariance/shape error after the checkpoint**. The discrepancy is much
larger in the clean-return lockout case. This is a bounded numerical comparison,
not a production correction or a claim about a universally better estimator.

## Baseline and target

Inspection started on `feat/long-outage-study` at `407e95a`, with a clean tree
and four unpushed follow-up commits. They were retained on the new branch
`feat/conditional-reference`. The protocol was committed as `d0111d1` before
implementation; the full reference ran from clean implementation `44a1d19`.
No push, remote change, production C++ change, or Q/R/prior/gate change was made.

The original [long-outage report](long_outage_results.md), manifest, metrics,
return records and all **409 artifact hashes** were verified locally. They
confirm 120 completed C++ replays, no numerical/output failures, median position
RMSE about 0.205 m nominal and 1.371 m dropout, four gated failures to settle,
three cases rejecting all returning GPS, and worst final error 26.5436 m.
Those failures remain unchanged. This reference does not rerun or improve them.

The [frozen protocol](conditional_reference_protocol.md) chose the lowest seed
rejecting all 51 returning GPS fixes (**83021**) and the lowest seed accepting
the first return and satisfying the existing settling criterion (**83001**).
Both recordings are consumed diagnostic data, not new held-out evidence.

Each reference starts from the gated EKF's full six-state Gaussian at **1.99 s,
after the gyro update**. It includes the last wheel update at 1.95 s and GPS at
1.8 s; neither 2.0 s observation is reused. It conditions on the actual wheel
and gyro observations through each report time. At **22 s**, gyro and wheel
are included, but returning GPS is excluded. The primary comparison is with
the stored C++ pre-GPS snapshot. Observations used after the checkpoint number
362, 962, 1,562 and 2,402 at 5, 10, 15 and 22 s, respectively.

## Independent reference and credibility

During GPS absence, unwrapped yaw, speed, yaw rate and bias form a linear
Gaussian subsystem. Python independently filters those four states, samples
conditional paths backward from each endpoint, and integrates the nonlinear
Euler position increments along each path. Initial position is sampled
conditional on the initial latent states, preserving the checkpoint's full
cross-covariance. Bounded steps, old-state Euler semantics, diffusion intensities,
measurement variances and timestamp ordering match the frozen discrete model.
The derivation and assumptions are in the protocol; the implementation is
[conditional_reference.py](../tools/conditional_reference.py).

Each earlier endpoint uses its own observation prefix. Backward sampling does
not give that endpoint future observations. Inference receives neither truth,
true bias, fault labels nor returning GPS. The baseline audit hashes truth files
for integrity but does not read their values into inference. The checkpoint
Gaussian is an assumption shared with the EKF, not a verified true posterior.

Samples are independent, unweighted conditional paths. There is no resampling,
weight collapse, ancestry duplication, or weight reset; ESS/ancestry fields are
explicitly not applicable. This exploits the linear subsystem rather than
trying to prevent particle collapse by changing the model. Finite samples
remain a numerical reference, not an exact posterior certificate.

The bounded pilot completed five distributions in 0.819 s with 163.625 MiB peak
RSS and estimated a 57.5 s full run. The frozen full matrix used **512, 2,048,
8,192 samples**, reference seeds **91001, 91002, 91003**, both recordings, and
five endpoints: **18 runs / 90 distributions**. All completed in **9.242 s**,
with **192.578 MiB** peak RSS, below the 600 s / 512 MiB limits. There were no
failed or unstarted distributions, and no sample-count escalation.

All **70/70 predefined stability and precision checks passed**; zero comparisons
were inconclusive at the specified resolution. The worst stability discrepancy
used 0.522 of its permitted limit; worst precision used 0.769. These checks
compare the two largest counts using batch Monte Carlo standard errors (MCSE),
and limit uncertainty at the largest count. They do not prove tail convergence
or constitute simultaneous confidence guarantees. Each run and replicate range
is retained in [convergence.json](../results/conditional-reference-study/convergence.json).

## Position distributions

Tables pool the three largest-count replicate estimates with equal weights.
Covariances are averages of the three own-mean sample covariances; they do not
include between-replicate mean variation. MCSE uses 16 independent batches per
run, pooled as `sqrt(sum(SE^2))/3`. Means are in meters.

| Recording | Time (s) | EKF mean (x, y) | Reference mean (x, y) | Mean difference norm (m) | Mass inside EKF ellipse |
|---|---:|---|---|---:|---:|
| 83001, non-lockout | 1.99 | (2.9523, 0.1024) | (2.9540, 0.1017) | 0.0018 | 95.28% |
| 83001 | 5 | (6.0154, 1.0451) | (5.9937, 1.0384) | 0.0227 | 94.30% |
| 83001 | 10 | (6.2945, 2.7974) | (6.2677, 2.7311) | 0.0715 | 92.83% |
| 83001 | 15 | (5.4128, 4.0776) | (5.4192, 3.9442) | 0.1336 | 91.63% |
| 83001 | 22, pre-GPS | (5.5492, 4.8648) | (5.5710, 4.6000) | **0.2657** | **84.92%** |
| 83021, lockout | 1.99 | (2.6513, 0.0145) | (2.6513, 0.0147) | 0.0002 | 94.87% |
| 83021 | 5 | (9.3585, -0.0486) | (9.3057, -0.0397) | 0.0535 | 91.12% |
| 83021 | 10 | (17.1738, -1.6357) | (16.9476, -1.6152) | 0.2272 | 74.40% |
| 83021 | 15 | (17.2386, -0.4764) | (17.0169, -0.4982) | 0.2228 | 75.83% |
| 83021 | 22, pre-GPS | (23.4933, -2.1712) | (22.4265, -1.9193) | **1.0961** | **45.85%** |

The ellipse uses the **marginal 2x2 position covariance**, with squared
Mahalanobis threshold 5.99146 (nominal 95% under a 2D Gaussian). Its sample mass
is a conditional-distribution diagnostic, not frequentist coverage across
independent navigation trials. At 22 s the x/y mean MCSE is (0.00972, 0.00333) m
for 83001 and (0.00930, 0.03407) m for 83021. Ellipse-mass MCSE is 0.00249 and
0.00277 respectively (0.249 and 0.277 percentage points).

The following are 22 s marginal covariance eigenvalues and principal directions.
Directions are degrees counterclockwise from world +x, modulo 180 degrees;
eigenvector signs have no geometric meaning. All variances are in m².

| Recording | Distribution | Narrow / wide eigenvalue | Narrow / wide direction |
|---|---|---|---|
| 83001 | EKF | 0.11520 / 2.15090 | 80.66° / 170.66° |
| 83001 | Reference | 0.22397 / 1.89398 | 77.99° / 167.99° |
| 83021 | EKF | 0.04327 / 35.01675 | 170.02° / 80.02° |
| 83021 | Reference | 2.07208 / 28.98361 | 171.22° / 81.22° |

Projecting onto the **EKF's** principal directions separates its reported
variance from reference variance and mean displacement:

| Recording | Direction | EKF variance | Reference own-mean variance (ratio to EKF) | Second moment about EKF mean (ratio) |
|---|---|---:|---:|---:|
| 83001 | Narrow | 0.11520 | 0.22760 (1.98×) | 0.29403 (2.55×) |
| 83001 | Wide | 2.15090 | 1.89035 (0.879×) | 1.89470 (0.881×) |
| 83021 | Narrow | 0.04327 | 2.08394 (**48.16×**) | 3.28127 (**75.83×**) |
| 83021 | Wide | 35.01675 | 28.97175 (0.827×) | 28.97579 (0.827×) |

The narrow own-mean variance spans 0.21979–0.23289 m² across the three 83001
replicates (pooled MCSE 0.00273 m²), and 2.05197–2.12359 m² for 83021
(MCSE 0.03741 m²). The large lockout mismatch is not explained by Monte Carlo
variation or just by a displaced mean. Second moments also include displacement
and are deliberately not labeled covariance. Raw eigensystems, directional
moments, mean differences, heading/bias summaries and MCSE are saved at **every**
endpoint/count/seed in [comparisons.json](../results/conditional-reference-study/comparisons.json).

The [position sample plot](../results/conditional-reference-study/position-shape.png)
shows pronounced curvature in the lockout case. Its covariance ellipse cannot
fully represent that shape. The non-lockout case has smaller discrepancies,
but accepting GPS later does not establish that its outage uncertainty was
correct. No claim about a fix's subsequent recovery performance follows here.

## Heading/bias and agreement

The independently filtered four-state means and covariances agree with the saved
C++ values within **7.66e-15** at the report times. This localizes the disagreement
to the nonlinear position approximation under the shared starting Gaussian;
it does not validate that Gaussian against truth. At 22 s:

| Recording | EKF yaw (rad) | Reference circular mean range (rad) | Reference resultant range | EKF bias mean / std (rad/s) | Reference bias mean range (rad/s) |
|---|---:|---|---|---|---|
| 83001 | -1.40692 | -1.42358 to -1.40391 | 0.81943–0.82009 | 0.000103 / 0.02855 | -0.000096 to 0.000831 |
| 83021 | 2.12779 | 2.11356 to 2.12948 | 0.82167–0.82446 | -0.002425 / 0.02853 | -0.002557 to -0.001817 |

Ranges are variation across the three N=8192 reference runs. The shared latent
Gaussian implies yaw resultants 0.82166/0.82161; sampled bias standard deviations
range 0.02867–0.02877 / 0.02836–0.02853 rad/s. These finite-sample summaries do
not establish reliable heading or bias observability, nor reveal true bias.

## Validation, reproducibility and stop

**23/23 C++ tests and 71/71 Python tests passed**, including 15 new independent
reference/analysis tests. No skips; 260 existing matplotlib/pyparsing deprecation
warnings. Runtime warnings were treated as errors. A temporary mutation removing
checkpoint position/latent correlations made the correlated sampling regression
fail; restoration passed. A local NumPy thin-matrix multiplication warning was
isolated and avoided with equivalent direct contraction. No production defect
was found or silently fixed. See [validation](conditional_reference_validation.md).
No C++ changed, so sanitizers were not rerun. No hosted CI or Linux run is claimed.

Host: macOS 15.7 arm64, Python 3.11.9, NumPy 2.2.6, SciPy 1.15.3,
matplotlib 3.10.3, pytest 8.3.5. Existing Debug C++17 build: AppleClang
17.0.0.17000604, CMake 3.31.6, Eigen 3.4.0, GoogleTest 1.17.0.

```bash
.venv/bin/ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest -q -W error::RuntimeWarning
.venv/bin/python -W error::RuntimeWarning tools/run_conditional_reference.py \
  --pilot --output results/reference-pilot-new
.venv/bin/python -W error::RuntimeWarning tools/run_conditional_reference.py \
  --pilot-results results/reference-pilot-new --output results/reference-study-new
```

Use unused output paths. The runner requires the original hashed baseline at
`results/long-outage-study` (or `--study PATH`); reconstruction instructions are
in the [README](../README.md#conditional-outage-reference). Actual output is in
`results/conditional-reference-study`, with about 16.44 MB retained:

- [Manifest](../results/conditional-reference-study/manifest.json): source/input/output hashes, clean commit, commands, versions, budgets and counts.
- [Samples and ellipse](../results/conditional-reference-study/position-shape.png), [differences over time](../results/conditional-reference-study/distribution-over-time.png), [convergence](../results/conditional-reference-study/convergence.png).
- Per-case folders: full checkpoint, exact observation prefix, C++ snapshots, latent agreement and all 90 compressed sample arrays.
- [Post-run audit](../results/conditional-reference-validation/artifact-review.json): 108 reference artifact hashes, 63 frozen source hashes and all 409 unchanged baseline hashes verified, plus both 2,402-observation prefixes.

All plots were inspected; generated artifacts remain outside Git. Personal
learning/study notes remain outside the repository. Commits add the protocol,
independent reference, comparison/plot runner, tests and technical reports only.

The chosen resolution supports mean and covariance/shape approximation error
in these two conditional cases. It does not establish correct initial uncertainty,
rare-tail accuracy, broader performance, an appropriate recovery policy, or that
a Gaussian correction would capture the curved distribution. **Stop for review.**
The next proposed task is to derive and test an **offline position-moment
correction against these frozen conditional distributions**, with the production
estimator and gate still fixed, before considering a production change.
