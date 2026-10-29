# Conditional trajectory mixture design

This is the design-only review approved by Ryan on 2026-10-18 in
[issue #5](https://github.com/rzcheng/Navtrace/issues/5). It starts from commit
`53be2f72d15f8d485640eaa09868210231b3e886` on the isolated
`codex/trajectory-mixture-design` branch. The approval permits this note and
independent mathematical review. It supersedes the previous stop boundary only
for that work; it does not authorize implementation or numerical execution.

**Recommendation: go only to proposing a separately approved, bounded synthetic
prototype of this one variant.** The conditional representation is coherent for
the existing GPS-free discrete model and retains the shared-heading dependence
in the analytical example. Finite-component accuracy and prior-proposal weight
collapse remain unresolved. No prototype, estimator adoption, recording
comparison, or recovery policy follows automatically from this recommendation.

## Evidence and scope

The [prediction closure diagnostic](prediction_closure.md) established a
prediction-only covariance-loss mechanism in the stopped Gaussian recursion:
projection can erase the nonlinear relation between accumulated displacement
and a shared heading. The [repaired comparison](position_moment_repaired_results.md)
retains unfavorable covariance and ellipse-mass results. Neither result proves
that the proposed representation fixes those failures, and this note attributes
no part of recording 83021's error to a particular mechanism.

The model and timing here follow [model.md](model.md), the
[conditional reference protocol](conditional_reference_protocol.md), and the
read-only source inspection of [position_moments.py](../tools/position_moments.py)
and [conditional_reference.py](../tools/conditional_reference.py). The completed
[M3](m3_results.md), [long-outage](long_outage_results.md), and
[conditional-reference](conditional_reference_results.md) results remain in
force, including their limits and consumed-data labels. No saved sample arrays
are used for this design.

Only one variant is considered: a fixed symbolic number M of independent
prior-proposed angular paths, conditional Gaussian position and speed, and
likelihood weights. There is no resampling, merging, rejuvenation, adaptive M,
covariance inflation, alternative proposal, or use of reconstructed Gaussian
moments as the next filtering state. M is not selected here. Equations below
specify probability laws and symbolic identities, not executable pseudocode.

## State and conditional law

The physical state order remains

\[
x=(p_x,p_y,\theta,v,\omega,b)^T,
\qquad a=(\theta,\omega,b)^T,
\qquad \ell=(p_x,p_y,v)^T.
\]

Position is in meters, time in seconds, heading in radians counterclockwise
from world +x, and v is signed body-forward speed. Rate and bias are in rad/s.
The angular heading is a real, unwrapped lift centered on the checkpoint's
stored angle. Trigonometric position increments are periodic; wrapping is only
a display operation and must not replace that lift in cross-covariances. This
is the same local Gaussian checkpoint interpretation as the reference, not a
claim of a globally Gaussian circular heading.

Let k index propagation substeps and let Y denote exactly the observations
received so far. Component i represents a persistent angular path
\(A_{0:k}^{(i)}=(a_0^{(i)},\ldots,a_k^{(i)})\), with immutable identity i,
weight \(w_i\geq0\), \(\sum_iw_i=1\), and

\[
\ell_k\mid A_{0:k}^{(i)},Y
\sim\mathcal N(m_i,C_i),\qquad C_i\in\mathbb R^{3\times3}.
\]

The covariance includes position/speed and x/y cross-covariances. A Gaussian
here can be a degenerate Gaussian measure on an affine support. Current angular
values are points within a component, not three more Gaussian coordinates.
The represented joint law is the finite mixture

\[
\widehat\pi(dA,d\ell)=
\sum_{i=1}^{M}w_i\,\delta_{A^{(i)}}(dA)\,
\mathcal N(d\ell;m_i,C_i).
\]

The same component continues after every step and observation. Retaining its
conditional statistics retains the effect of its earlier angles even if the
full path is not kept in working memory. Replacing this mixture by its single
six-state Gaussian moments would remove the defining property of this design.

## Correlated checkpoint and singular support

Permute the full checkpoint mean and covariance into blocks (a, ell):

\[
\mu=\begin{pmatrix}\mu_a\\\mu_\ell\end{pmatrix},\qquad
P=\begin{pmatrix}A&B^T\\B&D\end{pmatrix},
\quad A=P_{aa},\ B=P_{\ell a},\ D=P_{\ell\ell}.
\]

For positive-definite A, the exact Gaussian split is

\[
a_0\sim\mathcal N(\mu_a,A),\qquad
m_0(a_0)=\mu_\ell+BA^{-1}(a_0-\mu_a),\qquad
C_0=D-BA^{-1}B^T.
\]

The inverse notation describes the mathematical map; it is not an instruction
to form a numerical inverse. In the proposed ensemble, each initial angular
component is a draw from this marginal, has conditional mean \(m_0(a_0^{(i)})\),
conditional covariance C0, and initial weight 1/M. There is no independent draw
of ell and no diagonalization of its covariance.

For a singular positive-semidefinite A, let \(A^\dagger\) be its
Moore-Penrose inverse on its range. The support-aware extension requires

\[
a_0-\mu_a\in\operatorname{range}(A),\qquad
B(I-A^\dagger A)=0,\qquad
C_0=D-BA^\dagger B^T\succeq0,
\]

and uses

\[
m_0(a_0)=\mu_\ell+BA^\dagger(a_0-\mu_a).
\]

A symmetric PSD full joint P implies the compatibility condition: a direction
with zero angular variance cannot have nonzero covariance with ell. Equivalently,
for a rank-r factor \(A=LL^T\), the marginal is the law of
\(a_0=\mu_a+L\xi\), \(\xi\sim\mathcal N(0,I_r)\). This law lives only on the
stated support; an off-support angular value is invalid, not an input to a
pseudoinverse formula that makes it valid. C0 may also be singular and need not
be made positive definite. When A is zero, B must be zero, a is deterministic,
and the conditional Gaussian of ell retains D. No angular randomness or extra
noise is added to that limit.

These identities preserve the full initial Gaussian in the continuous mixture.
Writing \(K_0=BA^\dagger\), total covariance gives

\[
\operatorname{Cov}(\ell,a)=K_0A=B,\qquad
\operatorname{Cov}(\ell)=C_0+K_0AK_0^T=D.
\]

A finite ensemble replaces the angular integration by empirical integration;
it does not in general reproduce the checkpoint mean or covariance exactly.
There is no unmentioned moment-matching step to enforce that equality.

The singular formulas are algebraic, not a completed numerical support policy.
A future implementation would need a separately reviewed rank/support decision
and stable conditional-covariance construction, including nearly singular
inputs. Material indefiniteness, incompatible cross-covariance, or unresolved
rank cannot be repaired by clipping eigenvalues, jitter, extra diffusion, or
silently deleting a block. The current note sets no new tolerance.

## Propagation with the existing discrete model

For one positive substep h, define

\[
T_h=\begin{pmatrix}1&h&0\\0&1&0\\0&0&1\end{pmatrix},\qquad
Q_a(h)=\operatorname{diag}(0,q_\omega h,q_bh),
\]

\[
F_i(h)=\begin{pmatrix}
1&0&h\cos\theta_k^{(i)}\\
0&1&h\sin\theta_k^{(i)}\\
0&0&1
\end{pmatrix},\qquad
Q_\ell(h)=\operatorname{diag}(0,0,q_vh).
\]

The angular proposal is exactly the prior transition

\[
a_{k+1}^{(i)}=T_h a_k^{(i)}+\eta_{a,k}^{(i)},
\qquad\eta_{a,k}^{(i)}\sim\mathcal N(0,Q_a(h)),
\]

while the conditional linear transition is

\[
\ell_{k+1}=F_i(h)\ell_k+\eta_{\ell,k},
\qquad\eta_{\ell,k}\sim\mathcal N(0,Q_\ell(h)).
\]

All physical diffusion coordinates are independent across steps and independent
of the checkpoint and observation noise. Measurement noises are independent
across events and sensor channels, and independent of checkpoint and process
noise. The angular transition is autonomous: it depends on a, not ell. Its new
diffusion is independent of conditional ell.
Therefore extending a component by the prior transition leaves its weight
unchanged and gives exactly

\[
m_i^-=F_i(h)m_i,\qquad
C_i^-=F_i(h)C_iF_i(h)^T+Q_\ell(h).
\]

This is conditional Gaussian propagation without a nonlinear Gaussian
projection. Both position increments use the old angle and old speed.
Heading uses the old rate. The new speed, rate, and bias diffusion enter at the
end of the substep; new rate noise affects heading only on later substeps,
and new speed noise affects position only on later substeps. Putting a
post-step angle or speed into the displacement, or adding integrated noise
terms, would change the model. Zero diffusion coordinates remain deterministic.
The singular angular transition has a well-defined measure; no full-dimensional
transition density or inverse of Qa is required for prior-proposal cancellation.

Timestamps remain integer microseconds. A positive interval from one accepted
input timestamp to the next is converted to seconds and partitioned into
\(n=\lceil\Delta t/\text{max_step}\rceil\) equal steps \(h=\Delta t/n\), as in
the existing model. Zero time adds no step or diffusion. Existing time/gap
bounds remain fixed. Events must arrive in their prescribed chronological
order, with gyro then wheel at a shared timestamp; they are never sorted or
interpolated. Reports are restricted to the checkpoint or actual observation
timestamps; a requested report must not insert propagation boundaries and thus
alter the discrete path law. Invalid, delayed, duplicate, or unsupported
sensor input is an error.

## Causal likelihoods and observations

The next observation follows propagation to its timestamp. For both sensors,
measurement variance R is the existing finite, strictly positive variance.
No sensor value is a motion control. Each likelihood below uses the component's
pre-observation state; after it is used, it must not be applied again.

For a gyro observation zg with variance Rg,

\[
r_{g,i}=z_g-(\omega_i+b_i),\qquad
L_{g,i}=\mathcal N(z_g;\omega_i+b_i,R_g).
\]

The weight changes by \(\widetilde w_i=w_iL_{g,i}\), followed by normalization.
Within a fixed angular path, zg is independent of ell, so its conditional m_i
and C_i do not change. Angular path coordinates are not moved by the observation.
There is no angular within-component covariance to add to Rg. Gyro reweighting
can still change the mixture's position, speed, heading, rate, and bias moments
through their association with the paths. Omitting this weight change would
lose those effects.

For a wheel observation zw with variance Rw, let \(H_v=(0,0,1)\). Then

\[
r_{w,i}=z_w-H_vm_i^-,\qquad
S_{w,i}=H_vC_i^-H_v^T+R_w>0,
\qquad L_{w,i}=\mathcal N(z_w;H_vm_i^-,S_{w,i}),
\]

\[
K_i=C_i^-H_v^T/S_{w,i},\qquad
m_i^+=m_i^-+K_ir_{w,i},
\]

\[
C_i^+=(I-K_iH_v)C_i^-(I-K_iH_v)^T+K_iR_wK_i^T.
\]

Here the minus/plus signs mean immediately before/after this wheel observation,
including any earlier same-time gyro likelihood. The weight changes by
\(\widetilde w_i=w_iL_{w,i}\). The predictive likelihood must include conditional
speed variance; using Rw alone or a likelihood evaluated after the update is
wrong. The full conditional gain changes position through its position/speed
covariance, and the full Joseph covariance update preserves those correlations.
Reweighting also changes the angular marginal when wheel predictions differ
between paths. Updating ell without the wheel likelihood would not give the
joint posterior. Deterministic conditional speed is permitted because Rw is
positive; it is not a zero-noise measurement limit.

At a shared timestamp, the gyro likelihood is applied first, then the wheel
likelihood and conditional update, with no intervening propagation. A report
after that timestamp uses both. Only the received prefix after the checkpoint
is used: checkpoint observations are not reused, earlier reports are not revised
using later observations, and returning GPS is outside the candidate's input
contract. Truth, fault labels, actual bias, and an outage schedule are not inputs.

### Why the conditional equations are exact

Let q be the angular prior path law through a given prefix:

\[
q(dA_{0:k})=p(da_0)\prod_{j<k}p(da_{j+1}\mid a_j).
\]

Given a path, the initial conditional law is Gaussian, every ell transition is
linear with independent Gaussian diffusion, and wheel observations are linear
Gaussian. Gyro observations depend only on the fixed path. By induction, the
conditional ell law is Gaussian with the moments above. If Le denotes the
corresponding gyro or integrated wheel predictive likelihood at event e, then

\[
p(dA_{0:k},d\ell_k\mid Y_{1:E})=
\frac{q(dA_{0:k})\prod_{e=1}^{E}L_e(A_{0:k_e})}{Z_E}
\,\mathcal N(d\ell_k;m_k(A),C_k(A)),
\]

where \(k_e\leq k\), \(Z_E\) is the evidence, and each conditional filter uses
only the prefix through e. This is a path-space identity, including deterministic
transition supports, rather than a claim that the current angle alone makes
position conditionally Gaussian. Prior proposals cancel the transition part
of the importance ratio on that support; only observation likelihoods remain.

The integral identity is exact for the specified Gaussian checkpoint and
discrete model. Its finite M self-normalized importance representation is an
approximation. Finite-ensemble means and covariance are not exact posterior
moments or unbiased estimators in general. Under independent prior draws, a
fixed finite prefix, valid support, finite required moments, and positive finite
evidence, the usual law of large numbers for the weighted numerator and
denominator yields consistency as M grows. This gives no finite-M error bound,
no uniform-in-time guarantee, and no evidence that a practical M avoids collapse.
The present work neither evaluates that limit numerically nor selects M.

## Full six-state reconstruction

Use block order \(z=(\ell,a)\). For component i put

\[
\mu_i=\begin{pmatrix}m_i\\a_i\end{pmatrix},\qquad
V_i=\begin{pmatrix}C_i&0\\0&0\end{pmatrix},\qquad
\bar\mu=\sum_iw_i\mu_i.
\]

The exact moments of the represented finite mixture are

\[
P_z=\sum_iw_i\left[V_i+(\mu_i-\bar\mu)(\mu_i-\bar\mu)^T\right].
\]

A fixed permutation from \((p_x,p_y,v,\theta,\omega,b)\) to the original
\((p_x,p_y,\theta,v,\omega,b)\) gives the reported mean and covariance.
There is no sample-covariance divisor adjustment: this expression is the
covariance of the normalized mixture itself, not an unbiased-sample correction.
The block identities are

\[
P_{\ell\ell}=\sum_iw_iC_i+
\sum_iw_i(m_i-\bar m)(m_i-\bar m)^T,
\]

\[
P_{\ell a}=\sum_iw_i(m_i-\bar m)(a_i-\bar a)^T,
\qquad P_{aa}=\sum_iw_i(a_i-\bar a)(a_i-\bar a)^T,
\qquad P_{a\ell}=P_{\ell a}^T.
\]

For example, position/speed covariance contains both terms,

\[
P_{pv}=\sum_iw_i\left[C_{pv,i}+
(m_{p,i}-\bar p)(m_{v,i}-\bar v)\right],
\]

while position with any \(d\in\{\theta,\omega,b\}\) has

\[
P_{pd}=\sum_iw_i(m_{p,i}-\bar p)(a_{d,i}-\bar a_d).
\]

Speed/angular, angular/angular, and x/y correlations follow from the same full
blocks; none may be zeroed merely because a is fixed within one component.
The zero conditional ell/a covariance reflects conditioning on the entire
angular path. The between-component term restores dependence when those paths
are integrated out.

The formulas are symmetric and PSD when every Ci is symmetric PSD and weights
are finite, nonnegative, and normalized. That is an algebraic property, not a
calibration test. Angular covariance rank is at most min(3,M-1), and the full
between-component covariance rank is at most M-1. Concentrated weights may
remove almost all represented angular spread even while each conditional Ci
remains healthy. A finite mixture can miss nonlinear dependence, tail mass, or
narrow directions despite passing PSD and symmetry checks.

A future implementation must validate conditional and reconstructed covariance
and reject material violations. Only the existing roundoff symmetrization
convention is allowed; no eigenvalue clipping or inflation is implied. The
unwrapped angular covariance is a lift-dependent Euclidean summary. Wrapping
each component before forming it gives a different, potentially discontinuous
quantity. A separately labeled circular mean would use the argument of
\(\sum_iw_i e^{i\theta_i}\), and is undefined when that resultant is zero; it
does not replace the unwrapped cross-covariances or certify circular accuracy.

## Shared-heading walkthrough without numerical execution

Use exactly the analytical case in [the completed diagnostic](prediction_closure.md):
initial position zero, deterministic signed speed v, deterministic zero rate and
bias, zero diffusion, no observations, and one shared
\(\theta\sim\mathcal N(0,s)\). Take N equal steps with \(h=T/N\). No fixture
values or additional cases are selected here.

Every proposed component keeps \(a_n^{(i)}=(\theta_i,0,0)^T\) for all n, with
\(w_i=1/M\). Its conditional ell covariance is zero and induction gives

\[
m_{i,n}=\left(nhv\cos\theta_i,\ nhv\sin\theta_i,\ v\right)^T.
\]

Thus its endpoint is \((Tv\cos\theta_i,Tv\sin\theta_i,v)^T\) for every N.
Write \(c_i=\cos\theta_i\), \(s_i=\sin\theta_i\), and use Covw and Varw for
the centered covariance and variance under the same normalized component
weights. At the endpoint,

\[
\bar p=Tv\begin{pmatrix}\bar c\\\bar s\end{pmatrix},\qquad
P_{pp}=T^2v^2
\begin{pmatrix}
\operatorname{Var}_w(c)&\operatorname{Cov}_w(c,s)\\
\operatorname{Cov}_w(s,c)&\operatorname{Var}_w(s)
\end{pmatrix},
\]

\[
P_{p\theta}=Tv\begin{pmatrix}
\operatorname{Cov}_w(c,\theta)\\
\operatorname{Cov}_w(s,\theta)
\end{pmatrix},\qquad
P_{\theta\theta}=\operatorname{Var}_w(\theta).
\]

Speed, rate, and bias have zero covariance with everything in this case.
The angular mean is the finite ensemble's mean, not necessarily zero.
For any steps j and k, the component displacement increments obey

\[
\operatorname{Cov}_w(\Delta p_j,\Delta p_k)
=h^2v^2\operatorname{Cov}_w((c,s)^T).
\]

All N squared pairs contribute to the accumulated covariance, giving the
factor \(N^2h^2=T^2\). The cross-time terms remain because the same angle and
identity persist. There is no independent redraw at each step, and no Gaussian
projection that converts zero position/heading covariance into independence.

Replacing empirical angular integration by the original continuous Gaussian
integral gives the already derived identities

\[
E[\cos\theta]=e^{-s/2},\quad E[\sin\theta]=0,\quad
\operatorname{Var}(\cos\theta)=\frac{(1-e^{-s})^2}{2},\quad
\operatorname{Var}(\sin\theta)=\frac{1-e^{-2s}}{2},
\]

\[
\operatorname{Cov}(\cos\theta,\sin\theta)=0,\quad
\operatorname{Cov}(\cos\theta,\theta)=0,\quad
\operatorname{Cov}(\sin\theta,\theta)=s e^{-s/2},\quad
\operatorname{Var}(\theta)=s.
\]

These recover the exact full moments in the earlier note, including its x
variance proportional to T squared rather than T squared divided by N. The
persistent representation therefore retains the dependence that its Gaussian
projection lost in this example. A finite ensemble need not integrate any of
these identities exactly; spurious odd empirical moments and covariance error
can remain. The algebra establishes partition independence only for fixed
components and weights in this special zero-diffusion, constant-heading case.
It does not make finite-M integration error vanish. Redrawing the ensemble for
each N would add a different integration error and obscure that distinction.

At s=0 the angular support is a point, all components coincide, and the full
covariance is zero without a variance ratio or added noise. General turning,
diffusion, or measurement sequences retain the specified Euler partition and
can depend on it. No general partition invariance, recording-specific
improvement, or covariance-calibration claim follows from this example.

## Weight concentration and numerical failure

For either scalar likelihood, with its appropriate ri and Si,

\[
\log L_i=-\tfrac12\left[\log(2\pi)+\log S_i+r_i^2/S_i\right].
\]

If the previous normalized log weights are \(\lambda_i\), define

\[
u_i=\lambda_i+\log L_i,\qquad c=\max_i u_i,\qquad
\log Z_{\mathrm{inc}}=c+\log\sum_i e^{u_i-c},\qquad
\lambda_i^+=u_i-\log Z_{\mathrm{inc}}.
\]

This log normalization is algebraically equivalent to multiplying and
normalizing weights and avoids direct products of tiny likelihoods. The
increment is the finite-mixture predictive evidence, not an exact likelihood
of the true filtering law; its cumulative sum must be labeled accordingly.
Every finite log weight and its associated angular state, conditional mean,
conditional covariance, and identity must be retained even if exponentiation
for normalization summaries or reported weights produces zero. Log weights
remain authoritative. Dropping or freezing such a component changes the method
and prevents later relative-weight recovery.

With finite valid states and the existing positive sensor variances, every
true component likelihood and every initially positive path weight remains
positive at any finite prefix. Singular transition or conditional covariance
supports do not invalidate this fact. Any retained log weight becoming
nonfinite, a nonfinite normalizer, or an invalid reported weight therefore
indicates invalid or unresolved arithmetic/inference. A finite log weight whose
exponential rounds to zero is the distinct reporting-underflow case above.
Such arithmetic failure does not establish a zero component likelihood under
this model. Extremely separated
finite logs may also underflow during summation or output and lose represented
mass numerically. That limitation must be recorded rather than silently
renormalizing a deliberately truncated ensemble or calling it resolved.

Numerically stable normalization cannot cure statistical collapse. Useful
mandatory diagnostics for any later prototype include

\[
N_{\mathrm{eff}}=\frac{1}{\sum_iw_i^2},\qquad
w_{\max}=\max_iw_i,
\]

together with log-weight range, numerical zero/underflow counts, likelihood
normalizers, observation counts, and component identities. ESS describes
concentration of these importance weights, not a number of independent
posterior draws or an error/coverage certificate. All M original identities can
survive with essentially one effective path; a unique-ancestry count alone
would hide that problem. Conditional Gaussian uncertainty does not restore
missing angular paths.

Prior proposals ignore the new observation when extending the path. Repeated
gyro likelihoods can favor a small subset of rate/bias trajectories, and wheel
likelihoods can further favor paths through checkpoint correlations. Early
poor angular support or a weakly represented rare path cannot be corrected by
moving existing angular coordinates. An initially low-weight path can gain
relative weight later if retained, but there is no guarantee that useful paths
were proposed at all. No resampling or other rescue is available in this variant.

No numeric ESS cutoff, precision target, acceptable underflow level, or horizon
is invented here. Without a separately frozen assessment those are reported
concentration/numerical diagnostics, and finite-M accuracy remains unresolved.
A later prototype must report invalid input/support, arithmetic failure, and
unresolved finite-ensemble inference distinctly, retaining every planned case
and failure. A failed event must not yield a fabricated successful snapshot:
retain the last valid prefix and its timestamp and mark the requested later
result unavailable with the component/event and reason. Do not drop an invalid
component, reset weights, restart at a favorable time, add noise, or substitute
the EKF/reference automatically. Concentration alone is evidence to report,
not permission for a new method or a favorable early stop.

## Ancestry, storage, and symbolic cost

There is no branching or duplicated ancestry because there is no resampling.
Each component keeps one stable identity through every substep. For current
filtering, \((a_i,m_i,C_i,\lambda_i,i)\), the current timestamp and observation
phase, and sufficient proposal-stream state are sufficient. Old angular states
influence current conditional position through m and C. They need not be
revisited to advance the filter, but their actual values cannot in general be
reconstructed from current moments and identity alone.

An audit that needs actual paths must preserve angular substep histories or a
reproducible account of the checkpoint draws, proposal randomness, step
partition and observation prefix. A future protocol must freeze that account
and its hashes rather than assume a seed alone identifies all software and
ordering choices. Immutable prefix snapshots are needed to retain earlier
reported distributions; later weights must not be retroactively substituted.
This design offers no historical smoothing output. Retaining full histories
for audit does not authorize an endpoint-conditioned backward redraw.

Let E be the number of processed gyro/wheel events, S the total positive-time
Euler substeps under the existing partition, and R the number of reports.
With fixed state dimensions, initialization costs O(M) plus constant-size
checkpoint factorizations. Each substep costs O(M) for angular proposal and
conditional mean/covariance propagation. Each event costs O(M) for likelihoods,
normalization, and the optional conditional wheel update. A full six-state
moment report costs O(M). Total algebraic work is therefore

\[
O\bigl(M(1+S+E+R)\bigr).
\]

S is determined by event gaps and the unchanged step bound; it cannot be omitted
when counting event costs. Same-time events add measurement work but no
substeps. These counts describe constant-size matrix algebra and proposed
random draws; no runtime, memory bytes, throughput, or practical M is measured.

Current filtering state uses O(M) storage. Retained scalar diagnostic records
add O(E) archival storage, and six-state moment reports add O(R). Saving the
full component mixture at every report adds O(MR). A full angular archive adds
O(M(S+1));
retaining all conditional substep/event snapshots would add O(M(S+E)) instead
of the current-state memory bound. An archived observation stream uses O(E).
No growing ancestor tree or backward Gaussian filter history is required for
current filtering.

The existing reference filters the four-state linear subsystem
\((\theta,v,\omega,b)\), then draws independent posterior paths backward for
each endpoint's own prefix and integrates position along them. It analytically
uses observations before drawing a path and has no importance weights. This
proposal instead extends the same angular prior paths forward, integrates
speed analytically conditional on each path, and accumulates likelihood weights
as data arrive. It does not rebuild the reference for each report, use its
samples, condition an earlier report on later data, or gain the reference's
absence of weight degeneracy. The two constructions target the same specified
conditional model in their ideal limits, with different finite approximations
and different failure risks.

## Approximation limits and bounded recommendation

The mathematical result is limited but useful: the correlated checkpoint split,
conditional propagation, gyro likelihood, wheel likelihood/update, and mixture
moment identities are exact algebra for the assumed checkpoint and discrete
GPS-free model. Conditioning on a full angular path avoids recursive Gaussian
projection of its relation to position. The fixed finite set of angular paths
and self-normalized weights are the new inference approximation.

The unresolved obligations are:

- Numerical rank/support handling, stable covariance arithmetic, and failure
  reporting have been specified as requirements but have no implementation or
  validation here. Singular algebra does not settle nearly singular numerics.
- No finite M, duration, or event sequence has been shown to give usable weight
  behavior, accurate moments, tails, or calibrated uncertainty. Long sequences
  can defeat this minimal prior-proposal variant. A larger nominal ensemble
  does not by itself certify that its relevant angular support is represented.
- The checkpoint is an assumed local joint Gaussian, not a verified posterior.
  Euler discretization, model mismatch, identifiability limits, and the physical
  interpretation of a broad heading lift remain. Exact conditional calculations
  do not remove them or establish navigation performance.
- The shared-heading result is structural and symbolic. It neither bounds the
  finite mixture's error nor attributes or resolves the measured errors in
  consumed recordings. All prior reference outputs, failed comparisons,
  configurations, tuning, and seed-consumption labels remain unchanged.
- GPS return, gating, recovery, production integration, and any materially
  different proposal or resampling method remain outside the design.

The single recommendation is **go to proposing a separate bounded synthetic
feasibility prototype of this exact variant**, because its conditional algebra
is coherent and it retains the specific temporal dependence symbolically.
Implementation and practical feasibility remain unapproved and unestablished.

That one later proposal would have to freeze synthetic inputs, component count,
randomization and replication, numerical support rules, budget, accuracy and
weight-concentration assessments, and failure/acceptance criteria before any
code or execution. It would cover the correlated and deterministic limits,
causal timing and both sensor likelihoods, the existing shared-heading
mechanism, complete six-state moments, and transparent degeneracy reporting.
It would be an offline GPS-free synthetic feasibility check, with independent
review and no recording/reference sampling or production changes. If acceptable
behavior requires changing the proposal or adding resampling, report that
limitation and stop; such a change needs its own reviewed scope. No fixture,
finite M, numeric threshold, or experimental budget is chosen in this note.

## Work performed and independent review

The design author read the approved issue, repository instructions, linked
specifications and completed reports, and the two existing offline source
modules. The derivations above were checked manually through Gaussian
conditioning, conditional independence, total covariance, and the shared-angle
sum. No executable derivation, numerical calculation, test, fixture, quadrature,
sampling, replay, simulation, benchmark, or build was performed for this note.
No saved sample array was decoded or analyzed. The preservation audit read
artifact bytes only to compare file hashes. Only this design document is authored
here; file reads and non-experimental provenance/diff checks are the permitted
audit.

### Independent review disposition

The independent mathematical reviewer accepted the design on 2026-10-18 after
reading the complete draft and checking the Gaussian checkpoint split,
support conditions, conditional transitions, both likelihoods, full covariance,
and shared-heading calculation against the existing model and source. The
reviewer separately checked the finite-M limits, weight failures, ancestry,
and symbolic costs. The parent also inspected the full note and manually
checked those equations against the C++ prediction/replay rules and both offline
modules. The review establishes symbolic coherence, with no experimental
evidence claimed.

Review clarified that finite log weights and all component states survive
reported exponential underflow; mixture covariance has no sample-divisor
correction; and forward filtering memory differs from a full path archive.
The parent then made the report-time restriction explicit, distinguished any
nonfinite retained log weight from a reported zero weight, included initialization
in the work count, and labeled retained diagnostic storage as archival. The
reviewer also requested explicit independence of measurement noise across events
and channels; that assumption is now stated. These are clarifications of the
same variant, not changes to its model or scope. No mathematical disagreement
or design-review blocker remained. Numerical support handling, finite-M error,
and practical weight behavior remain unresolved obligations, not passed checks.

Both reviews support the single recommendation to propose a separately approved
synthetic feasibility prototype. They do not endorse implementation under this
approval, navigation performance, a new proposal method, or recovery behavior.

### Preservation audit and stop boundary

The parent compared file hashes with the completed diagnostic's saved manifests.
All 87 tracked baseline files matched. All 541 files in the four preserved study
directories matched: 410 long-outage files, 109 conditional-reference files,
and 11 files in each position-moment comparison attempt. Their file lists also
matched. The 12 diagnostic evidence artifacts and 24 development-demo artifacts
matched, as did their saved manifest checks. These are byte-preservation checks,
not new calculations from the recordings or samples.

The original checkout remains clean at `c834bf3`; the integration branch remains
clean at `53be2f7`. The isolated branch adds only this note. The parent inspected
the complete diff and checked whitespace; production, tests, tuning, previous
protocols/results, and saved artifacts are unchanged. No build, CTest, pytest,
demo, sanitizer, hosted CI, experiment, or numerical feasibility check was run
for this design-only milestone. No push, merge, release, or remote change was
made.

Work stops with this reviewed design. The next decision is whether to commission
the separately frozen synthetic prototype proposal outlined above; its finite
component budget, numerical rules, implementation and execution still require
review and approval.
