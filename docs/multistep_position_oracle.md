# Position moments across steps

[Issue #11](https://github.com/rzcheng/Navtrace/issues/11) is a design-only follow-up
to the completed [observed-path experiment](observed_trajectory_results.md).
There is an analytical route to the full six-state endpoint mean and covariance
under that experiment's Gaussian checkpoint and GPS-free discrete model. It
conditions the physical Gaussian history on the received gyro/wheel observations,
then evaluates Gaussian trigonometric moments. It does not use the candidate's
posterior, conditional endpoint maps or saved mixtures as an oracle.

This note specifies that route and a bounded possible implementation. No oracle
was implemented or executed for this review. Numerical resolution and runtime
remain unmeasured. Issue #10's **defer**, its four inconclusive precision checks,
consumed groups 96001–96004, and the earlier study's **no-go** remain unchanged.
A new oracle would not retroactively change those decisions.

## Model and exactness

Use physical order `(px, py, theta, v, omega, b)`, meters, seconds and radians,
with signed forward speed and counterclockwise heading from world +x. Here
`theta` is an **unwrapped lift** of heading, as in the frozen synthetic study.
Trigonometric displacement is invariant to wrapping; a full six-state mean and
covariance with this lifted yaw is not the mean/covariance of wrapped yaw in the
production estimator.

For positive substep lengths h_j, j=0,...,N-1, the physical model is

```
p_(j+1)     = p_j + h_j v_j [cos(theta_j), sin(theta_j)]
theta_(j+1) = theta_j + h_j omega_j
v_(j+1)     = v_j + eta_v,j
omega_(j+1) = omega_j + eta_omega,j
b_(j+1)     = b_j + eta_b,j
```

The independent end-step increments have covariance
`h_j diag(q_v,q_omega,q_b)`. All checkpoint correlations are retained. This is
exactly the stated Euler–Maruyama model, including its omission of within-step
integrated diffusion. It is not an exact continuous-motion model. A new rate or
speed increment cannot affect that same step's displacement, and a new rate
increment cannot affect that step's yaw increment.

The checkpoint is a stipulated joint Gaussian. Gyro observes `omega+b`, wheel
observes `v`, and independent sensor errors have the supplied strictly positive
variances. There are no GPS likelihoods, gate-selection events, truth labels or
nonlinear observations. Gaussian conditioning therefore remains exact for the
linear physical variables. The evolved position is generally non-Gaussian.

The equations below give exact first and second moments in real arithmetic
under these assumptions. They do not establish an exact checkpoint posterior
from earlier navigation history, a Gaussian position distribution, ellipse
coverage, calibration, or a better estimator. They do not authorize a correction
or a recovery policy.

## Causal schedule and independent physical history

Each call starts from the same fixed checkpoint and receives only an already
received event prefix. Use integer microsecond timestamps. For each positive
inter-event gap dt, use the unchanged `ceil(dt/max_step)` equal subdivision.
Record the actual substep sizes and event-to-step indices. Do not introduce a
report boundary, change the partition, sort delayed inputs or reuse candidate
schedule-building code. Reports are at the checkpoint for the empty prefix or
at the final received event.

Reject checkpoint-time events, duplicates, delayed/reordered events and
unsupported sensors. At equal timestamps gyro precedes wheel, with no additional
propagation. A gyro-only prefix cannot see the following wheel. A later call can
condition on that now-received wheel and revise its own inferred earlier latent
history; an earlier returned report stays immutable. Do not pass the posterior
from one prefix as the next prefix's checkpoint or apply any observation twice.

Let `t_j = sum_(r<j) h_r`. Define a physical Gaussian vector

```
g = (x_0, eta_0, ..., eta_(N-1)),       dimension D = 6 + 3N
m_g = (m_0, 0, ..., 0)
P_g = blockdiag(P_0, Q_0, ..., Q_(N-1))
Q_j = h_j diag(q_v,q_omega,q_b)
```

The checkpoint has all six physical coordinates; zero process coordinates can
remain as exactly zero rows. The following are linear functions of g:

```
v_j     = v_0     + sum_(k<j) eta_v,k
omega_j = omega_0 + sum_(k<j) eta_omega,k
b_j     = b_0     + sum_(k<j) eta_b,k
theta_j = theta_0 + t_j omega_0
          + sum_(k<j) (t_j - t_(k+1)) eta_omega,k
```

In the last sum the k=j-1 coefficient is zero. These expressions, and direct
physical selectors for p_0, are the independent construction. At an observation
boundary n, the gyro row is the sum of the omega_n and b_n rows; a wheel row is
the v_n row. Stack these rows into H and values into y, with diagonal sensor R.
This construction does not import the candidate's angular/linear split,
posterior factors, path maps, conditional moments or sampled coordinates.

## Conditioning and declared support

Mathematically, with `S = H P_g H^T + R`,

```
K   = P_g H^T solve(S, I)
m_c = m_g + K (y - H m_g)
P_c = (I - K H) P_g (I - K H)^T + K R K^T
```

The displayed identity for K is notation; implement Cholesky solves on S with
the cross-covariance right-hand side, never form S inverse. Positive R makes S
positive definite in exact arithmetic even when the physical Gaussian support
is singular. Empty H leaves the prior unchanged. Return log evidence as an explicit
diagnostic: with residual r=y-H m_g and S factor L_S, it is
`-0.5 [E log(2pi)+2 sum(log(diag(L_S)))+r^T solve(S,r)]`, or zero for
the empty prefix. This uses the same conditioning calculation; it must not
condition the moments a second time.

For the proposed implementation, require a declared finite checkpoint factor
`B_0` of shape 6-by-r, 0<=r<=6, together with m_0 and redundant full P_0. In exact
arithmetic `P_0 = B_0 B_0^T`. This admits declared rotated singular support and
the deterministic checkpoint without inferring numerical rank. An arbitrary
PSD covariance without a declared factor is outside this interface. Never find
its support by an eigenvalue cutoff, pseudoinverse, clipping or added jitter.

Construct L from B_0 and one process column `sqrt(q h)` for each strictly positive
physical diffusion term. Omit exactly zero columns only; never discard a tiny
positive term. Then `P_g = L L^T`. After solving for K, retain the Joseph factor

```
L_c = [ (I - K H) L, K sqrt(R) ] .
```

No factorization of the singular physical prior or posterior is required.
Extract all conditional covariances as dot products of physical-map rows times
L_c. In particular,

```
Var(theta_j + s theta_k) = ||(a_theta,j + s a_theta,k) L_c||^2
```

for s=+1 or -1 and physical selector rows a. This avoids subtracting nearly equal
variances to get a negative phase variance. Keep full correlations after
conditioning: formerly independent process terms are generally dependent then.
Also condition p_0 through its correlations. Keeping its prior mean/covariance
in the endpoint assembly would be wrong.

For the published O checkpoint, fix B_0 directly from the **prior input**
factors J, L_a and K_0 in [trajectory_mixture_proposal.md](trajectory_mixture_proposal.md),
section O. With angular row indices a=(2,4,5), linear indices ell=(0,1,3), and
three angular columns followed by three conditional linear columns,

```
B_0[a,   :3] = L_a       B_0[a,   3:] = 0
B_0[ell, :3] = K_0 L_a   B_0[ell, 3:] = J
```

P_0 is independently assembled from the published covariance blocks, not copied
from a candidate posterior. These prior factors are part of the frozen input
law; their use does not reuse candidate inference. Record both constructions,
provenance and the actual residual `B_0 B_0^T - P_0`. Require its maximum absolute
entry <= `1e-10 max(1,max(abs(P_0)))`. This consistency check is not proof of
relative accuracy for a tiny variance. The floating computation uses the
factor-defined law; claims cannot resolve differences below the input/factor
discrepancy. Even the empty-prefix covariance follows this factor-defined law;
do not silently return supplied P_0 there and reconstructed covariance elsewhere.
No free choice of a replacement O factor or changed prior is allowed.

## Gaussian identities and displacement means

In the remaining equations every mean and covariance refers to g conditioned
on this call's complete received prefix. For jointly Gaussian real A, B and T,
write `phi_T = exp(i mu_T - Var(T)/2)`. Then

```
E[A exp(iT)]   = (mu_A + i Cov(A,T)) phi_T
E[A B exp(iT)] = [Cov(A,B)
                  + (mu_A+i Cov(A,T))(mu_B+i Cov(B,T))] phi_T .
```

One independent derivation starts from
`E[exp(u A + w B + iT)]`. Its logarithm is the Gaussian quadratic form in
(u,w,i), including every cross term. Differentiating once in u, or once in
both u and w, at u=w=0 gives these identities. It requires no motion-map
linearization, Gaussian projection of position, or independent-step assumption.

Define complex displacement rate `z_j = v_j exp(i theta_j)` and its mean

```
m_j = (mu_vj + i Cov(v_j,theta_j))
      exp(i mu_thetaj - Var(theta_j)/2) .
```

The real displacement rate is `d_j = (Re z_j, Im z_j)^T`.

## Pair moments, including different steps

For s=+1 or -1 let `T = theta_j + s theta_k`. The Gaussian identity gives

```
H_jk^s = [Cov(v_j,v_k)
          + (mu_vj+i Cov(v_j,T))(mu_vk+i Cov(v_k,T))]
         exp(i mu_T - Var(T)/2) .
```

Here `H^+ = E[z_j z_k]`, whereas `H^- = E[z_j conjugate(z_k)]`. Center them as

```
C_jk^+ = H_jk^+ - m_j m_k
C_jk^- = H_jk^- - m_j conjugate(m_k) .
```

The real covariance block, with rows for d_j and columns for d_k, is

```
Cov(d_j,d_k) = 1/2 [ Re(C^+ + C^-),  Im(C^+ - C^-)
                    Im(C^+ + C^-),  Re(C^- - C^+) ] .
```

Check `C_kj^+ = C_jk^+`, `C_kj^- = conjugate(C_jk^-)` and the transpose relation
between real blocks. In a literal sign check with deterministic theta_j=0 and
theta_k=pi/2, correlated speeds give `C^+=i Cov(v_j,v_k)` and
`C^-=-i Cov(v_j,v_k)`. The real block must be
`[[0,Cov(v_j,v_k)],[0,0]]`, not its transpose.

For j=k the minus phase vanishes and `H_jj^- = mu_vj^2 + Var(v_j)`.
The plus expression is
`[Var(v_j)+(mu_vj+2i Cov(v_j,theta_j))^2] exp(2i mu_thetaj-2 Var(theta_j))`.
These identities check the diagonal case but cannot replace the j!=k terms.
For one constant shared random heading and deterministic signed speeds, the
displacement covariance scales with A squared, where A=sum_j h_j v_j, rather
than with sum_j (h_j v_j)^2. Turning and correlated random speed require the
general pair terms; they do not obey that simpler scaling rule.

### Centered evaluation with less cancellation

The raw centered definitions above establish the mathematics. Prefer this
algebraically equivalent evaluation to subtracting two large raw moments:

```
a   = mu_vj + i Cov(v_j,theta_j)
b_s = mu_vk + i s Cov(v_k,theta_k)
d   = Cov(v_j,theta_k)       e = Cov(v_k,theta_j)
A_s = -Var(theta_j+s theta_k)/2
B   = -(Var(theta_j)+Var(theta_k))/2

C_jk^s = exp(i(mu_thetaj+s mu_thetak)) * {
           [exp(A_s)-exp(B)] a b_s
           + exp(A_s) [Cov(v_j,v_k)+i e a+i s d b_s-s d e]
         } .
```

Both damping exponents are nonpositive. Evaluate their difference by
`exp(A_s)*[-expm1(B-A_s)]` when A_s>=B, otherwise
`exp(B)*expm1(A_s-B)`. Equality gives zero. Do not form a separate exponential
of a potentially large cross-covariance and multiply it by an underflowed
attenuation. The bracket follows by expanding `(a+i s d)(b_s+i e)`.

This reduces one source of cancellation. It does not prevent cancellation
between bracket terms, between C^+ and C^- in a narrow real variance, or between
initial-position cross terms and displacement terms. Neither this rewrite nor
agreement with the raw expression certifies floating-point accuracy.

## Position and final latent cross-covariance

For a real Gaussian vector X, which can be p_0 or the final latent vector
`u_N = (theta_N,v_N,omega_N,b_N)`, define the **unconjugated** complex cross moment
`b_Xj = E[(X-mu_X)(z_j-m_j)]`, a column vector. The notation below uses
Cov(X,z_j) for this definition; a convention that conjugates the second
argument would give the wrong imaginary sign.

```
b_Xj = Cov(X,z_j)
     = [Cov(X,v_j)
        + i Cov(X,theta_j)(mu_vj+i Cov(v_j,theta_j))]
       exp(i mu_thetaj-Var(theta_j)/2) .
```

Thus `Cov(X,d_j) = [Re b_Xj, Im b_Xj]`. Subtracting the product of means in the
Gaussian identity cancels all mu_X terms; it does not remove the X/theta or
X/speed correlations. Assemble the endpoint as

```
mu_pN = mu_p0 + sum_j h_j [Re m_j, Im m_j]^T

P_pp = P_p0p0 + sum_j h_j [Cov(p_0,d_j)+Cov(p_0,d_j)^T]
       + sum_(j,k) h_j h_k Cov(d_j,d_k)

P_pu = P_p0uN + sum_j h_j Cov(u_N,d_j)^T
P_uu = Cov(u_N,u_N) .
```

The final mean is `(mu_pN,mu_uN)` and covariance is the full block matrix with
P_pp, P_pu, its transpose and P_uu. Retain every j=k diagonal term. For
the remaining j<k terms use
`Cov(d_j,d_k)+Cov(d_j,d_k)^T`, not twice a generally nonsymmetric cross block.
No covariance correlation is dropped. Deterministic zero speed with no speed
diffusion gives zero displacement; zero **mean** speed alone does not.

## Numerical contract and unresolved results

Use float64 and complex128 with the existing local NumPy dependency. Validate
finite inputs, shapes, nonnegative diffusion, positive sensor variances,
integer times, chronological sensor order and the fixed schedule bounds. Check
input dimensions and event/substep limits before dense allocation or unbounded
array coercion; consume an event iterable only up to its fixed limit. A
positive process scale that underflows to zero is unresolved, not zero support.
Arithmetic overflow, invalid operations, nonfinite outputs or materially
invalid covariance are failures/refusals with saved diagnostics, not repairs.

Apply the unchanged `1e-10 max(1,max(abs(P)))` symmetry/PSD tolerance. Record
asymmetry before any allowed roundoff symmetrization. A small negative eigenvalue
within that tolerance is a roundoff diagnostic, not a reason to clip it or a
proof that its direction is resolved. S is the only SPD matrix requiring a
solve; use the existing resolution test
`lambda_min > 1e-12 max(1,lambda_max)` and Cholesky. A failed resolution check is
unresolved. Do not apply an SPD rank threshold to the singular physical history
or require the endpoint to be invertible merely to return its moments.

Record factor/P_0 discrepancy, S spectrum and solve residuals, phase variances,
underflow occurrences, pair symmetry residuals, endpoint symmetry/eigenvalues
and absolute sums of the terms entering each covariance entry. Use compensated
or pairwise real/imaginary summation in the same fixed order. Relative to a
small final entry, large sums expose cancellation; they do not by themselves
bound the error. Damping underflow is recorded and cannot establish exact zero
moments or an accurate narrow variance.

For any requested directional variance, retain the direct projected-term sum
as well as the projection of the assembled matrix. Residual checks, two forms
agreeing, and positive covariance are diagnostics only. A certified error bound
would also have to cover input representation, conditioning, dot products,
trigonometric/exponential evaluation and summation. This design does not claim
such a bound or introduce arbitrary precision/interval dependencies.

A future prototype can report completed moments and diagnostics without claiming
certified resolution in every direction. Its status must separate invalid
input, arithmetic/covariance failure, unresolved solve/support/precision, and
completed fixture checks. In particular, a small variance accepted by the
absolute PSD tolerance must not automatically become a whitening denominator.
Eligibility for any new comparison, its required precision and any unresolved
position direction need a later reviewed decision. No fallback eigenvalue
floor, covariance inflation, tolerance weakening or alternative candidate is
part of this design.

## Independent deterministic validation proposed for later approval

The following tests are proposed, not executed. Expected values must come from
these identities or separately written small physical constructions, never the
new builder's own intermediate values or candidate outputs. General conditioning
fixtures use at most eight substeps and six events; one explicitly reducible
100-step control below exercises the proposed dimension bound without sampling.
Freeze test values and acceptance before execution. Do not select them from the
consumed study outputs.

| check | fixed construction and independent expectation |
|---|---|
| Empty prefix | Published O m_0, P_0 and declared B_0, no events. Return factor-defined checkpoint moments, with the P_0 residual retained; log evidence zero. Also a zero-column deterministic factor. |
| One step | Only the two published 10000 us O events, q unchanged. Compare full six-state moments and evidence with the frozen independently reviewed nine-dimensional one-step oracle. Do not alter that oracle. |
| Literal pair signs | theta_j=0, theta_k=pi/2, speed means (1.5,-0.5), speed covariance `[[0.04,0.01],[0.01,0.09]]`. Expected cross block is `[[0,0.01],[0,0]]`; swapping steps transposes it. |
| Pair limits | The same scalar Gaussian identities give j=k plus/minus limits, plus symmetry and minus conjugate symmetry. Use nonzero speed/heading covariance from the declared rank-one fixture below. |
| Linear motion | Mean `(0.5,-0.25,alpha,1.5,0,0)`, alpha=0 and pi/2 as separate cases; only p_x,p_y,v are uncertain through the published J factor, q=(0.04,0,0). Events at 10000 and 30000 us, gyro 0 then wheel 1.4 and 1.6 respectively, R=(0.02^2,0.1^2). Independently build a small linear Gaussian endpoint/observation joint covariance and condition once. |
| Shared heading | m=(0,0,0,2,0,0), heading variance V in {0,0.4,1e-8,1e-12}, other covariance and all q zero. A single independent gyro=0 with variance 0.02^2 at 1000000 us gives 100 steps, with no conditioning effect. Total displacement length A=2. Use the stable closed forms below. |
| Partition invariance | For shared heading V=0.4, compare a final gyro at 990000 us with the same endpoint plus an earlier independent gyro at 333333 us. Total distance is A=1.98 for both schedules; all rates remain deterministic. The schedules have 99 and 100 steps respectively, so both fit the bound. |
| Scalar conditioned history | Use the declared rank-one fixture below, with its scalar precision-sum oracle. Check the complete six-state mean/covariance, including conditioned p_0 and every final latent cross block. |
| End-step timing | Two 0.01 s steps, m=(0,0,0,1.5,0.1,-0.02), B_0 zero, q=(0.04,0.0004,0.0001); gyro=0.09 and wheel=1.4 at 20000 us. Independently write all six process coordinates and endpoint/observation coefficients. First-step rate noise affects final yaw but neither old heading; last-step speed/rate noise does not directly enter displacement. |
| Received prefixes | Use only the rank-one fixture's first gyro, then gyro plus same-time wheel, then its final event. Direct scalar conditioning uses exactly that prefix. Preserve earlier return values; reject delayed, duplicate, wheel-before-gyro, checkpoint-time and unsupported events. |
| Signed and deterministic limits | Shared-heading V=0.4 with speeds -2 and 0 separately; shift its mean heading by 2pi. Check sign reversal of mean, unchanged displacement covariance, deterministic zero displacement and the unwrapped final yaw mean. A zero-mean random speed case uses the same literal pair speed covariance, so its variance must not vanish. |
| Characteristic derivatives and refusal | Independently finite-difference the Gaussian characteristic function for the rank-one variables to check first/second moment identities. Exercise bad shape/nonfinite/negative variance, inconsistent factor, unresolved S, scale underflow and bounds without repair. |

For shared heading at mean zero,

```
E[p_x] = A exp(-V/2)       E[p_y] = 0
Var(p_x) = A^2/2 [1-exp(-V)]^2
Var(p_y) = A^2/2 [1-exp(-2V)]
Cov(p_x,p_y) = 0
Cov(theta,p_x) = 0
Cov(theta,p_y) = A V exp(-V/2) .
```

Evaluate `1-exp(-V)` with `-expm1(-V)`. The x variance is order V squared,
so an absolute 1e-10 check alone cannot establish its accuracy at small V.
For the V=1e-8 and 1e-12 controls, report relative variance error against this
stable target. A result may be labelled resolved for this fixture only if that
relative error is <=1e-7; otherwise retain it as unresolved, without widening
the criterion or adding a special-case production formula. An unresolved narrow
control is an allowed reported limitation, not a successful precision claim.
The V=0 case has exact declared zero support and no relative denominator.

For the scalar conditioned history use physical
`m=(0.5,-0.25,0.2,1.5,0.1,-0.02)`, B_0 a single column
`(0.2,-0.1,0.3,0.15,0.04,-0.02)^T`, `P_0=B_0 B_0^T`, q=0 and max_step=0.01 s.
Events are gyro=0.09 and wheel=1.4 at 10000 us, then gyro=0.11 at 30000 us,
with the unchanged variances 0.02^2 and 0.1^2. Write `x_0=m+B_0 G`, G~N(0,1).
Each event is `d_e+c_e G+noise`, so directly compute

```
Var(G|y) = [1+sum_e c_e^2/R_e]^-1
E[G|y]   = Var(G|y) sum_e c_e (y_e-d_e)/R_e .
```

All old headings, speeds, p_0 and final latent coordinates are affine in this
one scalar. Compute endpoint moments from scalar characteristic derivatives
`E[exp(itG)]`, `E[G exp(itG)]`, `E[G^2 exp(itG)]` and explicit physical sums,
without using the general matrix conditioner or pair kernel. This checks
rotated singular support, nonidentical shared headings and observation-conditioned
correlations independently of the new implementation.

For ordinary exact fixture comparisons retain the existing absolute 1e-10
criterion; no relative relaxation. Finite differences use the existing 1e-6
perturbations and 1e-7
absolute/relative derivative tolerances. For the scalar conditioned fixture let
phi(t)=exp(i mu_G t-Var(G)t^2/2). Central-difference phi to check phi' and
central-difference the independently derived analytic
phi'(t)=(i mu_G-Var(G)t)phi(t) to check phi''. Use t=0, 0.3 and 0.3008,
fixed from the fixture's scalar phase coefficients. Compare the first/second
identity kernels with phi'/i and -phi'' respectively, retaining the constant
phase factor when needed. Do not use a raw second difference of phi: its
roundoff amplification at this step can exceed the unchanged tolerance even
without a formula bug. The narrow relative test above is additional precision evidence, not a substitute for the
ordinary tests. Explicitly report each failure or unresolved case.

Review test sensitivity with small temporary, restored mutations: drop j!=k
terms, swap the minus-phase conjugation, omit p_0 or final latent cross terms,
use new-state displacement, or condition twice. Each must fail its relevant
independent check. Do not commit these mutations, weaken expectations or use
candidate agreement as a replacement. A failed proposed check is evidence to
inspect, not permission to tune the physical law.

## Bounded possible implementation and stop

A later approval could authorize only these new files:

- `tools/multistep_position_oracle.py`: independent physical schedule, declared
  factor conditioning, analytic moments and immutable diagnostics.
- `tests/test_multistep_position_oracle.py`: the fixed independent checks above.
- `docs/multistep_position_oracle_validation.md`: actual checks, reviewed failures,
  resource measurements, preservation evidence and unresolved precision limits.

No candidate, production, sampler, old oracle or frozen input/protocol/result
file changes are proposed. The oracle has no random input, no RNG and no CLI
that launches a comparison. It does not open recordings or saved sample arrays.
The frozen old oracle may be imported read-only by the one-step validation, not
by the new implementation. The complete published O sequence is a dimension
reference, not an authorized numerical run in this proposal.

Bounds are N<=100 positive steps, E<=32 observations, D=6+3N<=306 physical
coordinates, checkpoint factor width <=6, max_step=0.01 s and max_gap=2 s.
The prior factor has at most 306 columns and the Joseph factor at most 338.
Only the E-by-E S matrix requires a solve. After physical row products, at most
10000 ordered step pairs are needed. Dense storage is bounded by these shapes;
practical time, peak memory and numerical precision are unmeasured. No claim of
long-prefix scalability follows from the size bounds.

Before any later execution, freeze this note and fixture expectations, review
the implementation independently, and create fresh
`results/multistep-position-oracle-checks-v1/` and
`results/multistep-position-oracle-demo-v1/`, which must not already exist.
Use the preserved command monitor with 900 s aggregate command time, 2 GiB
assessed per-invocation process-tree memory and 256 MiB retained validation/demo
output, including failures and final metadata. Retain its 100 ms sampling and
short-lived-process limitations. Save unique numbered attempts for demonstrated
implementation-bug repairs within those same caps, reviewed before repeating
only affected checks. No input/tolerance changes or increased budget.

Acceptance for that possible implementation would be independently reviewed
formulas and code, all ordinary deterministic checks passing, every narrow
precision result honestly resolved or unresolved, restored mutation checks,
full pytest, CTest, an unchanged real C++ development demo with inspected plots,
source/artifact hashes and resource evidence. If C++ changes prove necessary,
stop for scope review; they are not part of this proposal. No sanitizer or hosted
CI result may be inferred from a Python-only implementation.

Stop after that deterministic validation report. A failed check or unresolved
precision must be retained. Do not compare with #10's saved outputs, reopen
consumed streams, generate draws, change its verdict, or proceed to a numerical
study/navigation experiment without a separate scope decision. The next decision
from this design review is whether to approve exactly this bounded implementation
and deterministic validation, with no new experiment or candidate change.

## Analytical review record

Two separate native agents worked read-only: one derived the physical Gaussian
conditioning and all moment/cross-covariance identities; the other independently
checked the phase signs, full endpoint assembly, singular support, timing and
numerical limits. The parent checked the real/complex block expansion, expanded
the centered-pair identity, inspected the preserved timing/model definitions and
owns this specification and final Git operations.

The reviews agreed on the analytical route. The main reconciled design choice
was to require a declared prior factor plus redundant covariance instead of
inferring support from a covariance. This makes phase variance a factor-row
norm and admits a rotated rank-one fixture, while explicitly limiting accuracy
claims by input/factor discrepancy. Both reviewers accepted that distinction.
They also required an explicit unwrapped-yaw statement and refused to treat
positive covariance or a heuristic cancellation estimate as an accuracy proof.
No analytical blocker remains; finite-precision feasibility remains untested.

Draft review also clarified the unconjugated complex cross moment, limited the
square-distance statement to constant shared heading and deterministic speeds,
and replaced a cancellation-prone second finite difference with a first
difference of an independently derived analytic derivative. The parent corrected
the partition-control endpoint to keep both schedules inside the 100-step bound.
These changes affect the proposed checks and explanations, not the frozen model
or any existing result.

The design starts from clean local commit
`376d5022c3198f1d5d4ad45c019ccc9b1d6676fc` on
`codex/trajectory-mixture-design`. Draft review and final preservation evidence
are retained separately under
`results/multistep-position-oracle-design-v1/`. Only this technical note is added
to Git. No candidate/oracle execution, numerical test, CTest, pytest, demo or
sampling was performed in this design-only milestone. Historical test and study
results are not new validation of these formulas.
