# Precision rules for the saved moment comparison

[Issue #13](https://github.com/rzcheng/Navtrace/issues/13) is a design-only
follow-up approved on 2026-10-19. It starts from clean local
`367f6a9b62c7670a69a23cbec7316222e279ccd2` after #12. This note proposes two
separate later stages. Neither is authorized by this design milestone.

The next recommended stage is an independent interval reference and deterministic
validation of its numerical eligibility rules. Stop at that report. Only a
subsequent approval could authorize the one saved-output comparison below.
There is no mathematical blocker to defining these checks. Their numerical
resolution and full-history cost remain unmeasured.

## What the existing evidence establishes

The #12 oracle passed its ordinary fixed checks, but at heading variance
`V=1e-12 rad²` its narrow position variance has relative error
`4.843588053953116e-5`, above the unchanged `1e-7` criterion. The report remains
unresolved. Agreement between its matrix projection and direct term sum is not
independent evidence: both use the same pair kernel. PSD, Cholesky success,
compensated summation, a cancellation diagnostic, or agreement at two precisions
cannot establish the required error bound.

The #10 observed-path experiment remains **defer**: 106 of 110 scalar comparisons
passed and four were inconclusive. Its later position moments had no independent
oracle. The earlier #7 **no-go**, all consumed groups 96001–96004 and every old
protocol, result and failed artifact remain unchanged. New full-six comparisons
would be a post hoc supplement on consumed synthetic outputs, not fresh
validation, navigation trials, calibration or a replacement verdict.

A stable binary64 formula might eventually help the oracle, but this proposal
does not change it. Use a separately written rigorous enclosure to assess the
frozen implementation. If it cannot be certified, withhold the comparison;
do not substitute the enclosure midpoint as a new point oracle to rescue it.

## Exactly what is available

The immutable source root is `results/observed-trajectory-study-v1/`. Its manifest
binds 34 artifacts and 102 source files; the manifest itself is the 35th file.
The design inventory verified all these hashes and inspected NPZ headers only.
It did not load numerical snapshot payloads or calculate new comparisons.

| prefix, us | positive substeps | received events | saved groups | future role |
|---:|---:|---:|---|---|
| 0 | 0 | 0 | 96001–96004 | full-six checkpoint bridge |
| 10000 | 1 | 2 | 96001–96004 | full-six one-step bridge |
| 250000 | 25 | 9 | 96001–96004 | new full-six supplement |
| 500000 | 50 | 17 | 96001–96004 | new full-six supplement |
| 750000 | 75 | 24 | 96001–96004 | new full-six supplement |
| 1000000 | 100 | 32 | 96001–96004 | new full-six supplement |

Every `group-{group}-prefix-{timestamp}.npz` has little-endian float64 arrays:
`angular (2048,3)`, `conditional_mean (2048,3)`,
`conditional_covariance (2048,3,3)`, `logweights (2048,)`, `mean (6,)`,
`covariance (6,6)`, and four `(2048,)` density/importance vectors.
Angular order is `(yaw,rate,bias)` and conditional linear order is `(px,py,speed)`.
The inventory supplies exact row keys, snapshot hashes, phases, actual step sizes,
dimensions and historical statuses. `inputs.json` supplies checkpoint mean,
assembled covariance, J/K/La, diffusion and the 32 ordered observations.
`oracles.json` and `comparisons.json` retain historical targets and scores.

The snapshots suffice for full endpoint moments and their cross-covariances.
They do not contain full angular histories, standard coordinates or dense prefix
posteriors. Do not regenerate these, reopen RNG streams or reconstruct a new
candidate call. Path-wise claims, observation reweighting, resampling, distribution
coverage, ellipse mass and recordings are outside this proposal.

The primary statistic is the saved `mean` and `covariance` output, converted
exactly from its binary64 bytes. Do not replace it with a more accurate component
sum. The frozen candidate used uniform `1/2048` averages; exponentiating rounded
saved logweights would define slightly different arithmetic. A separate integrity
audit may reconstruct the exact uniform finite-mixture mean/covariance from the
saved components, with all within-component and between-component terms. It must
retain the difference from the saved output and use the existing `1e-10` absolute
check. No M-1 correction, new weights or pooled covariance is allowed. The saved
full covariance is already symmetrized by the candidate; require exact symmetry
for scoring and report any violation rather than silently repairing it.

## Fix the input law before asking about precision

The canonical target for the supplement is the stated discrete physical model
with the inputs actually supplied to #10, not a covariance rounded from a table.
Read JSON using binary64 round-to-nearest, ties-to-even, then embed every finite
value exactly as an integer ratio. Preserve the original JSON bytes too.

The saved assembled covariance is not exactly symmetric because of its original
floating arithmetic. Define `P10` by the documented input operation
`RN(RN(0.5*P_ij) + RN(0.5*P_ji))` for each entry, where RN is binary64 ties-to-even.
Implement this conversion independently with exact rational rounding and verify
it with small deterministic examples. Do not import candidate covariance/split
helpers. This reproduces the frozen input symmetrization rather than changing
the prior. Require a verified positive-definiteness certificate for P10 on O;
a failure means withholding, not a new factor or inferred lower rank.

Reproduce the timestamp schedule independently, including its binary64
`delta_us*1e-6`, `ceil(interval/max_step)` and `interval/count` operations.
Require bitwise agreement with each saved row's `step_sizes_s`, identical across
all four groups at a prefix. Interpret the agreed h values as exact dyadics in
the reference. Gyro precedes wheel at equal times; no report adds a step. Use
exact products `q*h` for independent end-step diffusion in the canonical law.
The physical recurrence otherwise uses real arithmetic, with old-state position
and heading and unwrapped yaw. Candidate arithmetic is an approximation to this
law, not permission to redefine it.

There are two separate questions about #12:

1. Its binary64 answer versus its declared-factor law. Fix the supplied B0 bytes
   from J/K/La before execution, preserving the physical row/column convention in
   #11. Save the independently assembled input and operation order. Define a
   factor-law reference from exact B0 entries and exact q*h, using the same h.
2. That factor-law reference versus the canonical P10 law. Propagate the full
   input-law difference through conditioning and every endpoint moment. A small
   entrywise checkpoint residual does not bound a narrow endpoint variance.

Rounded `sqrt(RN(q*h))` in #12, squared in its effective factor covariance, and
rounded map time sums are part of its arithmetic error against the factor-law
reference. They are not adopted into the canonical target. Record their raw
values and discrepancies; no unbounded sensitivity estimate replaces propagation.
Only the total error relative to P10 can authorize scoring. Also report the
arithmetic and prior-law contributions separately; do not assume their bounds
cancel. The future adapter freezes B0 with independently coded scalar binary64
multiply/add in ascending inner-index order for K*La. This is the declared input
for that attempt, not a claim of bit identity with every earlier BLAS product.
No alternative factor is tried if the total certificate fails.

## Independent enclosure construction

Propose a new optional dependency, `python-flint==0.9.0`, using Arb real/complex
balls at a fixed **256-bit** working precision. Its documented arithmetic tracks
rigorous error enclosures; this does not prove that our physical equations or
adapter are correct. Independent derivation, tests and review remain required.
No package was installed or imported during this design.

The proposed CPython 3.11 / macOS arm64 wheel is
`python_flint-0.9.0-cp310-abi3-macosx_11_0_arm64.whl`, SHA-256
`025fd4e77f2cbbf40f63b9cd7571aad3803deb57aecd72e7c397480fe84921a0`.
Pin it in a separate optional lock file; keep existing requirements and the
original virtualenv unchanged. A different platform or unavailable verified
wheel is a setup blocker, not permission to fetch a different release.

Use exact integer-ratio inputs, outward-rounded bounds, and the verified
`arb_mat.solve(..., algorithm="lu")` path. Never use `approx`, `chop`, a float
conversion as an enclosure endpoint, adaptive `showgood`, or automatic precision
escalation. Set the existing five environment thread controls before numerical
imports; immediately after importing flint, set ctx.threads=1 and ctx.prec=256
before any arithmetic. Serialize bounds as exact dyadic endpoints or outward decimal bounds
with a verified round-trip enclosure. Nonfinite or insufficiently narrow balls
are unresolved. No result may lose its radius during serialization or decisions.

Relevant primary references, inspected for this design:
[Arb arithmetic](https://python-flint.readthedocs.io/en/latest/arb.html),
[verified matrix solves](https://python-flint.readthedocs.io/en/latest/arb_mat.html),
[input and predicate semantics](https://python-flint.readthedocs.io/en/latest/general.html),
and the [pinned release and wheel](https://pypi.org/project/python-flint/0.9.0/).
These support the backend choice, not unmeasured performance claims.

Construct physical `g=(x0,eta0,...,etaN-1)` independently with
`P_g=blockdiag(P10,h0 Q,...,hN-1 Q)` and physical observation rows H. Use the
identities in #11, not candidate maps, posterior factors, Gaussian projection
or the #12 centered kernel. With `S=H P_g H^T+R`, solve once for the conditional
joint Gaussian of p0, all old `(theta_j,v_j)` pairs and the final latent four.
For their independently built selector matrix F:

```
mu_Y = F m_g + F P_g H^T solve(S, y-H m_g)
C_Y  = F P_g F^T - F P_g H^T solve(S, H P_g F^T).
```

The subtractive form is intentional independent arithmetic. Its cancellation
widens the certified enclosure; it never justifies clipping. Work directly with
these marginals (at most 206 rows), without factoring a possibly singular 306-row
physical covariance. Verified Cholesky pivots of S must be positive; retain the
solve residual enclosure, determinant/log-evidence enclosure and conditioning
bounds. Preserve the existing S resolution rule as well: require a verified
`lambda_min(S)>1e-12*max(1,lambda_max(S))` bound, using the same shifted-Cholesky
sufficient test described below. Positive R explains exact nonsingularity but
does not certify a floating solve. All divisions and square roots need proved domains.

Evaluate raw Gaussian characteristic moments in balls:

```
E[A exp(iT)] = (mu_A+i Cov(A,T)) exp(i mu_T-Var(T)/2)
E[A B exp(iT)] = (Cov(A,B)+(mu_A+i Cov(A,T))
                 *(mu_B+i Cov(B,T))) exp(i mu_T-Var(T)/2).
```

Use `T=theta_j+theta_k` and `theta_j-theta_k`, all j,k including unequal steps.
Form raw real second moments, include conditioned p0 and every final-latent
cross term, then subtract the full endpoint mean outer product. This differs
from #12's centered expm1 implementation. Exact zero phase rows may be simplified
algebraically before evaluation; a small interval may not be clipped to zero.
Phase variance can be nonnegative mathematically even if an enclosing lower
endpoint is negative; retain that enclosure, never replace it with a guessed
variance. A wide resulting endpoint enclosure leads to refusal.

This enclosure covers the real-arithmetic discrete model, transcendental
rounding, solves, cancellation and assembly under its declared exact inputs.
It does not enclose uncertainty about the model, navigation truth, the Gaussian
checkpoint assumption or arbitrary upstream data errors.

## Full covariance and directional eligibility

Freeze physical order `(px,py,yaw,speed,rate,bias)` and the existing numerical
unit scales (one in each stated unit). Do not select a favorable axis/subset from
candidate results. The canonical endpoint covariance C* must first satisfy the
existing denominator rule

```
lambda_min(C*) > 1e-12 * max(1, lambda_max(C*)).
```

Use verified bounds. One fixed sufficient check is a ball Cholesky proof that
`C* - tau I` is positive definite, with
`tau=1e-12*max(1,U)` and U an outward Frobenius upper bound on C*. This may refuse
a case that a sharper method would resolve; it may not silently retry a different
criterion. It preserves the old threshold and does not infer rank.

For the precision certificate choose T once from the canonical enclosure only:
round its symmetric midpoint to binary64 and perform the fixed-order lower
Cholesky; retain T's exact dyadic entries. If this tentative construction fails,
withhold. No candidate values participate. Apply T inverse through triangular
solves, not an explicit numerical inverse. Certify, with outward bounds,

```
A = T^-1 C* T^-T
alpha = 1 - upper(||A-I||_F) > 0
beta = upper(||T^-1(C12-C*)T^-T||_F)
gamma = upper(||T^-1(mu12-mu*)||_2).
```

For every nonzero direction d, relative variance error is at most beta/alpha.
More generally, for any nonzero u,v,

```
|u^T(C12-C*)v| / sqrt((u^T C* u)(v^T C* v)) <= beta/alpha.
```

This bounds every cross block on its variance scale; it is not relative error
of an individual cross-covariance entry, which can be zero. The normalized mean
error is bounded by `gamma/sqrt(alpha)`. These statements follow by substituting
`T^T u,T^T v`, using Cauchy–Schwarz and `A >= alpha I`. Thus a narrow direction
cannot hide behind a large absolute tolerance or a large marginal variance.

Require **both bounds <=1e-7**, with a positive certified denominator. Apply the
same operations separately to the arithmetic and law-bridge differences, using
the same canonical T and alpha, and report their sum bound as well. Use the more conservative of the direct total
bound and the summed bound for eligibility. Also require the total absolute
log-evidence error bound <=1e-10. Keep all #12 statuses, underflow/cancellation
and covariance diagnostics; a #12 error/refusal prevents eligibility even if
another target enclosure is available.

Require canonical enclosure uncertainty itself to be small: using the exact
midpoint vector/matrix as the point estimate in the same certificate, its mean
and covariance bounds must each be <=1e-10. Reference width is numerical
uncertainty, not Monte Carlo SE. A certificate not proved at the fixed precision
is `withheld_precision`; a certified #12 bound above the limit is also withheld,
with the measured reason, not automatically a candidate performance failure.

Singular primitive support does not imply the same nonlinear endpoint support.
A variance is exactly zero only with an explicit algebraic constant/support
proof. For deterministic and rotated-singular tests, report raw bounded moments
and the declared support; withhold full-six whitening. Never use a pseudoinverse,
eigenvalue floor, invented rank, jitter or a reduced coordinate fallback.

## Saved-output scoring, only after a later approval

Before reading any candidate snapshot payload, finish the six oracle eligibility
records in increasing prefix order. Inputs and fixed row metadata may be read
first. An ineligible prefix withholds all four new full-six row scores. Continue
to the next prescribed prefix within the fixed cap; do not replace failed rows,
retry with another precision, or choose only successful prefixes for the verdict.

For an eligible prefix, obtain a verified lower Cholesky enclosure G of the
canonical C* in the fixed physical order. With saved outputs treated as exact
binary64 inputs, compute interval-valued

```
d_m,r = solve(G, saved_mean_r - mu*)
D_P,r = G^-1 saved_covariance_r G^-T - I.
```

Use triangular solves. Keep original-unit differences and all 21 unique
covariance entries, including the eight position/latent cross entries. The
statistic being scored remains the stored output; a rigorous component audit
checks provenance/rounding without substituting a corrected output. Require
its differences from saved means/covariances <=1e-10 absolute, and require the
same variance-scaled mean/covariance bounds <=1e-7 under the prefix certificate.
A proved violation of the frozen absolute reconstruction limit or a hash/schema
contract is an integrity failure. An undecidable reconstruction bound is withheld.
Failure to prove the additional scaled 1e-7 bound is `withheld_precision`, not
a candidate accuracy failure. Preserve these distinctions in terminal precedence.
Candidate generation roundoff is part of the algorithm's saved error; it is not
an uncertainty distribution that the new evaluator can estimate from missing paths.

For each of the 27 scalar errors (six means, 21 covariance entries), use exactly
four independent groups at that prefix:

```
d_bar = sum_r d_r/4
se = sqrt(sum_r (d_r-d_bar)^2/12)
```

The constants remain epsilon=0.10 for means and 0.20 for covariance entries.
All inherited numerical threshold literals are interpreted as their frozen
binary64 values, embedded exactly; no threshold is re-rounded through a printed
results table. Interval squares use the mathematical nonnegative square range
(including exact zero when the argument straddles zero), not an arbitrary clamp
of an uncertain covariance. This also applies to norms and the SE square root.
Pass only if the entire enclosure satisfies `4se<=epsilon/2`,
`abs(d_bar)+4se<=epsilon`, and every `abs(d_r)<=2epsilon`.
Fail only if a certified lower bound establishes `abs(d_bar)-4se>epsilon` or
some `abs(d_r)>2epsilon`. Otherwise label inconclusive; distinguish statistical
insufficient precision from numerical threshold overlap. A missing/withheld
replicate never reduces the denominator or produces an aggregate pass. Shared
oracle uncertainty is propagated through all four errors, not divided by two
or treated as independent replicate noise. Outward interval dependency may
widen the result; that is a reason to withhold, not to discard it.

Require width <=1e-7 for each normalized scalar and each resulting four-SE
quantity before statistical classification. Otherwise withhold numerical
classification even when a midpoint looks favorable. A wide enclosure already
proving a failure may retain that certified failure, with its width recorded.

Retain the original covariance spectral guards: every group <=0.5 and the
four-group mean <=0.3. Freeze this conservative norm enclosure for symmetric D:
lower bound is the largest certified column 2-norm lower bound; upper bound is
the smaller of the Frobenius upper bound and the maximum absolute row-sum upper
bound. These bound the true spectral norm without an unverified eigensolver.
Pass if upper<=threshold; fail if lower>threshold; otherwise inconclusive.
No favorable norm midpoint or extra norm algorithm is introduced after inspection.

The planned denominator is always **24 rows**, **648 individual scalar errors**,
**162 four-group scalar assessments** and **30 spectral guards** (24 individual,
six aggregate). Six oracle eligibility records and 24 component integrity checks
are additional gates, not extra navigation trials. Report eligible/withheld counts
against these fixed totals. The checkpoint and first-step contribute 54 bridge
scalar assessments; the four later prefixes contribute 108 supplemental ones.
Later full-six Cholesky coordinates differ from the old latent-four coordinates:
these are not the old 110 scores plus a few new independent ones.

The supplement's own decision is pass only if every required gate, row, scalar
and norm passes; no-go for certified candidate accuracy/integrity failure or an
actual implementation/resource failure; defer for unresolved eligibility,
missing results or inconclusive precision without a separate failure. Separate
execution status (`not-started`, `completed`, `invalid`, `failed`, `unresolved`),
numerical eligibility (`eligible`/reasoned `withheld`) and statistical assessment.
Even a new pass leaves #10's historical defer intact. Four-SE limits are descriptive
engineering rules, not confidence coverage or independent navigation evidence.

## Stage A proposed next: implementation and deterministic validation

A later approval could add only:

- `requirements-oracle-lock.txt` for the pinned optional backend.
- `tools/position_moment_enclosures.py` for independent physical raw moments.
- `tools/moment_precision.py` for enclosures, certificates and scalar/norm rules.
- `tests/test_position_moment_enclosures.py` and `tests/test_moment_precision.py`.
- `docs/position_moment_enclosure_validation.md` for actual evidence and limits.

No existing implementation/test/protocol is edited. The enclosure module imports
neither candidate nor #12 kernels. Tests may read/call the frozen #12 public API
only on the fixed deterministic fixtures. No saved #10 numerical payload is used
for test expected values; inputs/schema hashes are permitted. No full O sequence
or new saved-output comparison is run in Stage A.

Before first execution freeze independently reviewed source and exact fixtures.
Reuse the 12 fixed fixture families and six restored mutations in #11/#12; ordinary
absolute limits remain 1e-10, derivative step 1e-6/tolerance 1e-7, and narrow relative
limit 1e-7. General conditioned fixtures stay at <=8 substeps and <=6 events.
Only the published reducible shared-heading control reaches 100 steps.
For the rotated rank-one scalar fixture use its exact factor-Gram law; do not
pretend that an independently rounded redundant covariance is the same singular
law. For shared heading, the canonical axis variance is the exact binary64 V;
fix B0 with independently verified RN(sqrt(V)), and also evaluate its exact
factor variance B0². Use each corresponding variance in its independent stable
formula, retain both laws and their discrepancy, and preserve the old reported
#12 result. Deterministic zero support has its explicit algebraic proof. The
canonical assembled-P10 semantics apply to O, whose full support must be certified. Standalone certificate tests also cover a deliberately inaccurate
point proposal without calling an estimator. Add these
fixed certificate tests without selecting them from study results:

| check | independent expectation |
|---|---|
| binary input and bounds | Exact dyadic 0.1, asymmetric 2x2 input rounding, saved schema/order rejection; outward serialization contains the original bound. |
| raw Gaussian identities | Independently derived scalar characteristic derivatives and explicit affine physical constructions; interval contains the exact target, not merely an overlapping wrong point estimate. |
| cancellation | Published V=0, 0.4, 1e-8, 1e-12 shared-heading formulas, evaluated independently with stable expm1 targets; require the reference relative enclosure width <=1e-10 for positive variances. Preserve the old #12 unresolved result. |
| certificate scale | Exact diagonal C=diag(1,2^-32), with point covariance off-diagonals set to `+c*2^-16` and `-c*2^-16` in separate symmetric cases, c=0.5e-7 and 2e-7; the diagonals are unchanged. The Frobenius certificate gives sqrt(2)*c, so eligibility accepts the first magnitude and refuses the second. Full-six support gate is tested separately. |
| input-law bridge | Fixed P=diag(1,2^-32) and B=diag(1,2^-16*(1+2^-20)); derive B B^T-P rationally, retain the difference and reject its narrow-relative eligibility. Include a two-step q*h rounding bridge with exact rational expected values. |
| hidden narrow error | Embed `[[1,1-2^-32],[1-2^-32,1]]` with I4. Add `2^-36` to both off-diagonal entries of the proposed covariance. Both are SPD and satisfy the denominator threshold; diagonals agree and absolute entry error is below 1e-10, but the narrow relative error is exactly 1/16. The precision certificate must refuse. |
| covariance correlations | The published rank-one scalar-conditioned fixture and literal pair-sign fixture; wrong conjugation or omitted p0, cross-step or final-latent terms must be detected. |
| rank and domain | Exact deterministic and rotated rank-one fixtures refuse full-six whitening; interval containing zero does not prove zero support; invalid R and unproved solve/Cholesky domains refuse without repair. |
| thresholds and groups | Four fixed error groups all 0, all 0.3 for mean error, and (-0.1,0.1,-0.1,0.1); respectively pass, fail, inconclusive. Add exact threshold-touching interval cases, missing fourth group, shared oracle offset and symmetry/schema failures. |
| norm bounds | Exact diagonal matrices with norm 0.25, 0.6 and 0.5 for the 0.5 guard; a fixed symmetric off-diagonal case verifies conservative inconclusive handling without changing the guard. |

Freeze all concrete matrix entries, precision, operation orders and interval
expected inclusions in the reviewed test file before execution. Interval target
containment and tightness are separate checks; an enormous interval cannot pass
because it happens to contain the answer. For the all-direction certificate,
use the stated exact rational examples; do not attempt to sample directions as
a substitute for the proven bound. Boundary cases must demonstrate that uncertain
comparisons never become pass by treating an undecidable predicate as false.

Use fresh `results/position-moment-enclosure-checks-v1/` and
`results/position-moment-enclosure-demo-v1/`, absent at initialization. Retain the
reviewed command monitor's 900 s aggregate, 2 GiB assessed process-tree memory
and 256 MiB validation/demo output caps. Download/install of the single pinned
wheel uses a separate fresh local dependency directory, bounded at 128 MiB
including the wheel and unpacked files, 120 s through final flush and 2 GiB
assessed process-tree memory; no source build, dependency upgrade or fallback.
Use a separate monitored setup ledger and durable terminal receipt, including
failed setup and final metadata, with the same RSS sampling caveat. Record these
costs separately rather than hiding them in validation totals. The original
virtualenv is read-only. Failure ends setup; retain its partial files and receipt.

Run applicable focused checks, full pytest with RuntimeWarning fatal, CTest and
the unchanged real C++ development demo; inspect both figures and hashes. Retain
numbered failed attempts. Demonstrated implementation-bug repairs may repeat
only affected checks after independent review within the same caps; no fixture,
precision, criterion or budget changes. No C++ edits or claimed sanitizer/hosted
CI result. A new backend succeeding at its narrow controls does not rewrite the
frozen binary64 oracle's result. Stop at the independently reviewed Stage A report.

## Stage B proposed later: one saved-output supplement

This needs a separate approval after Stage A evidence. It could add only
`tools/run_saved_moment_comparison.py`, `tests/test_saved_moment_comparison.py`
and `docs/saved_moment_comparison_results.md`. Freeze the reviewed runner, manifest,
input semantics, all criteria and inventory before any full-history execution.
Runner validation must use small deterministic inputs/mocked full inventories,
not a concealed full O dry run or numerical peek at saved moments. Use fresh
`results/saved-moment-comparison-checks-v1/` and
`results/saved-moment-comparison-demo-v1/`, absent together with the comparison
root at Stage B initialization. Give this stage its own monitored prelaunch
ledger: 900 s aggregate command time, 2 GiB assessed process-tree memory and
256 MiB retained checks/demo output, including failures and final metadata.
Run focused runner tests, full pytest, CTest and the unchanged development demo;
review the logs, figures, preservation and source before a clean committed
launch freeze. Never reuse Stage A's spent validation accounting or run unmetered
prelaunch work. Stage B validation adds no new dependency or full-history probe.

If separately approved, make exactly one launch in fresh
`results/saved-moment-comparison-v1/`. The new proposed offline cap is **600 s
through final flush, 512 MiB assessed wrapper-plus-worker memory and 64 MiB output**,
with 4 MiB metadata reserve. This is a separate proposed workload/cap, not a
retroactive increase of #10's 120 s experiment cap. No child worker descendants,
retry, larger precision, alternative factor, candidate execution or new draw.
Retain the 100 ms RSS sampling/short-lived-descendant caveat.

Within that one launch, compute the canonical/factor reference and frozen #12
answer once per each of the six prefixes (in order), then assess the 24 saved
rows. No oracle rerun per group. Maximum N=100, E=32, D=306, selected marginal
rows=206 and ordered displacement pairs=10000; no dense trajectory-sized sample
array is created. The exact operation count, interval width and practical
runtime are not yet measured; bounded dimensions are not feasibility evidence.
If the cap or a gate prevents work, preserve every planned row and not-started
reason. Full-history feasibility is an outcome of this attempt, not a promise.

Before reading payloads verify every source and artifact hash below, plus the
new clean reviewed source lock and Stage A evidence. Verify again afterward.
Store all exact input semantics, B0/schedule bytes, precision/backend hashes,
reference enclosures, #12 outputs and diagnostics, separate/total error bounds,
T and full-six G enclosures, every row/status and original-unit/normalized
interval, four-group/norm decisions and complete resource ledger. Do not retain
huge intermediate histories to evade the output cap. Two fixed figures show
eligibility/status at every prescribed prefix and normalized error intervals
for all four groups; unavailable values remain visible. Parent and independent
reviewer inspect all tables, intervals, figures and preservation evidence.
Stop at the supplemental report. No production or recovery decision follows.

## Preservation and review

| preserved item | SHA-256 |
|---|---|
| #10 study manifest | `3c4299251bfbc6080264a5d1861680b980a2e29105b0aabb2f58957b5fc05e2d` |
| #10 inputs.json | `c78c3826adc7034d827ee9b7e3733e682c517ee01738ab1a5e22bc2393c2de7e` |
| #10 inventory.json | `8bc70d398462bfd9202a039117936e514e366ee2e2412effd887458589d347a4` |
| #10 comparisons.json | `5ad4261e3c06e0b109c7a1db2e257e31519ab36ace1e36379abc5f7cacd9f66c` |
| #10 oracles.json | `6eb2db4f203a7ca2a4aba03d4032b8e8377f288b70821bd4c8293af20753e2dd` |
| #12 oracle | `8a315d6d6b5ff46567e5e60d5b308b0414a6a3bce8049e45f08e507f3aca3cf5` |
| #12 tests | `feae2e7c8217fb4e1c0c7c349cf62a6b4e9effabfb2d49f2d2f41052041fd8be` |
| #12 specification | `411b448230d93787fd450f161f52ea4d424123c0b80e247c80d8854506344d48` |
| #12 validation report | `f1a895274eb91b33818a08ebe75353f1e242c7bd2f7f4342669cf68555b9595c` |
| #12 evidence manifest | `a27cfc83a95186b65d4281d5a6b2b980f70b235fad8abc5aff467c922a58bcde` |
| design saved-artifact inventory | `2f0313772e46652edf9ed8409796370100a5c7ba1bfe406d2774986430b5a3fe` |

The design inventory includes every one of the 24 snapshot hashes and headers.
Baseline preservation covers 107 tracked files, 5,001 prior worktree artifacts
and 579 older artifact hashes. Both protected checkouts are clean: the original
at `c834bf3e5b9b421564c2e0050092895abe12dd43`, earlier integration at
`53be2f72d15f8d485640eaa09868210231b3e886`. Design evidence is retained under
`results/saved-moment-comparison-design-v1/`; only this note is added to Git.

Two native agents independently analyzed numerical certification and reviewed
source/artifact semantics. The parent reconciled the input-law, timing, support,
stored-statistic and full-six whitening distinctions and owns this specification.
Final review acceptance and preservation hashes are recorded in the design
evidence. No numerical module, oracle, candidate, test, demo, comparison or study
was executed for this design. Only source reads, metadata/schema/hash checks and
primary backend documentation inspection were performed. Nothing was installed,
pushed, merged or released.

The next decision is whether to approve **Stage A only**, including its single
pinned optional dependency and bounded deterministic validation. Stage B remains
behind a separate report and approval gate; it must not start automatically.
