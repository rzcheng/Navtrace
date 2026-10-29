# Check the exact moment targets

Ryan approved the bounded validation-proof repair, push and tested merge on
2026-10-29. This follows issue #14 and its failed Stage A report at `474373d`.
Issue #15 covers the required CI dependency setup. The new branch starts from
that exact report commit; earlier branches, reports, protocols and artifacts
remain intact. This is a repair to deterministic validation, not Stage B.

## Problem and decision

The old tests asked one independently rounded 256-bit ball to contain another.
Two valid enclosures of the same real value need not nest. Twenty-two tests
stopped at that predicate, and a norm test used an unjustified ordering between
independent square-root bounds. Preserve all 23 failures. The decision here is
whether independent certificates establish the original exact-target membership
and tightness requirements for the same fixed physical fixtures.

The oracle remains byte-for-byte unchanged, including its fixed 256-bit Arb
context. The new test certificates use exact integer arithmetic and a separately
fixed rational grid of 2^-640. This is an explicit change to the test reference,
not a claim that both computations still use the former 256-bit target route.
No certificate parameter depends on an oracle output or a previous test result.

## Allowed changes

- Add `tests/exact_target_bounds.py` and `tests/test_exact_target_bounds.py`.
- Change the independent target representation and membership assertions in
  `tests/test_position_moment_enclosures.py` and `tests/test_moment_precision.py`.
- Add this protocol and a separate measured repair report.
- Update `.github/workflows/ci.yml` and add `requirements-oracle-ci-lock.txt`.
- Clarify historical local provenance and the same isolated backend setup in
  README without changing frozen runners or claiming a fresh public clone
  contains the saved study inputs.

Keep all physical means, covariances, factors, events, timestamps, process and
measurement noise, derivative steps, mutation definitions and accuracy limits
fixed. Keep both oracle modules, every earlier implementation, the original
optional lock, all prior reports and the frozen design unchanged. No saved
study payload supplies an expected value. No new study, full O sequence,
comparison, sampling campaign, candidate change or production retuning occurs.

## Independent proof rules

Keep rational physical conditioning and coefficients exact. Test certificates
must not import oracle kernels, flint, numpy or another numerical backend.
They use Python integers and Fraction only. All finite-precision target
arithmetic encloses its exact mathematical result by rounding the lower
endpoint down and upper endpoint up to the fixed grid. Negative rounding uses
integer floor/ceiling, not truncation. Every bound retains its radius.

A real certificate E=[e0,e1] has finite exact rational endpoints. For the
unchanged oracle interval A=[a0,a1], prove `a0 <= e0 <= e1 <= a1` using exact
rational comparisons. For rational targets q, compare `a0 <= q <= a1` directly.
Require the original independent width limits as well. Disjoint certificates
are a failed check; overlapping but nonnested certificates are unresolved and
must fail validation. Never substitute overlap, matching midpoints, an
absolute-error tolerance, padding A, or extra terms chosen after failure.

Use rectangular complex intervals with exact real/imaginary interval arithmetic.
For z in the input rectangle, set R=max|Re z|+max|Im z|. Require R<=16. Evaluate
S=sum(z^k/k!, k=0..256), with 257 summands, with outward rational interval operations. Bound each
component of exp(z)-S by

```
Texp = 16^257 / (257! * (1 - 16/258)).
```

The bound follows by dominating the absolute power-series tail with a geometric
series: successive term ratios after the first omitted term are <=16/258<1.
The fixed global tail avoids powering arbitrarily large rational denominators.
Check the represented input rectangle before polynomial evaluation; refuse a
rounded rectangle outside R<=16 rather than silently extending the domain.
The same bound applies simultaneously to every point of the input rectangle.
Sine and cosine are the imaginary and real parts of exp(i*x). For expm1, omit
the constant term (256 summands), retaining the same fixed tail proof. The fixed
series degree is never increased to obtain inclusion.

For positive rational x in [2^-128,2^128], reduce exactly to x=2^k*m with
1<=m<2. Put u=(m-1)/(m+1). Evaluate 256 terms of

```
log(m) = 2 * sum(u^(2*j+1)/(2*j+1), j=0..255) + remainder.
0 <= remainder <= Tlog = 2*3^-513 / (513*(1-1/9)).
```

The fixed tail follows from exact 0<=u<=1/3. Form u exactly before its
outward polynomial enclosure; do not use an unchecked rounded approximation
to justify the domain or the tail. Compute log(2) by the same fixed series with
u=1/3, then enclose k*log(2)+log(m), including negative k correctly. For interval
arguments, apply monotonicity to the exact endpoints and refuse if either
endpoint leaves the stated positive domain. log(1)=0 is an algebraic identity.

For true pi in Gaussian evidence, use Machin's identity
pi=16*atan(1/5)-4*atan(1/239). Use 256 alternating terms for each arctangent;
the next term gives the signed remainder interval. This follows from the
alternating-series remainder theorem and the tangent addition identity with
its principal-angle range. Fixture headings such as binary64 math.pi/2 remain
those exact binary64 inputs; they are not replaced by true pi.

The point constructor may retain an exact rational input under the bit guards.
Preserve exact singleton rational sums and zero products. Other real interval
products, divisions and power steps round outward on the fixed grid; the
transcendental polynomial loops force outward rounding. Exact physical
conditioning stays in Fraction arithmetic outside this helper.

Only input-based algebraic reductions are allowed: zero products, exact rational
sums, exp(0)=1, expm1(0)=0, sin(0)=0, cos(0)=1, and log(1)=0. Conjugate pairing
or zero support must follow from the independently written physical expression.
Identical nonsingleton interval bounds do not establish equal exact expressions. In particular,
never manufacture an exact zero because the oracle returned a zero-width ball.

For nonnegative square-root targets only, at each exact rational endpoint q
compute n=isqrt(floor(q*2^1280)). The exact bounds are
[n/2^640,(n+1)/2^640]; exact square equality permits a singleton. Apply
monotonicity at both interval endpoints and refuse any negative lower endpoint.
This is a test certificate, not variance flooring; the norm fixture below still
uses direct rational square inequalities.

Bound external rational numerator/denominator sizes at 65,536 bits, and guard
integer operations before allowing intermediates beyond 131,072 bits. Each
transcendental call has its fixed 256 terms and domain checks. A fresh proof context starts for each parametrized pytest test invocation,
including each proof-helper test. Limit it to 10,000,000 primitive operations.
Count each real endpoint rounding, real arithmetic operation and series
iteration before executing it; complex arithmetic counts its constituent real
operations. Physical Fraction conditioning is separately bounded by the fixed
fixture dimensions and parent command budget. Test the exhaustion branch by
setting a private test context counter at its fixed limit, then discard that
context; never lower or override the actual limit. Preflight products and shifts
(including grid shifts and negative rounding) against the intermediate bit cap.
Cache at most
4,096 immutable certificates per primitive, keyed only by the input certificate;
cache hits cannot alter a result. Once a cache is full, further fixed requests
are computed without storing additional entries; this is bounded optional
memoization, not an exhausted proof budget. Domain, bit-size or operation-budget
exhaustion is an explicit unresolved proof, never a clipped number or a zero. No adaptive term count or precision is
available. The parent also enforces the total execution limits below.

The norm fixture retains its `inconclusive` classification and upper bound 9/16.
For the returned lower-bound ball [l,u] and the unchanged exact binary64
δ=1e-10, prove `0 <= l <= u`, `u^2 <= 41/256`, and
`(l+δ)^2 >= 41/256`. These inequalities prove validity and tightness without a
rounded comparison to sqrt(41)/16. If the entire-ball proof fails, do not fall
back to testing only l. Preserve the failure for review.

For the fixed scale-certificate fixture, retain its existing predicates and
also prove b>=0 and b^2>=2*c^2, where b is the lower endpoint of the
returned covariance-bound ball and c is the original exact binary64
magnitude. This establishes that the whole reported bound is at least
sqrt(2)*|c|; comparison to only a target lower endpoint is insufficient.
Keep its original relative tightness test. This stronger proof obligation
was independently reviewed before any numerical execution.

## Acceptance and negative checks

Before numerical execution, independently review and freeze the complete source
and fixed test cases. Proof-helper tests must cover exact endpoints and negative
rounding, rational inclusion, exact zero identities, fixed elementary-function
identities/ranges, invalid domains and size guards. Include explicit examples
where overlap would accept a wrong point, where a target fits but the output is
too wide, and where an unresolved enclosure must not become pass. Independently
check the finite-series tail bounds with exact inequalities. No random probes.

Re-run all original fixture families. Reach all eight shared-heading width gates,
the historical narrow-oracle checks and the complete one-step certificate.
Detect and restore all six prescribed mutations using certified difference
bounds above the unchanged 1e-10 limit. Use exact interval subtraction for these
bounds; do not round the target into the oracle backend first. Keep the finite-
difference step 1e-6 and tolerance 1e-7, ordinary width 1e-10, positive shared-
heading relative width 1e-10, and old narrow criterion 1e-7 unchanged. Preserve
the exact existing representations: ordinary containment width is
Fraction(1,10**10), while shared-heading relative width, mutation and norm
tolerances retain the exact binary64 ratio of 1e-10. Other limits retain the
expressions in the original test source.

A pass requires every required assertion to be reached and pass, not only source
coverage or positive covariance. Record any failure or unresolved case. Repairs
of demonstrated implementation mistakes may repeat only affected checks after
independent review within the same resource limits. Any change to formulas,
physical inputs, degree, grid, domain, thresholds or budget needs a newly
reviewed scope; do not tune them after seeing outcomes.

## Execution and integration

Use fresh `results/position-moment-proof-repair-checks-v1/` and
`results/position-moment-proof-repair-demo-v1/`. Reuse the existing verified
isolated python-flint installation and original virtualenv read-only; do not
reinstall locally. Verify old manifests and environment before and after work.
The unchanged reviewed command monitor enforces aggregate 900 seconds, 2 GiB
assessed memory and 256 MiB retained validation/build/demo output, including
failures and final metadata. Retain its 100 ms sampling caveat. Include any
fresh dependency archives in retained output; never charge reused read-only
files as newly generated output or conceal a new install.

Run the proof-helper checks first, focused oracle tests, full pytest with
RuntimeWarning fatal, Debug CMake build, all CTest checks and the unchanged real
C++ development demo. Inspect both demo figures, exact output records and the
final diff. Existing regression smoke paths are development checks, not new
navigation evidence. No C++ changes are planned; hosted CI retains sanitizers.

CI uses Python 3.11 on ubuntu-24.04 x64 and macos-15 arm64. Install only the
hash-verified python-flint 0.9.0 binary wheels in `.deps/oracle`, with no fallback
source build or skipped oracle tests. Keep original dependencies and all existing
C++/demo/development smoke stages. Each hosted job has a fixed 30-minute timeout;
record its actual status separately from the local 900-second ledger. The two
wheel hashes and platform names are frozen in the CI lock and workflow before
execution. Primary references are the official PyPI 0.9.0 JSON metadata and
GitHub's hosted-runner reference.

Publish a new branch and create a draft PR only after the local reviewed repair.
Remote main was verified as `06e05f259e1d5e0415173127bfd21c2aebfddf64`; the starting
repair commit has 32 unpublished ancestor commits beyond it. Independent review
found personal study-guide additions in one unpublished ancestor, even though
the final tree removes them. Keep that entire original local history intact;
do not push it. Create the publication branch from verified remote main and
apply the reviewed final tree as one integration commit, recording its relation
to the preserved local source commit. This does not rewrite published history
or discard local evidence. Preserve all technical reports and no-go/defer
outcomes. Review the complete integration diff and publication ancestry, not
only the repair. Recheck remote base/head identities, all configured CI
jobs and required protections before marking ready and merging. No formal PR
approval, protection bypass, security change or published-history rewrite.

Stop at the repair/integration report. A successful repair certifies only the
fixed tests; it does not change prior study conclusions or authorize Stage B.
