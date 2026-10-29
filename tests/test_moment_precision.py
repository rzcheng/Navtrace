"""Frozen exact arithmetic examples for the Stage A precision/refusal rules.

No study payload, estimator run, random direction or adaptive tolerance supplies
an expectation. Rational examples below expose scale, law and threshold errors.
"""
from fractions import Fraction
import json
import math
import os
import struct
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import pytest
import exact_target_bounds as proof
from tools import moment_precision as precision
from flint import arb, ctx
from tools import position_moment_enclosures as enclosure


def q(value):
    return Fraction.from_float(value) if isinstance(value, float) else Fraction(value)


def b(value):
    value = q(value)
    return arb(value.numerator)/arb(value.denominator)


def diagonal(values):
    return tuple(tuple(b(value) if i == j else arb(0) for j in range(len(values)))
                 for i, value in enumerate(values))


def reference(mean, covariance, evidence=0, full=True, law="canonical"):
    # Synthetic certificate records isolate its algebra; they do not claim a
    # history-builder output. Matching fixed empty-prefix provenance is explicit.
    return SimpleNamespace(mean=tuple(b(value) for value in mean), covariance=covariance,
                           log_evidence=b(evidence), whitening_allowed=full, model_law=law,
                           physical={"step_sizes_s": (), "events": (), "checkpoint_us": 0,
                                     "prior_mean": (Fraction(0),)*6, "process_variances": (),
                                     "checkpoint_factor": tuple(tuple(Fraction(i == j) for j in range(6)) for i in range(6))},
                           diagnostics={"checkpoint_law_difference": tuple((Fraction(0),)*6 for _ in range(6))})


def p(value):
    """Proof target only; b() keeps the original oracle-side construction."""
    return value if isinstance(value, proof.Real) else proof.point(q(value))


MEMBERSHIP_RECEIPTS = []


@pytest.fixture(autouse=True)
def fresh_proof_context(request):
    MEMBERSHIP_RECEIPTS.clear()
    with proof.ProofContext() as context:
        try:
            yield
        finally:
            emit("target_proof_context", test=request.node.nodeid,
                 operations=context.operations, memberships=MEMBERSHIP_RECEIPTS)


def actual_bounds(value):
    assert value.is_finite()
    def dyadic(part):
        mantissa, exponent = part.man_exp()
        mantissa, exponent = int(mantissa), int(exponent)
        if abs(exponent) > 1000000:
            raise proof.ProofUnresolved("actual endpoint exceeds the frozen representation bound")
        return Fraction(mantissa << exponent) if exponent >= 0 else Fraction(mantissa, 1 << -exponent)
    midpoint, radius = dyadic(value.mid()), dyadic(value.rad())
    return midpoint-radius, midpoint+radius


def encode(value):
    if isinstance(value, (proof.Real, proof.Complex)):
        return proof.serialize(value)
    if isinstance(value, arb):
        return precision.serialize_ball(value)
    if isinstance(value, Fraction):
        return proof.serialize(value)
    if hasattr(value, "items"):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    return value


def emit(name, **values):
    print("precision_validation "+json.dumps(encode({"check": name, **values}),
                                             sort_keys=True, allow_nan=False))


def contains(actual, expected, width_limit=Fraction(1, 10**10)):
    target = expected if isinstance(expected, proof.Real) else q(expected)
    receipt = proof.verify(actual_bounds(actual), target, width_limit=width_limit)
    MEMBERSHIP_RECEIPTS.append(receipt)
    assert receipt["status"] == "proved", proof.serialize(receipt)
    return receipt


def test_fixed_backend_context_and_exact_binary_point_one():
    assert ctx.prec == 256 and ctx.threads == 1
    actual = precision.exact(.1)
    expected = Fraction(3602879701896397, 36028797018963968)
    assert precision.bounds(actual) == (expected, expected)
    assert expected != Fraction(1, 10)
    emit("exact_binary_input", point_one=actual, exact_rational=expected)


@pytest.mark.parametrize("value,expected", [
    (Fraction(1)+Fraction(1, 2**53), 1.),
    (Fraction(1)+Fraction(3, 2**53), float.fromhex("0x1.0000000000002p+0")),
    (-(Fraction(1)+Fraction(1, 2**53)), -1.),
    (Fraction(1, 2**1075), 0.),
    (Fraction(3, 2**1075), float.fromhex("0x0.0000000000002p-1022")),
    (Fraction(1, 2**1074), float.fromhex("0x0.0000000000001p-1022")),
])
def test_binary64_rounding_ties_even_and_subnormals(value, expected):
    assert struct.pack(">d", precision.binary64_round(value)) == struct.pack(">d", expected)


def test_asymmetric_symmetrization_preserves_each_rounding_operation():
    following = float.fromhex("0x1.0000000000001p+0")
    actual = precision.symmetrize_binary64(((2., 1.), (following, 3.)))
    assert actual == ((2., 1.), (1., 3.))  # exact halfway sum rounds to even.
    tiny = float.fromhex("0x0.0000000000001p-1022")
    subnormal = precision.symmetrize_binary64(((0., tiny), (3*tiny, 0.)))
    # Each half rounds first: 0.5 ulp ->0; 1.5 ulp ->2; final sum2 ulp.
    assert subnormal[0][1] == subnormal[1][0] == 2*tiny
    emit("binary_symmetrization", ordinary=actual, subnormal=subnormal)


def test_outward_bound_serialization_contains_original_radius():
    original = precision.interval(Fraction(1, 10), Fraction(1, 10)+Fraction(1, 2**100))
    data = precision.serialize_ball(original)
    decoded = precision.deserialize_ball(json.loads(json.dumps(data)))
    lower, upper = precision.bounds(original)
    decoded_lower, decoded_upper = precision.bounds(decoded)
    assert decoded_lower <= lower <= upper <= decoded_upper
    assert decoded_lower < decoded_upper
    assert decoded.contains(original)
    assert precision.width(decoded) <= Fraction(1, 2**95)
    assert set(data) == {"lower", "upper"}
    assert set(data["lower"]) == {"mantissa", "exponent"}
    emit("serialized_enclosure", original=original, decoded=decoded)


@pytest.mark.parametrize("data", [
    {"lower": {"mantissa": "0", "exponent": 0}},
    {"lower": {"mantissa": 0, "exponent": 0}, "upper": {"mantissa": "1", "exponent": 0}},
    {"lower": {"mantissa": "0", "exponent": True}, "upper": {"mantissa": "1", "exponent": 0}},
    {"lower": {"mantissa": "2", "exponent": 0}, "upper": {"mantissa": "1", "exponent": 0}},
])
def test_serialization_schema_and_endpoint_order_are_not_repaired(data):
    with pytest.raises(precision.InvalidInput):
        precision.deserialize_ball(data)


def test_square_range_does_not_treat_an_uncertain_sign_as_negative_variance():
    result = precision.square_range(precision.interval(-2, 3))
    low, high = precision.bounds(result)
    assert low <= 0 and high >= 9
    assert result.contains(0) and result.contains(9)
    assert precision.width(result) < 10
    with pytest.raises(precision.UnresolvedArithmetic):
        precision.sqrt_range(precision.interval(-1, 1))
    with pytest.raises(precision.UnresolvedArithmetic):
        precision.verified_cholesky(((precision.interval(-Fraction(1, 2**100), Fraction(1, 2**100)),),))


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("magnitude,status", [(.5e-7, "eligible"), (2e-7, "withheld_precision")])
def test_all_direction_certificate_scale_uses_exact_off_diagonal_fixture(sign, magnitude, status):
    scale = Fraction(1, 2**16)
    covariance = diagonal((1, scale**2))
    off_diagonal = sign*q(magnitude)*scale
    point = ((b(1), b(off_diagonal)), (b(off_diagonal), b(scale**2)))
    result = precision.certificate((0, 0), point, (0, 0), covariance)
    assert result["status"] == status
    expected = p(2).sqrt()*p(q(magnitude))
    # alpha=1 exactly and beta=||[[0,c],[c,0]]||F=sqrt(2)|c|.
    contains(result["alpha"], 1)
    lower, upper = expected.lo, expected.hi
    bound_lower, bound_upper = precision.bounds(result["covariance_bound"])
    assert bound_lower >= lower
    assert bound_upper-lower <= q(1e-10)*upper
    exact_squared_target = 2*q(magnitude)**2
    assert bound_lower >= 0
    assert bound_lower**2 >= exact_squared_target
    contains(result["mean_bound"], 0)
    assert precision.bounds(result["reference_covariance_bound"])[1] <= q(1e-10)
    emit("certificate_scale", sign=sign, magnitude=magnitude, expected=expected, result=result,
         exact_squared_target=exact_squared_target, lower_bound_squared=bound_lower**2)


def test_input_factor_law_bridge_cannot_hide_behind_absolute_smallness():
    small = Fraction(1, 2**32)
    factor_entry = Fraction(1, 2**16)*(1+Fraction(1, 2**20))
    canonical_covariance = diagonal((1, small, 1, 1, 1, 1))
    factor_covariance = diagonal((1, factor_entry**2, 1, 1, 1, 1))
    canonical = reference((0,)*6, canonical_covariance)
    factor = reference((0,)*6, factor_covariance, law="factor")
    result = precision.certify_oracle((0,)*6, factor_covariance, 0, canonical, factor)
    exact_difference = factor_entry**2-small
    exact_relative = exact_difference/small
    assert exact_difference < q(1e-10)
    assert exact_relative > q(1e-7)
    assert result["status"] == "withheld_precision"
    contains(result["law_bridge"]["covariance_bound"], exact_relative)
    contains(result["arithmetic"]["covariance_bound"], 0)
    contains(result["direct_total"]["covariance_bound"], exact_relative)
    assert result["T"] == tuple(tuple(b(1 if i == j else 0) if i != 1 else
                                      b(Fraction(1, 2**16) if j == 1 else 0)
                                      for j in range(6)) for i in range(6))
    emit("factor_law_bridge", exact_difference=exact_difference, exact_relative=exact_relative, result=result)


def test_two_step_process_rounding_is_arithmetic_against_exact_qh():
    mean = (0.,)*6
    covariance = tuple(tuple(float(i == j) for j in range(6)) for i in range(6))
    factor = covariance
    diffusion = (.04, .0004, .0001)
    events = ((20000, "gyro", 0., .0004),)
    canonical = enclosure.condition_prefix(mean, covariance, factor, diffusion, events)
    factored = enclosure.condition_prefix(mean, covariance, factor, diffusion, events, law="factor")
    rows = canonical.physical["diffusion_rounding"]
    assert len(rows) == 6
    for index, row in enumerate(rows):
        channel = index % 3
        exact_product = q(diffusion[channel])*q(.01)
        rounded_product = float(exact_product)
        root = math.sqrt(rounded_product)
        halfway_low = (q(math.nextafter(root, -math.inf))+q(root))/2
        halfway_high = (q(root)+q(math.nextafter(root, math.inf)))/2
        assert halfway_low**2 < q(rounded_product) < halfway_high**2
        exact_difference = q(root)**2-exact_product
        assert row["exact_qh"] == exact_product
        assert row["rounded_qh"] == rounded_product
        assert row["rounded_root"] == root
        assert row["effective_factor_variance"] == q(root)**2
        assert row["difference"] == exact_difference
    assert any(row["difference"] != 0 for row in rows)
    # Both rigorous laws use exact q*h; the root-rounding discrepancy is not
    # incorrectly charged as a canonical-vs-factor prior-law difference.
    for i in range(6):
        contains(canonical.mean[i]-factored.mean[i], 0)
        for j in range(6):
            contains(canonical.covariance[i][j]-factored.covariance[i][j], 0)
    emit("two_step_process_law", rounding=rows)


def test_hidden_narrow_error_is_one_sixteenth_even_with_matching_diagonals():
    a, delta = Fraction(1, 2**32), Fraction(1, 2**36)
    covariance = [list(row) for row in diagonal((1,)*6)]
    covariance[0][1] = covariance[1][0] = b(1-a)
    point = [list(row) for row in covariance]
    point[0][1] = point[1][0] = b(1-a+delta)
    assert delta < q(1e-10)
    assert delta/a == Fraction(1, 16)
    precision.verified_cholesky(covariance, require_resolution=True)
    precision.verified_cholesky(point, require_resolution=True)
    result = precision.certificate((0,)*6, point, (0,)*6, covariance)
    assert result["status"] == "withheld_precision"
    assert precision.bounds(result["covariance_bound"])[0] >= Fraction(1, 16)
    emit("hidden_narrow_error", exact_entry_error=delta, exact_relative_error=delta/a, result=result)


def test_total_certificate_keeps_sum_bound_when_arithmetic_and_bridge_cancel():
    canonical = reference((0,)*6, diagonal((1,)*6))
    displaced = [0]*6
    displaced[0] = Fraction(3, 4)*q(1e-7)
    factor = reference(displaced, canonical.covariance, law="factor")
    result = precision.certify_oracle((0,)*6, canonical.covariance, 0, canonical, factor)
    assert result["status"] == "withheld_precision"
    contains(result["direct_total"]["mean_bound"], 0)
    contains(result["law_bridge"]["mean_bound"], displaced[0])
    contains(result["arithmetic"]["mean_bound"], displaced[0])
    contains(result["summed_mean_bound"], 2*displaced[0])
    contains(result["mean_bound"], 2*displaced[0])
    emit("noncancelling_bound", result=result)


def test_reference_width_evidence_and_point_refusal_are_independent_gates():
    base = reference((0,)*6, diagonal((1,)*6))
    broad_mean = list(base.mean)
    broad_mean[0] = precision.interval(-Fraction(1, 2**30), Fraction(1, 2**30))
    broad = reference((0,)*6, base.covariance)
    broad.mean = tuple(broad_mean)
    broad_factor = reference((0,)*6, base.covariance, law="factor")
    broad_factor.mean = tuple(broad_mean)
    factor = reference((0,)*6, base.covariance, law="factor")
    result = precision.certify_oracle((0,)*6, base.covariance, 0, broad, broad_factor)
    assert result["status"] == "withheld_precision" and result["reason"] == "reference_width"
    evidence = precision.certify_oracle((0,)*6, base.covariance, 2e-10, base, factor)
    assert evidence["status"] == "withheld_precision"
    assert precision.bounds(evidence["evidence_bound"])[0] > q(1e-10)
    for state in ("invalid", "failed", "unresolved", "not-started"):
        refused = precision.certify_oracle((0,)*6, base.covariance, 0, base, factor,
                                           point_status=state, point_diagnostics={"preserved": state})
        assert refused["status"] == "withheld_precision"
        assert refused["point_status"] == state and refused["reason"] == "point_oracle_refused"
        assert refused["point_diagnostics"]["preserved"] == state
    emit("separate_eligibility_gates", width=result, evidence=evidence)


def test_full_six_support_has_no_pseudoinverse_floor_or_tiny_denominator():
    zero = reference((0,)*6, diagonal((0,)*6), full=False)
    zero_factor = reference((0,)*6, zero.covariance, full=False, law="factor")
    refused = precision.certify_oracle((0,)*6, zero.covariance, 0, zero, zero_factor)
    assert refused["status"] == "withheld_precision"
    with pytest.raises(precision.UnresolvedArithmetic):
        precision.normalized_errors((0,)*6, zero.covariance, zero)
    for covariance in (diagonal((1, 1, 1, 1, 1, 0)), diagonal((1, 1, 1, 1, 1, Fraction(1, 10**14)))):
        result = precision.certificate((0,)*6, covariance, (0,)*6, covariance)
        assert result["status"] == "withheld_precision"
    emit("full_six_support_refusal", declared_zero=refused)


@pytest.mark.parametrize("values,status", [
    ((0., 0., 0., 0.), "pass"), ((.3, .3, .3, .3), "fail"),
    ((-.1, .1, -.1, .1), "inconclusive"),
])
def test_fixed_four_group_examples(values, status):
    result = precision.group_assessment(values, .1)
    assert result["status"] == status
    expected_mean = sum((q(value) for value in values), Fraction(0))/4
    expected_variance = sum(((q(value)-expected_mean)**2 for value in values), Fraction(0))/12
    contains(result["mean"], expected_mean)
    contains(result["se"], p(expected_variance).sqrt())
    emit("four_groups", values=values, result=result)


def test_group_threshold_touches_are_not_coerced_to_pass():
    epsilon, radius = q(.1), Fraction(1, 2**40)
    touching = precision.group_assessment((.1,)*4, .1)
    assert touching["status"] == "pass"  # closed <= limit, exactly zero SE.
    uncertain = precision.interval(epsilon-radius, epsilon+radius)
    crossed = precision.group_assessment((uncertain,)*4, .1)
    assert crossed["status"] == "inconclusive"
    assert crossed["reason"] == "threshold_overlap"
    assert crossed["width_ok"]
    emit("group_boundary", exact=touching, uncertain=crossed)


def test_missing_groups_and_shared_uncertainty_are_never_independent_trials():
    for values in ((0, 0, 0), (0, 0, 0, None), (0, 0, 0, 0, 0)):
        result = precision.group_assessment(values, .1)
        assert result["status"] == "withheld_missing"
    radius = Fraction(1, 2**20)
    common = precision.interval(-radius, radius)
    result = precision.group_assessment((common,)*4, .1)
    assert result["status"] == "withheld_precision"
    assert precision.width(result["mean"]) >= 2*radius
    assert not result["width_ok"]
    wide_fail = precision.group_assessment((precision.interval(1, 2),)*4, .1)
    assert wide_fail["status"] == "fail" and not wide_fail["width_ok"]
    emit("shared_uncertainty", result=result, wide_certified_failure=wide_fail)


@pytest.mark.parametrize("value,status", [(.25, "pass"), (.6, "fail"), (.5, "pass")])
def test_exact_diagonal_spectral_guard(value, status):
    matrix = ((b(value), b(0)), (b(0), b(0)))
    result = precision.norm_assessment(matrix, .5)
    assert result["status"] == status
    contains(result["lower"], q(value))
    contains(result["upper"], q(value))
    emit("diagonal_norm", value=value, result=result)


def test_fixed_off_diagonal_norm_is_conservatively_inconclusive():
    # Exact eigenvalues are 9/16 and1/16; column norm sqrt(41)/16<1/2.
    # Row sum9/16 is the upper bound. The frozen enclosure straddles1/2,
    # even though an extra eigensolver could establish failure. Do not add it.
    matrix = ((b(Fraction(5, 16)), b(Fraction(1, 4))),
              (b(Fraction(1, 4)), b(Fraction(5, 16))))
    result = precision.norm_assessment(matrix, .5)
    assert result["status"] == "inconclusive"
    lower, upper = actual_bounds(result["lower"])
    delta = q(1e-10)
    assert 0 <= lower <= upper
    assert upper*upper <= Fraction(41, 256)
    assert (lower+delta)*(lower+delta) >= Fraction(41, 256)
    contains(result["upper"], Fraction(9, 16))
    emit("conservative_norm", result=result, exact_eigenvalues=(Fraction(9, 16), Fraction(1, 16)),
         lower_bound_interval=(lower, upper), norm_squared=Fraction(41, 256), tightness_delta=delta)


def test_norm_interval_touching_threshold_is_inconclusive():
    radius = Fraction(1, 2**40)
    uncertain = precision.interval(Fraction(1, 2)-radius, Fraction(1, 2)+radius)
    result = precision.norm_assessment(((uncertain, b(0)), (b(0), b(0))), .5)
    assert result["status"] == "inconclusive"


def test_normalized_errors_keep_all_full_six_entries_and_original_units():
    scales = (1, 2, 4, 8, 16, 32)
    covariance = diagonal(tuple(value**2 for value in scales))
    base = reference((0,)*6, covariance)
    mean = tuple(q(value)*Fraction(index+1, 100) for index, value in enumerate(scales))
    saved = [list(row) for row in covariance]
    for i in range(6):
        for j in range(i, 6):
            difference = Fraction(i+j+1, 1000)*scales[i]*scales[j]
            saved[i][j] += b(difference)
            saved[j][i] = saved[i][j]
    result = precision.normalized_errors(mean, saved, base)
    for i in range(6):
        contains(result["mean"][i], Fraction(i+1, 100))
        contains(result["original_mean"][i], mean[i])
        for j in range(6):
            contains(result["covariance"][i][j], Fraction(i+j+1, 1000))
            contains(result["original_covariance"][i][j], Fraction(i+j+1, 1000)*scales[i]*scales[j])
    emit("full_six_normalized_errors", unique_covariance_entries=21, position_latent_cross_entries=8, result=result)


def test_schema_symmetry_nonfinite_and_context_failures_keep_distinct_statuses():
    base = reference((0,)*6, diagonal((1,)*6))
    asymmetric = [list(row) for row in base.covariance]
    asymmetric[0][1] = b(Fraction(1, 2**50))
    with pytest.raises(precision.InvalidInput):
        precision.normalized_errors((0,)*6, asymmetric, base)
    with pytest.raises(precision.InvalidInput):
        precision.normalized_errors((0,)*5, base.covariance, base)
    with pytest.raises(precision.InvalidInput):
        precision.normalized_errors((0,)*6, base.covariance, base,
            state_order=("py", "px", "yaw", "speed", "yaw_rate", "gyro_bias"))
    with pytest.raises(precision.InvalidInput):
        precision.group_assessment((0,)*4, .1001)
    with pytest.raises(precision.InvalidInput):
        precision.norm_assessment(diagonal((1, 1)), .5001)
    for value in (math.nan, math.inf, -math.inf, True):
        with pytest.raises(precision.InvalidInput) as invalid:
            precision.exact(value)
        assert invalid.value.status == "invalid"
    with pytest.raises(precision.UnresolvedArithmetic) as unresolved:
        precision.exact(arb("nan"))
    assert unresolved.value.status == "unresolved"
    original = ctx.prec
    try:
        ctx.prec = 128
        with pytest.raises(precision.UnresolvedArithmetic):
            precision.check_context()
    finally:
        ctx.prec = original
    assert ctx.prec == 256
    emit("failure_statuses", invalid="invalid", arithmetic="unresolved", context_restored=True)


def test_fixed_dyadic_representation_limit_refuses_before_fraction_expansion():
    assert precision.MAX_DYADIC_EXPONENT == 1000000
    tiny = arb((1, -1000001))
    with pytest.raises(precision.UnresolvedArithmetic) as failure:
        precision.bounds(tiny)
    assert failure.value.status == "unresolved"
    assert failure.value.diagnostics["exponent"] == -1000001
    assert failure.value.diagnostics["maximum_absolute_exponent"] == 1000000
    data = {"lower": {"mantissa": "1", "exponent": -1000001},
            "upper": {"mantissa": "1", "exponent": -1000001}}
    with pytest.raises(precision.UnresolvedArithmetic) as serialized:
        precision.deserialize_ball(data)
    assert serialized.value.status == "unresolved"
    emit("dyadic_resource_refusal", status=failure.value.status, diagnostics=failure.value.diagnostics)


def test_certificate_reference_law_and_prefix_metadata_must_match():
    canonical = reference((0,)*6, diagonal((1,)*6))
    factor = reference((0,)*6, canonical.covariance, law="factor")
    with pytest.raises(precision.InvalidInput):
        precision.certify_oracle((0,)*6, canonical.covariance, 0, canonical, canonical)
    for name, changed in (("step_sizes_s", (.01,)), ("events", ((10000, "gyro", 0., .0004),)),
                          ("checkpoint_us", 1), ("prior_mean", (1, 0, 0, 0, 0, 0)),
                          ("process_variances", ((Fraction(1, 100), 0, 0),)),
                          ("checkpoint_factor", ((Fraction(1),),)*6)):
        original = factor.physical[name]
        try:
            factor.physical[name] = changed
            with pytest.raises(precision.InvalidInput):
                precision.certify_oracle((0,)*6, canonical.covariance, 0, canonical, factor)
        finally:
            factor.physical[name] = original
    original_difference = factor.diagnostics["checkpoint_law_difference"]
    try:
        changed_difference = [list(row) for row in original_difference]
        changed_difference[0][0] = Fraction(1, 2**32)
        factor.diagnostics["checkpoint_law_difference"] = tuple(tuple(row) for row in changed_difference)
        with pytest.raises(precision.InvalidInput):
            precision.certify_oracle((0,)*6, canonical.covariance, 0, canonical, factor)
    finally:
        factor.diagnostics["checkpoint_law_difference"] = original_difference
    result = precision.certify_oracle((0,)*6, canonical.covariance, 0, canonical, factor)
    assert result["status"] == "eligible"
    emit("matching_reference_provenance", result=result)
