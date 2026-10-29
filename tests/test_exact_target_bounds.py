"""Fixed checks for rational proof rules; no oracle/backend imports or draws."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
import math

import pytest

import exact_target_bounds as proof


@pytest.fixture(autouse=True)
def proof_context():
    with proof.ProofContext() as context:
        yield context


def test_exact_binary_inputs_and_immutable_endpoints():
    assert proof.q(.1) == Fraction(3602879701896397, 36028797018963968)
    value = proof.point(Fraction(1, 3))
    assert value.lo == value.hi == Fraction(1, 3)
    assert value.lower == value.lo and value.upper == value.hi
    with pytest.raises((FrozenInstanceError, AttributeError)):
        value.lo = Fraction(0)
    complex_value = proof.Complex(value, Fraction(-1, 7))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        complex_value.imag = proof.point(0)
    assert proof.current_context().operations == 0


@pytest.mark.parametrize("numerator", [-2, -1, 1, 2])
def test_grid_rounding_uses_floor_and_ceiling_for_both_signs(numerator):
    value = proof.point(numerator)/3
    target = Fraction(numerator, 3)
    floor = (numerator*proof.GRID)//3
    assert value.lo == Fraction(floor, proof.GRID)
    assert value.hi == Fraction(floor+1, proof.GRID)
    assert value.contains(target)
    assert value.width() == Fraction(1, proof.GRID)


def test_input_intervals_and_singleton_sums_remain_exact():
    lower, upper = Fraction(1, 3), Fraction(2, 3)
    value = proof.interval(lower, upper)
    assert (value.lo, value.hi) == (lower, upper)
    summed = proof.point(lower)+proof.point(upper)
    assert summed.lo == summed.hi == 1
    # Nonzero products and quotients obey the grid even for rational points.
    product = proof.point(Fraction(1, 3))*1
    assert product.lo < lower < product.hi
    quotient = proof.point(Fraction(1, 3))/1
    assert quotient.lo < lower < quotient.hi


def test_nonsingleton_subtraction_does_not_manufacture_zero():
    value = proof.interval(Fraction(1, 3), Fraction(2, 3))
    difference = value-value
    assert difference.lo <= Fraction(-1, 3)
    assert difference.hi >= Fraction(1, 3)
    assert not difference.is_zero()
    assert (0*value).is_zero()
    assert (value*0).is_zero()


def test_rational_membership_margins_width_and_wrong_point_receipts():
    target = Fraction(1, 3)
    exact = proof.verify((target, target), target, Fraction(0))
    assert exact["status"] == "proved"
    assert exact["lower_margin"] == exact["upper_margin"] == 0
    inside = proof.verify((Fraction(0), Fraction(1)), target, Fraction(1))
    assert inside["status"] == "proved"
    assert inside["lower_margin"] == Fraction(1, 3)
    assert inside["upper_margin"] == Fraction(2, 3)
    wrong = proof.verify((Fraction(0), Fraction(0)), Fraction(1), Fraction(0))
    assert wrong["status"] == "disproved"
    # Merely overlapping a target enclosure must not accept this point output.
    unresolved = proof.verify((Fraction(0), Fraction(0)), proof.interval(-1, 1), Fraction(0))
    assert unresolved["status"] == "unresolved"
    assert unresolved["membership"] != "proved"
    wide = proof.verify((Fraction(-1), Fraction(1)), Fraction(0), proof.q(1e-10))
    assert wide["status"] == "failed_width" and wide["membership"] == "proved"
    with pytest.raises(TypeError):
        inside["status"] = "proved"
    encoded = json.loads(json.dumps(proof.serialize(inside)))
    assert encoded["target"]["lo"] == {"numerator": "1", "denominator": "3"}
    assert encoded["lower_margin"] == {"numerator": "1", "denominator": "3"}


def test_mutation_error_is_exact_interval_subtraction():
    actual = (Fraction(2), Fraction(3))
    target = proof.interval(Fraction(1, 3), Fraction(2, 3))
    error = proof.mutation_error(actual, target)
    assert (error.lo, error.hi) == (Fraction(4, 3), Fraction(8, 3))
    assert error.abs_bounds() == (Fraction(4, 3), Fraction(8, 3))
    crossing = proof.mutation_error((Fraction(0), Fraction(0)), proof.interval(-1, 1))
    assert crossing.abs_bounds() == (Fraction(0), Fraction(1))


def test_exact_zero_and_one_elementary_identities():
    zero = proof.point(0)
    assert proof.exp(zero).lo == proof.exp(zero).hi == 1
    assert proof.expm1(zero).is_zero()
    assert proof.sin(zero).is_zero()
    assert proof.cos(zero).lo == proof.cos(zero).hi == 1
    assert proof.log(proof.point(1)).is_zero()
    assert proof.sqrt(zero).is_zero()
    assert proof.exp(proof.Complex(0)).imag.is_zero()
    assert proof.exp(proof.Complex(1)).imag.is_zero()
    assert not proof.expm1(proof.interval(0, Fraction(1, 1 << 40))).is_zero()


def test_fixed_exponential_ranges_and_inverse_identity():
    positive, negative = proof.exp(1), proof.exp(-1)
    assert Fraction(8, 3) < positive.lo <= positive.hi < Fraction(11, 4)
    assert Fraction(1, 3) < negative.lo <= negative.hi < Fraction(1, 2)
    identity = positive*negative-1
    assert identity.contains(0)
    assert identity.width() < Fraction(1, 1 << 500)
    small = Fraction(1, 1 << 40)
    delta = proof.expm1(small)
    assert small < delta.lo <= delta.hi < small+small*small
    assert delta.width() < Fraction(1, 1 << 500)


def test_complex_exponential_rectangle_and_conjugate_identity():
    value = proof.Complex(Fraction(1, 3), Fraction(1, 4))
    result = proof.exp(value)
    conjugate = proof.exp(value.conjugate())
    assert (result.real-conjugate.real).contains(0)
    assert (result.imag+conjugate.imag).contains(0)
    assert result.real.lo > 1 and result.imag.lo > 0
    assert result.real.width() < Fraction(1, 1 << 500)
    assert result.imag.width() < Fraction(1, 1 << 500)
    # Check exact represented rectangle, before polynomial evaluation.
    with pytest.raises(proof.ProofUnresolved):
        proof.exp(proof.Complex(proof.interval(15, 16), Fraction(1, 1 << 700)))


def test_fixed_trigonometric_and_pi_ranges():
    sine, cosine = proof.sin(1), proof.cos(1)
    assert Fraction(5, 6) < sine.lo <= sine.hi < 1
    assert Fraction(1, 2) < cosine.lo <= cosine.hi < Fraction(13, 24)
    identity = sine*sine+cosine*cosine-1
    assert identity.contains(0)
    assert identity.width() < Fraction(1, 1 << 500)
    constant = proof.pi()
    assert Fraction(333, 106) < constant.lo <= constant.hi < Fraction(355, 113)
    assert constant.width() < Fraction(1, 1 << 500)
    # This binary64 input is preserved, not silently replaced by true pi.
    half_binary_pi = proof.point(math.pi/2)
    assert half_binary_pi.lo == Fraction.from_float(math.pi/2)
    assert proof.cos(half_binary_pi).lo > 0


def test_fixed_log_normalization_negative_exponents_and_monotonicity():
    log_two = proof.log(2)
    assert Fraction(69, 100) < log_two.lo <= log_two.hi < Fraction(7, 10)
    for exponent in (-128, -3, 3, 128):
        value = Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)
        residual = proof.log(value)-exponent*log_two
        assert residual.contains(0)
        assert residual.width() < Fraction(1, 1 << 490)
    enclosure = proof.log(proof.interval(1, 2))
    assert enclosure.lo == 0
    assert enclosure.hi >= log_two.hi
    assert proof.log(Fraction(3, 2)).lo > 0


def test_sqrt_exact_brackets_and_rational_square_proofs():
    for value in (Fraction(0), Fraction(1), Fraction(9, 16), Fraction(2)):
        result = proof.sqrt(value)
        assert 0 <= result.lo <= result.hi
        assert result.lo*result.lo <= value <= result.hi*result.hi
        assert result.width() <= Fraction(1, proof.GRID)
    assert proof.sqrt(Fraction(9, 16)).lo == proof.sqrt(Fraction(9, 16)).hi == Fraction(3, 4)
    ranged = proof.sqrt(proof.interval(Fraction(1, 4), Fraction(9, 16)))
    assert (ranged.lo, ranged.hi) == (Fraction(1, 2), Fraction(3, 4))


def test_fixed_series_tail_proofs_and_signed_machin_remainders():
    exp_tail = proof._exp_tail()
    first_exp = Fraction(16**257, math.factorial(257))
    ratio = Fraction(16, 258)
    assert 0 < ratio < 1
    assert exp_tail*(1-ratio) == first_exp
    assert 0 < exp_tail < Fraction(1, proof.GRID)
    log_tail = proof._log_tail()
    assert log_tail*(1-Fraction(1, 9)) == Fraction(2, 513*3**513)
    assert 0 < log_tail < Fraction(1, proof.GRID)
    # The first omitted alternating term is positive after 256 terms. Its
    # magnitude exceeds every later term and bounds the signed remainder.
    for denominator in (5, 239):
        first = Fraction(1, 513*denominator**513)
        next_term = Fraction(1, 515*denominator**515)
        assert 0 < next_term < first
        target = proof._atan_inverse(denominator)
        partial = sum((Fraction(1 if index % 2 == 0 else -1,
                                (2*index+1)*denominator**(2*index+1))
                       for index in range(256)), Fraction(0))
        assert target.lo <= partial
        assert target.hi >= partial+first
        assert target.width() < Fraction(1, 1 << 500)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "0.1"])
def test_malformed_scalar_inputs_are_rejected(value):
    with pytest.raises(proof.InvalidProofInput):
        proof.q(value)


def test_domains_are_refused_without_clipping():
    for value in (0, -1, Fraction(1, 1 << 129), 1 << 129):
        with pytest.raises(proof.ProofUnresolved):
            proof.log(value)
    with pytest.raises(proof.ProofUnresolved):
        proof.exp(17)
    with pytest.raises(proof.ProofUnresolved):
        proof.exp(proof.Complex(8, 9))
    with pytest.raises(proof.ProofUnresolved):
        proof.sqrt(proof.interval(Fraction(-1, 1 << 700), 1))
    with pytest.raises(proof.ProofUnresolved):
        proof.point(1)/proof.interval(-1, 1)
    with pytest.raises(proof.InvalidProofInput):
        proof.interval(2, 1)
    with pytest.raises(proof.InvalidProofInput):
        proof.verify((Fraction(1), Fraction(0)), 0)
    with pytest.raises(proof.InvalidProofInput):
        proof.verify((Fraction(0), Fraction(1)), 0, -1)


def test_fixed_integer_preflights_reject_inputs_products_and_grid_shifts():
    with pytest.raises(proof.ProofUnresolved):
        proof.point(1 << proof.INPUT_BITS)
    large = proof.point(1 << (proof.INPUT_BITS-1))
    with pytest.raises(proof.ProofUnresolved):
        large*large  # product fits, but its required grid shift would not
    ratio = Fraction((1 << (proof.INPUT_BITS-1))+1, (1 << (proof.INPUT_BITS-1))+3)
    with pytest.raises(proof.ProofUnresolved):
        proof.point(ratio)+proof.point(ratio)  # preflight cross-products plus carry
    with pytest.raises(proof.ProofUnresolved):
        proof._shift(1, proof.INTERMEDIATE_BITS)
    with pytest.raises(proof.ProofUnresolved):
        proof._multiply(Fraction(1 << (proof.INTERMEDIATE_BITS-1)), Fraction(2))


def test_cache_capacity_and_cache_hits_do_not_change_certificates(proof_context):
    first = proof.exp(Fraction(1, 4))
    operations = proof_context.operations
    assert proof.exp(Fraction(1, 4)) is first
    assert proof_context.operations == operations
    # Exercise storage capacity only, without thousands of transcendental calls.
    context = proof_context
    context._caches["capacity_fixture"] = {index: proof.point(index) for index in range(proof.CACHE_LIMIT)}
    context._remember("capacity_fixture", proof.CACHE_LIMIT, proof.point(0))
    assert len(context._caches["capacity_fixture"]) == proof.CACHE_LIMIT


def test_operation_exhaustion_uses_the_actual_fixed_limit(proof_context):
    proof_context._operations = proof.OPERATION_LIMIT
    with pytest.raises(proof.ProofUnresolved):
        proof.point(1)+proof.point(1)
    assert proof_context.operations == proof.OPERATION_LIMIT


def test_contexts_cannot_nest_or_reset_the_budget(proof_context):
    with pytest.raises(proof.InvalidProofInput):
        with proof.ProofContext():
            pass
    assert proof.current_context() is proof_context
