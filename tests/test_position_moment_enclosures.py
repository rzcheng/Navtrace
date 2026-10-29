"""Independent fixed laws for the rigorous position reference; no random draws.

Small conditioning targets use exact Fraction elimination. Endpoint targets are
polynomial/characteristic expansions written from physical coordinates, without
calling the new history builder or moment kernels. Runtime is owned by the
parent's preserved monitor; this module performs no study or artifact loading.
"""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
import math
import os
import struct

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pytest
import exact_target_bounds as proof

from tools import position_moment_enclosures as enclosure
from tools import moment_precision as precision
from tools import multistep_position_oracle as frozen
from flint import arb, acb, ctx

M0 = (.5, -.25, .2, 1.5, .1, -.02)
B1 = (.2, -.1, .3, .15, .04, -.02)
J = ((.2, 0., 0.), (.05, .15, 0.), (.04, -.02, .25))
LA = ((.3, 0., 0.), (.02, .08, 0.), (-.01, -.03, .04))
K0 = ((.4, .2, -.3), (-.2, .5, .1), (.3, -.4, .2))
RANK_EVENTS = ((10000, "gyro", .09, .02**2),
               (10000, "wheel", 1.4, .1**2),
               (30000, "gyro", .11, .02**2))


def q(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction.from_float(value)
    return Fraction(value)


def b(value):
    value = q(value)
    return arb(value.numerator)/arb(value.denominator)


def p(value):
    """Independent proof value; never pass it to an oracle constructor."""
    return value if isinstance(value, proof.Real) else proof.point(q(value))


def z(real=0, imag=0):
    return proof.Complex(p(real), p(imag))


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
    """Read the stored midpoint and radius exactly, without a target conversion."""
    assert value.is_finite()
    def dyadic(part):
        mantissa, exponent = part.man_exp()
        mantissa, exponent = int(mantissa), int(exponent)
        if abs(exponent) > 1000000:
            raise proof.ProofUnresolved("actual endpoint exceeds the frozen representation bound")
        return Fraction(mantissa << exponent) if exponent >= 0 else Fraction(mantissa, 1 << -exponent)
    midpoint, radius = dyadic(value.mid()), dyadic(value.rad())
    return midpoint-radius, midpoint+radius


def zeros(rows, columns):
    return [[Fraction(0) for _ in range(columns)] for _ in range(rows)]


def identity(size):
    matrix = zeros(size, size)
    for index in range(size):
        matrix[index][index] = Fraction(1)
    return matrix


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def dot(left, right):
    return sum((a*c for a, c in zip(left, right)), Fraction(0))


def product(left, right):
    columns = transpose(right)
    return [[dot(row, column) for column in columns] for row in left]


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def plus(left, right):
    return [[a+c for a, c in zip(row, other)] for row, other in zip(left, right)]


def minus(left, right):
    return [[a-c for a, c in zip(row, other)] for row, other in zip(left, right)]


def exact_inverse_and_determinant(matrix):
    """Small rational Gauss-Jordan target, independent of verified Arb LU."""
    count = len(matrix)
    rows = [list(row)+unit for row, unit in zip(matrix, identity(count))]
    determinant = Fraction(1)
    for column in range(count):
        pivot = next(index for index in range(column, count) if rows[index][column])
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        divisor = rows[column][column]
        determinant *= divisor
        rows[column] = [entry/divisor for entry in rows[column]]
        for row in range(count):
            if row != column:
                factor = rows[row][column]
                rows[row] = [entry-factor*other for entry, other in zip(rows[row], rows[column])]
    return [row[count:] for row in rows], determinant


def exact_condition(mean, covariance, observation, values, variances):
    count = len(values)
    if not count:
        return mean, covariance, p(0)
    cross = product(covariance, transpose(observation))
    innovation = product(observation, cross)
    for index, variance in enumerate(variances):
        innovation[index][index] += variance
    inverse, determinant = exact_inverse_and_determinant(innovation)
    gain = product(cross, inverse)
    residual = [value-predicted for value, predicted in zip(values, matvec(observation, mean))]
    conditioned_mean = [value+update for value, update in zip(mean, matvec(gain, residual))]
    conditioned_covariance = minus(covariance, product(gain, transpose(cross)))
    quadratic = dot(residual, matvec(inverse, residual))
    evidence = -(count*(2*proof.pi()).log()+p(determinant).log()+p(quadratic))/2
    return conditioned_mean, conditioned_covariance, evidence


def basis(size, index, scale=Fraction(1)):
    row = [Fraction(0)]*size
    row[index] = scale
    return row


def complex_dot(left, right):
    return sum((coefficient*p(value) for coefficient, value in zip(left, right)), z(0))


def term_mean(term, mean, covariance):
    """A term is (constant + linear dot G) exp(i frequency dot G)."""
    constant, linear, frequency = term
    phase_mean = dot(frequency, mean)
    phase_covariance = matvec(covariance, frequency)
    phi = z(-p(dot(frequency, phase_covariance))/2, p(phase_mean)).exp()
    tilted = constant+complex_dot(linear, mean)+z(0, 1)*complex_dot(linear, phase_covariance)
    return tilted*phi


def term_product(left, right, mean, covariance):
    a, linear_a, phase_a = left
    c, linear_c, phase_c = right
    phase = [x+y for x, y in zip(phase_a, phase_c)]
    cov_phase = matvec(covariance, phase)
    phi = z(-p(dot(phase, cov_phase))/2, p(dot(phase, mean))).exp()
    left_tilt = a+complex_dot(linear_a, mean)+z(0, 1)*complex_dot(linear_a, cov_phase)
    right_tilt = c+complex_dot(linear_c, mean)+z(0, 1)*complex_dot(linear_c, cov_phase)
    polynomial_covariance = sum((linear_a[i]*linear_c[j]*p(covariance[i][j])
        for i in range(len(mean)) for j in range(len(mean))), z(0))
    return (polynomial_covariance+left_tilt*right_tilt)*phi


def physical_terms(final_rows, old_heading_rows, old_speed_rows, steps):
    """Expand x/y as conjugate exponentials, preserving the full physical map."""
    dimension = len(final_rows[0])
    no_phase = [Fraction(0)]*dimension
    terms = [[(z(0), [z(value) for value in row], no_phase)] for row in final_rows]
    for heading, speed, step in zip(old_heading_rows, old_speed_rows, steps):
        for sign in (1, -1):
            phase = [sign*entry for entry in heading]
            terms[0].append((z(0), [z(step*entry/2) for entry in speed], phase))
            terms[1].append((z(0), [z(0, -sign*step*entry/2) for entry in speed], phase))
    return terms


def moments_of_terms(terms, mean, covariance):
    complex_means = [sum((term_mean(term, mean, covariance) for term in row), z(0)) for row in terms]
    for value in complex_means:
        assert value.imag.contains(0)
    means = [value.real for value in complex_means]
    covariances = []
    for row, left in enumerate(terms):
        result_row = []
        for column, right in enumerate(terms):
            raw = sum((term_product(a, c, mean, covariance) for a in left for c in right), z(0))
            assert raw.imag.contains(0)
            result_row.append(raw.real-means[row]*means[column])
        covariances.append(result_row)
    return means, covariances


def encode(value):
    if isinstance(value, (proof.Real, proof.Complex)):
        return proof.serialize(value)
    if isinstance(value, arb):
        return precision.serialize_ball(value)
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    return value


def emit(name, **values):
    print("enclosure_validation "+json.dumps(encode({"check": name, **values}), sort_keys=True, allow_nan=False))


def contained(actual, expected, *, absolute_width=Fraction(1, 10**10)):
    target = expected if isinstance(expected, proof.Real) else q(expected)
    receipt = proof.verify(actual_bounds(actual), target, width_limit=absolute_width)
    MEMBERSHIP_RECEIPTS.append(receipt)
    assert receipt["status"] == "proved", proof.serialize(receipt)
    return receipt


def check_result(name, result, target_mean, target_covariance, target_evidence):
    widths = []
    for actual, expected in zip(result.mean, target_mean):
        contained(actual, expected)
        low, high = precision.bounds(actual)
        widths.append(high-low)
    for row, expected_row in zip(result.covariance, target_covariance):
        for actual, expected in zip(row, expected_row):
            contained(actual, expected)
            low, high = precision.bounds(actual)
            widths.append(high-low)
    contained(result.log_evidence, target_evidence)
    emit(name, maximum_moment_width=max(widths), log_evidence=result.log_evidence,
         dimensions={"steps": result.diagnostics["step_count"],
                     "events": result.diagnostics["event_count"],
                     "physical": result.diagnostics["coordinate_count"],
                     "selected": result.diagnostics["marginal_count"]},
         support=result.support, whitening_allowed=result.whitening_allowed)
    for i in range(6):
        for j in range(6):
            assert precision.bounds(result.covariance[i][j]) == precision.bounds(result.covariance[j][i])
    if result.whitening_allowed:
        positive = precision.verified_cholesky(result.covariance, require_resolution=True)
        assert positive["resolution_proved"]
        emit(name+"_covariance_support", support="full", conditioning=positive)
    else:
        emit(name+"_covariance_support", support=result.support,
             independent_exact_law="declared physical Gaussian and polynomial characteristic target",
             whitening="withheld")


def rounded_sqrt(value):
    """A binary64 proposal verified against exact halfway-square boundaries."""
    if value == 0:
        return 0.
    root = math.sqrt(value)
    lo = (q(math.nextafter(root, -math.inf))+q(root))/2
    hi = (q(root)+q(math.nextafter(root, math.inf)))/2
    assert lo*lo < q(value) < hi*hi
    return root


def published_checkpoint():
    # Fixed old O covariance recipe; independent target conversion below records
    # every binary64 input exactly. No saved study payload is read.
    angular, linear = [2, 4, 5], [0, 1, 3]
    j, la, gain = np.array(J), np.array(LA), np.array(K0)
    mm = lambda a, c: np.einsum("ij,jk->ik", a, c, optimize=False)
    pa = mm(la, la.T)
    cross = mm(gain, pa)
    covariance = np.zeros((6, 6))
    covariance[np.ix_(angular, angular)] = pa
    covariance[np.ix_(linear, angular)] = cross
    covariance[np.ix_(angular, linear)] = cross.T
    covariance[np.ix_(linear, linear)] = mm(mm(gain, pa), gain.T)+mm(j, j.T)
    factor = np.zeros((6, 6))
    factor[angular, :3] = la
    factor[linear, 3:] = j
    for row, physical in enumerate(linear):
        for column in range(3):
            value = 0.
            for inner in range(3):
                value = value+K0[row][inner]*LA[inner][column]
            factor[physical, column] = value
    return np.array(M0), covariance, factor


def canonical_covariance(covariance):
    return [[q(.5*float(covariance[i][j])+.5*float(covariance[j][i]))
             for j in range(len(covariance))] for i in range(len(covariance))]


def factor_covariance(factor):
    factor = [[q(value) for value in row] for row in factor]
    return product(factor, transpose(factor)) if factor[0] else zeros(len(factor), len(factor))


def one_step_target(covariance):
    mean = [q(value) for value in M0]+[Fraction(0)]*3
    prior = zeros(9, 9)
    for i in range(6):
        prior[i][:6] = covariance[i]
    step = q(.01)
    for i, diffusion in enumerate((.04, .0004, .0001)):
        prior[6+i][6+i] = q(diffusion)*step
    observation = zeros(2, 9)
    for index in (4, 5, 7, 8):
        observation[0][index] = Fraction(1)
    for index in (3, 6):
        observation[1][index] = Fraction(1)
    mean, covariance, evidence = exact_condition(mean, prior, observation,
        [q(.09), q(1.4)], [q(.02**2), q(.1**2)])
    final = [basis(9, index) for index in range(6)]
    final[2][4] = step
    final[3][6] = final[4][7] = final[5][8] = Fraction(1)
    terms = physical_terms(final, [basis(9, 2)], [basis(9, 3)], [step])
    mean, covariance = moments_of_terms(terms, mean, covariance)
    return mean, covariance, evidence


def rank_target(events=RANK_EVENTS):
    # Independently conditioned scalar G; no matrix conditioner is used here.
    coefficients, residuals, variances = [], [], []
    for _, sensor, value, variance in events:
        coefficient = q(B1[4])+q(B1[5]) if sensor == "gyro" else q(B1[3])
        offset = q(M0[4])+q(M0[5]) if sensor == "gyro" else q(M0[3])
        coefficients.append(coefficient)
        residuals.append(q(value)-offset)
        variances.append(q(variance))
    scalar_precision = 1+sum((c*c/r for c, r in zip(coefficients, variances)), Fraction(0))
    information = sum((c*y/r for c, y, r in zip(coefficients, residuals, variances)), Fraction(0))
    scalar_variance = 1/scalar_precision
    scalar_mean = scalar_variance*information
    mean = [q(value)+q(loading)*scalar_mean for value, loading in zip(M0, B1)]
    covariance = [[q(a)*q(c)*scalar_variance for c in B1] for a in B1]
    quadratic = sum((y*y/r for y, r in zip(residuals, variances)), Fraction(0))-information**2/scalar_precision
    evidence = -(len(events)*(2*proof.pi()).log()
        +sum((p(r).log() for r in variances), p(0))+p(scalar_precision).log()+p(quadratic))/2
    steps = 0 if not events else (1 if len(events) <= 2 else 3)
    step = q(.01)
    final = identity(6)
    final[2][4] = steps*step
    headings, speeds = [], []
    for index in range(steps):
        row = basis(6, 2)
        row[4] = index*step
        headings.append(row)
        speeds.append(basis(6, 3))
    target_mean, target_covariance = moments_of_terms(
        physical_terms(final, headings, speeds, [step]*steps), mean, covariance)
    return target_mean, target_covariance, evidence


def rank_inputs(events=RANK_EVENTS):
    factor = np.array(B1)[:, None]
    return (np.array(M0), np.outer(B1, B1), factor, np.zeros(3), events)


def linear_target(alpha):
    # Independent primitives are (px0, py0, v0, dv0, dv1, dv2).
    step = q(.01)
    mean = [q(.5), q(-.25), q(1.5), Fraction(0), Fraction(0), Fraction(0)]
    covariance = zeros(6, 6)
    gram = factor_covariance(J)
    for i in range(3):
        covariance[i][:3] = gram[i]
    for i in range(3, 6):
        covariance[i][i] = q(.04)*step
    observation = zeros(4, 6)
    for index in (2, 3):
        observation[1][index] = Fraction(1)
    for index in (2, 3, 4, 5):
        observation[3][index] = Fraction(1)
    mean, covariance, evidence = exact_condition(mean, covariance, observation,
        [q(0), q(1.4), q(0), q(1.6)], [q(.02**2), q(.1**2), q(.02**2), q(.1**2)])
    direction = [p(alpha).cos(), p(alpha).sin()]
    rows = [[p(0) for _ in range(6)] for _ in range(6)]
    rows[0][0] = rows[1][1] = p(1)
    for coordinate in range(2):
        for column, multiplier in ((2, 3), (3, 2), (4, 1)):
            rows[coordinate][column] = p(multiplier*step)*direction[coordinate]
    for column in (2, 3, 4, 5):
        rows[3][column] = p(1)
    target_mean = [sum((coefficient*p(value) for coefficient, value in zip(row, mean)), p(0)) for row in rows]
    target_mean[2] = p(alpha)
    target_covariance = [[sum((rows[i][k]*rows[j][l]*p(covariance[k][l])
        for k in range(6) for l in range(6)), p(0)) for j in range(6)] for i in range(6)]
    return target_mean, target_covariance, evidence


def linear_inputs(alpha):
    mean = np.array([.5, -.25, alpha, 1.5, 0., 0.])
    factor = np.zeros((6, 3))
    factor[[0, 1, 3], :] = J
    covariance = np.einsum("ik,jk->ij", factor, factor, optimize=False)
    events = ((10000, "gyro", 0., .02**2), (10000, "wheel", 1.4, .1**2),
              (30000, "gyro", 0., .02**2), (30000, "wheel", 1.6, .1**2))
    return mean, covariance, factor, np.array([.04, 0., 0.]), events


def timing_target():
    step = q(.01)
    rate, speed = q(.1), q(1.5)
    prior = zeros(6, 6)
    for index, diffusion in enumerate((.04, .0004, .0001)*2):
        prior[index][index] = q(diffusion)*step
    observation = zeros(2, 6)
    for index in (1, 2, 4, 5):
        observation[0][index] = Fraction(1)
    for index in (0, 3):
        observation[1][index] = Fraction(1)
    mean, covariance, evidence = exact_condition([Fraction(0)]*6, prior, observation,
        [q(.09)-q(.1)-q(-.02), q(1.4)-q(1.5)], [q(.02**2), q(.1**2)])
    angle = p(step*rate)
    cosine, sine = angle.cos(), angle.sin()
    rows = [[p(0) for _ in range(6)] for _ in range(6)]
    rows[0][0], rows[1][0], rows[2][1] = p(step)*cosine, p(step)*sine, p(step)
    for coordinate, columns in ((3, (0, 3)), (4, (1, 4)), (5, (2, 5))):
        for column in columns:
            rows[coordinate][column] = p(1)
    offset = [p(step*speed)*(1+cosine), p(step*speed)*sine,
              p(2*step*rate), p(speed), p(rate), p(-.02)]
    target_mean = [constant+sum((coefficient*p(value) for coefficient, value in zip(row, mean)), p(0))
                   for constant, row in zip(offset, rows)]
    target_covariance = [[sum((rows[i][k]*rows[j][l]*p(covariance[k][l])
        for k in range(6) for l in range(6)), p(0)) for j in range(6)] for i in range(6)]
    return target_mean, target_covariance, evidence


def timing_inputs():
    return (np.array([0., 0., 0., 1.5, .1, -.02]), np.zeros((6, 6)), np.zeros((6, 0)),
            np.array([.04, .0004, .0001]),
            ((20000, "gyro", .09, .02**2), (20000, "wheel", 1.4, .1**2)))


def heading_inputs(variance, speed=2., heading=0., events=None):
    mean = np.array([0., 0., heading, speed, 0., 0.])
    factor = np.zeros((6, 1 if variance else 0))
    if variance:
        factor[2, 0] = rounded_sqrt(variance)
    covariance = np.zeros((6, 6))
    covariance[2, 2] = variance
    events = ((1000000, "gyro", 0., .02**2),) if events is None else events
    return mean, covariance, factor, np.zeros(3), events


def heading_target(variance, speed=2., heading=0., duration=None, event_count=1):
    duration = 100*q(.01) if duration is None else duration
    variance, distance = p(variance), p(q(speed)*duration)
    damping = (-variance/2).exp()
    cosine, sine = p(heading).cos(), p(heading).sin()
    rotation = [[cosine, -sine], [sine, cosine]]
    local_variances = [distance**2*(-(-variance).expm1())**2/2,
                       distance**2*(-(-2*variance).expm1())/2]
    means = [distance*damping*cosine, distance*damping*sine, p(heading), p(speed), p(0), p(0)]
    covariance = [[p(0) for _ in range(6)] for _ in range(6)]
    for i in range(2):
        for j in range(2):
            covariance[i][j] = sum((rotation[i][k]*rotation[j][k]*local_variances[k] for k in range(2)), p(0))
        covariance[i][2] = covariance[2][i] = rotation[i][1]*distance*variance*damping
    covariance[2][2] = variance
    evidence = -event_count*(2*proof.pi()*p(.02**2)).log()/2
    return means, covariance, evidence


def call_rank(events=RANK_EVENTS):
    return enclosure.condition_prefix(*rank_inputs(events), law="factor", support="declared_factor")


def test_empty_published_checkpoint_and_two_declared_laws():
    mean, covariance, factor = published_checkpoint()
    assembled_factor = enclosure.checkpoint_factor(J, LA, K0)
    for i in range(6):
        for j in range(6):
            assert struct.pack(">d", assembled_factor[i][j]) == struct.pack(">d", factor[i, j])
    emit("checkpoint_factor_assembly", ascending_scalar_binary64_bytes_match=True,
         factor=assembled_factor)
    canonical = enclosure.condition_prefix(mean, covariance, factor, [.04, .0004, .0001], [])
    factored = enclosure.condition_prefix(mean, covariance, factor, [.04, .0004, .0001], [], law="factor")
    check_result("empty_canonical", canonical, [q(value) for value in mean], canonical_covariance(covariance), Fraction(0))
    check_result("empty_factor", factored, [q(value) for value in mean], factor_covariance(factor), Fraction(0))
    assert canonical.whitening_allowed and factored.whitening_allowed
    bridge = [[factored.covariance[i][j]-canonical.covariance[i][j] for j in range(6)] for i in range(6)]
    exact_difference = minus(factor_covariance(factor), canonical_covariance(covariance))
    for i in range(6):
        for j in range(6):
            contained(bridge[i][j], exact_difference[i][j])
    emit("checkpoint_law_bridge", exact_difference=exact_difference, bridge=bridge)
    zero = enclosure.condition_prefix(mean, np.zeros((6, 6)), np.zeros((6, 0)), [0., 0., 0.], [],
                                      support="deterministic")
    check_result("empty_deterministic", zero, [q(value) for value in mean], zeros(6, 6), Fraction(0))
    assert not zero.whitening_allowed


@pytest.mark.parametrize("law", ["canonical", "factor"])
def test_published_one_step_against_independent_nine_dimensional_physical_law(law):
    mean, covariance, factor = published_checkpoint()
    inputs = (mean, covariance, factor, [.04, .0004, .0001], RANK_EVENTS[:2])
    exact_covariance = canonical_covariance(covariance) if law == "canonical" else factor_covariance(factor)
    result = enclosure.condition_prefix(*inputs, law=law)
    check_result("one_step_"+law, result, *one_step_target(exact_covariance))
    assert result.timestamp_us == 10000
    assert result.diagnostics["step_count"] == 1
    if law == "factor":
        old = frozen.condition_prefix(*inputs)
        assert max(abs(float(actual.mid())-value) for actual, value in zip(result.mean, old.mean)) <= 1e-10
        assert max(abs(float(result.covariance[i][j].mid())-old.covariance[i, j])
                   for i in range(6) for j in range(6)) <= 1e-10
        assert abs(float(result.log_evidence.mid())-old.log_evidence) <= 1e-10
        emit("frozen_one_step_bridge", old_mean=old.mean.tolist(), old_covariance=old.covariance.tolist(),
             old_log_evidence=old.log_evidence)
        canonical = enclosure.condition_prefix(*inputs, law="canonical")
        check_result("one_step_certificate_canonical", canonical,
                     *one_step_target(canonical_covariance(covariance)))
        assert canonical.model_law == "canonical" and result.model_law == "factor"
        for field in ("checkpoint_factor", "prior_mean", "process_variances", "step_sizes_s", "events", "checkpoint_us"):
            assert canonical.physical[field] == result.physical[field]
        assert canonical.diagnostics["checkpoint_law_difference"] == result.diagnostics["checkpoint_law_difference"]
        certification = precision.certify_oracle(old.mean, old.covariance, old.log_evidence,
            canonical, result, point_status=old.status,
            point_diagnostics={"precision_certified": old.diagnostics["precision_certified"],
                "underflow_counts": dict(old.diagnostics["displacement"]["underflow_counts"]),
                "dimensions": dict(old.diagnostics["dimensions"])})
        assert certification["status"] == "eligible"
        assert precision.bounds(certification["alpha"])[0] > 0
        for field in ("mean_bound", "covariance_bound"):
            assert precision.bounds(certification[field])[1] <= q(1e-7)
            assert precision.bounds(certification["reference_"+field])[1] <= q(1e-10)
            assert precision.bounds(certification[field])[1] >= precision.bounds(certification["direct_total"][field])[1]
            assert precision.bounds(certification[field])[1] >= precision.bounds(certification["summed_"+field])[1]
        assert precision.bounds(certification["evidence_bound"])[1] <= q(1e-10)
        for contribution in ("arithmetic", "law_bridge", "direct_total"):
            for field in ("mean_bound", "covariance_bound", "evidence_bound"):
                assert precision.bounds(certification[contribution][field])[0] >= 0
        emit("complete_one_step_certificate", result=certification,
             checkpoint_factor=canonical.physical["checkpoint_factor"],
             checkpoint_law_difference=canonical.diagnostics["checkpoint_law_difference"],
             step_sizes_s=canonical.physical["step_sizes_s"], timestamp_us=canonical.timestamp_us)


@pytest.mark.parametrize("heading", [0., math.pi/2])
def test_small_linear_physical_joint_law(heading):
    result = enclosure.condition_prefix(*linear_inputs(heading), law="factor", support="declared_factor")
    check_result(f"linear_heading_{heading}", result, *linear_target(heading))
    assert not result.whitening_allowed


@pytest.mark.parametrize("count", [1, 2, 3])
def test_scalar_conditioned_history_complete_six_state(count):
    check_result(f"rank_one_prefix_{count}", call_rank(RANK_EVENTS[:count]), *rank_target(RANK_EVENTS[:count]))


def test_end_step_noise_uses_old_heading_and_speed():
    result = enclosure.condition_prefix(*timing_inputs(), support="deterministic")
    check_result("two_step_physical_timing", result, *timing_target())
    assert result.diagnostics["step_count"] == 2
    assert result.diagnostics["event_count"] == 2


@pytest.mark.parametrize("variance", [0., .4, 1e-8, 1e-12])
@pytest.mark.parametrize("law", ["canonical", "factor"])
def test_shared_heading_hundred_steps_with_stable_narrow_targets(variance, law):
    inputs = heading_inputs(variance)
    support = "diagonal" if law == "canonical" else "declared_factor"
    result = enclosure.condition_prefix(*inputs, law=law, support=support)
    exact_variance = q(variance) if law == "canonical" else q(rounded_sqrt(variance))**2
    target = heading_target(exact_variance)
    check_result(f"shared_heading_{variance}_{law}", result, *target)
    assert not result.whitening_allowed
    assert result.diagnostics["step_count"] == 100
    assert result.diagnostics["coordinate_count"] == 306
    assert len(result.physical["step_sizes_s"]) == 100
    if variance:
        for axis in (0, 1):
            lower_target = target[1][axis][axis].lo
            assert lower_target > 0
            relative_width = precision.width(result.covariance[axis][axis])/lower_target
            assert relative_width <= q(1e-10)
            emit("narrow_enclosure", variance=variance, model_law=law, axis=axis,
                 exact_variance=exact_variance, covariance=result.covariance[axis][axis],
                 target=target[1][axis][axis], relative_width=relative_width, criterion=1e-10)
    if law == "canonical" and variance in (1e-8, 1e-12):
        old = frozen.condition_prefix(*inputs)
        error = (p(old.covariance[0, 0])-target[1][0][0])/target[1][0][0]
        error_lower, error_upper = error.abs_bounds()
        expected_status = "resolved" if variance == 1e-8 else "unresolved"
        if variance == 1e-8:
            assert error_upper <= q(1e-7)
        else:
            assert error_lower > q(1e-7)
        emit("historical_oracle_narrow_control", variance=variance, actual=old.covariance[0, 0],
             relative_error=error, status=expected_status, criterion=1e-7,
             historical_result_unchanged=True)


def test_shared_heading_partitions_use_their_exact_binary_substeps():
    first_events = ((990000, "gyro", 0., .02**2),)
    second_events = ((333333, "gyro", 0., .02**2), (990000, "gyro", 0., .02**2))
    first_h = (990000*1e-6)/99
    second_h = ((333333*1e-6)/34, (656667*1e-6)/66)
    durations = (99*q(first_h), 34*q(second_h[0])+66*q(second_h[1]))
    results = []
    for events, duration, count in zip((first_events, second_events), durations, (1, 2)):
        result = enclosure.condition_prefix(*heading_inputs(.4, events=events), support="diagonal")
        check_result(f"partition_{len(results)}", result,
                     *heading_target(q(.4), duration=duration, event_count=count))
        results.append(result)
    assert results[0].diagnostics["step_count"] == 99
    assert results[1].diagnostics["step_count"] == 100
    for left, right in zip(results[0].mean, results[1].mean):
        assert precision.bounds(precision.abs_range(left-right))[1] <= q(1e-10)
    for i in range(6):
        for j in range(6):
            assert precision.bounds(precision.abs_range(results[0].covariance[i][j]-results[1].covariance[i][j]))[1] <= q(1e-10)
    emit("partition_exact_durations", durations=durations, difference=durations[1]-durations[0])


def test_received_prefix_outputs_are_immutable_and_causal():
    first = call_rank(RANK_EVENTS[:1])
    snapshot = json.dumps(encode((first.mean, first.covariance, first.log_evidence)), sort_keys=True)
    second, final, repeat = call_rank(RANK_EVENTS[:2]), call_rank(), call_rank(RANK_EVENTS[:1])
    assert json.dumps(encode((first.mean, first.covariance, first.log_evidence)), sort_keys=True) == snapshot
    assert json.dumps(encode((repeat.mean, repeat.covariance, repeat.log_evidence)), sort_keys=True) == snapshot
    assert any(precision.bounds(precision.abs_range(a-c))[0] > q(1e-10) for a, c in zip(first.mean, second.mean))
    assert any(precision.bounds(precision.abs_range(a-c))[0] > q(1e-10) for a, c in zip(second.mean, final.mean))
    with pytest.raises(TypeError):
        first.mean[0] = arb(0)
    with pytest.raises(TypeError):
        first.covariance[0][0] = arb(0)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        first.log_evidence = arb(0)
    with pytest.raises(TypeError):
        first.diagnostics["changed"] = True
    with pytest.raises(TypeError):
        first.physical["prior_covariance"][0][0] = 0
    emit("causal_prefixes", first=first.mean, second=second.mean, final=final.mean)


@pytest.mark.parametrize("speed", [-2., 0., 2.])
def test_signed_zero_speed_and_unwrapped_heading_limits(speed):
    events = ((30000, "gyro", 0., .02**2),)
    results = []
    for heading in (0., 2*math.pi):
        result = enclosure.condition_prefix(*heading_inputs(.4, speed, heading, events), support="diagonal")
        check_result(f"signed_speed_{speed}_heading_{heading}", result,
                     *heading_target(q(.4), speed, heading, duration=3*q(.01)))
        results.append(result)
    contained(results[1].mean[2]-results[0].mean[2], q(2*math.pi))
    if speed == 0:
        for row in results[0].covariance[:2]:
            for value in row:
                contained(value, 0)


def scalar_parameters(events=RANK_EVENTS):
    precision_value, information = Fraction(1), Fraction(0)
    for timestamp, sensor, value, variance in events:
        coefficient = q(B1[4])+q(B1[5]) if sensor == "gyro" else q(B1[3])
        offset = q(M0[4])+q(M0[5]) if sensor == "gyro" else q(M0[3])
        precision_value += coefficient**2/q(variance)
        information += coefficient*(q(value)-offset)/q(variance)
    return information/precision_value, 1/precision_value


def test_raw_gaussian_identities_against_characteristic_derivatives():
    import cmath
    mean, variance = scalar_parameters()
    step, constant_phase = 1e-6, .2
    phi = lambda t: cmath.exp(1j*float(mean)*t-.5*float(variance)*t*t)
    derivative = lambda t: (1j*float(mean)-float(variance)*t)*phi(t)
    for t in (0., .3, .3008):
        phase_mean = q(constant_phase)+q(t)*mean
        phase_variance = q(t)**2*variance
        phase_cross = q(t)*variance
        first = enclosure.raw_gaussian_first(b(mean), b(phase_cross), b(phase_mean), b(phase_variance))
        second = enclosure.raw_gaussian_second(b(mean), b(mean), b(variance), b(phase_cross),
                                               b(phase_cross), b(phase_mean), b(phase_variance))
        independent_first = term_mean((z(0), [z(1)], [q(t)]), [mean], [[variance]])*z(0, q(constant_phase)).exp()
        independent_second = term_product((z(0), [z(1)], [q(t)]), (z(0), [z(1)], [q(0)]),
                                          [mean], [[variance]])*z(0, q(constant_phase)).exp()
        for actual, target in ((first, independent_first), (second, independent_second)):
            contained(actual.real, target.real)
            contained(actual.imag, target.imag)
        first_difference = (phi(t+step)-phi(t-step))/(2*step)
        second_difference = (derivative(t+step)-derivative(t-step))/(2*step)
        expected_first = cmath.exp(1j*constant_phase)*first_difference/1j
        expected_second = -cmath.exp(1j*constant_phase)*second_difference
        first_point = complex(float(first.real.mid()), float(first.imag.mid()))
        second_point = complex(float(second.real.mid()), float(second.imag.mid()))
        assert abs(first_point-expected_first) <= 1e-7+1e-7*abs(expected_first)
        assert abs(second_point-expected_second) <= 1e-7+1e-7*abs(expected_second)
        emit("raw_characteristic_derivatives", frequency=t, first_error=abs(first_point-expected_first),
             second_error=abs(second_point-expected_second), step=step, absolute_relative_tolerance=1e-7)


def literal_pair(zero_means=False):
    left = (b(0), b(0 if zero_means else 1.5))
    right = (b(math.pi/2), b(0 if zero_means else -.5))
    left_covariance = ((b(0), b(0)), (b(0), b(q(.2)**2)))
    right_covariance = ((b(0), b(0)), (b(0), b(q(.05)**2+q(math.sqrt(.0875))**2)))
    cross = ((b(0), b(0)), (b(0), b(q(.2)*q(.05))))
    return left, right, left_covariance, right_covariance, cross


@pytest.mark.parametrize("zero_means", [False, True])
def test_literal_pair_signs_zero_mean_speed_and_diagonal_limits(zero_means):
    inputs = literal_pair(zero_means)
    mean_product = q(0 if zero_means else 1.5)*q(0 if zero_means else -.5)
    for sign in (1, -1):
        result = enclosure.raw_displacement_pair(*inputs, sign)
        target = z(mean_product+q(.2)*q(.05))*z(0, sign*q(math.pi/2)).exp()
        contained(result.real, target.real)
        contained(result.imag, target.imag)
        swapped = enclosure.raw_displacement_pair(inputs[1], inputs[0], inputs[3], inputs[2],
                                                  tuple(zip(*inputs[4])), sign)
        comparison = swapped if sign == 1 else swapped.conjugate()
        assert (result-comparison).real.contains(0)
        assert (result-comparison).imag.contains(0)
    diagonal = enclosure.raw_displacement_pair(inputs[0], inputs[0], inputs[2], inputs[2], inputs[2], -1)
    contained(diagonal.real, q(0 if zero_means else 1.5)**2+q(.2)**2)
    contained(diagonal.imag, 0)
    emit("literal_pair", zero_speed_means=zero_means, exact_cross=q(.2)*q(.05))


@pytest.mark.parametrize("events", [
    ((10000, "wheel", 1.4, .01), (10000, "gyro", .09, .0004)),
    ((10000, "gyro", .09, .0004), (10000, "gyro", .09, .0004)),
    ((20000, "gyro", .09, .0004), (10000, "wheel", 1.4, .01)),
    ((0, "gyro", .09, .0004),), ((10000, "gps", .09, .0004),),
    ((10000., "gyro", .09, .0004),), ((True, "gyro", .09, .0004),),
    ((10000, "gyro", math.nan, .0004),), ((10000, "gyro", .09, 0.),),
    ((10000, "gyro", .09, -.0004),),
])
def test_invalid_sensor_prefixes_are_rejected(events):
    with pytest.raises(precision.InvalidInput) as error:
        call_rank(events)
    assert error.value.status == "invalid"


def test_declared_support_and_innovation_domains_refuse_without_repair():
    inputs = rank_inputs()
    with pytest.raises(precision.UnresolvedArithmetic):
        enclosure.condition_prefix(*inputs, law="factor", support="full")
    with pytest.raises(precision.InvalidInput):
        enclosure.condition_prefix(*inputs, law="canonical", support="declared_factor")
    with pytest.raises(precision.InvalidInput):
        enclosure.condition_prefix(*inputs, support="diagonal")
    with pytest.raises(precision.InvalidInput):
        enclosure.condition_prefix(*inputs, support="deterministic")
    zero = (np.zeros(6), np.zeros((6, 6)), np.zeros((6, 0)), [0., 0., 0.])
    with pytest.raises(precision.UnresolvedArithmetic) as unresolved:
        enclosure.condition_prefix(*zero, [(10000, "gyro", 0., 1e-14)], support="deterministic")
    assert unresolved.value.status == "unresolved"
    emit("support_refusals", rank_one_full="unresolved", innovation="unresolved")


def test_fixed_bounds_reject_before_unbounded_consumption():
    count = 0
    def many_events():
        nonlocal count
        for timestamp in range(1, 1000000):
            count += 1
            if count > 33:
                pytest.fail("event source consumed beyond the frozen lookahead")
            yield timestamp, "gyro", 0., .0004
    inputs = (np.zeros(6), np.zeros((6, 6)), np.zeros((6, 0)), [0., 0., 0.])
    with pytest.raises(precision.InvalidInput):
        enclosure.condition_prefix(*inputs, many_events(), support="deterministic")
    assert count <= 33
    for timestamp in (1000001, 2000001):
        with pytest.raises(precision.InvalidInput):
            enclosure.condition_prefix(*inputs, [(timestamp, "gyro", 0., .0004)], support="deterministic")
    class Unbounded:
        def __array__(self, *args, **kwargs):
            pytest.fail("unbounded input coerced before shape validation")
    with pytest.raises(precision.InvalidInput):
        enclosure.condition_prefix(Unbounded(), np.zeros((6, 6)), np.zeros((6, 0)), [0., 0., 0.], [])
    with pytest.raises(precision.InvalidInput):
        enclosure.condition_prefix(np.zeros(6), np.zeros((6, 6)), Unbounded(), [0., 0., 0.], [])
    emit("bounded_input_rejection", event_items_consumed=count, maximum_events=32, maximum_steps=100)


def test_raw_pair_correlations_and_conjugate_limits_from_scalar_conditioning():
    scalar_mean, scalar_variance = scalar_parameters()
    pairs, coefficients, terms = [], [], []
    for time in (Fraction(0), q(.01), 2*q(.01)):
        angle_offset = q(M0[2])+time*q(M0[4])
        angle_coefficient = q(B1[2])+time*q(B1[4])
        speed_coefficient = q(B1[3])
        pairs.append((b(angle_offset+angle_coefficient*scalar_mean),
                      b(q(M0[3])+speed_coefficient*scalar_mean)))
        coefficients.append((angle_coefficient, speed_coefficient))
        phase = z(0, angle_offset).exp()
        terms.append((phase*p(M0[3]), [phase*p(speed_coefficient)], [angle_coefficient]))
    for j in range(3):
        for k in range(3):
            left = tuple(tuple(b(a*c*scalar_variance) for c in coefficients[j]) for a in coefficients[j])
            right = tuple(tuple(b(a*c*scalar_variance) for c in coefficients[k]) for a in coefficients[k])
            cross = tuple(tuple(b(a*c*scalar_variance) for c in coefficients[k]) for a in coefficients[j])
            for sign in (1, -1):
                right_term = terms[k] if sign == 1 else (
                    terms[k][0].conjugate(), [value.conjugate() for value in terms[k][1]],
                    [-value for value in terms[k][2]])
                if j == k and sign == -1:
                    # The exact total phase is theta_j-theta_j=0. The product
                    # is v_j^2, independent of the numerical output interval.
                    square = (q(M0[3])+q(B1[3])*scalar_mean)**2+q(B1[3])**2*scalar_variance
                    expected = z(square)
                else:
                    expected = term_product(terms[j], right_term, [scalar_mean], [[scalar_variance]])
                result = enclosure.raw_displacement_pair(pairs[j], pairs[k], left, right, cross, sign)
                contained(result.real, expected.real)
                contained(result.imag, expected.imag)
                swapped = enclosure.raw_displacement_pair(pairs[k], pairs[j], right, left, tuple(zip(*cross)), sign)
                difference = result-(swapped if sign == 1 else swapped.conjugate())
                assert difference.real.contains(0) and difference.imag.contains(0)
            if j == k:
                expected_square = (q(M0[3])+q(B1[3])*scalar_mean)**2+q(B1[3])**2*scalar_variance
                contained(result.real, expected_square)  # last sign is conjugated minus.
                contained(result.imag, 0)
    emit("scalar_pair_limits", steps=3, ordered_pairs=9, complete_correlations=True)


def test_phase_variance_enclosure_crossing_zero_is_retained():
    radius = Fraction(1, 2**100)
    uncertain = precision.interval(-radius, radius)
    value = enclosure.raw_gaussian_first(1, 0, 0, uncertain)
    contained(value.real, p(-radius/2).exp(), absolute_width=q(1e-10))
    contained(value.real, p(radius/2).exp(), absolute_width=q(1e-10))
    assert precision.width(value.real) > 0
    assert precision.width(value.real) <= q(1e-10)
    contained(value.imag, 0)
    with pytest.raises(precision.UnresolvedArithmetic):
        enclosure.raw_gaussian_first(1, 0, 0, -1)
    emit("phase_interval_preserved", phase_variance=uncertain, result_real=value.real)


def test_requested_direction_includes_every_cross_block():
    result = call_rank()
    target = rank_target()[1]
    for direction in ((1., 0., 0., 0., 0., 0.), (1., -.3, .2, -.4, .5, .1)):
        actual = sum((b(direction[i])*b(direction[j])*result.covariance[i][j]
                      for i in range(6) for j in range(6)), arb(0))
        expected = sum((p(direction[i])*p(direction[j])*target[i][j]
                        for i in range(6) for j in range(6)), p(0))
        contained(actual, expected)
        emit("fixed_direction", direction=direction, actual=actual, independent_target=expected,
             full_six_whitening="withheld_declared_singular_support")


@pytest.mark.parametrize("kind", ["mean_shape", "covariance_shape", "factor_shape", "factor_width",
                                  "nonfinite_mean", "negative_diffusion", "material_asymmetry"])
def test_invalid_checkpoint_shapes_and_values_are_rejected(kind):
    mean, covariance, factor = published_checkpoint()
    diffusion = np.zeros(3)
    if kind == "mean_shape":
        mean = np.zeros(7)
    elif kind == "covariance_shape":
        covariance = np.zeros((6, 5))
    elif kind == "factor_shape":
        factor = np.zeros((5, 6))
    elif kind == "factor_width":
        factor = np.zeros((6, 7))
    elif kind == "nonfinite_mean":
        mean[0] = math.inf
    elif kind == "negative_diffusion":
        diffusion[0] = -.01
    elif kind == "material_asymmetry":
        covariance[0, 1] += .1
    with pytest.raises(precision.InvalidInput):
        enclosure.condition_prefix(mean, covariance, factor, diffusion, [])


def test_exact_positive_process_variance_is_not_binary64_underflowed():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    result = enclosure.condition_prefix(np.zeros(6), np.zeros((6, 6)), np.zeros((6, 0)),
        [tiny, 0., 0.], [(1, "gyro", 0., .0004)], support="deterministic")
    exact_variance = q(tiny)*q(1e-6)
    contained(result.covariance[3][3], exact_variance)
    assert precision.bounds(result.covariance[3][3])[0] > 0
    rounding = result.physical["diffusion_rounding"][0]
    assert rounding["rounded_qh"] == 0.
    assert rounding["difference"] == -exact_variance
    emit("positive_process_underflow_audit", exact_variance=exact_variance,
         result=result.covariance[3][3], old_factor_arithmetic=rounding)


def mutation_detected(name, invoke, expected):
    try:
        result = invoke()
    except precision.UnresolvedArithmetic as error:
        emit("mutation", mutation=name, detected=True, mechanism="arithmetic_invariant_refusal", reason=str(error))
        return
    differences = [proof.mutation_error(actual_bounds(actual), target)
                   for actual, target in zip(result.mean, expected[0])]
    differences += [proof.mutation_error(actual_bounds(result.covariance[i][j]), expected[1][i][j])
                    for i in range(6) for j in range(6)]
    differences += [proof.mutation_error(actual_bounds(result.log_evidence), expected[2])]
    lower_errors = [value.abs_bounds()[0] for value in differences]
    assert max(lower_errors) > q(1e-10), f"{name} escaped the independent fixed fixture"
    emit("mutation", mutation=name, detected=True, mechanism="independent_fixture_difference",
         maximum_certified_absolute_error=max(lower_errors), differences=differences)


def test_drop_cross_step_mutation_is_detected_and_restored(monkeypatch):
    target = rank_target()
    check_result("cross_step_mutation_baseline", call_rank(), *target)
    original = enclosure._raw_pair
    def diagonal_only(left_mean, right_mean, left_covariance, right_covariance, cross_covariance, sign, **kwargs):
        if tuple(map(precision.bounds, left_mean)) != tuple(map(precision.bounds, right_mean)):
            return acb(0)
        return original(left_mean, right_mean, left_covariance, right_covariance, cross_covariance, sign, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(enclosure, "_raw_pair", diagonal_only)
        mutation_detected("drop_different_steps", call_rank, target)
    assert enclosure._raw_pair is original
    check_result("cross_step_mutation_restored", call_rank(), *target)


def test_minus_phase_conjugation_mutation_is_detected_and_restored(monkeypatch):
    original = enclosure._raw_pair
    inputs = literal_pair()
    expected = z(q(1.5)*q(-.5)+q(.2)*q(.05))*z(0, -q(math.pi/2)).exp()
    baseline = enclosure.raw_displacement_pair(*inputs, -1)
    contained(baseline.real, expected.real)
    contained(baseline.imag, expected.imag)
    def wrong_conjugation(left_mean, right_mean, left_covariance, right_covariance, cross_covariance, sign, **kwargs):
        return original(left_mean, right_mean, left_covariance, right_covariance, cross_covariance, 1, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(enclosure, "_raw_pair", wrong_conjugation)
        changed = enclosure.raw_displacement_pair(*inputs, -1)
        difference = proof.mutation_error(actual_bounds(changed.imag), expected.imag)
        lower_error = difference.abs_bounds()[0]
        assert lower_error > q(1e-10)
        emit("mutation", mutation="minus_phase_conjugation", detected=True,
             mechanism="literal_pair_sign", certified_imaginary_error=lower_error, difference=difference)
    assert enclosure._raw_pair is original
    restored = enclosure.raw_displacement_pair(*inputs, -1)
    contained(restored.real, expected.real)
    contained(restored.imag, expected.imag)


@pytest.mark.parametrize("omit", ["initial_position", "final_latent"])
def test_cross_covariance_omission_mutations_are_detected_and_restored(monkeypatch, omit):
    original = enclosure._raw_cross
    target = rank_target()
    check_result("cross_mutation_baseline_"+omit, call_rank(), *target)
    def missing_cross(mu_x, mu_theta, mu_v, cov_x_theta, cov_x_v, cov_v_theta, var_theta, *, output_index=None):
        selected = output_index < 2 if omit == "initial_position" else output_index >= 2
        if selected:
            # Retain the raw mean product while dropping only the covariance.
            return mu_x*enclosure.raw_gaussian_first(mu_v, cov_v_theta, mu_theta, var_theta)
        return original(mu_x, mu_theta, mu_v, cov_x_theta, cov_x_v, cov_v_theta,
                        var_theta, output_index=output_index)
    with monkeypatch.context() as patch:
        patch.setattr(enclosure, "_raw_cross", missing_cross)
        mutation_detected("omit_"+omit+"_cross", call_rank, target)
    assert enclosure._raw_cross is original
    check_result("cross_mutation_restored_"+omit, call_rank(), *target)


def test_double_conditioning_mutation_is_detected_and_restored(monkeypatch):
    original = enclosure._condition_marginals
    target = rank_target()
    check_result("double_conditioning_baseline", call_rank(), *target)
    def twice(prior_mean, prior_covariance, selector, observation_rows, values, variances):
        # Two conditionings on the same received values equal one conditioning
        # on two copies with independent stated R. This stays at six reports.
        return original(prior_mean, prior_covariance, selector,
                        tuple(observation_rows)*2, tuple(values)*2, tuple(variances)*2)
    with monkeypatch.context() as patch:
        patch.setattr(enclosure, "_condition_marginals", twice)
        mutation_detected("condition_twice", call_rank, target)
    assert enclosure._condition_marginals is original
    check_result("double_conditioning_restored", call_rank(), *target)


def test_new_state_displacement_mutation_is_detected_and_restored(monkeypatch):
    original = enclosure._old_step_rows
    target = timing_target()
    invoke = lambda: enclosure.condition_prefix(*timing_inputs(), support="deterministic")
    check_result("new_state_mutation_baseline", invoke(), *target)
    def new_state(physical):
        return tuple(zip(physical["theta_rows"][1:], physical["speed_rows"][1:]))
    with monkeypatch.context() as patch:
        patch.setattr(enclosure, "_old_step_rows", new_state)
        mutation_detected("new_state_displacement", invoke, target)
    assert enclosure._old_step_rows is original
    check_result("new_state_mutation_restored", invoke(), *target)
