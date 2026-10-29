"""Fixed independent checks of physical position moments; no random draws.

Expected laws below are scalar characteristic derivatives or explicitly written
small linear maps. They do not use the new oracle's schedule, conditioner,
pair kernel, internal history, or a candidate posterior to calculate targets.
The JSON records are retained by the parent's monitored ``pytest -s`` run.
"""
from dataclasses import FrozenInstanceError
import cmath
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from tools import multistep_position_oracle as oracle

ATOL = 1e-10
M0 = np.array([.5, -.25, .2, 1.5, .1, -.02])
B1 = np.array([.2, -.1, .3, .15, .04, -.02])
J = np.array([[.2, 0., 0.], [.05, .15, 0.], [.04, -.02, .25]])
LA = np.array([[.3, 0., 0.], [.02, .08, 0.], [-.01, -.03, .04]])
K0 = np.array([[.4, .2, -.3], [-.2, .5, .1], [.3, -.4, .2]])
RANK_EVENTS = ((10000, "gyro", .09, .02**2),
               (10000, "wheel", 1.4, .1**2),
               (30000, "gyro", .11, .02**2))


def mm(left, right):
    return np.einsum("ij,jk->ik", left, right, optimize=False)


def mv(left, right):
    return np.einsum("ij,j->i", left, right, optimize=False)


def emit(name, **values):
    print("oracle_validation " + json.dumps({"check": name, **values}, sort_keys=True, allow_nan=False))


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0.)


def check_result(name, result, expected_mean, expected_covariance, expected_evidence):
    errors = {
        "mean_max_abs_error": float(np.max(np.abs(result.mean - expected_mean))),
        "covariance_max_abs_error": float(np.max(np.abs(result.covariance - expected_covariance))),
        "log_evidence_abs_error": abs(float(result.log_evidence - expected_evidence)),
        "symmetry_max_abs": float(np.max(np.abs(result.covariance - result.covariance.T))),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(result.covariance)[0]),
    }
    emit(name, **errors)
    close(result.mean, expected_mean)
    close(result.covariance, expected_covariance)
    close(result.log_evidence, expected_evidence)
    assert np.isfinite(result.mean).all()
    assert np.isfinite(result.covariance).all()
    scale = max(1., float(np.max(np.abs(result.covariance))))
    assert errors["symmetry_max_abs"] <= ATOL * scale
    assert errors["minimum_eigenvalue"] >= -ATOL * scale


def published_checkpoint():
    """Two independently written constructions of the published O prior."""
    angular, linear = [2, 4, 5], [0, 1, 3]
    pa = mm(LA, LA.T)
    cross = mm(K0, pa)
    p = np.zeros((6, 6))
    p[np.ix_(angular, angular)] = pa
    p[np.ix_(linear, angular)] = cross
    p[np.ix_(angular, linear)] = cross.T
    p[np.ix_(linear, linear)] = mm(mm(K0, pa), K0.T) + mm(J, J.T)
    factor = np.zeros((6, 6))
    factor[angular, :3] = LA
    factor[linear, :3] = mm(K0, LA)
    factor[linear, 3:] = J
    return M0.copy(), p, factor


def call_rank(events=RANK_EVENTS):
    factor = B1[:, None]
    return oracle.condition_prefix(M0, np.outer(B1, B1), factor, np.zeros(3), events)


def scalar_posterior(events):
    """Rank-one Gaussian law, using only scalar precision sums."""
    coefficients, residuals, variances = [], [], []
    for _, sensor, value, variance in events:
        coefficient = B1[4] + B1[5] if sensor == "gyro" else B1[3]
        offset = M0[4] + M0[5] if sensor == "gyro" else M0[3]
        coefficients.append(coefficient)
        residuals.append(value - offset)
        variances.append(variance)
    precision = 1. + math.fsum(c*c/r for c, r in zip(coefficients, variances))
    information = math.fsum(c*y/r for c, y, r in zip(coefficients, residuals, variances))
    variance = 1./precision
    mean = variance*information
    quadratic = math.fsum(y*y/r for y, r in zip(residuals, variances)) - information*information/precision
    log_evidence = -.5*(len(events)*math.log(2*math.pi)
        + math.fsum(math.log(r) for r in variances) + math.log(precision) + quadratic)
    return mean, variance, log_evidence


def scalar_characteristic_moments(t, mean, variance):
    """E[exp(itG)], E[G exp(itG)], E[G^2 exp(itG)], G Gaussian."""
    phi = cmath.exp(1j*mean*t - .5*variance*t*t)
    tilted = mean + 1j*variance*t
    return phi, tilted*phi, (variance + tilted*tilted)*phi


def term_expectation(terms, mean, variance):
    # A term (a,b,t) denotes (a+bG) exp(itG). No motion-pair identity is used.
    values = []
    for a, b, t in terms:
        moment0, moment1, _ = scalar_characteristic_moments(t, mean, variance)
        values.append(a*moment0+b*moment1)
    return complex(math.fsum(value.real for value in values), math.fsum(value.imag for value in values))


def term_product_expectation(left, right, mean, variance):
    values = []
    for a, b, t in left:
        for c, d, u in right:
            moment0, moment1, moment2 = scalar_characteristic_moments(t+u, mean, variance)
            values.append(a*c*moment0+(a*d+b*c)*moment1+b*d*moment2)
    return complex(math.fsum(value.real for value in values), math.fsum(value.imag for value in values))


def rank_target(events=RANK_EVENTS):
    """Explicit scalar path expansion at the three fixed received prefixes."""
    mean_g, variance_g, log_evidence = scalar_posterior(events)
    # The fixed events produce exactly these old physical boundaries. Do not
    # infer them with the implementation's schedule helper.
    old_times = () if not events else ((0.,) if len(events) <= 2 else (0., .01, .02))
    end_time = 0. if not events else (.01 if len(events) <= 2 else .03)
    terms = [[(complex(M0[k]), complex(B1[k]), 0.)] for k in range(6)]
    terms[2] = [(complex(M0[2]+end_time*M0[4]), complex(B1[2]+end_time*B1[4]), 0.)]
    for time in old_times:
        phase = cmath.exp(1j*(M0[2]+time*M0[4]))
        frequency = B1[2]+time*B1[4]
        a, b = .01*M0[3]*phase, .01*B1[3]*phase
        # Re(z)=(z+z*)/2 and Im(z)=(z-z*)/(2i).
        terms[0].extend(((a/2, b/2, frequency), (a.conjugate()/2, b.conjugate()/2, -frequency)))
        terms[1].extend(((a/(2j), b/(2j), frequency), (-a.conjugate()/(2j), -b.conjugate()/(2j), -frequency)))
    expected_mean_complex = np.array([term_expectation(row, mean_g, variance_g) for row in terms])
    assert np.max(np.abs(expected_mean_complex.imag)) < ATOL
    expected_mean = expected_mean_complex.real
    expected_covariance = np.empty((6, 6))
    for j in range(6):
        for k in range(6):
            product = term_product_expectation(terms[j], terms[k], mean_g, variance_g)
            assert abs(product.imag) < ATOL
            expected_covariance[j, k] = product.real-expected_mean[j]*expected_mean[k]
    return expected_mean, expected_covariance, log_evidence


def independent_linear_condition(mean, covariance, observations, values, noise, endpoint, offset):
    """Small explicit affine joint law; covariance subtraction, not Joseph factor."""
    residual = values-mv(observations, mean)
    innovation = mm(mm(observations, covariance), observations.T)+noise
    cross = mm(mm(endpoint, covariance), observations.T)
    expected_mean = offset+mv(endpoint, mean)+mv(cross, np.linalg.solve(innovation, residual))
    expected_covariance = mm(mm(endpoint, covariance), endpoint.T)-mm(cross, np.linalg.solve(innovation, cross.T))
    sign, logdet = np.linalg.slogdet(innovation)
    assert sign == 1
    log_evidence = -.5*(len(values)*math.log(2*math.pi)+logdet+float(residual @ np.linalg.solve(innovation, residual)))
    return expected_mean, expected_covariance, log_evidence


def linear_fixture(alpha):
    mean = np.array([.5, -.25, alpha, 1.5, 0., 0.])
    factor = np.zeros((6, 3))
    factor[[0, 1, 3], :] = J
    covariance = mm(factor, factor.T)
    events = ((10000, "gyro", 0., .02**2), (10000, "wheel", 1.4, .1**2),
              (30000, "gyro", 0., .02**2), (30000, "wheel", 1.6, .1**2))
    # Independent coordinates: (px0, py0, v0, dv0, dv1, dv2).
    primitive_mean = np.array([.5, -.25, 1.5, 0., 0., 0.])
    primitive_covariance = np.zeros((6, 6))
    primitive_covariance[:3, :3] = mm(J, J.T)
    primitive_covariance[3:, 3:] = np.eye(3)*.04*.01
    observation = np.zeros((4, 6))
    observation[1, [2, 3]] = 1.
    observation[3, [2, 3, 4, 5]] = 1.
    endpoint = np.zeros((6, 6))
    endpoint[0, 0] = endpoint[1, 1] = 1.
    direction = np.array([math.cos(alpha), math.sin(alpha)])
    endpoint[:2, 2] = .03*direction
    endpoint[:2, 3] = .02*direction
    endpoint[:2, 4] = .01*direction
    endpoint[3, [2, 3, 4, 5]] = 1.
    offset = np.array([0., 0., alpha, 0., 0., 0.])
    target = independent_linear_condition(primitive_mean, primitive_covariance, observation,
        np.array([0., 1.4, 0., 1.6]), np.diag([.02**2, .1**2, .02**2, .1**2]), endpoint, offset)
    return (mean, covariance, factor, np.array([.04, 0., 0.]), events), target


def timing_fixture():
    mean = np.array([0., 0., 0., 1.5, .1, -.02])
    events = ((20000, "gyro", .09, .02**2), (20000, "wheel", 1.4, .1**2))
    # Independent coordinates: (dv0,domega0,db0,dv1,domega1,db1).
    primitive_mean = np.zeros(6)
    primitive_covariance = np.diag([.04, .0004, .0001]*2)*.01
    observation = np.zeros((2, 6))
    observation[0, [1, 2, 4, 5]] = 1.
    observation[1, [0, 3]] = 1.
    endpoint = np.zeros((6, 6))
    endpoint[0, 0] = .01*math.cos(.001)
    endpoint[1, 0] = .01*math.sin(.001)
    endpoint[2, 1] = .01
    endpoint[3, [0, 3]] = 1.
    endpoint[4, [1, 4]] = 1.
    endpoint[5, [2, 5]] = 1.
    offset = np.array([.015+.015*math.cos(.001), .015*math.sin(.001), .002, 1.5, .1, -.02])
    target = independent_linear_condition(primitive_mean, primitive_covariance, observation,
        np.array([.09-.08, 1.4-1.5]), np.diag([.02**2, .1**2]), endpoint, offset)
    return (mean, np.zeros((6, 6)), np.zeros((6, 0)), np.array([.04, .0004, .0001]), events), target


def heading_fixture(variance, speed=2., heading=0., events=None):
    mean = np.array([0., 0., heading, speed, 0., 0.])
    factor = np.zeros((6, 1 if variance else 0))
    if variance:
        factor[2, 0] = math.sqrt(variance)
    covariance = np.zeros((6, 6))
    covariance[2, 2] = variance
    if events is None:
        events = ((1000000, "gyro", 0., .02**2),)
    return mean, covariance, factor, np.zeros(3), events


def heading_target(variance, speed=2., duration=1., heading=0., event_count=1):
    distance = speed*duration
    damping = math.exp(-variance/2)
    rotation = np.array([[math.cos(heading), -math.sin(heading)],
                         [math.sin(heading), math.cos(heading)]])
    mean = np.array([distance*damping, 0., heading, speed, 0., 0.])
    mean[:2] = mv(rotation, mean[:2])
    covariance = np.zeros((6, 6))
    raw_variances = np.array([distance*distance/2*(-math.expm1(-variance))**2,
                              distance*distance/2*(-math.expm1(-2*variance))])
    covariance[:2, :2] = mm(mm(rotation, np.diag(raw_variances)), rotation.T)
    covariance[2, 2] = variance
    covariance[:2, 2] = mv(rotation, np.array([0., distance*variance*damping]))
    covariance[2, :2] = covariance[:2, 2]
    log_evidence = -.5*event_count*math.log(2*math.pi*.02**2)
    return mean, covariance, log_evidence


def test_empty_published_checkpoint_and_factor_law():
    mean, covariance, factor = published_checkpoint()
    result = oracle.condition_prefix(mean, covariance, factor, [.04, .0004, .0001], [])
    check_result("empty_published", result, mean, mm(factor, factor.T), 0.)
    factor_law_covariance = result.covariance.copy()
    discrepancy = float(np.max(np.abs(mm(factor, factor.T)-covariance)))
    emit("published_factor_input", maximum_absolute_discrepancy=discrepancy)
    assert discrepancy <= ATOL
    deterministic = oracle.condition_prefix(mean, np.zeros((6, 6)), np.zeros((6, 0)), [0., 0., 0.], [])
    check_result("empty_deterministic", deterministic, mean, np.zeros((6, 6)), 0.)
    # The supplied covariance can differ within the frozen input consistency
    # allowance. Even the empty result must use the factor-defined law.
    perturbed = covariance.copy()
    perturbed[0, 0] += 5e-11
    result = oracle.condition_prefix(mean, perturbed, factor, [0., 0., 0.], [])
    close(result.covariance, mm(factor, factor.T))
    np.testing.assert_array_equal(result.covariance, factor_law_covariance)
    assert result.covariance[0, 0] != perturbed[0, 0]


def test_only_published_one_step_against_frozen_nine_dimensional_oracle():
    # The old one-step function imports a sibling module using its historical
    # absolute import. This imports it read-only; it does not run a study.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
    import run_trajectory_mixture_study as old
    mean, covariance, factor = published_checkpoint()
    events = RANK_EVENTS[:2]
    result = oracle.condition_prefix(mean, covariance, factor, [.04, .0004, .0001], events)
    target = old.observed_one_step_oracle()
    check_result("published_one_step", result, target["mean"], target["covariance"], target["log_evidence"])


@pytest.mark.parametrize("alpha", [0., math.pi/2])
def test_small_linear_physical_joint_law(alpha):
    inputs, target = linear_fixture(alpha)
    check_result(f"linear_heading_{alpha}", oracle.condition_prefix(*inputs), *target)


@pytest.mark.parametrize("variance", [0., .4, 1e-8, 1e-12])
def test_shared_heading_hundred_step_control_and_narrow_precision(variance):
    result = oracle.condition_prefix(*heading_fixture(variance))
    target = heading_target(variance)
    check_result(f"shared_heading_{variance}", result, *target)
    if variance in (1e-8, 1e-12):
        expected = float(target[1][0, 0])
        actual = float(result.covariance[0, 0])
        relative_error = abs(actual-expected)/expected
        projected = result.directional_variance([1., 0., 0., 0., 0., 0.])
        projected_relative_error = abs(projected["direct_term_sum"]-expected)/expected
        resolved = relative_error <= 1e-7 and projected_relative_error <= 1e-7
        emit("narrow_variance", heading_variance=variance, expected=expected,
             actual=actual, relative_error=relative_error,
             direct_term_sum=projected["direct_term_sum"],
             direct_relative_error=projected_relative_error,
             absolute_term_sum=projected["absolute_term_sum"],
             projection_difference=projected["difference"],
             status="resolved" if resolved else "unresolved", criterion=1e-7)
        # The frozen protocol explicitly permits an unresolved narrow control.
        # Recording it is required; it is never converted to an accuracy pass.
        assert math.isfinite(relative_error)


def test_shared_heading_partition_invariance():
    first_events = ((990000, "gyro", 0., .02**2),)
    second_events = ((333333, "gyro", 0., .02**2), (990000, "gyro", 0., .02**2))
    first = oracle.condition_prefix(*heading_fixture(.4, events=first_events))
    second = oracle.condition_prefix(*heading_fixture(.4, events=second_events))
    check_result("partition_99_steps", first, *heading_target(.4, duration=.99))
    check_result("partition_100_steps", second, *heading_target(.4, duration=.99, event_count=2))
    close(first.mean, second.mean)
    close(first.covariance, second.covariance)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_scalar_conditioned_history_complete_six_state(count):
    events = RANK_EVENTS[:count]
    check_result(f"rank_one_prefix_{count}", call_rank(events), *rank_target(events))


def test_end_step_noise_uses_old_heading_and_speed():
    inputs, target = timing_fixture()
    check_result("two_step_physical_timing", oracle.condition_prefix(*inputs), *target)


def test_received_prefix_outputs_are_immutable_and_do_not_see_future():
    first = call_rank(RANK_EVENTS[:1])
    saved_mean, saved_covariance = first.mean.copy(), first.covariance.copy()
    saved_evidence = first.log_evidence
    second, final = call_rank(RANK_EVENTS[:2]), call_rank()
    repeat = call_rank(RANK_EVENTS[:1])
    np.testing.assert_array_equal(first.mean, saved_mean)
    np.testing.assert_array_equal(first.covariance, saved_covariance)
    np.testing.assert_array_equal(repeat.mean, saved_mean)
    np.testing.assert_array_equal(repeat.covariance, saved_covariance)
    assert first.log_evidence == saved_evidence == repeat.log_evidence
    assert np.max(np.abs(second.mean-first.mean)) > 1e-10
    assert np.max(np.abs(final.mean-second.mean)) > 1e-10
    with pytest.raises(ValueError):
        first.mean[0] = 0.
    with pytest.raises(ValueError):
        first.covariance[0, 0] = 0.
    with pytest.raises(ValueError):
        first.mean.setflags(write=True)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        first.log_evidence = 0.
    with pytest.raises(TypeError):
        first.diagnostics["changed"] = True


@pytest.mark.parametrize("speed", [-2., 0., 2.])
def test_signed_zero_speed_and_unwrapped_heading_limits(speed):
    base = oracle.condition_prefix(*heading_fixture(.4, speed=speed))
    shifted = oracle.condition_prefix(*heading_fixture(.4, speed=speed, heading=2*math.pi))
    check_result(f"signed_speed_{speed}", base, *heading_target(.4, speed=speed))
    check_result(f"shifted_speed_{speed}", shifted, *heading_target(.4, speed=speed, heading=2*math.pi))
    close(base.mean[:2], shifted.mean[:2])
    close(base.covariance, shifted.covariance)
    close(shifted.mean[2]-base.mean[2], 2*math.pi)
    if speed == 0:
        np.testing.assert_array_equal(base.mean[:2], np.zeros(2))
        np.testing.assert_array_equal(base.covariance[:2], np.zeros((2, 6)))


def test_scalar_characteristic_derivatives_independent_finite_differences():
    mean, variance, _ = scalar_posterior(RANK_EVENTS)
    step = 1e-6
    phi = lambda t: cmath.exp(1j*mean*t-.5*variance*t*t)
    phi_prime = lambda t: (1j*mean-variance*t)*phi(t)
    for t in (0., .3, .3008):
        first_difference = (phi(t+step)-phi(t-step))/(2*step)
        second_difference = (phi_prime(t+step)-phi_prime(t-step))/(2*step)
        _, first_kernel, second_kernel = scalar_characteristic_moments(t, mean, variance)
        emit("scalar_characteristic_derivatives", t=t,
             first_error=abs(first_kernel-first_difference/1j),
             second_error=abs(second_kernel+second_difference))
        np.testing.assert_allclose(first_kernel, first_difference/1j, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(second_kernel, -second_difference, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("events", [
    ((10000, "wheel", 1.4, .01), (10000, "gyro", .09, .0004)),
    ((10000, "gyro", .09, .0004), (10000, "gyro", .09, .0004)),
    ((20000, "gyro", .09, .0004), (10000, "wheel", 1.4, .01)),
    ((0, "gyro", .09, .0004),),
    ((10000, "gps", .09, .0004),),
    ((10000., "gyro", .09, .0004),),
    ((True, "gyro", .09, .0004),),
    ((10000, "gyro", math.nan, .0004),),
    ((10000, "gyro", .09, 0.),),
    ((10000, "gyro", .09, -.0004),),
])
def test_invalid_sensor_prefixes_are_rejected(events):
    with pytest.raises(oracle.InvalidInput):
        call_rank(events)


@pytest.mark.parametrize("kind", ["mean_shape", "covariance_shape", "factor_shape", "factor_width",
                                  "nonfinite", "negative_covariance", "inconsistent_factor", "negative_diffusion"])
def test_invalid_checkpoint_and_diffusion_are_rejected(kind):
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
    elif kind == "nonfinite":
        mean[0] = math.inf
    elif kind == "negative_covariance":
        covariance[0, 0] = -1.
    elif kind == "inconsistent_factor":
        factor[0, 0] += .1
    elif kind == "negative_diffusion":
        diffusion[0] = -.01
    with pytest.raises(oracle.InvalidInput):
        oracle.condition_prefix(mean, covariance, factor, diffusion, [])


def test_unresolved_innovation_and_positive_scale_underflow_are_not_repaired():
    mean, covariance, factor = np.zeros(6), np.zeros((6, 6)), np.zeros((6, 0))
    with pytest.raises(oracle.UnresolvedInference):
        oracle.condition_prefix(mean, covariance, factor, [0., 0., 0.], [(10000, "gyro", 0., 1e-14)])
    with pytest.raises(oracle.UnresolvedInference):
        oracle.condition_prefix(mean, covariance, factor, [float.fromhex("0x0.0000000000001p-1022"), 0., 0.],
                                [(1, "gyro", 0., .0004)])


def test_event_and_substep_bounds_stop_before_unbounded_consumption():
    count = 0
    def events():
        nonlocal count
        for timestamp in range(1, 1000000):
            count += 1
            if count > 33:
                pytest.fail("event iterable consumed beyond the fixed 32-event lookahead")
            yield timestamp, "gyro", 0., .0004
    with pytest.raises(oracle.InvalidInput):
        oracle.condition_prefix(np.zeros(6), np.zeros((6, 6)), np.zeros((6, 0)), [0., 0., 0.], events())
    assert count <= 33
    with pytest.raises(oracle.InvalidInput):
        oracle.condition_prefix(*heading_fixture(.4, events=((1000001, "gyro", 0., .0004),)))
    with pytest.raises(oracle.InvalidInput):
        oracle.condition_prefix(*heading_fixture(.4, events=((2000001, "gyro", 0., .0004),)))


def literal_pair_inputs(zero_means=False):
    means = np.array([[0., 0. if zero_means else 1.5],
                      [math.pi/2, 0. if zero_means else -.5]])
    factors = np.zeros((2, 2, 2))
    # This is the declared square root of [[.04,.01],[.01,.09]].
    factors[0, 1] = [.2, 0.]
    factors[1, 1] = [.05, math.sqrt(.0875)]
    return means, factors


def test_literal_pair_signs_and_zero_mean_random_speed():
    for zero_means in (False, True):
        means, factors = literal_pair_inputs(zero_means)
        result = oracle.displacement_moments(means, factors)
        close(result["centered_plus"][0, 1], .01j)
        close(result["centered_minus"][0, 1], -.01j)
        close(result["covariance"][0, 1], [[0., .01], [0., 0.]])
        close(result["covariance"][1, 0], [[0., 0.], [.01, 0.]])
        close(result["covariance"][0, 0], [[.04, 0.], [0., 0.]])
        close(result["covariance"][1, 1], [[0., 0.], [0., .09]])
        close(result["mean_complex"], means[:, 1]*np.exp(1j*means[:, 0]))
        emit("literal_pair", zero_speed_means=zero_means,
             cross_max_abs_error=float(np.max(np.abs(result["covariance"][0, 1]-np.array([[0., .01], [0., 0.]])))))
        if zero_means:
            assert result["covariance"][0, 0, 0, 0] > 0.
            assert result["covariance"][1, 1, 1, 1] > 0.


def test_pair_limits_and_conjugate_symmetry_from_scalar_conditioned_law():
    mean_g, variance_g, _ = scalar_posterior(RANK_EVENTS)
    old_times = (0., .01, .02)
    means = np.array([[M0[2]+time*M0[4]+(B1[2]+time*B1[4])*mean_g,
                       M0[3]+B1[3]*mean_g] for time in old_times])
    factors = np.array([[[math.sqrt(variance_g)*(B1[2]+time*B1[4])],
                         [math.sqrt(variance_g)*B1[3]]] for time in old_times])
    result = oracle.displacement_moments(means, factors)
    scalar_terms = []
    for time in old_times:
        phase = cmath.exp(1j*(M0[2]+time*M0[4]))
        scalar_terms.append([(M0[3]*phase, B1[3]*phase, B1[2]+time*B1[4])])
    scalar_means = [term_expectation(row, mean_g, variance_g) for row in scalar_terms]
    expected_plus, expected_minus = np.empty((3, 3), complex), np.empty((3, 3), complex)
    for j in range(3):
        for k in range(3):
            conjugate_k = [(a.conjugate(), b.conjugate(), -t) for a, b, t in scalar_terms[k]]
            expected_plus[j, k] = term_product_expectation(scalar_terms[j], scalar_terms[k], mean_g, variance_g)-scalar_means[j]*scalar_means[k]
            expected_minus[j, k] = term_product_expectation(scalar_terms[j], conjugate_k, mean_g, variance_g)-scalar_means[j]*scalar_means[k].conjugate()
    close(result["mean_complex"], scalar_means)
    close(result["centered_plus"], expected_plus)
    close(result["centered_minus"], expected_minus)
    close(result["centered_plus"], result["centered_plus"].T)
    close(result["centered_minus"], result["centered_minus"].conjugate().T)
    close(result["covariance"], result["covariance"].transpose(1, 0, 3, 2))
    for j in range(3):
        theta, speed = means[j]
        variance_theta = factors[j, 0, 0]**2
        variance_speed = factors[j, 1, 0]**2
        cross = factors[j, 0, 0]*factors[j, 1, 0]
        raw_minus = result["centered_minus"][j, j]+abs(result["mean_complex"][j])**2
        raw_plus = result["centered_plus"][j, j]+result["mean_complex"][j]**2
        close(raw_minus, speed*speed+variance_speed)
        close(raw_plus, (variance_speed+(speed+2j*cross)**2)*cmath.exp(2j*theta-2*variance_theta))
    emit("rank_pair_limits", plus_max_abs_error=float(np.max(np.abs(result["centered_plus"]-expected_plus))),
         minus_max_abs_error=float(np.max(np.abs(result["centered_minus"]-expected_minus))))


def test_implementation_identity_kernels_against_independent_derivatives():
    mean, variance, _ = scalar_posterior(RANK_EVENTS)
    step, constant_phase = 1e-6, .2
    phi = lambda t: cmath.exp(1j*mean*t-.5*variance*t*t)
    phi_prime = lambda t: (1j*mean-variance*t)*phi(t)
    for t in (0., .3, .3008):
        first_difference = (phi(t+step)-phi(t-step))/(2*step)
        second_difference = (phi_prime(t+step)-phi_prime(t-step))/(2*step)
        first = oracle.gaussian_first(mean, constant_phase+t*mean, t*t*variance, t*variance)
        second = oracle.gaussian_second(mean, mean, variance, constant_phase+t*mean,
                                        t*t*variance, t*variance, t*variance)
        phase = cmath.exp(1j*constant_phase)
        np.testing.assert_allclose(first, phase*first_difference/1j, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(second, -phase*second_difference, atol=1e-7, rtol=1e-7)
        emit("implementation_derivative_kernels", t=t,
             first_error=abs(first-phase*first_difference/1j),
             second_error=abs(second+phase*second_difference))


def test_requested_direction_keeps_term_sum_and_matrix_projection():
    result = call_rank()
    expected_covariance = rank_target()[1]
    for direction in (np.array([1., 0., 0., 0., 0., 0.]),
                      np.array([1., -.3, .2, -.4, .5, .1])):
        diagnostics = result.directional_variance(direction)
        target = float(direction @ expected_covariance @ direction)
        close(diagnostics["direct_term_sum"], target)
        close(diagnostics["matrix_projection"], target)
        close(diagnostics["difference"], diagnostics["direct_term_sum"]-diagnostics["matrix_projection"])
        assert diagnostics["absolute_term_sum"] >= abs(diagnostics["direct_term_sum"])
        assert diagnostics["precision_certified"] is False
        recorded = dict(diagnostics)
        recorded["direction"] = recorded["direction"].tolist()
        recorded["underflow_counts"] = dict(recorded["underflow_counts"])
        emit("directional_variance", target=target, **recorded)
    with pytest.raises(oracle.InvalidInput):
        result.directional_variance([1., 0.])


def mutation_detected(name, invoke, expected):
    """A restored faulty computation must fail the fixed independent check."""
    try:
        result = invoke()
    except oracle.NumericalFailure as error:
        emit("mutation", mutation=name, detected=True, mechanism="numerical_failure", reason=str(error))
        return
    try:
        close(result.mean, expected[0])
        close(result.covariance, expected[1])
        close(result.log_evidence, expected[2])
    except AssertionError:
        emit("mutation", mutation=name, detected=True, mechanism="independent_fixture_difference",
             mean_max_abs_error=float(np.max(np.abs(result.mean-expected[0]))),
             covariance_max_abs_error=float(np.max(np.abs(result.covariance-expected[1]))),
             log_evidence_abs_error=abs(float(result.log_evidence-expected[2])))
        return
    pytest.fail(f"{name} escaped its fixed independent fixture")


def test_drop_cross_step_mutation_is_detected_and_restored(monkeypatch):
    original = oracle._centered_pair
    def diagonal_only(left_mean, right_mean, left_factor, right_factor, sign, diagnostics):
        if not np.array_equal(left_mean, right_mean):
            return 0j
        return original(left_mean, right_mean, left_factor, right_factor, sign, diagnostics)
    with monkeypatch.context() as patch:
        patch.setattr(oracle, "_centered_pair", diagonal_only)
        mutation_detected("drop_different_steps", call_rank, rank_target())
    assert oracle._centered_pair is original


def test_minus_phase_conjugation_mutation_is_detected_and_restored(monkeypatch):
    original = oracle._centered_pair
    def wrong_conjugation(left_mean, right_mean, left_factor, right_factor, sign, diagnostics):
        return original(left_mean, right_mean, left_factor, right_factor, 1, diagnostics)
    with monkeypatch.context() as patch:
        patch.setattr(oracle, "_centered_pair", wrong_conjugation)
        # Literal independent signs identify the specific error before endpoint
        # checks; a plus phase cannot stand in for the conjugated minus phase.
        try:
            pair = oracle.displacement_moments(*literal_pair_inputs())
        except oracle.NumericalFailure as error:
            emit("mutation", mutation="minus_phase_conjugation", detected=True,
                 mechanism="numerical_failure", reason=str(error))
        else:
            with pytest.raises(AssertionError):
                close(pair["centered_minus"][0, 1], -.01j)
            emit("mutation", mutation="minus_phase_conjugation", detected=True,
                 mechanism="literal_pair_sign", absolute_error=abs(pair["centered_minus"][0, 1]+.01j))
    assert oracle._centered_pair is original


@pytest.mark.parametrize("omit", ["initial_position", "final_latent"])
def test_cross_covariance_omission_mutations_are_detected_and_restored(monkeypatch, omit):
    original = oracle._cross_moment
    def missing_cross(step_mean, step_factor, x_factor, diagnostics):
        cross = np.array(original(step_mean, step_factor, x_factor, diagnostics), copy=True)
        if omit == "initial_position":
            cross[:2] = 0.
        else:
            cross[2:] = 0.
        return cross
    with monkeypatch.context() as patch:
        patch.setattr(oracle, "_cross_moment", missing_cross)
        mutation_detected(f"omit_{omit}_cross", call_rank, rank_target())
    assert oracle._cross_moment is original


def test_double_conditioning_mutation_is_detected_and_restored(monkeypatch):
    original = oracle._condition_physical
    def twice(prior_mean, prior_factor, observation, values, variances):
        mean, factor, evidence, diagnostics = original(prior_mean, prior_factor, observation, values, variances)
        mean, factor, second_evidence, second_diagnostics = original(mean, factor, observation, values, variances)
        return mean, factor, evidence+second_evidence, second_diagnostics
    with monkeypatch.context() as patch:
        patch.setattr(oracle, "_condition_physical", twice)
        mutation_detected("condition_twice", call_rank, rank_target())
    assert oracle._condition_physical is original


def test_new_state_displacement_mutation_is_detected_and_restored(monkeypatch):
    original = oracle._old_step_rows
    def use_new_state(physical):
        return np.stack((physical["theta"][1:], physical["speed"][1:]), axis=1)
    inputs, target = timing_fixture()
    with monkeypatch.context() as patch:
        patch.setattr(oracle, "_old_step_rows", use_new_state)
        mutation_detected("new_state_displacement", lambda: oracle.condition_prefix(*inputs), target)
    assert oracle._old_step_rows is original


def test_diagnostics_retain_support_timing_covariance_and_no_precision_claim():
    result = call_rank()
    diagnostic, physical = result.diagnostics, result.physical
    assert result.status == "completed"
    assert result.timestamp_us == 30000
    assert result.phase == "gyro"
    assert diagnostic["dimensions"] == {"steps": 3, "events": 3, "physical": 15,
        "prior_factor_columns": 1, "posterior_factor_columns": 4}
    close(physical["step_sizes_s"], [.01, .01, .01])
    np.testing.assert_array_equal(physical["observation_step_indices"], [1, 1, 3])
    assert diagnostic["precision_certified"] is False
    assert diagnostic["factor_covariance_residual_max_abs"] <= ATOL
    close(diagnostic["factor_covariance_residual"], np.zeros((6, 6)))
    assert diagnostic["conditioning"]["solve_residual_max_abs"] <= ATOL
    assert diagnostic["conditioning"]["innovation_eigenvalues"][0] > 1e-12
    assert np.min(diagnostic["phase_variances"]) >= 0.
    for key in ("plus_symmetry_residual", "minus_conjugate_residual", "block_transpose_residual"):
        assert diagnostic["displacement"][key] <= ATOL
    assert diagnostic["endpoint_covariance"]["asymmetry_before_symmetrization"] <= ATOL
    assert np.min(diagnostic["covariance_absolute_term_sums"] - np.abs(result.covariance)) >= -ATOL
    emit("diagnostics", factor_residual=diagnostic["factor_covariance_residual_max_abs"],
         solve_residual=diagnostic["conditioning"]["solve_residual_max_abs"],
         phase_variance_min=float(np.min(diagnostic["phase_variances"])),
         phase_variance_max=float(np.max(diagnostic["phase_variances"])),
         pair_symmetry={key: diagnostic["displacement"][key] for key in
             ("plus_symmetry_residual", "minus_conjugate_residual", "block_transpose_residual")},
         underflow_counts=dict(diagnostic["displacement"]["underflow_counts"]),
         dimensions=dict(diagnostic["dimensions"]), precision_certified=False)
    with pytest.raises(ValueError):
        physical["posterior_factor"][0, 0] = 0.
    with pytest.raises(ValueError):
        diagnostic["phase_variances"][0, 0, 0] = 0.
    with pytest.raises(TypeError):
        diagnostic["conditioning"]["solve_residual_max_abs"] = 1.


def test_fixed_hundred_step_dimension_bound_is_reported():
    result = oracle.condition_prefix(*heading_fixture(.4))
    assert result.diagnostics["dimensions"]["steps"] == 100
    assert result.diagnostics["dimensions"]["physical"] == 306
    assert result.physical["theta"].shape == (101, 306)
    assert result.physical["posterior_factor"].shape == (306, 2)
    assert len(result.physical["step_sizes_s"]) == 100


def test_shape_rejection_happens_before_array_coercion_or_factor_allocation():
    class Unbounded:
        def __array__(self, *args, **kwargs):
            pytest.fail("unbounded input was coerced before dimension validation")
    with pytest.raises(oracle.InvalidInput):
        oracle.condition_prefix(Unbounded(), np.zeros((6, 6)), np.zeros((6, 0)), [0., 0., 0.], [])
    with pytest.raises(oracle.InvalidInput):
        oracle.condition_prefix(np.zeros(6), np.zeros((6, 6)), Unbounded(), [0., 0., 0.], [])
    with pytest.raises(oracle.InvalidInput):
        oracle.displacement_moments(np.empty((101, 2)), np.empty((101, 2, 0)))
    with pytest.raises(oracle.InvalidInput):
        oracle.displacement_moments(np.empty((1, 2)), np.empty((1, 2, 339)))


def test_failure_statuses_remain_distinct_and_arithmetic_is_not_repaired():
    mean, covariance, factor = np.zeros(6), np.zeros((6, 6)), np.zeros((6, 0))
    with pytest.raises(oracle.InvalidInput) as invalid:
        oracle.condition_prefix(mean, covariance, factor, [-1., 0., 0.], [])
    assert invalid.value.status == "invalid"
    with pytest.raises(oracle.UnresolvedInference) as unresolved:
        oracle.condition_prefix(mean, covariance, factor, [0., 0., 0.], [(10000, "gyro", 0., 1e-14)])
    assert unresolved.value.status == "unresolved"
    assert unresolved.value.diagnostics["innovation_eigenvalues"][0] <= 1e-12
    huge_factor = np.zeros((6, 1))
    huge_factor[0, 0] = 1e308
    with pytest.raises(oracle.NumericalFailure) as failed:
        oracle.condition_prefix(mean, covariance, huge_factor, [0., 0., 0.], [])
    assert failed.value.status == "failed"
    emit("refusal_statuses", invalid=invalid.value.status, unresolved=unresolved.value.status,
         arithmetic=failed.value.status)
