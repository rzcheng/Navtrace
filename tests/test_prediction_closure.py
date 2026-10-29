"""Fixed prediction-only fixtures for the stopped Gaussian moment candidate."""
import hashlib
from itertools import product
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
import pytest

from tools import position_moments


DURATION_S = .1
MOMENT_ATOL = 5e-12
CANDIDATE_SHA256 = '7bd81a4fdd92296919f7aaf2d1b32cacdbe4d1e7d880a5168eb2697e7e3d8104'
FIXTURES = tuple(product((-2., 2.), (0., .4), (1, 2, 4, 8)))


@pytest.fixture(autouse=True)
def unchanged_candidate_source():
    """An in-memory mutation must never leave candidate source changes behind."""
    path = Path(position_moments.__file__)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == CANDIDATE_SHA256
    try:
        yield
    finally:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def initial_state(speed, yaw_variance):
    mean = np.array([0., 0., 0., speed, 0., 0.])
    covariance = np.zeros((6, 6))
    covariance[2, 2] = yaw_variance
    return mean, covariance


def analytical_moments(speed, yaw_variance, steps):
    """Separate exact trajectory moments and the predicted Gaussian recursion."""
    decay = math.exp(-yaw_variance)
    dx = speed**2*(1-decay)**2/2
    dy = speed**2*(1-math.exp(-2*yaw_variance))/2
    a = speed*yaw_variance*math.exp(-yaw_variance/2)
    b = speed**2*yaw_variance*decay
    mean = np.array([DURATION_S*speed*math.exp(-yaw_variance/2), 0., 0., speed, 0., 0.])
    exact = np.zeros((6, 6))
    exact[0, 0] = DURATION_S**2*dx
    exact[1, 1] = DURATION_S**2*dy
    exact[1, 2] = exact[2, 1] = DURATION_S*a
    exact[2, 2] = yaw_variance
    projected = exact.copy()
    projected[0, 0] = DURATION_S**2*dx/steps
    projected[1, 1] = DURATION_S**2*(dy/steps+(1-1/steps)*b)
    return mean, exact, projected


def endpoint_quadrature(speed, yaw_variance, order):
    """Integrate the full deterministic map with one shared real heading.

    The endpoint does not depend on the Euler partition in this fixture.
    Centered products are integrated directly, without characteristic moments
    or a conditional-reference sampler.
    """
    nodes, weights = hermgauss(order)
    heading = math.sqrt(2*yaw_variance)*nodes
    weights = weights/math.sqrt(math.pi)
    states = np.column_stack([
        DURATION_S*speed*np.cos(heading),
        DURATION_S*speed*np.sin(heading),
        heading,
        np.full(order, speed),
        np.zeros(order),
        np.zeros(order),
    ])
    mean = np.einsum('n,ni->i', weights, states, optimize=False)
    centered = states-mean
    covariance = np.einsum('n,ni,nj->ij', weights, centered, centered, optimize=False)
    return mean, covariance


def prediction_paths(speed, yaw_variance, steps):
    """Use the same explicit substep bound for a bulk call and repeated calls."""
    mean, covariance = initial_state(speed, yaw_variance)
    step_s = DURATION_S/steps
    bulk = position_moments.predict(mean, covariance, DURATION_S, np.zeros(3), max_step_s=step_s)
    repeated_mean, repeated_covariance = mean.copy(), covariance.copy()
    for _ in range(steps):
        repeated_mean, repeated_covariance = position_moments.predict(
            repeated_mean, repeated_covariance, step_s, np.zeros(3), max_step_s=step_s)
    return bulk, (repeated_mean, repeated_covariance)


def assert_covariance_valid(covariance):
    assert covariance.shape == (6, 6)
    assert np.isfinite(covariance).all()
    tolerance = 1e-10*max(1., float(np.abs(covariance).max()))
    assert float(np.abs(covariance-covariance.T).max()) <= tolerance
    assert float(np.linalg.eigvalsh(covariance).min()) >= -tolerance


@pytest.mark.parametrize('speed,yaw_variance,steps', FIXTURES)
def test_prediction_closure_full_state(speed, yaw_variance, steps):
    expected_mean, exact_covariance, projected_covariance = analytical_moments(speed, yaw_variance, steps)
    low_mean, low_covariance = endpoint_quadrature(speed, yaw_variance, 64)
    high_mean, high_covariance = endpoint_quadrature(speed, yaw_variance, 128)
    for mean, covariance in [(low_mean, low_covariance), (high_mean, high_covariance)]:
        np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=MOMENT_ATOL)
        np.testing.assert_allclose(covariance, exact_covariance, rtol=0, atol=MOMENT_ATOL)
        assert_covariance_valid(covariance)
    np.testing.assert_allclose(low_mean, high_mean, rtol=0, atol=MOMENT_ATOL)
    np.testing.assert_allclose(low_covariance, high_covariance, rtol=0, atol=MOMENT_ATOL)

    paths = prediction_paths(speed, yaw_variance, steps)
    initial_mean, initial_covariance = initial_state(speed, yaw_variance)
    for mean, covariance in paths:
        np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=MOMENT_ATOL)
        np.testing.assert_allclose(covariance, projected_covariance, rtol=0, atol=MOMENT_ATOL)
        assert_covariance_valid(covariance)
        np.testing.assert_array_equal(mean[2:], initial_mean[2:])
        np.testing.assert_array_equal(covariance[2:, 2:], initial_covariance[2:, 2:])
        np.testing.assert_allclose(covariance[1, 2], exact_covariance[1, 2], rtol=0, atol=MOMENT_ATOL)
        if yaw_variance == 0:
            np.testing.assert_array_equal(covariance, np.zeros((6, 6)))
        else:
            assert np.sign(covariance[1, 2]) == np.sign(speed)
            np.testing.assert_allclose(covariance[0, 0]/exact_covariance[0, 0], 1/steps,
                                       rtol=0, atol=MOMENT_ATOL)
            if steps > 1:
                # A passing regression confirms this covariance loss; it does
                # not mean the repeated projection has exact endpoint moments.
                assert covariance[0, 0] < exact_covariance[0, 0]
                assert covariance[1, 1] < exact_covariance[1, 1]
        if steps == 1:
            np.testing.assert_allclose(covariance, high_covariance, rtol=0, atol=MOMENT_ATOL)
    np.testing.assert_allclose(paths[0][0], paths[1][0], rtol=0, atol=MOMENT_ATOL)
    np.testing.assert_allclose(paths[0][1], paths[1][1], rtol=0, atol=MOMENT_ATOL)


@pytest.mark.parametrize('speed,steps', tuple(product((-2., 2.), (2, 4, 8))))
def test_displacement_position_cross_mutation_is_detected(monkeypatch, speed, steps):
    yaw_variance = .4
    _, _, expected_covariance = analytical_moments(speed, yaw_variance, steps)
    original = position_moments._displacement_moments

    def omit_position_cross(mean, covariance):
        increment, cross, spread = original(mean, covariance)
        changed_cross = cross.copy()
        changed_cross[:2, :] = 0.
        return increment, changed_cross, spread

    # First confirm the frozen, unmutated candidate passes the same full-P check.
    for _, covariance in prediction_paths(speed, yaw_variance, steps):
        np.testing.assert_allclose(covariance, expected_covariance, rtol=0, atol=MOMENT_ATOL)

    with monkeypatch.context() as patch:
        patch.setattr(position_moments, '_displacement_moments', omit_position_cross)
        for repeated in (False, True):
            mean, covariance = initial_state(speed, yaw_variance)
            step_s = DURATION_S/steps
            try:
                if repeated:
                    for _ in range(steps):
                        mean, covariance = position_moments.predict(
                            mean, covariance, step_s, np.zeros(3), max_step_s=step_s)
                else:
                    mean, covariance = position_moments.predict(
                        mean, covariance, DURATION_S, np.zeros(3), max_step_s=step_s)
            except ValueError as error:
                # Dropping this variance contribution while keeping Py,theta
                # can violate PSD. Existing validation must remain enabled.
                assert str(error) == 'covariance is not positive semidefinite'
            else:
                with pytest.raises(AssertionError):
                    np.testing.assert_allclose(covariance, expected_covariance, rtol=0, atol=MOMENT_ATOL)
    assert position_moments._displacement_moments is original
    for _, covariance in prediction_paths(speed, yaw_variance, steps):
        np.testing.assert_allclose(covariance, expected_covariance, rtol=0, atol=MOMENT_ATOL)
