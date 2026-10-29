"""Independent checks of the offline Gaussian projection, without recordings."""
import math

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from tools.conditional_reference import Observation
from tools.position_moments import observe, predict, run_candidate


MEAN = np.array([1., -2., .8, 2.3, -.4, .07])
DIFFUSION = np.array([.25, .01, 1e-7])


def correlated_covariance():
    factor = np.array([[.5, 0, 0, 0, 0, 0], [.1, .4, 0, 0, 0, 0],
                       [.1, -.08, .7, 0, 0, 0], [.2, .1, -.25, .6, 0, 0],
                       [-.1, .04, .12, -.09, .3, 0], [.02, -.03, .06, .05, -.08, .2]])
    return factor@factor.T


def assert_valid(covariance):
    tolerance = 1e-10*max(1., np.abs(covariance).max())
    assert np.isfinite(covariance).all()
    assert np.abs(covariance-covariance.T).max() <= tolerance
    assert np.linalg.eigvalsh(covariance).min() >= -tolerance


def conditional_yaw_quadrature(mean, covariance, step, diffusion, order=96):
    """Integrate over yaw; the remaining conditional map is affine in X.

    This oracle uses Gaussian conditioning and Gauss-Hermite quadrature, not
    characteristic-function or centered trigonometric-moment formulas.
    """
    nodes, weights = hermgauss(order)
    weights = weights/np.sqrt(np.pi)
    yaw_std = np.sqrt(covariance[2, 2])
    slope = covariance[:, 2]/yaw_std
    conditional_covariance = covariance-np.outer(slope, slope)
    means, covariances = [], []
    for node in nodes:
        conditional_mean = mean+np.sqrt(2)*node*slope
        angle = mean[2]+np.sqrt(2)*node*yaw_std
        transform = np.eye(6)
        transform[0, 3] = step*np.cos(angle)
        transform[1, 3] = step*np.sin(angle)
        transform[2, 4] = step
        means.append(transform@conditional_mean)
        covariances.append(transform@conditional_covariance@transform.T)
    means, covariances = np.array(means), np.array(covariances)
    result_mean = np.einsum('n,ni->i', weights, means)
    centered = means-result_mean
    result_covariance = (np.einsum('n,nij->ij', weights, covariances)
                         +np.einsum('n,ni,nj->ij', weights, centered, centered)
                         +np.diag(np.r_[np.zeros(3), diffusion*step]))
    return result_mean, result_covariance


@pytest.mark.parametrize('speed,yaw_scale', [(2.3, 1.), (-2.3, 1.), (0., 1.),
                                            (2.3, 2.), (-2.3, 1e-5)])
def test_full_prediction_against_independent_yaw_quadrature(speed, yaw_scale):
    mean = MEAN.copy()
    mean[3] = speed
    covariance = correlated_covariance()
    scaling = np.ones(6)
    scaling[2] = yaw_scale
    covariance *= np.outer(scaling, scaling)
    expected = conditional_yaw_quadrature(mean, covariance, .1, DIFFUSION)
    # Two quadrature orders certify the integration error independently. The
    # 5e-12 absolute tolerance covers float summation of these O(1) moments.
    finer = conditional_yaw_quadrature(mean, covariance, .1, DIFFUSION, 128)
    actual = predict(mean, covariance, .1, DIFFUSION, max_step_s=.1)
    for reference, refined, result in zip(expected, finer, actual):
        np.testing.assert_allclose(reference, refined, rtol=3e-12, atol=5e-12)
        np.testing.assert_allclose(result, refined, rtol=3e-12, atol=5e-12)
    assert_valid(actual[1])


def deterministic_euler(mean, step):
    result = mean.copy()
    result[0] += step*mean[3]*math.cos(mean[2])
    result[1] += step*mean[3]*math.sin(mean[2])
    result[2] += step*mean[4]
    return result


def test_finite_difference_and_small_covariance_jacobian_limit():
    step = .1
    jacobian = np.eye(6)
    jacobian[0, 2] = -step*MEAN[3]*np.sin(MEAN[2])
    jacobian[1, 2] = step*MEAN[3]*np.cos(MEAN[2])
    jacobian[0, 3] = step*np.cos(MEAN[2])
    jacobian[1, 3] = step*np.sin(MEAN[2])
    jacobian[2, 4] = step
    epsilon = 1e-6
    numerical = np.column_stack([
        (deterministic_euler(MEAN+epsilon*direction, step)
         -deterministic_euler(MEAN-epsilon*direction, step))/(2*epsilon)
        for direction in np.eye(6)])
    np.testing.assert_allclose(jacobian, numerical, rtol=0, atol=4e-10)
    covariance = correlated_covariance()
    linear_limit = jacobian@covariance@jacobian.T
    errors = []
    for scale in (1e-3, 1e-4, 1e-5, 1e-7):
        mean, propagated = predict(MEAN, scale*covariance, step, [0, 0, 0], max_step_s=step)
        errors.append(np.max(np.abs(propagated/scale-linear_limit)))
        assert_valid(propagated)
    # Smooth-map Taylor expansion gives O(scale) error after dividing P by
    # scale. The final 2e-8 bound exceeds its leading coefficient at 1e-7.
    assert errors[1] < .11*errors[0]
    assert errors[2] < .11*errors[1]
    np.testing.assert_allclose(propagated/scale, linear_limit, rtol=0, atol=2e-8)
    np.testing.assert_allclose(mean, deterministic_euler(MEAN, step), rtol=0, atol=1e-8)


def test_position_cross_covariance_uses_next_yaw_including_rate():
    covariance = correlated_covariance()
    step = .1
    _, actual = predict(MEAN, covariance, step, [0, 0, 0], max_step_s=step)
    _, reference = conditional_yaw_quadrature(MEAN, covariance, step, np.zeros(3))
    np.testing.assert_allclose(actual[:2, 2:], reference[:2, 2:], rtol=0, atol=5e-12)
    # Isolate the next-yaw/rate term with an additional oracle calculation at
    # zero yaw-rate coupling, leaving position increments exactly unchanged.
    nodes, weights = hermgauss(96)
    weights /= np.sqrt(np.pi)
    slope = covariance[:, 2]/np.sqrt(covariance[2, 2])
    conditional_covariance = covariance-np.outer(slope, slope)
    terms = []
    for node in nodes:
        conditional_mean = MEAN+np.sqrt(2)*node*slope
        angle = conditional_mean[2]
        direction = np.array([np.cos(angle), np.sin(angle)])
        terms.append(direction*(conditional_covariance[3, 4]
                                +conditional_mean[3]*(conditional_mean[4]-MEAN[4])))
    displacement_rate_covariance = np.einsum('n,ni->i', weights, np.array(terms))
    assert np.linalg.norm(step**2*displacement_rate_covariance) > 1e-4


@pytest.mark.parametrize('yaw_variance', [0., 1e-20, 1e-14, 1e-8])
def test_small_yaw_variance_keeps_narrow_displacement_variance(yaw_variance):
    mean = np.array([0., 0., 0., -3., 0., 0.])
    covariance = np.zeros((6, 6))
    covariance[2, 2] = yaw_variance
    updated, propagated = predict(mean, covariance, .1, [0, 0, 0], max_step_s=.1)
    # For fixed speed, Var(v cos theta)=v^2*(1-exp(-s))^2/2.
    # Its leading O(s^2) term must survive even at s=1e-20.
    expected_x = .5*.1**2*9*np.expm1(-yaw_variance)**2
    expected_y = -.5*.1**2*9*np.expm1(-2*yaw_variance)
    assert propagated[0, 0] == pytest.approx(expected_x, rel=5e-15, abs=0)
    assert propagated[1, 1] == pytest.approx(expected_y, rel=5e-15, abs=0)
    assert updated[0] == pytest.approx(-.3*np.exp(-yaw_variance/2), abs=1e-16)
    assert_valid(propagated)


@pytest.mark.parametrize('speed', [-3., 0., 3.])
def test_deterministic_yaw_and_signed_speed(speed):
    mean = MEAN.copy()
    mean[3] = speed
    mean[2] = 7.
    updated, covariance = predict(mean, np.zeros((6, 6)), .1, [0, 0, 0], max_step_s=.1)
    np.testing.assert_allclose(updated, deterministic_euler(mean, .1), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(covariance, np.zeros((6, 6)))
    assert updated[2] > 2*np.pi


def test_old_state_increment_and_end_step_diffusion():
    covariance = np.zeros((6, 6))
    updated, propagated = predict(MEAN, covariance, .1, DIFFUSION, max_step_s=.1)
    np.testing.assert_allclose(updated, deterministic_euler(MEAN, .1), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(propagated, np.diag(np.r_[np.zeros(3), .1*DIFFUSION]))
    _, after_second = predict(updated, propagated, .1, [0, 0, 0], max_step_s=.1)
    assert np.trace(after_second[:2, :2]) > 0
    assert after_second[2, 2] > 0


def test_equal_subdivision_matches_repeated_steps_without_yaw_wrapping():
    covariance = correlated_covariance()
    actual = predict(MEAN, covariance, .025, DIFFUSION)
    expected = (MEAN.copy(), covariance.copy())
    for _ in range(3):
        expected = predict(*expected, .025/3, DIFFUSION)
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)


def test_two_pi_shift_preserves_covariance_and_position():
    covariance = correlated_covariance()
    shifted = MEAN.copy()
    shifted[2] += 2*np.pi
    original = predict(MEAN, covariance, .073, DIFFUSION)
    changed = predict(shifted, covariance, .073, DIFFUSION)
    expected = original[0].copy()
    expected[2] += 2*np.pi
    np.testing.assert_allclose(changed[0][[0, 1, 3, 4, 5]], expected[[0, 1, 3, 4, 5]], rtol=0, atol=2e-15)
    np.testing.assert_allclose(changed[1], original[1], rtol=0, atol=2e-15)
    # Two recursions of n additions, the initial 2*pi shift and two final
    # subtractions each incur at most eps times this magnitude bound. This
    # yaw-only bound accounts for unwrapped accumulation at different scales.
    steps = math.ceil(.073/.01)
    magnitude_bound = max(abs(MEAN[2]), abs(shifted[2]))+.073*abs(MEAN[4])
    yaw_roundoff_bound = (2*steps+3)*np.finfo(float).eps*magnitude_bound
    assert abs((changed[0][2]-original[0][2])-2*np.pi) <= yaw_roundoff_bound


@pytest.mark.parametrize('sensor', ['wheel', 'gyro'])
def test_correlated_scalar_update_against_hand_conditioning(sensor):
    vector = np.array([1., -2., 3., 2., -1., 4.])
    covariance = np.diag([2., 3., 4., 5., 6., 7.])+np.outer(vector, vector)
    mean = np.array([1., 2., .3, -1., .2, -.4])
    if sensor == 'wheel':
        cross = np.array([2., -4., 6., 9., -2., 8.])
        predicted, innovation_variance = -1., 12.
    else:
        cross = np.array([3., -6., 9., 6., 3., 19.])
        predicted, innovation_variance = -.2, 25.
    # R=3, z=2. The subtraction form is an independent algebraic check of
    # the implementation's Joseph form, with nonzero position/latent terms.
    expected_mean = mean+cross*(2.-predicted)/innovation_variance
    expected_covariance = covariance-np.outer(cross, cross)/innovation_variance
    updated, posterior = observe(mean, covariance, sensor, 2., 3.)
    np.testing.assert_allclose(updated, expected_mean, rtol=0, atol=1e-14)
    np.testing.assert_allclose(posterior, expected_covariance, rtol=0, atol=1e-14)
    assert_valid(posterior)


def test_gaussian_closure_covariance_misses_residual_dependence():
    mean = np.array([0., 0., 0., 1., 0., 0.])
    covariance = np.diag([0., 0., .4, .5, 0., 0.])
    projected = predict(mean, covariance, .1, [0, 0, 0], max_step_s=.1)
    projected_covariances, exact_covariances = [], []
    for measurement in (1., 4.):
        _, posterior = observe(*projected, 'wheel', measurement, .5)
        projected_covariances.append(posterior)
        # Constant speed over this single zero-diffusion step means conditioning
        # before propagation is exact. v|z has mean 1+(z-1)/2, variance 1/4.
        conditioned_mean = mean.copy()
        conditioned_mean[3] = 1+(measurement-1)/2
        conditioned_covariance = covariance.copy()
        conditioned_covariance[3, 3] = .25
        _, exact = conditional_yaw_quadrature(conditioned_mean, conditioned_covariance, .1, np.zeros(3))
        exact_covariances.append(exact)
        expected_y_variance = .01*(.25+conditioned_mean[3]**2)*(1-np.exp(-.8))/2
        assert exact[1, 1] == pytest.approx(expected_y_variance, abs=2e-15)
    np.testing.assert_array_equal(*projected_covariances)
    assert exact_covariances[1][1, 1]-exact_covariances[0][1, 1] > .01
    assert abs(projected_covariances[1][1, 1]-exact_covariances[1][1, 1]) > .01


def test_zero_dt_and_all_operations_copy_inputs():
    mean, covariance, diffusion = MEAN.copy(), correlated_covariance(), DIFFUSION.copy()
    saved = (mean.copy(), covariance.copy(), diffusion.copy())
    outputs = [predict(mean, covariance, 0., diffusion), predict(mean, covariance, .01, diffusion),
               observe(mean, covariance, 'wheel', 2., .1)]
    np.testing.assert_array_equal(outputs[0][0], mean)
    np.testing.assert_array_equal(outputs[0][1], covariance)
    for result_mean, result_covariance in outputs:
        assert not np.shares_memory(result_mean, mean)
        assert not np.shares_memory(result_covariance, covariance)
        result_mean[:] = 0
        result_covariance[:] = 0
    for original, unchanged in zip((mean, covariance, diffusion), saved):
        np.testing.assert_array_equal(original, unchanged)
    with pytest.raises(ValueError):
        predict(mean, covariance, 3., diffusion)
    with pytest.raises(ValueError):
        observe(mean, covariance, 'wheel', np.inf, .1)
    for original, unchanged in zip((mean, covariance, diffusion), saved):
        np.testing.assert_array_equal(original, unchanged)


def test_causal_reports_include_all_same_time_events_and_are_independent_copies():
    covariance = correlated_covariance()
    events = [Observation(10000, 'gyro', .1, .02), Observation(10000, 'wheel', 2., .1),
              Observation(20000, 'gyro', -.2, .01), Observation(30000, 'wheel', -8., .2)]
    snapshots = run_candidate(MEAN, covariance, 0, events, [0, 10000, 30000], DIFFUSION)
    prefix = run_candidate(MEAN, covariance, 0, events[:2], [0, 10000], DIFFUSION)
    changed = run_candidate(MEAN, covariance, 0, events[:2]+[
        Observation(20000, 'gyro', 3., .02), Observation(30000, 'wheel', 99., .2)],
        [0, 10000, 30000], DIFFUSION)
    expected = predict(MEAN, covariance, .01, DIFFUSION)
    for event in events[:2]:
        expected = observe(*expected, event.sensor, event.value, event.variance)
    for timestamp, count in [(0, 0), (10000, 2), (30000, 4)]:
        assert snapshots[timestamp]['observations'] == count
        assert_valid(snapshots[timestamp]['covariance'])
    for name, reference in zip(('mean', 'covariance'), expected):
        np.testing.assert_array_equal(snapshots[10000][name], reference)
        np.testing.assert_array_equal(snapshots[10000][name], prefix[10000][name])
        np.testing.assert_array_equal(snapshots[10000][name], changed[10000][name])
    np.testing.assert_array_equal(snapshots[0]['mean'], MEAN)
    np.testing.assert_array_equal(snapshots[0]['covariance'], covariance)
    assert not np.array_equal(snapshots[30000]['mean'], changed[30000]['mean'])
    snapshots[10000]['mean'][:] = 100
    snapshots[10000]['covariance'][:] = 100
    np.testing.assert_array_equal(snapshots[0]['mean'], MEAN)
    np.testing.assert_array_equal(snapshots[0]['covariance'], covariance)


@pytest.mark.parametrize('events,reports', [
    ([Observation(0, 'gyro', 0., 1.)], [0]),
    ([Observation(10000, 'gps', 0., 1.)], [10000]),
    ([Observation(10000, 'wheel', 0., 1.)]*2, [10000]),
    ([Observation(10000, 'wheel', 0., 1.), Observation(10000, 'gyro', 0., 1.)], [10000]),
    ([Observation(20000, 'gyro', 0., 1.), Observation(10000, 'wheel', 0., 1.)], [20000]),
    ([Observation(10000, 'gyro', 0., 1.)], [0]),
    ([Observation(10000, 'gyro', 0., 1.)], [5000, 10000]),
    ([Observation(10000., 'gyro', 0., 1.)], [10000]),
    ([Observation(True, 'gyro', 0., 1.)], [10000]),
    ([], [10000]), ([], []), ([], [0, 0]), ([], [10000, 0]), ([], [-1]), ([], [0.]),
    ([Observation(3000000, 'gyro', 0., 1.)], [3000000]),
    ([Observation(10000, 'gyro', np.nan, 1.)], [10000]),
    ([Observation(10000, 'wheel', 0., 0.)], [10000]),
])
def test_invalid_replay_inputs(events, reports):
    covariance = correlated_covariance()
    with pytest.raises(ValueError):
        run_candidate(MEAN, covariance, 0, events, reports, DIFFUSION)
    np.testing.assert_array_equal(covariance, correlated_covariance())


@pytest.mark.parametrize('field,value', [
    ('dt_s', -.01), ('dt_s', np.inf), ('dt_s', np.nan), ('dt_s', 2.01),
    ('diffusion', [-1, 0, 0]), ('diffusion', [0, 0]), ('diffusion', [0, np.nan, 0]),
    ('max_step_s', 0.), ('max_step_s', .2), ('max_step_s', np.nan),
    ('max_gap_s', .005), ('max_gap_s', 61.), ('max_gap_s', np.inf),
])
def test_invalid_prediction_configuration(field, value):
    arguments = dict(dt_s=.01, diffusion=DIFFUSION, max_step_s=.01, max_gap_s=2.)
    arguments[field] = value
    with pytest.raises(ValueError):
        predict(MEAN, correlated_covariance(), **arguments)


@pytest.mark.parametrize('sensor,value,variance', [('gps', 1., 1.), ('wheel', np.nan, 1.),
    ('gyro', np.inf, 1.), ('wheel', 1., 0.), ('gyro', 1., -1.), ('wheel', 1., np.nan)])
def test_invalid_observation(sensor, value, variance):
    with pytest.raises(ValueError):
        observe(MEAN, correlated_covariance(), sensor, value, variance)


@pytest.mark.parametrize('covariance', [np.eye(5), np.full((6, 6), np.nan),
    np.diag([-1e-5, 1, 1, 1, 1, 1]), np.eye(6)+np.triu(np.ones((6, 6)), 1)*1e-5])
def test_invalid_covariance_is_rejected(covariance):
    for operation in (lambda: predict(MEAN, covariance, 0., DIFFUSION),
                      lambda: observe(MEAN, covariance, 'gyro', 0., 1.)):
        with pytest.raises(ValueError):
            operation()


def test_covariance_roundoff_allowance_is_scaled_and_does_not_clip():
    covariance = np.diag([-5e-7, 1e4, 1, 1, 1, 1])
    _, result = predict(MEAN, covariance, 0., DIFFUSION)
    assert result[0, 0] == -5e-7
    covariance[0, 0] = -2e-6
    with pytest.raises(ValueError, match='positive semidefinite'):
        predict(MEAN, covariance, 0., DIFFUSION)
    covariance = np.eye(6)
    covariance[0, 1] = 1e-12
    _, symmetric = predict(MEAN, covariance, .01, DIFFUSION)
    np.testing.assert_array_equal(symmetric, symmetric.T)


def test_budget_callback_runs_during_events_and_steps_and_propagates_failure():
    calls = []
    events = [Observation(30000, 'gyro', .1, .02), Observation(30000, 'wheel', 2., .1)]
    result = run_candidate(MEAN, correlated_covariance(), 0, events, [30000], DIFFUSION,
                           check_budget=lambda: calls.append(1))
    assert len(calls) == 5  # Three substeps and two observations.
    assert result[30000]['observations'] == 2
    def exhausted():
        raise TimeoutError('test budget exhausted')
    with pytest.raises(TimeoutError, match='test budget exhausted'):
        run_candidate(MEAN, correlated_covariance(), 0, events, [30000], DIFFUSION,
                      check_budget=exhausted)
    with pytest.raises(ValueError, match='callable'):
        run_candidate(MEAN, correlated_covariance(), 0, [], [0], DIFFUSION, check_budget=1)


def test_invalid_mean_and_checkpoint_and_arithmetic_overflow():
    for mean in (np.zeros(5), np.full(6, np.nan)):
        with pytest.raises(ValueError, match='mean'):
            predict(mean, np.eye(6), .01, DIFFUSION)
    for checkpoint in (-1, 0., True):
        with pytest.raises(ValueError, match='timestamp'):
            run_candidate(MEAN, np.eye(6), checkpoint, [], [0], DIFFUSION)
    enormous = MEAN.copy()
    enormous[3] = 1e308
    with pytest.raises(ValueError, match='arithmetic'):
        predict(enormous, np.eye(6), .01, DIFFUSION)
