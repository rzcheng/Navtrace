import numpy as np
import pytest
from scipy.integrate import quad

from tools.conditional_reference import (Observation, backward_parameters, build_history,
    circular_summary, normal_log_density, observe, sample_endpoint)


def assert_gaussian_sample(samples, mean, covariance):
    # Six Gaussian standard errors; fixed seeds, no tolerance chosen from results.
    n = len(samples)
    assert np.all(np.abs(samples.mean(axis=0)-mean) <= 6*np.sqrt(np.diag(covariance)/n))
    actual = np.cov(samples, rowvar=False)
    se = np.sqrt((covariance**2+np.outer(np.diag(covariance), np.diag(covariance)))/(n-1))
    assert np.all(np.abs(actual-covariance) <= 6*se+1e-12)


def test_initial_samples_preserve_full_correlated_gaussian():
    scales = np.array([.4, .3, .05, .2, .02, .01])
    covariance = np.outer(scales, scales)*(.4*np.ones((6, 6))+.6*np.eye(6))
    mean = np.array([2., -1., .3, 1., .1, .02])
    history = build_history(mean, covariance, 0, [], 0, [.25, .01, 1e-7])
    samples = sample_endpoint(history, 0, 100000, 31001)
    assert_gaussian_sample(samples, mean, covariance)


def test_hand_scalar_update_and_density_normalization():
    mean = np.array([0., 10., 0., 0.]); covariance = np.diag([1., 4., 1., 1.])
    updated, posterior, likelihood = observe(mean, covariance, 'wheel', 12., 9.)
    assert updated[1] == pytest.approx(10+8/13)
    assert posterior[1, 1] == pytest.approx(36/13)
    assert likelihood == pytest.approx(-.5*(np.log(2*np.pi*13)+4/13))
    assert normal_log_density(3, 9) == pytest.approx(-np.log(3*np.sqrt(2*np.pi))-.5)
    integral, error = quad(lambda x: np.exp(normal_log_density(x, 2.5)), -np.inf, np.inf)
    assert integral == pytest.approx(1., abs=1e-10) and error < 1e-8


def test_linear_filter_matches_independent_joint_gaussian_conditioning():
    mean = np.array([0., 0., .2, 1., -.1, .03])
    covariance = np.diag([.1, .1, .09, .4, .04, .01])
    covariance[2, 5] = covariance[5, 2] = .015
    events = [Observation(100000, 'gyro', .15, .02), Observation(100000, 'wheel', 1.7, .1)]
    history = build_history(mean, covariance, 0, events, 100000, [.25, .01, .001], max_step_s=.1)
    a = np.array([[1., 0., .1, 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    c0 = covariance[2:, 2:]; c1 = a@c0@a.T+np.diag([0., .025, .001, .0001])
    joint_covariance = np.block([[c0, c0@a.T], [a@c0, c1]])
    joint_mean = np.r_[mean[2:], a@mean[2:]]
    h = np.zeros((2, 8)); h[0, 6:] = 1; h[1, 5] = 1
    innovation = h@joint_covariance@h.T+np.diag([.02, .1])
    gain = np.linalg.solve(innovation, h@joint_covariance).T
    expected_mean = joint_mean+gain@(np.array([.15, 1.7])-h@joint_mean)
    expected_covariance = joint_covariance-gain@h@joint_covariance
    np.testing.assert_allclose(history.means[-1], expected_mean[4:], atol=1e-12)
    np.testing.assert_allclose(history.covariances[-1], expected_covariance[4:, 4:], atol=1e-12)
    # Independently check the backward noise conditional, including cross-time covariance.
    rng = np.random.default_rng(31002); n = 100000
    current = expected_mean[4:, None]+np.einsum('ij,jn->in', np.linalg.cholesky(expected_covariance[4:, 4:]), rng.normal(size=(4, n)))
    predicted, noise_gain, noise_factor, noise_std = backward_parameters(mean[2:], c0, .1, np.array([.25, .01, .001]))
    noise = (np.einsum('ij,jn->in', noise_gain, current-predicted[:, None])
             +np.einsum('ij,jn->in', noise_factor, rng.normal(size=(3, n))))
    previous = current.copy(); previous[1:] -= noise_std[:, None]*noise; previous[0] -= .1*previous[2]
    assert_gaussian_sample(np.vstack([previous, current]).T, expected_mean, expected_covariance)


def test_no_observation_position_moments_use_start_of_step():
    mean = np.array([1., 2., .4, 3., .5, 0.])
    covariance = np.diag([.01, .02, .16, .25, .03, .001])
    history = build_history(mean, covariance, 0, [], 100000, [5., .1, .001], max_step_s=.1)
    samples = sample_endpoint(history, 100000, 150000, 31003)
    # E[cos theta], E[sin theta] and second trigonometric moments for Gaussian theta.
    angle, var, speed, speed2, step = .4, .16, 3., 9.25, .1
    direction = np.exp(-var/2)*np.array([np.cos(angle), np.sin(angle)])
    second = .5*np.array([[1+np.exp(-2*var)*np.cos(2*angle), np.exp(-2*var)*np.sin(2*angle)],
                         [np.exp(-2*var)*np.sin(2*angle), 1-np.exp(-2*var)*np.cos(2*angle)]])
    expected_mean = mean[:2]+step*speed*direction
    expected_covariance = covariance[:2, :2]+step**2*(speed2*second-speed**2*np.outer(direction, direction))
    # Fixed conservative Monte Carlo bounds at N=150000: .003 m and .0008 m².
    np.testing.assert_allclose(samples[:, :2].mean(axis=0), expected_mean, atol=.003, rtol=0)
    np.testing.assert_allclose(np.cov(samples[:, :2], rowvar=False), expected_covariance, atol=.0008, rtol=0)


def test_zero_diffusion_and_same_timestamp_do_not_add_transitions():
    events = [Observation(10000, 'gyro', .1, .02), Observation(10000, 'wheel', 1., .1)]
    history = build_history(np.zeros(6), np.eye(6), 0, events, 10000, [0., 0., 0.])
    assert history.steps == [.01] and history.observation_counts[10000] == 2
    samples = sample_endpoint(history, 10000, 1000, 31004)
    assert np.isfinite(samples).all()


def test_future_data_do_not_change_earlier_reference_samples():
    events = [Observation(10000, 'gyro', .2, .01), Observation(20000, 'wheel', 1., .1)]
    changed = [events[0], Observation(20000, 'wheel', 50., .1)]
    first = build_history(np.zeros(6), np.eye(6), 0, events, 20000, [.25, .01, 1e-7])
    second = build_history(np.zeros(6), np.eye(6), 0, changed, 20000, [.25, .01, 1e-7])
    np.testing.assert_array_equal(sample_endpoint(first, 10000, 512, 31005), sample_endpoint(second, 10000, 512, 31005))
    assert not np.array_equal(sample_endpoint(first, 20000, 512, 31005), sample_endpoint(second, 20000, 512, 31005))


def test_circular_statistics_and_explicit_invalid_inputs():
    summary = circular_summary([np.pi-.01, -np.pi+.01])
    assert abs(abs(summary['mean_rad'])-np.pi) < 1e-12
    assert summary['resultant'] == pytest.approx(np.cos(.01))
    assert circular_summary([0., np.pi])['mean_rad'] is None
    for residual, variance in [(np.nan, 1), (0, 0), (0, -1), (1e308, 1e-300)]:
        with pytest.raises(ValueError): normal_log_density(residual, variance)
    with pytest.raises(ValueError, match='positive definite'):
        build_history(np.zeros(6), np.zeros((6, 6)), 0, [], 0, [0, 0, 0])
    with pytest.raises(ValueError, match='duplicate'):
        build_history(np.zeros(6), np.eye(6), 0, [Observation(10000, 'wheel', 1., .1)]*2, 10000, [0, 0, 0])
    with pytest.raises(ValueError, match='diffusion'):
        build_history(np.zeros(6), np.eye(6), 0, [], 0, [-1, 0, 0])
    with pytest.raises(ValueError, match='gap'):
        build_history(np.zeros(6), np.eye(6), 0, [], 3000000, [0, 0, 0])


def test_budget_failure_stops_sampling_explicitly():
    history = build_history(np.zeros(6), np.eye(6), 0, [], 10000, [.25, .01, 1e-7])
    def exhausted():
        raise TimeoutError('test budget exhausted')
    with pytest.raises(TimeoutError, match='budget exhausted'):
        sample_endpoint(history, 10000, 512, 31006, check_budget=exhausted)


def test_current_wheel_observation_conditions_past_speed_and_initial_position():
    # At one step, v is constant. A wheel fix changes both its past value and
    # the correlated initial x distribution, before position integration.
    mean = np.array([0., 0., 0., 1., 0., 0.])
    covariance = np.diag([.2, .2, .04, .4, .001, .001])
    covariance[0, 3] = covariance[3, 0] = .1
    history = build_history(mean, covariance, 0, [Observation(100000, 'wheel', 2., .5)],
                            100000, [0., 0., 0.], max_step_s=.1)
    samples = sample_endpoint(history, 100000, 100000, 31007)
    v_mean, v_var, x_mean, x_var, xv_cov = 1+.4/.9, .4-.4**2/.9, .1/.9, .2-.1**2/.9, .1-.1*.4/.9
    cosine = np.exp(-.04/2)
    cos2 = (1+np.exp(-2*.04))/2
    sin2 = (1-np.exp(-2*.04))/2
    expected_mean = np.array([x_mean+.1*v_mean*cosine, 0.])
    expected_covariance = np.diag([
        x_var+.01*((v_var+v_mean**2)*cos2-(v_mean*cosine)**2)+.2*xv_cov*cosine,
        .2+.01*(v_var+v_mean**2)*sin2])
    position = samples[:, :2]; n = len(samples)
    assert np.all(np.abs(position.mean(axis=0)-expected_mean) <= 6*np.sqrt(np.diag(expected_covariance)/n))
    centered = position-expected_mean
    # Fourth-moment Monte Carlo SE for this non-Gaussian position distribution.
    products = centered[:, :, None]*centered[:, None, :]
    se = products.std(axis=0, ddof=1)/np.sqrt(n)
    assert np.all(np.abs(np.cov(position, rowvar=False)-expected_covariance) <= 6*se)
