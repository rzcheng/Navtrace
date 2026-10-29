"""Fixed-coordinate checks of the received-prefix proposal; no random draws.

Expected Gaussian laws start with physical checkpoint/noise coordinates. They
never use the prototype's observation matrices, layout, or path-map helpers.
"""
from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from tools import observed_trajectory as observed
from tools.observed_trajectory import (InvalidInput, NumericalFailure, Observation,
                                       UnresolvedInference, condition_prefix)

A = [2, 4, 5]
L = [0, 1, 3]
LA = np.array([[.3, 0, 0], [.02, .08, 0], [-.01, -.03, .04]])
J = np.array([[.2, 0, 0], [.05, .15, 0], [.04, -.02, .25]])
K = np.array([[.4, .2, -.3], [-.2, .5, .1], [.3, -.4, .2]])
MU_A = np.array([.2, .1, -.02])
MU_L = np.array([.5, -.25, 1.5])
DIFFUSION = np.array([.04, .0004, .0001])
H = .01
EVENTS = (Observation(10000, 'gyro', .09, .02**2),
          Observation(10000, 'wheel', 1.4, .1**2),
          Observation(20000, 'gyro', .11, .02**2),
          Observation(20000, 'wheel', 1.6, .1**2))


def mm(left, right):
    return np.einsum('ij,jk->ik', left, right, optimize=False)


def mv(matrix, vector):
    return np.einsum('ij,j->i', matrix, vector, optimize=False)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def psd(covariance):
    assert np.isfinite(covariance).all()
    if covariance.size:
        tolerance = 1e-10*max(1., float(np.max(np.abs(covariance))))
        assert np.max(np.abs(covariance-covariance.T)) <= tolerance
        assert np.linalg.eigvalsh(covariance)[0] >= -tolerance


def checkpoint():
    mean, covariance = np.zeros(6), np.zeros((6, 6))
    mean[A], mean[L] = MU_A, MU_L
    angular = mm(LA, LA.T)
    cross = mm(K, angular)
    covariance[np.ix_(A, A)] = angular
    covariance[np.ix_(L, A)] = cross
    covariance[np.ix_(A, L)] = cross.T
    covariance[np.ix_(L, L)] = mm(mm(K, angular), K.T)+mm(J, J.T)
    return mean, covariance


def coordinates(dimension):
    return np.array([np.zeros(dimension), np.ones(dimension),
                     np.array([1. if index % 2 == 0 else -1. for index in range(dimension)])])


def independent_log_density(point, mean, covariance):
    dimension = len(point)
    if not dimension:
        return 0.
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    residual = point-mean
    quadratic = float(np.sum(residual*np.linalg.solve(covariance, residual)))
    return -.5*(dimension*math.log(2*math.pi)+logdet+quadratic)


def physical_oracle(event_count=4):
    """Independent 9D/12D law: physical x0 then (dv,domega,db) per step."""
    events = EVENTS[:event_count]
    steps = 1 if event_count <= 2 else 2
    mean, covariance = checkpoint()
    dimension = 6+3*steps
    physical_mean = np.zeros(dimension)
    physical_mean[:6] = mean
    physical_covariance = np.zeros((dimension, dimension))
    physical_covariance[:6, :6] = covariance
    for step in range(steps):
        start = 6+3*step
        physical_covariance[start:start+3, start:start+3] = np.diag(DIFFUSION*H)
    sensor = np.zeros((len(events), dimension))
    values = np.array([event.value for event in events])
    noise = np.diag([event.variance for event in events])
    for row, event in enumerate(events):
        count = event.timestamp_us//10000
        if event.sensor == 'gyro':
            sensor[row, 4:6] = 1.
            for step in range(count):
                sensor[row, 6+3*step+1:6+3*step+3] = 1.
        else:
            sensor[row, 3] = 1.
            for step in range(count):
                sensor[row, 6+3*step] = 1.
    # Batch conditioning of the PHYSICAL joint, independently of the prototype.
    cross = mm(physical_covariance, sensor.T)
    innovation = mm(sensor, cross)+noise
    gain = np.linalg.solve(innovation, cross.T).T
    posterior_mean = physical_mean+mv(gain, values-mv(sensor, physical_mean))
    residual_map = np.eye(dimension)-mm(gain, sensor)
    posterior_covariance = (mm(mm(residual_map, physical_covariance), residual_map.T)
                            +mm(mm(gain, noise), gain.T))
    # Explicit fixture-only change of coordinates, not prototype layout/maps.
    du, dz = 3+2*steps, 3+steps
    transform = np.zeros((dimension, du+dz))
    transform[np.ix_(A, range(3))] = LA
    transform[np.ix_(L, range(3))] = mm(K, LA)
    transform[np.ix_(L, range(du, du+3))] = J
    for step in range(steps):
        transform[6+3*step+1, 3+2*step] = math.sqrt(DIFFUSION[1]*H)
        transform[6+3*step+2, 4+2*step] = math.sqrt(DIFFUSION[2]*H)
        transform[6+3*step, du+3+step] = math.sqrt(DIFFUSION[0]*H)
    inverse_transform = np.linalg.solve(transform, np.eye(dimension))
    primitive_mean = mv(inverse_transform, posterior_mean-physical_mean)
    primitive_covariance = mm(mm(inverse_transform, posterior_covariance), inverse_transform.T)
    # A separate information-form identity checks every induced primitive cross block.
    primitive_sensor = mm(sensor, transform)
    information = np.eye(dimension)+mm(primitive_sensor.T, np.linalg.solve(noise, primitive_sensor))
    information_covariance = np.linalg.solve(information, np.eye(dimension))
    information_mean = mv(information_covariance, mv(primitive_sensor.T,
        np.linalg.solve(noise, values-mv(sensor, physical_mean))))
    close(primitive_mean, information_mean)
    close(primitive_covariance, information_covariance)
    evidence = independent_log_density(values, mv(sensor, physical_mean), innovation)
    return dict(mean=posterior_mean, covariance=posterior_covariance, prior_mean=physical_mean,
                prior_covariance=physical_covariance, sensor=sensor, values=values, noise=noise,
                transform=transform, inverse_transform=inverse_transform, du=du, dz=dz,
                primitive_mean=primitive_mean, primitive_covariance=primitive_covariance,
                evidence=evidence, steps=steps)


def physical_point(u, z, steps):
    """Explicit independent physical generator at fixed, supplied coordinates."""
    point = np.zeros(6+3*steps)
    point[A] = MU_A+mv(LA, u[:3])
    point[L] = MU_L+mv(K, point[A]-MU_A)+mv(J, z[:3])
    for step in range(steps):
        point[6+3*step:9+3*step] = [math.sqrt(DIFFUSION[0]*H)*z[3+step],
            math.sqrt(DIFFUSION[1]*H)*u[3+2*step],
            math.sqrt(DIFFUSION[2]*H)*u[4+2*step]]
    return point


def direct_physical_endpoint(point, steps):
    state = point[:6].copy()
    paths = [state[A].copy()]
    for step in range(steps):
        angle, speed, rate = state[2:5].copy()
        state[:2] += H*speed*np.array([math.cos(angle), math.sin(angle)])
        state[2] += H*rate
        state[[3, 4, 5]] += point[6+3*step:9+3*step]
        paths.append(state[A].copy())
    return state, np.array(paths)


def conditional_physical_endpoint(oracle, u):
    """Condition the physical Gaussian on U, then use a physical affine endpoint map."""
    du = oracle['du']
    angular_selector = oracle['inverse_transform'][:du]
    angular_mean = oracle['primitive_mean'][:du]
    angular_covariance = oracle['primitive_covariance'][:du, :du]
    cross = mm(oracle['covariance'], angular_selector.T)
    gain = np.linalg.solve(angular_covariance, cross.T).T
    mean = oracle['mean']+mv(gain, u-angular_mean)
    covariance = oracle['covariance']-mm(gain, cross.T)
    # Given U, theta/rate/bias are fixed. Every position coefficient uses old theta.
    endpoint = np.zeros((6, len(mean)))
    endpoint[:, :6] = np.eye(6)
    angle, rate = mean[2], mean[4]
    for step in range(oracle['steps']):
        direction = np.array([math.cos(angle), math.sin(angle)])
        endpoint[:2, 3] += H*direction
        for earlier in range(step):
            endpoint[:2, 6+3*earlier] += H*direction
        endpoint[2, 4] += H
        for earlier in range(step):
            endpoint[2, 6+3*earlier+1] += H
        endpoint[3, 6+3*step] = 1.
        endpoint[4, 6+3*step+1] = 1.
        endpoint[5, 6+3*step+2] = 1.
        angle += H*rate
        rate += mean[6+3*step+1]
    return mv(endpoint, mean), mm(mm(endpoint, covariance), endpoint.T)


def verify_joint_and_endpoint(event_count=4):
    oracle = physical_oracle(event_count)
    plan = condition_prefix(*checkpoint(), DIFFUSION, EVENTS[:event_count])
    du = oracle['du']
    close(plan.angular_mean, oracle['primitive_mean'][:du])
    close(plan.angular_covariance, oracle['primitive_covariance'][:du, :du])
    close(plan.log_evidence, oracle['evidence'])
    cuu = oracle['primitive_covariance'][:du, :du]
    czu = oracle['primitive_covariance'][du:, :du]
    conditional_covariance = oracle['primitive_covariance'][du:, du:]-mm(czu, np.linalg.solve(cuu, czu.T))
    close(plan.residual_covariance, conditional_covariance)
    report = plan.evaluate(coordinates(du))
    for index, u in enumerate(report['angular_coordinates']):
        expected_mean, expected_covariance = conditional_physical_endpoint(oracle, u)
        close(report['angular_paths'][index, -1], expected_mean[A])
        close(report['conditional_mean'][index], expected_mean[L])
        close(report['conditional_covariance'][index], expected_covariance[np.ix_(L, L)])
        expected_z = oracle['primitive_mean'][du:]+mv(czu, np.linalg.solve(cuu, u-oracle['primitive_mean'][:du]))
        close(report['residual_mean'][index], expected_z)
    return plan, report, oracle


@pytest.mark.parametrize('event_count', [1, 2, 3, 4])
def test_independent_physical_joint_and_conditional_endpoint(event_count):
    plan, report, oracle = verify_joint_and_endpoint(event_count)
    for covariance in (plan.angular_covariance, plan.residual_covariance, report['covariance']):
        psd(covariance)
    for covariance in report['conditional_covariance']:
        psd(covariance)
    if event_count == 4:
        # Observation conditioning couples initially independent process innovations.
        assert abs(oracle['primitive_covariance'][3, 5]) > 1e-12
        assert np.max(np.abs(oracle['primitive_covariance'][:oracle['du'], oracle['du']:])) > 1e-12


def test_scalar_wheel_conditioning_and_gyro_ambiguity():
    mean, covariance = np.zeros(6), np.zeros((6, 6))
    mean[3:5] = [1.4, .1]
    variance_a, coupling, variance_z, variance_r = .25, .5, .16, .09
    covariance[4, 4] = variance_a
    covariance[3, 4] = covariance[4, 3] = coupling*variance_a
    covariance[3, 3] = coupling**2*variance_a+variance_z
    value = 1.1
    plan = condition_prefix(mean, covariance, [0, 0, 0], [Observation(10000, 'wheel', value, variance_r)])
    total = coupling**2*variance_a+variance_z+variance_r
    close(plan.angular_mean, [math.sqrt(variance_a)*coupling*(value-mean[3])/total])
    close(plan.angular_covariance, [[(variance_z+variance_r)/total]])
    assert plan.angular_mean[0] != 0.
    covariance = np.zeros((6, 6))
    covariance[4, 4], covariance[5, 5] = .04, .09
    mean[5] = -.02
    gyro = Observation(10000, 'gyro', .15, .01)
    plan = condition_prefix(mean, covariance, [0, 0, 0], [gyro])
    loading = np.array([.2, .3])
    innovation = .04+.09+.01
    close(plan.angular_mean, loading*(gyro.value-mean[4]-mean[5])/innovation)
    close(plan.angular_covariance, np.eye(2)-np.outer(loading, loading)/innovation)
    null_direction = np.array([.3, -.2])
    close(mv(plan.angular_covariance, null_direction), null_direction)


def verify_importance():
    plan = condition_prefix(*checkpoint(), DIFFUSION, EVENTS)
    report = plan.evaluate(coordinates(plan.angular_dimension))
    u = report['angular_coordinates']
    z = coordinates(plan.residual_dimension)
    terms = plan.density_terms(u, z)
    oracle = physical_oracle()
    primitive_sensor = mm(oracle['sensor'], oracle['transform'])
    hu, hz = primitive_sensor[:, :oracle['du']], primitive_sensor[:, oracle['du']:]
    residual = oracle['values']-mv(oracle['sensor'], oracle['prior_mean'])
    marginal = oracle['noise']+mm(hz, hz.T)
    for index in range(len(u)):
        close(terms['log_prior'][index], independent_log_density(u[index], np.zeros(len(u[index])), np.eye(len(u[index]))))
        close(terms['log_likelihood'][index], independent_log_density(residual, mv(hu, u[index]), marginal))
        close(terms['log_proposal'][index], independent_log_density(u[index], oracle['primitive_mean'][:oracle['du']],
            oracle['primitive_covariance'][:oracle['du'], :oracle['du']]))
        close(terms['log_importance'][index], oracle['evidence'])
        close(terms['log_joint_importance'][index], oracle['evidence'])
    close(report['logweights'], np.full(3, -math.log(3)))
    close(report['log_importance'], np.full(3, oracle['evidence']))
    return plan, report, oracle


def test_normalized_marginal_and_full_joint_importance_identities():
    verify_importance()


def test_direct_physical_maps_and_finite_difference_derivatives():
    plan = condition_prefix(*checkpoint(), DIFFUSION, EVENTS)
    u, z = coordinates(plan.angular_dimension), coordinates(plan.residual_dimension)
    maps = plan.path_maps(u)
    step = 1e-6
    for row in range(len(u)):
        physical = physical_point(u[row], z[row], 2)
        expected, paths = direct_physical_endpoint(physical, 2)
        close(maps['angular_paths'][row], paths)
        close(maps['offset'][row]+mv(maps['residual_map'][row], z[row]), expected[L])
        for column in range(plan.residual_dimension):
            delta = np.zeros(plan.residual_dimension)
            delta[column] = step
            plus = direct_physical_endpoint(physical_point(u[row], z[row]+delta, 2), 2)[0][L]
            minus = direct_physical_endpoint(physical_point(u[row], z[row]-delta, 2), 2)[0][L]
            np.testing.assert_allclose((plus-minus)/(2*step), maps['residual_map'][row, :, column],
                                       atol=1e-7, rtol=1e-7)
    # Last speed diffusion changes final speed, never current displacement.
    close(maps['residual_map'][:, :2, -1], np.zeros((3, 2)))
    first = condition_prefix(*checkpoint(), DIFFUSION, EVENTS[:2])
    origin = np.zeros((1, first.angular_dimension))
    changed = origin.copy()
    changed[0, 3] = 1.
    close(first.path_maps(origin)['angular_paths'][:, -1, 0], first.path_maps(changed)['angular_paths'][:, -1, 0])
    origin = np.zeros((1, plan.angular_dimension))
    changed = origin.copy()
    changed[0, 3] = 1.
    close(plan.path_maps(changed)['angular_paths'][0, -1, 0]-plan.path_maps(origin)['angular_paths'][0, -1, 0],
          H*math.sqrt(DIFFUSION[1]*H))
    # It changes theta_2, not the old theta_1 used for the second displacement.
    close(plan.path_maps(changed)['offset'][:, :2], plan.path_maps(origin)['offset'][:, :2])


def test_gaussian_density_finite_difference_gradients():
    plan = condition_prefix(*checkpoint(), DIFFUSION, EVENTS)
    oracle = physical_oracle()
    point = coordinates(plan.angular_dimension)[2]
    primitive_sensor = mm(oracle['sensor'], oracle['transform'])
    hu, hz = primitive_sensor[:, :oracle['du']], primitive_sensor[:, oracle['du']:]
    marginal = oracle['noise']+mm(hz, hz.T)
    residual = oracle['values']-mv(oracle['sensor'], oracle['prior_mean'])
    expected = {
        'log_prior': -point,
        'log_proposal': -np.linalg.solve(oracle['primitive_covariance'][:oracle['du'], :oracle['du']],
                                        point-oracle['primitive_mean'][:oracle['du']]),
        'log_likelihood': mv(hu.T, np.linalg.solve(marginal, residual-mv(hu, point))),
    }
    close(expected['log_prior']+expected['log_likelihood']-expected['log_proposal'], np.zeros(len(point)))
    hessian = (-np.eye(len(point))-mm(hu.T, np.linalg.solve(marginal, hu))
               +np.linalg.solve(plan.angular_covariance, np.eye(len(point))))
    close(hessian, np.zeros_like(hessian))
    step = 1e-6
    for column in range(len(point)):
        delta = np.zeros(len(point))
        delta[column] = step
        plus, minus = plan.density_terms((point+delta)[None, :]), plan.density_terms((point-delta)[None, :])
        for key, gradient in expected.items():
            np.testing.assert_allclose((plus[key][0]-minus[key][0])/(2*step), gradient[column], atol=1e-7, rtol=1e-7)


def test_full_six_state_within_and_between_covariances():
    plan, report, oracle = verify_joint_and_endpoint()
    components, within = [], []
    for u in report['angular_coordinates']:
        mean, covariance = conditional_physical_endpoint(oracle, u)
        components.append(mean)
        within.append(covariance)
    mean = np.array([math.fsum(row[index] for row in components)/len(components) for index in range(6)])
    covariance = np.zeros((6, 6))
    for row, conditional in zip(components, within):
        covariance += (conditional+np.outer(row-mean, row-mean))/len(components)
    close(report['mean'], mean)
    close(report['covariance'], covariance)
    assert np.max(np.abs(covariance[np.ix_(A, L)])) > 1e-12


@pytest.mark.parametrize('angular,linear', [(False, False), (True, False), (False, True)])
def test_empty_primitive_and_empty_prefix_limits(angular, linear):
    mean, covariance = np.zeros(6), np.zeros((6, 6))
    mean[2], mean[3] = .2, -1.
    if angular:
        covariance[2, 2] = .4
    if linear:
        covariance[np.ix_(L, L)] = mm(J, J.T)
    for events in ((), (Observation(10000, 'gyro', .03, .01), Observation(10000, 'wheel', -.8, .04))):
        plan = condition_prefix(mean, covariance, [0, 0, 0], events)
        report = plan.evaluate(coordinates(plan.angular_dimension))
        terms = plan.density_terms(report['angular_coordinates'], coordinates(plan.residual_dimension))
        close(terms['log_importance'], np.full(3, plan.log_evidence))
        close(terms['log_joint_importance'], np.full(3, plan.log_evidence))
        psd(report['covariance'])
        if not events:
            assert plan.log_evidence == 0.
            close(plan.angular_covariance, np.eye(plan.angular_dimension))
            close(plan.residual_covariance, np.eye(plan.residual_dimension))
        if not angular and not linear:
            close(report['covariance'], np.zeros((6, 6)))


@pytest.mark.parametrize('speed', [-2., 0., 2.])
def test_signed_speed_periodic_displacement_and_unwrapped_yaw(speed):
    mean, covariance = np.zeros(6), np.zeros((6, 6))
    mean[2], mean[3], mean[4] = .3, speed, .2
    event = Observation(20000, 'gyro', .2, .01)
    first = condition_prefix(mean, covariance, [0, 0, 0], [event]).evaluate(np.empty((1, 0)))
    mean[2] += 2*math.pi
    second = condition_prefix(mean, covariance, [0, 0, 0], [event]).evaluate(np.empty((1, 0)))
    close(first['conditional_mean'], second['conditional_mean'])
    close(second['angular_paths'][:, :, 0]-first['angular_paths'][:, :, 0], np.full((1, 3), 2*math.pi))
    expected = speed*H*np.array([math.cos(.3)+math.cos(.3+H*.2), math.sin(.3)+math.sin(.3+H*.2)])
    close(first['conditional_mean'][0, :2], expected)


def test_causal_prefixes_repeatability_and_immutable_outputs():
    mean, covariance = checkpoint()
    gyro = condition_prefix(mean, covariance, DIFFUSION, EVENTS[:1])
    before = gyro.evaluate(coordinates(gyro.angular_dimension))
    saved = {key: value.copy() for key, value in before.items() if isinstance(value, np.ndarray)}
    wheel = condition_prefix(mean, covariance, DIFFUSION, EVENTS[:2])
    later = wheel.evaluate(coordinates(wheel.angular_dimension))
    repeated = condition_prefix(mean, covariance, DIFFUSION, EVENTS[:1]).evaluate(coordinates(gyro.angular_dimension))
    for key, value in saved.items():
        np.testing.assert_array_equal(before[key], value)
        np.testing.assert_array_equal(repeated[key], value)
    assert np.max(np.abs(later['angular_coordinates']-before['angular_coordinates'])) > 1e-12
    # Wheel also moves angular innovations through their gyro-induced correlations.
    assert np.max(np.abs(wheel.angular_mean[3:]-gyro.angular_mean[3:])) > 1e-12
    mean[:] = 99.
    covariance[:] = 99.
    np.testing.assert_array_equal(gyro.evaluate(coordinates(gyro.angular_dimension))['mean'], before['mean'])
    with pytest.raises(FrozenInstanceError):
        gyro.phase = 'changed'
    with pytest.raises(TypeError):
        before['mean'] = np.zeros(6)
    with pytest.raises(ValueError):
        before['mean'][0] = 0.
    with pytest.raises(ValueError):
        gyro.angular_mean.setflags(write=True)


@pytest.mark.parametrize('events', [
    (Observation(0, 'gyro', 0., .1),),
    (Observation(10000, 'wheel', 0., .1), Observation(10000, 'gyro', 0., .1)),
    (Observation(10000, 'gyro', 0., .1), Observation(10000, 'gyro', 0., .1)),
    (Observation(20000, 'gyro', 0., .1), Observation(10000, 'wheel', 0., .1)),
    (Observation(10000.0, 'gyro', 0., .1),),
    (Observation(10000, 'gps', 0., .1),),
    (Observation(10000, 'gyro', 0., 0.),),
    (Observation(10000, 'gyro', math.nan, .1),),
    (Observation(3000000, 'gyro', 0., .1),),
])
def test_invalid_event_prefix_is_rejected(events):
    with pytest.raises(InvalidInput):
        condition_prefix(*checkpoint(), DIFFUSION, events)


def test_resource_bounds_precede_large_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid size reached Gaussian construction')
    with monkeypatch.context() as guard:
        guard.setattr(observed, 'split_checkpoint', forbidden)
        with pytest.raises(InvalidInput, match='event bound'):
            condition_prefix(*checkpoint(), DIFFUSION,
                             [Observation(index+1, 'gyro', 0., .1) for index in range(65)])
        with pytest.raises(InvalidInput, match='substep bound'):
            condition_prefix(*checkpoint(), DIFFUSION, [Observation(1290000, 'gyro', 0., .1)])
    plan = condition_prefix(*checkpoint(), DIFFUSION, ())
    monkeypatch.setattr(observed.PrefixPosterior, 'path_maps', forbidden)
    with pytest.raises(InvalidInput, match='row bound|row'):
        plan.evaluate(np.zeros((2049, plan.angular_dimension)))


def test_support_resolution_and_nonfinite_inputs():
    mean, covariance = np.zeros(6), np.zeros((6, 6))
    for angular in (np.outer([.3, .08, 0.], [.3, .08, 0.]), np.diag([.09, .01, 1e-16])):
        covariance[np.ix_(A, A)] = angular
        with pytest.raises(UnresolvedInference):
            condition_prefix(mean, covariance, [0, 0, 0], ())
    covariance[:] = 0.
    with pytest.raises(UnresolvedInference):
        condition_prefix(mean, covariance, [0, 0, 0], [Observation(10000, 'gyro', 0., 1e-16)])
    plan = condition_prefix(mean, covariance, [0, 1e-20, 1e-20], [Observation(10000, 'gyro', 0., .1)])
    assert plan.angular_dimension == 2  # Positive diffusion is never removed by a rank threshold.
    with pytest.raises(InvalidInput):
        plan.evaluate([[math.nan, 0.]])
    with pytest.raises(InvalidInput):
        plan.evaluate(np.empty((0, 2)))
    with pytest.raises(InvalidInput):
        plan.density_terms([[0., 0.]], np.empty((2, 0)))
    with pytest.raises(InvalidInput):
        condition_prefix(mean, covariance, [-1, 0, 0], ())
    with pytest.raises(InvalidInput):
        condition_prefix(mean, covariance, [0, 0, 0], (), checkpoint_us=True)
    with pytest.raises(NumericalFailure):
        plan.density_terms([[1e308, 0.]])


@pytest.mark.parametrize('mutation', ['omit_likelihood', 'double_likelihood', 'omit_prior_normalizer',
                                     'omit_proposal_normalizer'])
def test_importance_mutations_are_detected_and_restored(monkeypatch, mutation):
    original = observed._standard_log_density
    original_gaussian = observed._gaussian_log_density
    with monkeypatch.context() as changed:
        if mutation == 'omit_likelihood':
            changed.setattr(observed, '_importance_ratio', lambda prior, likelihood, proposal: prior-proposal)
        elif mutation == 'double_likelihood':
            changed.setattr(observed, '_importance_ratio', lambda prior, likelihood, proposal: prior+2*likelihood-proposal)
        elif mutation == 'omit_prior_normalizer':
            changed.setattr(observed, '_standard_log_density',
                            lambda points: original(points)+.5*points.shape[1]*math.log(2*math.pi))
        else:
            def without_proposal_normalizer(points, mean, factor):
                value = original_gaussian(points, mean, factor)
                if factor.shape == (7, 7):  # The frozen two-step proposal has seven U coordinates.
                    value += .5*(7*math.log(2*math.pi)+np.linalg.slogdet(mm(factor, factor.T))[1])
                return value
            changed.setattr(observed, '_gaussian_log_density', without_proposal_normalizer)
        with pytest.raises(AssertionError):
            verify_importance()
    verify_importance()


@pytest.mark.parametrize('mutation', ['checkpoint_cross', 'conditional_cross', 'new_angle'])
def test_covariance_and_timing_mutations_are_detected_and_restored(monkeypatch, mutation):
    original_covariance = observed._residual_covariance
    with monkeypatch.context() as changed:
        if mutation == 'checkpoint_cross':
            changed.setattr(observed, '_checkpoint_linear_map', lambda split, factor: np.zeros((3, factor.shape[1])))
        elif mutation == 'conditional_cross':
            changed.setattr(observed, '_residual_covariance',
                lambda gain, design, variances: np.diag(np.diag(original_covariance(gain, design, variances))))
        else:
            changed.setattr(observed, '_old_directions', lambda paths:
                np.stack((np.cos(paths[:, 1:, 0]), np.sin(paths[:, 1:, 0])), axis=-1))
        with pytest.raises(AssertionError):
            verify_joint_and_endpoint()
    verify_joint_and_endpoint()



def test_coordinate_width_is_checked_before_conversion(monkeypatch):
    plan = condition_prefix(*checkpoint(), DIFFUSION, ())

    class ConversionTrap:
        def __array__(self, *args, **kwargs):
            raise AssertionError('unbounded coordinate input was converted')

    with pytest.raises(InvalidInput):
        plan.evaluate(ConversionTrap())
    with pytest.raises(InvalidInput):
        plan.evaluate([ConversionTrap()])
    original = np.asarray

    def guarded_conversion(value, *args, **kwargs):
        if isinstance(value, list):
            raise AssertionError('wrong-width row reached dtype conversion')
        return original(value, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(np, 'asarray', guarded_conversion)
        with pytest.raises(InvalidInput, match='width'):
            plan.evaluate([[0.]*(plan.angular_dimension+1)])
        with pytest.raises(InvalidInput, match='width'):
            plan.evaluate([[0.]*plan.angular_dimension, []])
