"""Independent deterministic checks of the frozen conditional path mixture.

Only control seed 94999 is used here. Study randomization belongs to the gated
runner, and no recorded samples or earlier candidate/reference helpers enter.
"""
from copy import deepcopy
import math

import numpy as np
import pytest

from tools import trajectory_mixture as candidate
from tools.trajectory_mixture import (InvalidInput, NumericalFailure, Observation,
                                      TrajectoryMixture, UnresolvedInference,
                                      run_events, split_checkpoint)


A_INDEX = [2, 4, 5]
L_INDEX = [0, 1, 3]
TOLERANCE = 1e-10
J = np.array([[.2, 0, 0], [.05, .15, 0], [.04, -.02, .25]])
LA = np.array([[.3, 0, 0], [.02, .08, 0], [-.01, -.03, .04]])
K = np.array([[.4, .2, -.3], [-.2, .5, .1], [.3, -.4, .2]])
MU_A = np.array([.2, .1, -.02])
MU_L = np.array([.5, -.25, 1.5])
DIFFUSION = np.array([.04, .0004, .0001])


def control_rng(case_id, phase):
    return np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([20261018, case_id, 94999, phase])))


def checkpoint(case='O'):
    mean, covariance = np.zeros(6), np.zeros((6, 6))
    if case == 'H':
        mean[3] = 2.
        covariance[2, 2] = .4
    else:
        mean[L_INDEX] = MU_L
        mean[A_INDEX] = MU_A if case == 'O' else [.2, 0, 0]
        covariance[np.ix_(L_INDEX, L_INDEX)] = J@J.T
        if case == 'O':
            angular = LA@LA.T
            cross = K@angular
            covariance[np.ix_(A_INDEX, A_INDEX)] = angular
            covariance[np.ix_(L_INDEX, A_INDEX)] = cross
            covariance[np.ix_(A_INDEX, L_INDEX)] = cross.T
            covariance[np.ix_(L_INDEX, L_INDEX)] += K@angular@K.T
    return mean, covariance


def initialize(case='O'):
    case_id = {'L': 1, 'H': 2, 'O': 3}[case]
    diffusion = {'L': [.04, 0, 0], 'H': [0, 0, 0], 'O': DIFFUSION}[case]
    return TrajectoryMixture.initialize(*checkpoint(case), 2048,
                                        control_rng(case_id, 0), control_rng(case_id, 1),
                                        diffusion)


def declared_pair(centered=False):
    offset = LA@np.array([1., -1., 1.])
    angular = np.array([MU_A, MU_A]) if centered else np.array([MU_A+offset, MU_A-offset])
    means = np.array([MU_L+K@(point-MU_A) for point in angular])
    covariances = np.array([J@J.T, J@J.T])
    return TrajectoryMixture.from_components(
        angular, means, covariances, np.full(2, -math.log(2)), np.array([0, 1]),
        control_rng(3, 1), DIFFUSION)


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=TOLERANCE)


def assert_covariance(covariance):
    scale = TOLERANCE*max(1., float(np.abs(covariance).max()))
    assert np.isfinite(covariance).all()
    assert np.abs(covariance-covariance.T).max() <= scale
    assert np.linalg.eigvalsh(covariance).min() >= -scale


def state_copy(mixture):
    return {key: value.copy() for key, value in mixture.snapshot().items()
            if isinstance(value, np.ndarray)}, deepcopy(mixture.transition_rng.bit_generator.state)


def assert_state_preserved(mixture, saved):
    arrays, rng_state = saved
    current = mixture.snapshot()
    for key, value in arrays.items():
        np.testing.assert_array_equal(current[key], value)
    assert mixture.transition_rng.bit_generator.state == rng_state


def verify_correlated_split():
    mean, covariance = checkpoint()
    split = split_checkpoint(mean, covariance)
    points = np.array([MU_A, MU_A+LA@np.array([1., -1., 1.])])
    expected_means = np.array([MU_L+K@(point-MU_A) for point in points])
    assert_close(split.conditional_means(points), expected_means)
    assert_close(split.conditional_gain, K)
    assert_close(split.conditional_covariance, J@J.T)
    angular = covariance[np.ix_(A_INDEX, A_INDEX)]
    reconstructed_cross = split.conditional_gain@angular
    reconstructed_linear = split.conditional_covariance+split.conditional_gain@angular@split.conditional_gain.T
    assert_close(reconstructed_cross, covariance[np.ix_(L_INDEX, A_INDEX)])
    assert_close(reconstructed_linear, covariance[np.ix_(L_INDEX, L_INDEX)])
    assert_covariance(split.conditional_covariance)


def test_full_correlated_checkpoint_split_from_input_covariance():
    verify_correlated_split()


def test_declared_deterministic_support_and_no_unneeded_draws():
    initial_rng, transition_rng = control_rng(1, 0), control_rng(1, 1)
    before_initial = deepcopy(initial_rng.bit_generator.state)
    before_transition = deepcopy(transition_rng.bit_generator.state)
    linear = TrajectoryMixture.initialize(*checkpoint('L'), 2048, initial_rng,
                                          transition_rng, [.04, 0, 0])
    assert initial_rng.bit_generator.state == before_initial
    assert transition_rng.bit_generator.state == before_transition
    assert_close(linear.full_moments()[0], checkpoint('L')[0])
    assert_close(linear.full_moments()[1], checkpoint('L')[1])
    np.testing.assert_array_equal(linear.angular, np.tile([.2, 0, 0], (2048, 1)))
    mean, covariance = checkpoint('H')
    split = split_checkpoint(mean, covariance)
    np.testing.assert_array_equal(split.angular_active, [0])
    np.testing.assert_array_equal(split.conditional_active, [])
    np.testing.assert_array_equal(split.conditional_covariance, np.zeros((3, 3)))
    with pytest.raises(InvalidInput, match='outside deterministic'):
        split.conditional_means(np.array([[0., .03, 0.]]))


def test_invalid_and_unresolved_checkpoint_support_are_distinct():
    mean, covariance = checkpoint('L')
    covariance[0, 2] = covariance[2, 0] = 1e-6
    with pytest.raises(InvalidInput, match='deterministic'):
        split_checkpoint(mean, covariance)
    for angular in (np.outer([.3, .08, 0.], [.3, .08, 0.]),
                    np.diag([.3**2, .08**2, 1e-16])):
        mean, covariance = checkpoint('L')
        covariance[np.ix_(A_INDEX, A_INDEX)] = angular
        with pytest.raises(UnresolvedInference, match='active support'):
            split_checkpoint(mean, covariance)


def test_failed_initialization_restores_initial_rng():
    rng = control_rng(3, 0)
    saved = deepcopy(rng.bit_generator.state)
    with pytest.raises(InvalidInput, match='diffusion'):
        TrajectoryMixture.initialize(*checkpoint(), 2048, rng, control_rng(3, 1), [-1, 0, 0])
    assert rng.bit_generator.state == saved


def verify_two_steps():
    mixture = declared_pair(centered=True)
    step = .01
    initial_covariance = J@J.T
    old_angle, old_rate = MU_A[:2]
    first_angle = old_angle+step*old_rate
    first_innovation = np.array([0., .03, -.01])
    first_transform = np.eye(3)
    first_transform[:2, 2] = step*np.array([math.cos(old_angle), math.sin(old_angle)])
    first_covariance = first_transform@initial_covariance@first_transform.T
    first_covariance[2, 2] += DIFFUSION[0]*step
    mixture.predict_step(step, angular_innovation=first_innovation)
    assert_close(mixture.conditional_mean, np.tile(first_transform@MU_L, (2, 1)))
    assert_close(mixture.conditional_covariance, np.tile(first_covariance, (2, 1, 1)))
    assert_close(mixture.angular, np.tile([first_angle, old_rate+.03, MU_A[2]-.01], (2, 1)))
    # Independent expanded map: ell_2 = T ell_0 + b1 eta_v1 + b2 eta_v2.
    transform = np.eye(3)
    transform[:2, 2] = step*np.array([math.cos(old_angle)+math.cos(first_angle),
                                     math.sin(old_angle)+math.sin(first_angle)])
    first_noise = np.array([step*math.cos(first_angle), step*math.sin(first_angle), 1.])
    second_noise = np.array([0., 0., 1.])
    expected_covariance = (transform@initial_covariance@transform.T
                           +DIFFUSION[0]*step*(np.outer(first_noise, first_noise)
                                              +np.outer(second_noise, second_noise)))
    mixture.predict_step(step, angular_innovation=np.zeros(3))
    assert_close(mixture.conditional_mean, np.tile(transform@MU_L, (2, 1)))
    assert_close(mixture.conditional_covariance, np.tile(expected_covariance, (2, 1, 1)))
    expected_angular = [first_angle+step*(old_rate+.03), old_rate+.03, MU_A[2]-.01]
    assert_close(mixture.angular, np.tile(expected_angular, (2, 1)))
    for covariance in mixture.conditional_covariance:
        assert_covariance(covariance)
    assert_covariance(mixture.full_moments()[1])


def test_old_state_euler_and_independent_end_step_diffusion():
    verify_two_steps()


def test_zero_step_is_a_noop_and_innovation_support_is_checked():
    mixture = declared_pair(centered=True)
    saved = state_copy(mixture)
    before_count = mixture.substeps
    mixture.predict_step(0.)
    assert_state_preserved(mixture, saved)
    assert mixture.substeps == before_count
    with pytest.raises(InvalidInput, match='transition support'):
        mixture.predict_step(.01, angular_innovation=[.01, 0, 0])
    assert_state_preserved(mixture, saved)


def independent_empirical_moments(points):
    count = len(points)
    mean = np.array([math.fsum(float(row[j]) for row in points)/count for j in range(6)])
    centered = points-mean
    covariance = np.empty((6, 6))
    for j in range(6):
        for k in range(6):
            covariance[j, k] = math.fsum(float(row[j])*float(row[k]) for row in centered)/count
    return mean, covariance


def test_shared_heading_branches_match_same_empirical_endpoint():
    first = initialize('H')
    eighth = first.clone()
    angles = first.angular[:, 0].copy()
    first_rng = deepcopy(first.transition_rng.bit_generator.state)
    endpoint = np.zeros((2048, 6))
    endpoint[:, 0] = .1*2*np.cos(angles)
    endpoint[:, 1] = .1*2*np.sin(angles)
    endpoint[:, 2] = angles
    endpoint[:, 3] = 2.
    expected = independent_empirical_moments(endpoint)
    first.predict_step(.1)
    for _ in range(8):
        eighth.predict_step(.1/8)
    for branch in (first, eighth):
        assert_close(branch.conditional_mean[:, :2], endpoint[:, :2])
        np.testing.assert_array_equal(branch.angular[:, 0], angles)
        np.testing.assert_array_equal(branch.conditional_covariance, np.zeros((2048, 3, 3)))
        np.testing.assert_array_equal(branch.conditional_mean[:, 2], np.full(2048, 2.))
        actual = branch.full_moments()
        for value, target in zip(actual, expected):
            assert_close(value, target)
        np.testing.assert_array_equal(actual[1][[3, 4, 5], :], np.zeros((3, 6)))
        assert_covariance(actual[1])
        assert branch.transition_rng.bit_generator.state == first_rng
        assert_close(np.exp(branch.logweights), np.full(2048, 1/2048))
    assert_close(first.full_moments()[1], eighth.full_moments()[1])
    assert first.substeps == 1 and eighth.substeps == 8


def normal_density(value, mean, variance):
    return math.exp(-(value-mean)**2/(2*variance))/math.sqrt(2*math.pi*variance)


def verify_measurements_and_reconstruction():
    mixture = declared_pair()
    before_angular = mixture.angular.copy()
    before_mean = mixture.conditional_mean.copy()
    before_covariance = mixture.conditional_covariance.copy()
    gyro_likelihood = np.array([normal_density(.09, row[1]+row[2], .02**2)
                               for row in before_angular])
    gyro_evidence = .5*sum(gyro_likelihood)
    gyro_weights = .5*gyro_likelihood/gyro_evidence
    mixture.observe('gyro', .09, .02**2)
    np.testing.assert_array_equal(mixture.angular, before_angular)
    np.testing.assert_array_equal(mixture.conditional_mean, before_mean)
    np.testing.assert_array_equal(mixture.conditional_covariance, before_covariance)
    assert_close(np.exp(mixture.logweights), gyro_weights)
    assert_close(mixture.log_evidence, math.log(gyro_evidence))
    # The frozen pair's rate/bias offsets cancel, so equal gyro weights are valid.
    expected_mean, expected_covariance, wheel_likelihood = [], [], []
    for mean, covariance in zip(before_mean, before_covariance):
        innovation = covariance[2, 2]+.1**2
        cross = covariance[:, 2]
        expected_mean.append(mean+cross*(1.4-mean[2])/innovation)
        expected_covariance.append(covariance-np.outer(cross, cross)/innovation)
        wheel_likelihood.append(normal_density(1.4, mean[2], innovation))
    expected_mean, expected_covariance = np.array(expected_mean), np.array(expected_covariance)
    wheel_evidence = float(np.dot(gyro_weights, wheel_likelihood))
    expected_weights = gyro_weights*np.array(wheel_likelihood)/wheel_evidence
    mixture.observe('wheel', 1.4, .1**2)
    np.testing.assert_array_equal(mixture.angular, before_angular)
    assert_close(mixture.conditional_mean, expected_mean)
    assert_close(mixture.conditional_covariance, expected_covariance)
    assert_close(np.exp(mixture.logweights), expected_weights)
    assert_close(mixture.log_evidence, math.log(gyro_evidence)+math.log(wheel_evidence))
    component_means, component_covariances = np.zeros((2, 6)), np.zeros((2, 6, 6))
    component_means[:, A_INDEX] = before_angular
    component_means[:, L_INDEX] = expected_mean
    for i in range(2):
        component_covariances[i][np.ix_(L_INDEX, L_INDEX)] = expected_covariance[i]
    full_mean = expected_weights@component_means
    difference = component_means[0]-component_means[1]
    within = sum(weight*covariance for weight, covariance in zip(expected_weights, component_covariances))
    full_covariance = within+expected_weights[0]*expected_weights[1]*np.outer(difference, difference)
    raw_second = sum(weight*(covariance+np.outer(mean, mean))
                     for weight, mean, covariance in zip(expected_weights, component_means, component_covariances))
    assert_close(raw_second-np.outer(full_mean, full_mean), full_covariance)
    actual_mean, actual_covariance = mixture.full_moments()
    assert_close(actual_mean, full_mean)
    assert_close(actual_covariance, full_covariance)
    assert_covariance(actual_covariance)
    for covariance in mixture.conditional_covariance:
        assert_covariance(covariance)


def test_gyro_wheel_likelihoods_conditionals_and_full_reconstruction():
    verify_measurements_and_reconstruction()


def test_finite_log_weights_and_all_states_survive_reporting_underflow():
    mixture = declared_pair()
    angular, means = mixture.angular.copy(), mixture.conditional_mean.copy()
    covariance, identities = mixture.conditional_covariance.copy(), mixture.ids.copy()
    mixture.apply_log_likelihood([0., -1000.])
    snapshot = mixture.snapshot()
    assert np.isfinite(snapshot['logweights']).all()
    assert snapshot['underflow_count'] > 0
    assert len(snapshot['ids']) == 2
    mixture.apply_log_likelihood([-1000., 0.])
    assert_close(np.exp(mixture.logweights), [.5, .5])
    np.testing.assert_array_equal(mixture.angular, angular)
    np.testing.assert_array_equal(mixture.conditional_mean, means)
    np.testing.assert_array_equal(mixture.conditional_covariance, covariance)
    np.testing.assert_array_equal(mixture.ids, identities)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_log_likelihoods_fail_transactionally(invalid):
    mixture = declared_pair()
    saved = state_copy(mixture)
    evidence = mixture.log_evidence
    with pytest.raises(NumericalFailure, match='finite'):
        mixture.apply_log_likelihood([0., invalid])
    assert_state_preserved(mixture, saved)
    assert mixture.log_evidence == evidence


def observed_events():
    events = [Observation(10000, 'gyro', .09, .02**2),
              Observation(10000, 'wheel', 1.4, .1**2)]
    for j in range(1, 21):
        events.append(Observation(50000*j, 'gyro', .08+.01*(-1)**j, .02**2))
        if j % 2 == 0:
            k = j//2
            events.append(Observation(100000*k, 'wheel', 1.5+.1*(-1)**k, .1**2))
    return events


REPORTS = [0, 10000, 250000, 500000, 750000, 1000000]


def test_reports_and_changed_future_values_do_not_change_earlier_prefixes():
    full, final_only, changed = initialize(), initialize(), initialize()
    full_snapshots, diagnostics = run_events(full, observed_events(), REPORTS)
    final_snapshots, _ = run_events(final_only, observed_events(), [1000000])
    for key in ('angular', 'conditional_mean', 'conditional_covariance', 'logweights', 'ids', 'mean', 'covariance'):
        np.testing.assert_array_equal(full_snapshots[1000000][key], final_snapshots[1000000][key])
    assert full.transition_rng.bit_generator.state == final_only.transition_rng.bit_generator.state
    changed_events = observed_events()
    changed_events[-2] = Observation(1000000, 'gyro', 9., .02**2)
    changed_events[-1] = Observation(1000000, 'wheel', -9., .1**2)
    changed_snapshots, _ = run_events(changed, changed_events, REPORTS)
    for timestamp in REPORTS[:-1]:
        for key in ('angular', 'conditional_mean', 'conditional_covariance', 'logweights', 'ids', 'mean', 'covariance'):
            np.testing.assert_array_equal(full_snapshots[timestamp][key], changed_snapshots[timestamp][key])
    assert diagnostics[0]['phase'] == 'gyro' and diagnostics[1]['phase'] == 'wheel'
    assert full_snapshots[10000]['phase'] == 'wheel'
    assert diagnostics[0]['substeps'] == diagnostics[1]['substeps']
    assert full_snapshots[1000000]['observation_count'] == len(observed_events())


@pytest.mark.parametrize('bad_events', [
    [Observation(10000, 'wheel', 1.4, .1**2), Observation(10000, 'gyro', .09, .02**2)],
    [Observation(10000, 'gyro', .09, .02**2), Observation(10000, 'gyro', .09, .02**2)]])
def test_invalid_input_order_is_rejected_before_any_proposal(bad_events):
    mixture = initialize()
    saved = state_copy(mixture)
    before_time = mixture.timestamp_us
    with pytest.raises(InvalidInput, match='out-of-order'):
        run_events(mixture, bad_events, [10000])
    assert_state_preserved(mixture, saved)
    assert mixture.timestamp_us == before_time and mixture.observation_count == 0


@pytest.mark.parametrize('event', [Observation(10000, 'gyro', .09, 0.),
                                   Observation(10000, 'gyro', float('nan'), .02**2)])
def test_invalid_observation_is_transactional(event):
    mixture = initialize()
    saved = state_copy(mixture)
    with pytest.raises(InvalidInput):
        mixture.process(event)
    assert_state_preserved(mixture, saved)
    assert mixture.timestamp_us == 0 and mixture.observation_count == 0


def test_failure_after_proposal_restores_arrays_time_and_rng(monkeypatch):
    mixture = initialize()
    saved = state_copy(mixture)

    def failed_likelihood(*args):
        raise NumericalFailure('injected likelihood failure after propagation')

    with monkeypatch.context() as context:
        context.setattr(candidate, '_normal_log_density', failed_likelihood)
        with pytest.raises(NumericalFailure, match='injected likelihood'):
            mixture.process(observed_events()[0])
    assert_state_preserved(mixture, saved)
    assert mixture.timestamp_us == 0 and mixture.observation_count == 0 and mixture.substeps == 0


def test_non_event_report_is_rejected_without_prediction():
    mixture = initialize()
    saved = state_copy(mixture)
    with pytest.raises(InvalidInput, match='every report'):
        run_events(mixture, observed_events(), [0, 1, 1000000])
    assert_state_preserved(mixture, saved)


def test_mutation_dropping_checkpoint_cross_block_is_detected_and_restored(monkeypatch):
    original = candidate.split_checkpoint

    def dropped_cross(mean, covariance):
        changed = covariance.copy()
        changed[np.ix_(L_INDEX, A_INDEX)] = 0
        changed[np.ix_(A_INDEX, L_INDEX)] = 0
        return original(mean, changed)

    with monkeypatch.context() as context:
        context.setattr(candidate, 'split_checkpoint', dropped_cross)
        # Call through the module so the isolated source mutation is exercised.
        with pytest.raises(AssertionError):
            actual = candidate.split_checkpoint(*checkpoint())
            assert_close(actual.conditional_gain, K)
    verify_correlated_split()


def test_mutation_using_new_angle_for_displacement_is_detected_and_restored(monkeypatch):
    original = candidate._conditional_transition

    def new_angle(angular, mean, covariance, step, speed_diffusion):
        changed = angular.copy()
        changed[:, 0] += step*changed[:, 1]
        return original(changed, mean, covariance, step, speed_diffusion)

    with monkeypatch.context() as context:
        context.setattr(candidate, '_conditional_transition', new_angle)
        with pytest.raises(AssertionError):
            verify_two_steps()
    verify_two_steps()


def test_mutation_omitting_wheel_likelihood_is_detected_and_restored(monkeypatch):
    original = candidate._wheel_update

    def omitted_weight(mean, covariance, value, variance):
        updated_mean, updated_covariance, likelihood = original(mean, covariance, value, variance)
        return updated_mean, updated_covariance, np.zeros_like(likelihood)

    with monkeypatch.context() as context:
        context.setattr(candidate, '_wheel_update', omitted_weight)
        with pytest.raises(AssertionError):
            verify_measurements_and_reconstruction()
    verify_measurements_and_reconstruction()


def test_frozen_control_streams_use_component_major_order():
    initial_stream = control_rng(3, 0)
    standard = initial_stream.standard_normal((2048, 3))
    expected_angular = MU_A+np.einsum('ni,ji->nj', standard, LA, optimize=False)
    expected_conditional = MU_L+np.einsum('ij,nj->ni', K, expected_angular-MU_A, optimize=False)
    transition_stream = control_rng(3, 1)
    standard_noise = transition_stream.standard_normal((2048, 2))
    expected_after = expected_angular.copy()
    expected_after[:, 0] += .01*expected_angular[:, 1]
    expected_after[:, 1:] += standard_noise*np.sqrt(DIFFUSION[1:]*.01)
    mixture = initialize()
    assert_close(mixture.angular, expected_angular)
    assert_close(mixture.conditional_mean, expected_conditional)
    mixture.process(observed_events()[0])
    assert_close(mixture.angular, expected_after)
    assert mixture.transition_rng.bit_generator.state == transition_stream.bit_generator.state


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_retained_log_weights_are_never_dropped(invalid):
    mixture = declared_pair()
    with pytest.raises(NumericalFailure, match='retained log weight'):
        TrajectoryMixture.from_components(
            mixture.angular, mixture.conditional_mean, mixture.conditional_covariance,
            [0., invalid], mixture.ids, control_rng(3, 1), DIFFUSION)
    mixture.logweights[1] = invalid
    with pytest.raises(NumericalFailure, match='retained log weight'):
        mixture.snapshot()
    assert len(mixture.ids) == 2
