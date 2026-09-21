import numpy as np
import pytest

from tools.consistency import ensemble_nees, elapsed_settling, innovation_directions, state_statistics


def test_diagonal_nees_wrapped_heading_and_nominal_coverage():
    truth = np.zeros((2, 6))
    estimate = np.array([[2, 3, -np.pi + .01, 0, 0, .03], [6, 0, 0, 0, 0, 0]])
    truth[0, 2] = np.pi - .01
    covariance = np.tile(np.diag([4., 9., .04, 1., .01, .0009]), (2, 1, 1))
    result = state_statistics(estimate, truth, covariance)
    assert result["position_nees"] == pytest.approx([2, 9])
    assert result["full_nees"][0] == pytest.approx(2 + .02**2 / .04 + 1)
    assert result["heading_error"][0] == pytest.approx(.02)
    assert result["heading_std"] == pytest.approx([.2, .2])
    assert result["bias_error"] == pytest.approx([.03, 0])
    assert result["bias_std"] == pytest.approx([.03, .03])
    assert result["position_covered_95"].tolist() == [1, 0]
    assert result["valid_full"].all() and result["valid_position"].all()


def test_correlated_full_nees_uses_marginal_position_block():
    covariance = np.eye(6)
    covariance[0, 2] = covariance[2, 0] = .5
    estimate = np.array([[1, 0, 0, 0, 0, 0]])
    result = state_statistics(estimate, np.zeros((1, 6)), covariance[None])
    assert result["full_nees"][0] == pytest.approx(4 / 3)
    assert result["position_nees"][0] == pytest.approx(1)  # Inverse(P)'s position block would give 4/3.
    covariance[4, 5] = covariance[5, 4] = -.8
    assert state_statistics(estimate, np.zeros((1, 6)), covariance[None])["rate_bias_correlation"][0] == pytest.approx(-.8)


def test_invalid_and_singular_covariances_are_undefined_without_repair():
    covariance = np.tile(np.eye(6), (5, 1, 1))
    covariance[0, 0, 0] = 0  # Singular position block as well as full covariance.
    covariance[1, 5, 5] = -1  # Position diagnostic remains mathematically defined.
    covariance[2, 0, 1] = .1  # Materially asymmetric.
    covariance[3, 0, 0] = np.nan
    covariance[4, 5, 5] = 0  # Singular full covariance but valid position marginal.
    result = state_statistics(np.zeros((5, 6)), np.zeros((5, 6)), covariance)
    assert not result["valid_full"].any()
    assert result["valid_position"].tolist() == [False, True, False, False, True]
    assert np.isnan(result["full_nees"]).all()
    assert np.isnan(result["position_nees"][[0, 2, 3]]).all()
    assert np.isnan(result["position_covered_95"][[0, 2, 3]]).all()
    assert np.isnan(result["bias_std"][1])
    assert np.isnan(result["rate_bias_correlation"][[1, 4]]).all()
    with pytest.raises(ValueError, match="N x 6"):
        state_statistics(np.zeros((1, 3)), np.zeros((1, 3)), np.eye(3)[None])


def test_rotated_innovation_contributions_use_full_covariance():
    rotation = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2)
    covariance = rotation @ np.diag([1., 9.]) @ rotation.T
    residual = rotation @ np.array([2., 3.])
    result = innovation_directions(residual, covariance)
    assert result["eigenvalues"] == pytest.approx([1, 9])
    assert result["contributions"] == pytest.approx([4, 1])
    assert result["nis"] == pytest.approx(5)
    assert sum(result["contributions"]) == pytest.approx(result["nis"])
    np.testing.assert_allclose(result["eigenvectors"] @ result["projections"], residual, atol=1e-14)
    with pytest.raises(ValueError, match="singular"):
        innovation_directions([1, 1], [[1, 1], [1, 1]])
    with pytest.raises(ValueError, match="symmetric"):
        innovation_directions([1, 1], [[1, .1], [0, 1]])
    with pytest.raises(ValueError, match="finite"):
        innovation_directions([np.nan, 1], np.eye(2))


def test_ensemble_reference_uses_independent_run_count_and_reports_failures():
    result = ensemble_nees([1., 3.], dimension=2)
    assert result["independent_runs"] == 2 and result["finite_runs"] == 2
    assert result["failed_runs"] == 0 and result["mean_nees"] == 2
    # Fixed numerical reference: chi-square quantiles for df=4, divided by N=2.
    assert result["reference_lower"] == pytest.approx(.24220927854396496)
    assert result["reference_upper"] == pytest.approx(5.571643390938898)
    failure = ensemble_nees([1., np.nan, np.inf], dimension=2)
    assert failure["independent_runs"] == 3 and failure["failed_runs"] == 2
    assert failure["finite_mean_nees"] == 1
    assert failure["mean_nees"] is None and failure["reference_lower"] is None
    assert ensemble_nees([np.nan], dimension=6)["finite_mean_nees"] is None
    with pytest.raises(ValueError, match="one-dimensional"):
        ensemble_nees(np.ones((30, 100)), dimension=6)
    with pytest.raises(ValueError, match="negative"):
        ensemble_nees([-1], dimension=2)


def test_elapsed_settling_preserves_inclusive_201_sample_requirement():
    times = np.arange(202, dtype=np.int64) * 10_000
    result = elapsed_settling(times, np.ones(202), 0, 2_010_000)
    assert result["settling_status"] == "within_limit_at_return"
    assert result["settling_start_delay_s"] == 0
    assert result["settling_confirmation_delay_s"] == 2
    errors = np.r_[np.zeros(200), [1.01, 1.01]]
    result = elapsed_settling(times, errors, 0, 2_010_000)
    assert result["settling_status"] == "not_observed"
    assert result["settling_start_delay_s"] is None
    assert elapsed_settling(times[:200], errors[:200], 0, 1_990_000)["settling_status"] == "insufficient_followup"


def test_elapsed_settling_handles_irregular_grid_by_duration():
    times = np.array([0, 5_000, 12_000, 19_000, 25_000, 34_000])
    result = elapsed_settling(times, np.ones(len(times)), 0, 34_000, dwell_us=20_000)
    assert result["settling_confirmation_delay_s"] == .025
    assert result["gaps_exceeding_limit"] == 0
    assert result["maximum_observed_gap_us"] == 9_000
    # A missing exact reference sample starts the observed dwell later.
    result = elapsed_settling(times, np.ones(len(times)), 1_000, 34_000, dwell_us=20_000)
    assert result["settling_status"] == "recovered"
    assert result["settling_start_delay_s"] == .004
    assert result["settling_confirmation_delay_s"] == .024


def test_elapsed_settling_resets_on_gaps_and_threshold_crossings():
    times = np.array([0, 10_000, 20_000, 40_000, 50_000, 60_000, 70_000])
    result = elapsed_settling(times, np.zeros(len(times)), 0, 70_000, dwell_us=30_000)
    assert result["settling_status"] == "recovered"
    assert result["settling_start_delay_s"] == .04
    assert result["settling_confirmation_delay_s"] == .07
    assert result["gaps_exceeding_limit"] == 1
    errors = np.zeros(len(times))
    errors[-2] = 1.01
    result = elapsed_settling(times, errors, 0, 70_000, dwell_us=30_000)
    assert result["settling_status"] == "not_observed"
    assert result["settling_confirmation_delay_s"] is None
    with pytest.raises(ValueError, match="increasing"):
        elapsed_settling([0, 0], [0, 0], 0, 0)
    with pytest.raises(ValueError, match="increasing"):
        elapsed_settling(np.array([10, 0], dtype=np.uint64), [0, 0], 0, 10)
    with pytest.raises(ValueError, match="span"):
        elapsed_settling(times, errors, 0, 80_000)
