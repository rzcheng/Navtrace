import copy
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.position_geometry import position_geometry
from tools.run_long_outage import ROOT, read_pre_gps, run_study, summarize, validate_protocol


def protocol():
    return json.loads((ROOT/'scenarios/long_outage.json').read_text())


def test_geometry_keeps_bias_second_moment_and_centered_covariance_separate():
    result = position_geometry([[1, 0], [3, 0]], [np.diag([2., 4.])]*2)
    np.testing.assert_allclose(result['mean_error'], [2, 0])
    np.testing.assert_allclose(result['centered_sample_covariance'], [[2, 0], [0, 0]])
    np.testing.assert_allclose(result['error_second_moment'], [[5, 0], [0, 0]])
    np.testing.assert_allclose(result['aggregate_ratio_eigenvalues'], [0, 2.5])
    # Identical biased errors have zero spread but still a nonzero second moment.
    biased = position_geometry([[2, 0], [2, 0]], [np.eye(2)]*2)
    assert biased['centered_sample_covariance'] == [[0., 0.], [0., 0.]]
    assert biased['error_second_moment'] == [[4., 0.], [0., 0.]]


def test_geometry_ratios_are_rotation_invariant_and_not_mean_nees():
    errors = np.array([[1., 2.], [-1., 0.]])
    covariances = np.array([[[1., .2], [.2, 2.]], [[4., .5], [.5, 3.]]])
    angle = .73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    original = position_geometry(errors, covariances)
    rotated = position_geometry(errors @ rotation.T, rotation @ covariances @ rotation.T)
    np.testing.assert_allclose(original['aggregate_ratio_eigenvalues'], rotated['aggregate_ratio_eigenvalues'])
    simple = position_geometry([[1, 0], [2, 0]], [np.eye(2), 4*np.eye(2)])
    # Aggregate ratio trace = (1+4)/(1+4) = 1, per-run NEES mean is also 1 here.
    assert sum(simple['aggregate_ratio_eigenvalues']) == pytest.approx(1)
    different = position_geometry([[1, 0], [1, 0]], [np.eye(2), 4*np.eye(2)])
    assert sum(different['aggregate_ratio_eigenvalues']) == pytest.approx(.4)
    assert sum(different['aggregate_ratio_eigenvalues']) != pytest.approx((1+1/4)/2)


def test_invalid_geometry_preserves_planned_count_and_single_run_has_no_sample_covariance():
    result = position_geometry([[1, 0], [np.nan, 0], [0, 0]], [np.eye(2), np.eye(2), -np.eye(2)])
    assert result['planned_runs'] == 3 and result['valid_runs'] == 1 and result['invalid_runs'] == 2
    assert result['error_second_moment'] is None
    single = position_geometry([[1, 0]], [np.eye(2)])
    assert single['centered_sample_covariance'] is None
    assert single['error_second_moment'] == [[1, 0], [0, 0]]


def test_pre_gps_reader_does_not_use_post_update_values(tmp_path):
    row = {'timestamp_us': 22000000}
    for phase, value in [('pre', 3), ('post', 9)]:
        row.update({f'{phase}_x{i}': value+i for i in range(6)})
        row.update({f'{phase}_p{i}{j}': value if i == j else 0 for i in range(6) for j in range(6)})
    path = tmp_path/'gps.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    state, covariance = read_pre_gps(path, 22000000)
    np.testing.assert_array_equal(state, [3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(covariance, 3*np.eye(6))
    with pytest.raises(ValueError, match='exactly one'):
        read_pre_gps(path, 0)


def test_failed_runs_and_missing_phase_cannot_disappear_from_summary():
    config = protocol()
    runs = [dict(seed=1, scenario=c, method=m, status='failed')
            for c in ['nominal', 'dropout'] for m in config['methods']]
    points = [dict(**r, timestamp_us=t, phase=phase) for r in runs
              for t, phase in [(t, 'post_events') for t in config['fixed_times_us']] + [(22000000, 'pre_gps')]]
    ensemble, geometry, summary = summarize(runs, points, [1], config)
    assert all(r['failed_runs'] == 1 and r['mean_nees'] is None for r in ensemble)
    assert all(r['planned_runs'] == 1 and r['invalid_runs'] == 1 for r in geometry)
    assert all(r['failed_runs'] == 1 and r['position_rmse_m_median'] is None for r in summary)
    with pytest.raises(ValueError, match='fixed-time matrix'):
        summarize(runs, points[:-1], [1], config)
    with pytest.raises(ValueError, match='run matrix'):
        summarize(runs+runs[:1], points, [1], config)


def test_protocol_and_seed_roles_are_fixed():
    config = protocol(); validate_protocol(config)
    assert len(config['evaluation_seeds']) == 30
    assert set(config['evaluation_seeds']).isdisjoint(range(81001, 81011))
    assert set(config['evaluation_seeds']).isdisjoint(range(82001, 82031))
    changed = copy.deepcopy(config); changed['evaluation_seeds'][0] = changed['development_seed']
    with pytest.raises(ValueError, match='seed roles'):
        validate_protocol(changed)
    changed = copy.deepcopy(config); changed['settling']['dwell_us'] = 1000000
    with pytest.raises(ValueError, match='design'):
        validate_protocol(changed)


def test_actual_long_outage_smoke_pairs_inputs_and_keeps_pre_gps_phase(tmp_path):
    binary = Path(os.environ.get('NAVTRACE_BINARY', ROOT/'build/navtrace_replay'))
    if not binary.exists():
        pytest.skip('build C++ first')
    output = tmp_path/'study'
    runs, ensemble = run_study(binary, output, smoke=True)
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['completed_replays'] == manifest['attempted_replays'] == 4
    assert manifest['failed_replays'] == manifest['not_started_replays'] == 0
    assert len(manifest['uncompressed_output_sha256']) == 8
    assert manifest['seeds'] == [protocol()['development_seed']]
    folder = output/f'seed-{manifest["seeds"][0]}/inputs'
    with (folder/'nominal.csv').open() as stream:
        nominal = list(csv.reader(stream))
    with (folder/'dropout.csv').open() as stream:
        dropout = list(csv.reader(stream))
    expected = [nominal[0]] + [r for r in nominal[1:] if not (r[1] == 'gps' and 2000000 <= int(r[0]) < 22000000)]
    assert dropout == expected and len(nominal)-len(dropout) == 100
    assert all('truth' not in ' '.join(c) and 'annotations' not in ' '.join(c) for c in manifest['commands'])
    assert all(r['independent_runs'] == 1 for r in ensemble)
    for run in runs:
        assert run['clean_gps_considered'] == (161 if run['scenario'] == 'nominal' else 61)
        assert run['outage_samples'] == (0 if run['scenario'] == 'nominal' else 2000)
    points = json.loads((output/'fixed-time-statistics.json').read_text())
    before = next(p for p in points if p['scenario'] == 'dropout' and p['method'] == 'ekf' and p['phase'] == 'pre_gps')
    after = next(p for p in points if p['scenario'] == 'dropout' and p['method'] == 'ekf' and p['phase'] == 'post_events' and p['timestamp_us'] == 22000000)
    assert before['p00'] > after['p00']
    assert before['error_x'] != after['error_x']
    for point in points:
        assert point['narrow_nees_contribution']+point['wide_nees_contribution'] == pytest.approx(point['position_nees'])
    with pytest.raises(FileExistsError):
        run_study(binary, output, smoke=True)


def test_replay_failures_are_saved_without_shrinking_the_ensemble(tmp_path, monkeypatch):
    binary = Path(os.environ.get('NAVTRACE_BINARY', ROOT/'build/navtrace_replay')).resolve()
    if not binary.exists():
        pytest.skip('build C++ first')
    original_run = subprocess.run
    def fail_replay(command, **kwargs):
        if command[0] == str(binary):
            return subprocess.CompletedProcess(command, 1, '', 'injected replay failure')
        return original_run(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', fail_replay)
    output = tmp_path/'failed-study'
    runs, ensemble = run_study(binary, output, smoke=True)
    manifest = json.loads((output/'manifest.json').read_text())
    assert len(runs) == 4 and manifest['failed_replays'] == 4
    assert manifest['completed_replays'] == manifest['not_started_replays'] == 0
    assert manifest['status'] == 'complete_with_failures'
    assert all(r['independent_runs'] == r['failed_runs'] == 1 for r in ensemble)
    assert all(r['mean_nees'] is None and r['reference_lower'] is None for r in ensemble)
