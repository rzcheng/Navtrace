import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from tools.conditional_metrics import METRICS, compare_samples, convergence, distribution_metrics
from tools.run_conditional_reference import ROOT, STATE_NAMES, read_observations, read_snapshots, run_study


def test_own_covariance_and_ekf_centered_second_moment_are_different():
    samples = np.zeros((2, 6)); samples[:, 0] = [1, 3]
    p = np.diag([2., 4., 1., 1., 1., 1.])
    row = distribution_metrics(samples, np.zeros(6), p)
    assert row['mean_x_m'] == 2 and row['mean_difference_norm_m'] == 2
    assert row['variance_narrow_m2'] == 2
    assert row['second_moment_narrow_m2'] == 5
    assert row['ellipse_mass'] == 1  # Both have x²/2 <= 5.991...
    with pytest.raises(ValueError, match='batches'):
        compare_samples(samples, np.zeros(6), p, 16)


def test_mass_uses_marginal_position_covariance_and_circular_heading():
    samples = np.zeros((4, 6)); samples[:, 0] = [2.4, -2.4, 2.5, -2.5]
    samples[:, 2] = [np.pi-.01, -np.pi+.01]*2
    p = np.eye(6); p[0, 2] = p[2, 0] = .8
    row = distribution_metrics(samples, np.zeros(6), p)
    assert row['ellipse_mass'] == .5
    assert abs(abs(row['heading_mean_rad'])-np.pi) < 1e-12


def test_saved_input_boundary_excludes_checkpoint_and_returning_gps(tmp_path):
    path = tmp_path/'observations.csv'
    path.write_text('timestamp_us,sensor,value1,cov11\n1990000,gyro,1,.1\n2000000,gyro,2,.1\n2000000,wheel,3,.2\n22000000,gyro,4,.1\n22000000,wheel,5,.2\n22000000,gps,999,.3\n22010000,gyro,999,.1\n')
    rows = read_observations(path, 1990000, 22000000)
    assert [(r.timestamp_us, r.sensor, r.value) for r in rows] == [
        (2000000, 'gyro', 2.), (2000000, 'wheel', 3.),
        (22000000, 'gyro', 4.), (22000000, 'wheel', 5.)]
    trajectory = tmp_path/'states.csv.gz'
    fields = ['timestamp_us', 'sensor']+STATE_NAMES+[f'p{i}{j}' for i in range(6) for j in range(6)]
    with gzip.open(trajectory, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for sensor, value in [('gyro', 1), ('wheel', 2), ('gps', 3)]:
            row = dict(timestamp_us=22000000, sensor=sensor)
            row.update({name:value for name in STATE_NAMES})
            row.update({f'p{i}{j}':int(i == j) for i in range(6) for j in range(6)})
            writer.writerow(row)
    snapshots = read_snapshots(trajectory, [22000000], 22000000)
    np.testing.assert_array_equal(snapshots[22000000][0], np.full(6, 2.))


def test_convergence_counts_failures_and_rejects_missing_entries():
    protocol = json.loads((ROOT/'scenarios/conditional_reference.json').read_text())
    rows = [dict(recording_seed=s, sample_count=n, reference_seed=r, timestamp_us=t, status='failed')
            for s in protocol['recording_seeds'] for n in protocol['sample_counts']
            for r in protocol['reference_seeds'] for t in protocol['report_times_us']]
    result = convergence(rows, protocol)
    assert len(result) == 70 and all(not r['resolved'] for r in result)
    with pytest.raises(ValueError, match='matrix'):
        convergence(rows[:-1], protocol)
    with pytest.raises(ValueError, match='matrix'):
        convergence(rows+rows[:1], protocol)


def test_full_study_requires_a_successful_pilot_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='pilot-results'):
        run_study(tmp_path/'missing-baseline', tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_convergence_pools_independent_replicate_errors_and_flags_instability():
    protocol = json.loads((ROOT/'scenarios/conditional_reference.json').read_text())
    rows = []
    for s in protocol['recording_seeds']:
        for n in protocol['sample_counts']:
            for r in protocol['reference_seeds']:
                for t in protocol['report_times_us']:
                    row = dict(recording_seed=s, sample_count=n, reference_seed=r, timestamp_us=t,
                               status='complete', ekf_covariance=np.eye(2).tolist())
                    row.update({name: .5 for name in METRICS})
                    row.update({name+'_mcse': .001 for name in METRICS})
                    rows.append(row)
    result = convergence(rows, protocol)
    assert all(r['resolved'] for r in result)
    assert result[0]['high_mcse'] == pytest.approx(.001/np.sqrt(3))
    target = next(r for r in rows if r['recording_seed'] == 83001 and r['sample_count'] == 2048
                  and r['reference_seed'] == 91001 and r['timestamp_us'] == 1990000)
    target['mean_x_m'] += 1.
    failures = [r for r in convergence(rows, protocol) if not r['resolved']]
    assert len(failures) == 1 and failures[0]['metric'] == 'mean_x_m'
    assert failures[0]['medium_high_difference'] == pytest.approx(1/3)
