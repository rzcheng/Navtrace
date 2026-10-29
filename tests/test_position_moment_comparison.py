import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import position_moment_metrics as metrics
from tools import run_position_moment_comparison as runner


@pytest.fixture
def protocol():
    return json.loads((runner.ROOT/runner.PROTOCOL_PATH).read_text())


def hand_samples(count=32):
    samples = np.zeros((count, 6))
    samples[:, 0] = np.arange(count)/8-1
    samples[:, 1] = samples[:, 0]*.3+np.tile([-1, 1], count//2)
    samples[:, 2] = np.tile([np.pi-.01, -np.pi+.01], count//2)
    return samples


def test_metrics_match_independent_arithmetic_and_batch_formula():
    samples = hand_samples(64)
    mean = np.zeros(6)
    covariance = np.diag([2., 4., 1., 1., 1., 1.])
    candidate = mean.copy(); candidate[0] = 1.
    result = metrics.compare_samples(samples, mean, covariance, candidate, covariance)
    position = samples[:, :2]
    pbar = position.mean(axis=0)
    dx, dy = (position-pbar).T
    expected = dict(mean_x_m=pbar[0], mean_y_m=pbar[1],
        variance_narrow_m2=sum(dx*dx)/63, variance_wide_m2=sum(dy*dy)/63,
        covariance_mixed_m2=sum(dx*dy)/63,
        second_moment_narrow_m2=sum(position[:, 0]**2)/64,
        second_moment_wide_m2=sum(position[:, 1]**2)/64)
    for name, value in expected.items():
        assert result['metrics'][name]['value'] == pytest.approx(value)
    batch_variances = [sum((b[:, 0]-b[:, 0].mean())**2)/(len(b)-1) for b in np.split(position, 16)]
    assert result['metrics']['variance_narrow_m2']['mcse'] == pytest.approx(np.std(batch_variances, ddof=1)/4)
    assert result['metrics']['variance_narrow_m2']['value'] != pytest.approx(np.mean(batch_variances))
    assert result['metrics']['second_moment_narrow_m2']['value'] != pytest.approx(result['metrics']['variance_narrow_m2']['value'])
    # Saved wrapped yaw is irrelevant to every primary metric and description.
    changed = samples.copy(); changed[:, 2:] = 999.
    assert result == metrics.compare_samples(changed, mean, covariance, candidate, covariance)


def test_common_axis_sign_and_rotated_mixed_covariance():
    angle = 2.3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    p = rotation @ np.diag([1., 3.]) @ rotation.T
    values, axes = metrics.principal_axes(p)
    np.testing.assert_allclose(axes.T @ p @ axes, np.diag(values), atol=1e-14)
    for column in range(2):
        assert axes[np.argmax(np.abs(axes[:, column])), column] > 0
    samples = hand_samples()
    full = np.eye(6); full[:2, :2] = p
    row = metrics.compare_samples(samples, np.zeros(6), full, np.zeros(6), full)
    expected = axes.T @ np.cov(samples[:, :2], rowvar=False, ddof=1) @ axes
    assert row['metrics']['covariance_mixed_m2']['value'] == pytest.approx(expected[0, 1])


def test_ellipse_uses_marginal_and_candidate_center_and_fails_independently():
    samples = np.zeros((32, 6)); samples[:, 0] = np.tile([2.4, -2.4, 2.5, -2.5], 8)
    p = np.eye(6); p[0, 2] = p[2, 0] = .8
    mean = np.zeros(6)
    candidate_mean = mean.copy(); candidate_mean[0] = 5
    row = metrics.compare_samples(samples, mean, p, candidate_mean, p)
    assert row['metrics']['baseline_ellipse_mass']['value'] == .5
    assert row['metrics']['candidate_ellipse_mass']['value'] == 0
    for candidate_p in [None, np.zeros((6, 6)), -np.eye(6), np.full((6, 6), np.nan)]:
        row = metrics.compare_samples(samples, mean, p, mean if candidate_p is not None else None, candidate_p)
        assert row['status'] == 'failed'
        assert row['metrics']['candidate_ellipse_mass']['status'] == 'failed'
        assert all(row['metrics'][name]['status'] == 'complete' for name in metrics.METRICS[:-1])
    singular_baseline = p.copy(); singular_baseline[0, 0] = 0
    row = metrics.compare_samples(samples, mean, singular_baseline, mean, p)
    assert row['metrics']['baseline_ellipse_mass']['status'] == 'failed'
    assert row['metrics']['candidate_ellipse_mass']['status'] == 'complete'


def complete_rows(protocol):
    rows = metrics.planned_rows(protocol)
    for row in rows:
        row['status'] = 'complete'
        row['metrics'] = {name: dict(status='complete', value=.5, mcse=.001) for name in metrics.METRICS}
        method = {name: .5 for name in metrics.MOMENTS}
        method.update(position_mean=[.5, .5], position_covariance=np.eye(2).tolist())
        row['description'] = dict(position_mean=[.5, .5], position_covariance=np.eye(2).tolist(),
                                  baseline=method, candidate=copy.deepcopy(method))
    return rows


def test_pooling_all_counts_and_all_ninety_resolution_checks(protocol):
    rows = complete_rows(protocol)
    pooled, outcomes = metrics.summarize(rows, protocol)
    assert len(pooled) == 30 and len(outcomes) == 90
    assert all(o['status'] == 'complete' and o['resolved'] for o in outcomes)
    assert outcomes[0]['high']['mcse'] == pytest.approx(.001/np.sqrt(3))
    assert outcomes[0]['agreement_band'] == pytest.approx(4*.001/np.sqrt(3)+1e-12)
    mixed = next(o for o in outcomes if o['metric'] == 'covariance_mixed_m2')
    assert mixed['precision_limit'] == pytest.approx(.1*np.sqrt(.5*.5))
    mass = next(o for o in outcomes if o['metric'] == 'candidate_ellipse_mass')
    assert mass['precision_limit'] == .02
    assert mass['difference_from_nominal'] == pytest.approx(-.45)
    assert not mass['nominal_within_agreement_band']
    group = [r for r in rows if (r['recording_seed'], r['timestamp_us'], r['sample_count']) == (83001, 1990000, 8192)]
    for index, row in enumerate(group):
        row['description']['position_mean'] = [index*100., 0.]
        row['description']['position_covariance'] = (np.eye(2)*(index+1)).tolist()
    pooled, _ = metrics.summarize(rows, protocol)
    target = next(p for p in pooled if (p['recording_seed'], p['timestamp_us'], p['sample_count']) == (83001, 1990000, 8192))
    np.testing.assert_array_equal(target['description']['position_covariance'], np.eye(2)*2)
    np.testing.assert_array_equal(target['description']['position_mean'], [100, 0])


def test_resolution_is_metric_specific_and_missing_work_is_retained(protocol):
    rows = complete_rows(protocol)
    target = next(r for r in rows if (r['recording_seed'], r['timestamp_us'], r['sample_count']) == (83001, 1990000, 8192))
    target['status'] = 'failed'
    target['metrics']['candidate_ellipse_mass'] = dict(status='failed', error='singular')
    _, outcomes = metrics.summarize(rows, protocol)
    failures = [o for o in outcomes if not o['resolved']]
    assert len(failures) == 1 and failures[0]['metric'] == 'candidate_ellipse_mass'
    target['metrics']['mean_x_m']['value'] += 1.
    _, outcomes = metrics.summarize(rows, protocol)
    assert len([o for o in outcomes if not o['resolved']]) == 2
    assert next(o for o in outcomes if o['recording_seed'] == 83001 and o['timestamp_us'] == 1990000 and o['metric'] == 'mean_x_m')['medium_high_difference'] == pytest.approx(1/3)
    _, missing = metrics.summarize(metrics.planned_rows(protocol), protocol)
    assert len(missing) == 90 and all(o['status'] == 'not_started' for o in missing)
    with pytest.raises(ValueError, match='matrix'):
        metrics.summarize(rows[:-1], protocol)
    with pytest.raises(ValueError, match='matrix'):
        metrics.summarize(rows+rows[:1], protocol)


def test_mixed_precision_uses_geometric_scale_and_moment_floor(protocol):
    rows = complete_rows(protocol)
    for row in rows:
        row['metrics']['covariance_mixed_m2'].update(value=0., mcse=.001)
        row['metrics']['variance_narrow_m2'].update(value=2., mcse=.001)
        row['metrics']['variance_wide_m2'].update(value=8., mcse=.001)
        row['metrics']['second_moment_narrow_m2'].update(value=0., mcse=0.)
    _, outcomes = metrics.summarize(rows, protocol)
    assert next(o for o in outcomes if o['metric'] == 'covariance_mixed_m2')['precision_limit'] == pytest.approx(.4)
    assert next(o for o in outcomes if o['metric'] == 'second_moment_narrow_m2')['precision_limit'] == pytest.approx(1e-13)


def test_latent_checks_angle_count_and_exact_checkpoint():
    baseline = (np.zeros(6), np.eye(6))
    snapshot = dict(mean=np.zeros(6), covariance=np.eye(6), observations=362)
    snapshot['mean'][2] = 2*np.pi
    assert runner.validate_candidate(snapshot, baseline, 362, 1e-9)['max_mean_difference'] == 0
    with pytest.raises(ValueError, match='count'):
        runner.validate_candidate(snapshot, baseline, 1, 1e-9)
    with pytest.raises(ValueError, match='checkpoint'):
        runner.validate_candidate(snapshot, baseline, 362, 1e-9, checkpoint=True)
    snapshot['covariance'][4, 4] += 1e-8
    with pytest.raises(ValueError, match='latent'):
        runner.validate_candidate(snapshot, baseline, 362, 1e-9)


def test_budget_checks_wall_and_peak_memory(monkeypatch):
    budget = runner.Budget(started=0.)
    monkeypatch.setattr(runner.time, 'monotonic', lambda: 601.)
    with pytest.raises(TimeoutError):
        budget.check()
    monkeypatch.setattr(runner.time, 'monotonic', lambda: 1.)
    monkeypatch.setattr(runner, 'peak_memory_mib', lambda: 513.)
    with pytest.raises(MemoryError):
        budget.check()


def test_preflight_refuses_paths_and_dirty_tree_before_output(tmp_path, monkeypatch):
    root = tmp_path/'repo'; root.mkdir()
    old = tmp_path/'old'; old.mkdir()
    ref = tmp_path/'ref'; ref.mkdir()
    out = tmp_path/'out'
    monkeypatch.setattr(runner, 'git', lambda *args: b'?? surprise.py\n')
    with pytest.raises(ValueError, match='clean committed'):
        runner.preflight(old, ref, out, runner.Budget(), root)
    assert not out.exists()
    with pytest.raises(FileExistsError):
        runner.preflight(old, ref, old, runner.Budget(), root)
    with pytest.raises(ValueError, match='outside'):
        runner.preflight(old, ref, ref/'out', runner.Budget(), root)
    with pytest.raises(ValueError, match='ignored results'):
        runner.preflight(old, ref, root/'tools/new-output', runner.Budget(), root)
    link = tmp_path/'alias'; link.symlink_to(ref, target_is_directory=True)
    with pytest.raises(ValueError, match='outside'):
        runner.preflight(old, ref, link/'out', runner.Budget(), root)
    monkeypatch.setattr(runner, 'git', lambda *args: b'')
    (root/'scenarios').mkdir(); (root/runner.PROTOCOL_PATH).write_text('{}')
    with pytest.raises(ValueError, match='pinned protocol'):
        runner.preflight(old, ref, out, runner.Budget(), root)
    assert not out.exists()


def test_audit_uses_historical_sources_and_verifies_every_artifact(tmp_path, monkeypatch):
    root = tmp_path/'input'; root.mkdir()
    (root/'protocol.json').write_bytes(b'old protocol')
    (root/'artifact').write_bytes(b'frozen data')
    sha = lambda b: hashlib.sha256(b).hexdigest()
    manifest = dict(git_commit='historical', git_dirty=False, status='complete', protocol_sha256=sha(b'old protocol'),
                    source_sha256={'source.py': sha(b'historical source')},
                    artifacts_sha256={'artifact': sha(b'frozen data'), 'protocol.json': sha(b'old protocol')})
    runner.json_write(root/'manifest.json', manifest)
    spec = dict(manifest_sha256=runner.sha256(root/'manifest.json'), git_commit='historical',
                protocol_sha256=sha(b'old protocol'), expected_source_count=1, expected_artifact_count=2)
    requested = []
    def git(_root, *args):
        requested.append(args)
        return b'historical source'
    monkeypatch.setattr(runner, 'git', git)
    runner.audit_input(root, tmp_path, spec, lambda: None)
    assert requested == [('show', 'historical:source.py')]
    (root/'artifact').write_text('changed')
    with pytest.raises(ValueError, match='artifact hash'):
        runner.audit_input(root, tmp_path, spec, lambda: None)
    (root/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest hash'):
        runner.audit_input(root, tmp_path, spec, lambda: None)
    with pytest.raises(ValueError, match='escapes'):
        runner.safe_member(root, '../other')


@pytest.fixture
def fake_run(tmp_path, monkeypatch, protocol):
    # Small deterministic hand fixtures stand in for saved distributions. No RNG.
    protocol = copy.deepcopy(protocol); protocol['sample_counts'] = [32, 64, 128]
    root = tmp_path/'repo'; (root/'scenarios').mkdir(parents=True); (root/'docs').mkdir()
    (root/runner.PROTOCOL_PATH).write_text(json.dumps(protocol))
    (root/protocol['specification_path']).write_text('synthetic test specification')
    reference = tmp_path/'reference'; reference.mkdir()
    for row in metrics.planned_rows(protocol):
        case = reference/f"case-{row['recording_seed']}"; case.mkdir(exist_ok=True)
        np.savez(case/f"n{row['sample_count']}-r{row['reference_seed']}-t{row['timestamp_us']}.npz",
                 states=hand_samples(row['sample_count']))
    metadata = dict(git_commit='test', git_branch='test', git_dirty=False, source_sha256={}, input_roots={}, inputs={})
    monkeypatch.setattr(runner, 'preflight', lambda *args: (protocol, metadata))
    baseline = {t: (np.zeros(6), np.eye(6)) for t in protocol['report_times_us']}
    monkeypatch.setattr(runner, 'load_case', lambda *args: (baseline, [], [0., 0., 0.], .01, 2.))
    def candidate(*args, **kwargs):
        return {t: dict(mean=np.zeros(6), covariance=np.eye(6), observations=n)
                for t, n in zip(protocol['report_times_us'], protocol['expected_observation_counts'])}
    candidate_module = SimpleNamespace(run_candidate=candidate)
    monkeypatch.setitem(sys.modules, 'tools.position_moments', candidate_module)
    monkeypatch.setattr(runner, 'recheck_inputs', lambda *args: args[-1].check())
    import tools.plot_position_moments as plots
    actual_plot = plots.plot_results
    monkeypatch.setattr(plots, 'plot_results', lambda output, *args: (output/'plot.png').write_bytes(b'test plot'))
    return dict(root=root, reference=reference, long_outage=tmp_path/'long', output=tmp_path/'out',
                protocol=protocol, candidate=candidate_module, plot=actual_plot)


def call_fake_run(fake_run):
    return runner.run_comparison(fake_run['long_outage'], fake_run['reference'], fake_run['output'], root=fake_run['root'])


def test_synthetic_complete_runner_matrix_snapshots_and_output_hashes(fake_run):
    rows, manifest = call_fake_run(fake_run)
    assert manifest['status'] == 'complete'
    assert len(rows) == 90 and manifest['completed_metrics'] == 810
    assert manifest['final_input_audit_status'] == 'complete'
    assert manifest['output_hash_status'] == 'complete'
    assert len(json.loads((fake_run['output']/'resolution.json').read_text())) == 90
    cases = json.loads((fake_run['output']/'case-snapshots.json').read_text())
    assert len(cases) == 10 and all(c['status'] == 'complete' for c in cases)
    for name, digest in manifest['artifacts_sha256'].items():
        assert runner.sha256(fake_run['output']/name) == digest
    assert 'manifest.json' not in manifest['artifacts_sha256']


@pytest.mark.parametrize('failure', ['nan', 'negative_covariance', 'exception'])
def test_invalid_candidate_preserves_valid_reference_metrics_and_manifest(fake_run, failure):
    original = fake_run['candidate'].run_candidate
    def invalid(*args, **kwargs):
        if failure == 'exception':
            raise ValueError('deliberate candidate failure')
        result = original(*args, **kwargs)
        for snapshot in result.values():
            if failure == 'nan':
                snapshot['mean'][0] = np.nan
            else:
                snapshot['covariance'][0, 0] = -1.
        return result
    fake_run['candidate'].run_candidate = invalid
    rows, manifest = call_fake_run(fake_run)
    assert manifest['status'] == 'failed' and manifest['completed_metrics'] == 720
    assert all(row['metrics']['candidate_ellipse_mass']['status'] == 'failed' for row in rows)
    assert all(row['metrics'][name]['status'] == 'complete' for row in rows for name in metrics.METRICS[:-1])
    assert len(json.loads((fake_run['output']/'resolution.json').read_text())) == 90
    cases = json.loads((fake_run['output']/'case-snapshots.json').read_text())
    assert all(case['status'] == 'failed' and 'candidate' not in case for case in cases)
    assert json.loads((fake_run['output']/'manifest.json').read_text())['status'] == 'failed'


def test_budget_stop_retains_complete_failed_and_unstarted_rows(fake_run, monkeypatch):
    original = runner.compare_samples
    calls = []
    def stop_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise TimeoutError('deliberate fixture budget stop')
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'compare_samples', stop_second)
    rows, manifest = call_fake_run(fake_run)
    assert manifest['status'] == 'failed' and manifest['budget_stopped']
    assert [r['status'] for r in rows[:3]] == ['complete', 'failed', 'not_started']
    assert manifest['completed_distributions'] == 1 and manifest['failed_distributions'] == 1
    assert manifest['not_started_distributions'] == 88
    assert all(m['status'] == 'failed' for m in rows[1]['metrics'].values())
    assert len(json.loads((fake_run['output']/'resolution.json').read_text())) == 90
    assert manifest['final_input_audit_status'] == 'not_started'


def test_compare_rejects_bad_shapes_batches_and_checks_budget():
    with pytest.raises(ValueError, match='N x 6'):
        metrics.compare_samples(np.zeros((32, 2)), np.zeros(6), np.eye(6), None, None)
    with pytest.raises(ValueError, match='batches'):
        metrics.compare_samples(hand_samples(16), np.zeros(6), np.eye(6), None, None)
    calls = []
    metrics.compare_samples(hand_samples(), np.zeros(6), np.eye(6), None, None, check_budget=lambda: calls.append(1))
    assert len(calls) == 16


def test_plotting_synthetic_shapes_has_equal_axes_and_full_ranges(fake_run, monkeypatch):
    import tools.plot_position_moments as plots
    original_save = plots.save_figure
    inspected = []
    def inspect(fig, path, callback):
        if path.name == 'position-shape.png':
            for ax in fig.axes:
                assert ax.get_aspect() == 1.
                np.testing.assert_array_equal(ax.get_xlim(), ax.get_ylim())
                assert ax.get_xlim()[1] > hand_samples(128)[:, :2].max()
                inspected.append(True)
        return original_save(fig, path, callback)
    monkeypatch.setattr(plots, 'save_figure', inspect)
    monkeypatch.setattr(plots, 'plot_results', fake_run['plot'])
    _, manifest = call_fake_run(fake_run)
    assert manifest['status'] == 'complete' and len(inspected) == 10
    for name in ['position-shape.png', 'moment-errors.png', 'ellipse-mass.png', 'resolution.png']:
        assert (fake_run['output']/name).stat().st_size > 1000


@pytest.fixture
def guarded_inputs(tmp_path, monkeypatch, protocol):
    """Small real manifests and file hashes; mocked Git bytes are explicitly historical."""
    root, old, ref = [tmp_path/name for name in ['repo', 'old', 'ref']]
    for path in [root/'docs', root/'scenarios', old, ref]:
        path.mkdir(parents=True)
    (root/'source.py').write_bytes(b'preserved current source')
    (root/'scenarios/filter.conf').write_bytes(b'fixed config')
    (root/'docs/position_moment_protocol.md').write_bytes(b'fixed specification')
    (old/'filter.conf').write_bytes(b'fixed config')
    protocol = copy.deepcopy(protocol)
    protocol.update(recording_seeds=[1], report_times_us=[10], sample_counts=[32], reference_seeds=[3],
                    preserved_source_sha256={'source.py': runner.sha256(root/'source.py')},
                    filter_sha256=runner.sha256(root/'scenarios/filter.conf'),
                    specification_sha256=runner.sha256(root/'docs/position_moment_protocol.md'))
    for name, base in [('long_outage', old), ('conditional_reference', ref)]:
        (base/'protocol.json').write_bytes(b'old fixed protocol')
        if name == 'conditional_reference':
            (base/'case-1').mkdir()
            (base/'case-1/n32-r3-t10.npz').write_bytes(b'hash-only test sample placeholder')
        artifacts = {str(p.relative_to(base)): runner.sha256(p) for p in base.rglob('*') if p.is_file()}
        manifest = dict(git_commit=name+'-old-commit', git_dirty=False, status='complete',
                        protocol_sha256=runner.sha256(base/'protocol.json'),
                        artifacts_sha256=artifacts,
                        source_sha256={'source.py': hashlib.sha256(b'historical source').hexdigest()})
        runner.json_write(base/'manifest.json', manifest)
        protocol['frozen_inputs'][name] = dict(manifest_sha256=runner.sha256(base/'manifest.json'),
            git_commit=manifest['git_commit'], protocol_sha256=manifest['protocol_sha256'],
            expected_source_count=1, expected_artifact_count=len(artifacts), expected_sample_arrays=1)
    runner.json_write(root/runner.PROTOCOL_PATH, protocol)
    committed = {str(runner.PROTOCOL_PATH): (root/runner.PROTOCOL_PATH).read_bytes(),
                 protocol['specification_path']: (root/protocol['specification_path']).read_bytes()}
    monkeypatch.setattr(runner, 'PROTOCOL_SHA256', runner.sha256(root/runner.PROTOCOL_PATH))
    def git(_root, *args):
        if args[0] == 'status':
            return b''
        if args[0] == 'show':
            commit, path = args[1].split(':', 1)
            return committed[path] if commit == 'HEAD' else b'historical source'
        if args[0] == 'ls-files':
            return b'source.py\0scenarios/filter.conf\0scenarios/position_moments.json\0docs/position_moment_protocol.md\0'
        if args[0] == 'rev-parse':
            return b'current-implementation\n'
        if args[0] == 'branch':
            return b'test-branch\n'
        raise AssertionError(args)
    monkeypatch.setattr(runner, 'git', git)
    return root, old, ref, tmp_path/'output', protocol, committed


def test_preflight_audits_real_file_hashes_and_historical_git_bytes(guarded_inputs):
    root, old, ref, output, protocol, _ = guarded_inputs
    actual, metadata = runner.preflight(old, ref, output, runner.Budget(), root)
    assert actual == protocol
    assert metadata['verified_sample_count'] == 1 and metadata['verified_artifact_count'] == 4
    assert metadata['git_commit'] == 'current-implementation'
    assert not output.exists()  # Successful preflight itself never starts a comparison.


@pytest.mark.parametrize('changed', ['committed_protocol', 'specification', 'preserved_source', 'saved_filter', 'missing_artifact'])
def test_preflight_refuses_changed_frozen_inputs_without_creating_output(guarded_inputs, changed):
    root, old, ref, output, protocol, committed = guarded_inputs
    if changed == 'committed_protocol':
        committed[str(runner.PROTOCOL_PATH)] = b'changed'
    elif changed == 'specification':
        (root/protocol['specification_path']).write_text('changed')
    elif changed == 'preserved_source':
        (root/'source.py').write_text('changed')
    elif changed == 'saved_filter':
        (old/'filter.conf').write_text('changed')
    else:
        (ref/'case-1/n32-r3-t10.npz').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.preflight(old, ref, output, runner.Budget(), root)
    assert not output.exists()


def test_preflight_budget_failure_creates_no_output(guarded_inputs):
    root, old, ref, output, _, _ = guarded_inputs
    class Stop:
        def check(self):
            raise TimeoutError('audit budget stop')
    with pytest.raises(TimeoutError):
        runner.preflight(old, ref, output, Stop(), root)
    assert not output.exists()


def test_output_hashing_failure_is_a_failed_run(fake_run, monkeypatch):
    original = runner.sha256
    def fail_output(path, check_budget=lambda: None):
        if Path(path).name == 'comparisons.json':
            raise TimeoutError('output hashing budget stop')
        return original(path, check_budget)
    monkeypatch.setattr(runner, 'sha256', fail_output)
    _, manifest = call_fake_run(fake_run)
    assert manifest['status'] == 'failed' and manifest['output_hash_status'] == 'failed'
    assert manifest['budget_stopped']
    assert json.loads((fake_run['output']/'manifest.json').read_text())['status'] == 'failed'



def test_cli_existing_output_is_a_preflight_refusal(tmp_path, monkeypatch, capsys):
    output = tmp_path/'existing'; output.mkdir()
    monkeypatch.setattr(sys, 'argv', ['comparison', '--long-outage', str(tmp_path/'missing-old'),
                                     '--reference', str(tmp_path/'missing-ref'), '--output', str(output)])
    assert runner.main() == 2
    error = capsys.readouterr().err
    assert 'Comparison refused before output:' in error
    assert 'output must be fresh' in error
    assert 'failed after output creation' not in error
    assert not list(output.iterdir())


def test_cli_newly_created_output_failure_is_identified(tmp_path, monkeypatch, capsys):
    output = tmp_path/'new-output'
    def fail_after_creation(_old, _reference, target, **kwargs):
        target.mkdir()
        raise RuntimeError('deliberate failure after creation')
    monkeypatch.setattr(runner, 'run_comparison', fail_after_creation)
    monkeypatch.setattr(sys, 'argv', ['comparison', '--long-outage', str(tmp_path/'old'),
                                     '--reference', str(tmp_path/'ref'), '--output', str(output)])
    assert runner.main() == 2
    error = capsys.readouterr().err
    assert 'Comparison failed after output creation:' in error
    assert 'deliberate failure after creation' in error
    assert 'refused before output' not in error



def scalar_position_metrics(position, baseline_mean, baseline_p, candidate_mean, candidate_p, cosine, sine):
    """Independent scalar sums in the analytically known rotated baseline basis."""
    points = [(float(row[0]), float(row[1])) for row in position]
    count = len(points)
    mean_x = math.fsum(x for x, _ in points)/count
    mean_y = math.fsum(y for _, y in points)/count
    narrow = [cosine*(x-mean_x)+sine*(y-mean_y) for x, y in points]
    wide = [-sine*(x-mean_x)+cosine*(y-mean_y) for x, y in points]
    displaced_narrow = [cosine*(x-baseline_mean[0])+sine*(y-baseline_mean[1]) for x, y in points]
    displaced_wide = [-sine*(x-baseline_mean[0])+cosine*(y-baseline_mean[1]) for x, y in points]

    def mass(mean, p):
        # Solve the 2 x 2 marginal system by scalar elimination, independently
        # of the implementation's Cholesky factorization and batch solve.
        a, b, d = float(p[0, 0]), float(p[0, 1]), float(p[1, 1])
        inside = 0
        for x, y in points:
            dx, dy = x-float(mean[0]), y-float(mean[1])
            solution_y = (dy-b*dx/a)/(d-b*b/a)
            solution_x = (dx-b*solution_y)/a
            inside += dx*solution_x+dy*solution_y <= 5.991464547107979
        return inside/count

    return dict(mean_x_m=mean_x, mean_y_m=mean_y,
                variance_narrow_m2=math.fsum(x*x for x in narrow)/(count-1),
                variance_wide_m2=math.fsum(y*y for y in wide)/(count-1),
                covariance_mixed_m2=math.fsum(x*y for x, y in zip(narrow, wide))/(count-1),
                second_moment_narrow_m2=math.fsum(x*x for x in displaced_narrow)/count,
                second_moment_wide_m2=math.fsum(y*y for y in displaced_wide)/count,
                baseline_ellipse_mass=mass(baseline_mean, baseline_p),
                candidate_ellipse_mass=mass(candidate_mean, candidate_p))


@pytest.mark.parametrize('count', [512, 2048, 8192])
def test_actual_counts_match_scalar_metrics_and_batch_mcse(count):
    # Deterministic finite points with changing batch means and correlated axes.
    # The actual frozen counts exercise the local large-matrix BLAS dispatch.
    samples = np.zeros((count, 6))
    for index in range(count):
        x = (index % 37-18)/8+index/(4*count)
        y = ((index*7) % 29-14)/7+.3*x
        samples[index, :2] = [x, y]
    cosine, sine = math.cos(.37), math.sin(.37)
    baseline_mean = np.array([.7, -.4, 0., 0., 0., 0.])
    baseline_p = np.eye(6)
    baseline_p[:2, :2] = [[cosine*cosine+4*sine*sine, -3*cosine*sine],
                          [-3*cosine*sine, sine*sine+4*cosine*cosine]]
    candidate_mean = np.array([1.2, .35, 0., 0., 0., 0.])
    candidate_p = np.eye(6); candidate_p[:2, :2] = [[2., .4], [.4, 1.2]]
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        row = metrics.compare_samples(samples, baseline_mean, baseline_p, candidate_mean, candidate_p)
    assert row['status'] == 'complete' and row['batches'] == 16
    np.testing.assert_allclose(row['description']['common_axes'], [[cosine, -sine], [sine, cosine]],
                               rtol=0, atol=1e-14)
    expected = scalar_position_metrics(samples, baseline_mean, baseline_p, candidate_mean, candidate_p, cosine, sine)
    batch_size = count//16
    batches = [scalar_position_metrics(samples[start:start+batch_size], baseline_mean, baseline_p,
                                      candidate_mean, candidate_p, cosine, sine)
               for start in range(0, count, batch_size)]
    for name in metrics.METRICS:
        result = row['metrics'][name]
        batch_values = [batch[name] for batch in batches]
        batch_mean = math.fsum(batch_values)/16
        mcse = math.sqrt(math.fsum((value-batch_mean)**2 for value in batch_values)/15)/4
        assert result['status'] == 'complete'
        # These bounds cover floating-point reductions of bounded scalar values;
        # they do not depend on saved samples or candidate performance.
        assert result['value'] == pytest.approx(expected[name], rel=0, abs=1e-12)
        np.testing.assert_allclose(result['batch_values'], batch_values, rtol=0, atol=1e-12)
        assert result['mcse'] == pytest.approx(mcse, rel=0, abs=1e-13)
        assert mcse > 0  # Every metric exercises uncertainty across real batches.
        if name in metrics.METRICS[-2:]:
            assert result['value'] == expected[name]
            assert result['batch_values'] == batch_values
    # Full-sample covariance is not silently replaced by average batch covariance.
    assert abs(expected['variance_narrow_m2']-math.fsum(b['variance_narrow_m2'] for b in batches)/16) > 1e-5
