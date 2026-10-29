"""Independent frozen-oracle, assessment, and failure-inventory checks."""
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_trajectory_mixture_study as study
from trajectory_mixture import Observation, run_events


def test_linear_batch_oracle_and_candidate_all_prescribed_reports():
    oracles = study.linear_batch_oracle()
    candidate = study.new_candidate("L", 0)  # L has no angular random draws.
    snapshots, events = run_events(candidate, [Observation(*e) for e in study.fixture("L")[3]],
                                    study.REPORTS["L"])
    assert len(events) == 6
    for timestamp, snapshot in snapshots.items():
        expected = oracles[timestamp]
        np.testing.assert_allclose(snapshot["mean"], expected["mean"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(snapshot["covariance"], expected["covariance"], atol=1e-10, rtol=0)
        assert abs(snapshot["log_evidence"] - expected["log_evidence"]) <= 1e-10
        _, dp = study.comparison(snapshot, expected)
        np.testing.assert_allclose(dp, 0, atol=1e-10, rtol=0)
        assert snapshot["observation_count"] == 2 * (timestamp // 100000)
        assert snapshot["substeps"] == timestamp // 10000


def test_full_one_step_oracle_matches_independent_latent_filter():
    full = study.observed_one_step_oracle()
    latent = study.latent_oracle()[10000]
    indices = [2, 3, 4, 5]
    np.testing.assert_allclose(full["mean"][indices], latent["mean"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(full["covariance"][np.ix_(indices, indices)],
                               latent["covariance"], atol=1e-10, rtol=0)
    assert abs(full["log_evidence"] - latent["log_evidence"]) <= 1e-10
    assert np.linalg.eigvalsh(full["covariance"])[0] > 0
    np.testing.assert_allclose(full["covariance"], full["covariance"].T, atol=1e-10, rtol=0)
    # Conditioning must retain induced state/noise correlations; dropping these is wrong.
    assert abs(full["conditioned_covariance9"][3, 6]) > 0
    assert abs(full["conditioned_covariance9"][4, 7]) > 0


def test_nine_dimensional_conditioning_information_identity():
    mean, covariance, diffusion, _ = study.fixture("O")
    prior_mean = np.r_[mean, [0., 0., 0.]]
    prior = np.zeros((9, 9))
    prior[:6, :6] = covariance
    prior[6:, 6:] = np.diag(diffusion*.01)
    h = np.zeros((2, 9))
    h[0, [4, 5, 7, 8]] = 1
    h[1, [3, 6]] = 1
    precision = np.linalg.inv(prior) + h.T @ np.diag([1/.02**2, 1/.1**2]) @ h
    information = np.linalg.solve(prior, prior_mean) + h.T @ np.array([.09/.02**2, 1.4/.1**2])
    expected_cov = np.linalg.inv(precision)
    expected_mean = np.linalg.solve(precision, information)
    oracle = study.observed_one_step_oracle()
    np.testing.assert_allclose(oracle["conditioned_mean9"], expected_mean, atol=1e-10, rtol=0)
    np.testing.assert_allclose(oracle["conditioned_covariance9"], expected_cov, atol=1e-10, rtol=0)


def test_heading_oracle_symmetry_support_and_empirical_map():
    initial = study.heading_oracle(0)
    endpoint = study.heading_oracle(100000)
    assert initial["support"] == [2]
    assert endpoint["support"] == [0, 1, 2]
    assert np.linalg.eigvalsh(endpoint["covariance"][:3, :3])[0] > 0
    assert endpoint["mean"][1] == 0
    assert endpoint["covariance"][0, 1] == 0
    assert endpoint["covariance"][0, 2] == 0
    # Declared H control seed only; study seeds are never opened in tests.
    candidate = study.new_candidate("H", 94999)
    angles = candidate.angular[:, 0].copy()
    candidate.predict_step(.1)
    mean, covariance, states = study.heading_empirical(angles, 100000)
    actual_mean, actual_covariance = candidate.full_moments()
    np.testing.assert_allclose(actual_mean, mean, atol=1e-10, rtol=0)
    np.testing.assert_allclose(actual_covariance, covariance, atol=1e-10, rtol=0)
    np.testing.assert_allclose(candidate.conditional_mean, states[:, [0, 1, 3]], atol=1e-10, rtol=0)


def test_frozen_schedule_and_oracle_support_inventory():
    events = study.fixture("O")[3]
    assert len(events) == 32
    assert events[:2] == [(10000, "gyro", .09, .02**2), (10000, "wheel", 1.4, .1**2)]
    assert events[-2:] == [(1000000, "gyro", .09, .02**2), (1000000, "wheel", 1.6, .1**2)]
    oracle = study.oracle_inventory()
    assert len(oracle["O"][10000]["mean"]) == 6
    assert len(oracle["O"][250000]["mean"]) == 4
    for case in oracle.values():
        for item in case.values():
            p = item["covariance"]
            np.testing.assert_allclose(p, p.T, atol=1e-10, rtol=0)
            assert np.linalg.eigvalsh(p)[0] >= -1e-10


def test_four_group_rule_precision_failure_and_missing():
    assert study.assess_four([.01]*4, .1)["assessment"] == "pass"
    assert study.assess_four([.11]*4, .1)["assessment"] == "fail"
    assert study.assess_four([-.09, .09, -.09, .09], .1)["assessment"] == "inconclusive"
    result = study.assess_four([.201, -.067, -.067, -.067], .1)
    assert result["assessment"] == "fail"  # Individual ceiling overrides pooled mean.
    assert study.assess_four([0, 0, 0, None], .1)["assessment"] == "inconclusive"
    sample = study.assess_four([.01, .02, .03, .04], .1)
    assert sample["se"] == pytest.approx(math.sqrt(.0005/12), abs=1e-15)


def test_whitening_uses_full_correlations():
    oracle = study.observed_one_step_oracle()
    snapshot = {"mean": oracle["mean"].copy(), "covariance": oracle["covariance"].copy()}
    metrics, dp = study.comparison(snapshot, oracle)
    assert max(abs(m["value"]) for m in metrics.values()) < 1e-10
    np.testing.assert_allclose(dp, 0, atol=1e-10, rtol=0)
    snapshot["covariance"] = np.diag(np.diag(snapshot["covariance"]))
    _, broken = study.comparison(snapshot, oracle)
    assert np.linalg.norm(broken, 2) > .5


def test_resource_stop_retains_thirteen_rows_all_reports_and_paired_branch():
    inventory = study.planned_inventory()
    assert len(inventory["rows"]) == 13
    assert sum(len(row["reports"]) for row in inventory["rows"]) == 44
    inventory["rows"][0]["execution"] = "completed"
    inventory["rows"][1]["execution"] = "running"
    study.mark_resource_stop(inventory, "injected resource stop")
    assert inventory["rows"][0]["execution"] == "completed"
    assert inventory["rows"][1]["execution"] == "failed"
    assert inventory["rows"][2]["branch"] == "H8"
    assert inventory["rows"][2]["execution"] == "not-started"
    assert inventory["decision"] == "no-go"
    assert all(metric["value"] is None for row in inventory["rows"]
               for report in row["reports"].values() for metric in report["metrics"].values())
    study.aggregate_inventory(inventory)
    assert inventory["decision"] == "no-go"
    assert len(inventory["aggregates"]) == 8
    assert all(m["assessment"] == "inconclusive" for g in inventory["aggregates"].values()
               for m in g["metrics"].values())


def test_output_reserve_rejects_before_writing_and_preserves_prior_file(tmp_path):
    existing = tmp_path / "kept.txt"
    existing.write_bytes(b"saved")
    target = tmp_path / "new.json"
    with pytest.raises(study.OutputLimit):
        study.save_json(target, {"data": "too large"}, [tmp_path], 10, 4)
    assert existing.read_bytes() == b"saved"
    assert not target.exists()


def test_monitor_accounting_failure_is_a_terminal_record(monkeypatch, tmp_path):
    class BrokenAccounting:
        def __init__(self):
            raise study.AccountingError("injected unsupported accounting")
    monkeypatch.setattr(study, "DarwinAccounting", BrokenAccounting)
    result = study.monitor_command(["not-launched"], tmp_path / "log", [tmp_path], 1, 1000, 1000)
    assert result["status"] == "unresolved"
    assert result["returncode"] is None
    assert not (tmp_path / "log").exists()
    assert result["elapsed_s"] >= 0


def test_no_stochastic_streams_in_inventory_or_resource_stop(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("inventory must not open a stream")
    monkeypatch.setattr(np.random, "PCG64", forbidden)
    inventory = study.planned_inventory()
    study.mark_resource_stop(inventory, "prelaunch gate failure")
    assert len(inventory["rows"]) == 13


def completed_inventory():
    inventory = study.planned_inventory()
    inventory['global_resource'] = 'completed'
    for row in inventory['rows']:
        row['execution'] = 'completed'
        for report in row['reports'].values():
            report['execution'] = 'completed'
            report['covariance_whitened_error'] = np.zeros((len(report['support']),)*2).tolist()
            report['covariance_spectral_norm'] = 0.
            for name, metric in report['metrics'].items():
                metric.update(value=0., epsilon=.5 if name == 'log_evidence' else (.1 if name.startswith('mean_') else .2))
    return inventory


def test_missing_paired_branch_or_incomplete_run_blocks_aggregate_claims():
    for case, branch in [('H', 'H8'), ('O', 'O')]:
        inventory = completed_inventory()
        row = next(r for r in inventory['rows'] if r['case'] == case and r['branch'] == branch)
        row['execution'] = 'unresolved'
        study.aggregate_inventory(inventory)
        assert inventory['decision'] == 'defer'
        assert all(metric['assessment'] == 'inconclusive' for key, group in inventory['aggregates'].items()
                   if key.startswith(case+'/') for metric in group['metrics'].values())
        assert row['reports']['0']['metrics']['mean_2']['value'] == 0.


def test_known_concentration_or_accuracy_failure_survives_later_unresolved_stop():
    for kind in ('concentration', 'accuracy'):
        inventory = completed_inventory()
        row = next(r for r in inventory['rows'] if r['case'] == 'O')
        row['execution'] = 'unresolved'
        if kind == 'concentration':
            row['events'] = [{'ess': 100., 'max_weight': .01}]
        else:
            row['reports']['0']['metrics']['mean_0']['value'] = .201
        study.aggregate_inventory(inventory)
        assert row['assessment'] == 'fail'
        assert inventory['decision'] == 'no-go'


def test_unresolved_oracle_support_is_rejected_before_cholesky(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unresolved support must not be inverted or factored')
    monkeypatch.setattr(np.linalg, 'cholesky', forbidden)
    with pytest.raises(study.OracleUnresolved):
        study.resolved_oracle_factor(np.diag([.3**2, .08**2, 1e-16]))
    with pytest.raises(ValueError, match='indefinite'):
        study.resolved_oracle_factor(np.diag([.3**2, .08**2, -1e-3]))


def test_final_memory_query_failure_cannot_claim_completed_resource_gate(monkeypatch):
    def unavailable(*args):
        raise OSError('injected final accounting failure')
    monkeypatch.setattr(study.resource, 'getrusage', unavailable)
    inventory = study.planned_inventory()
    inventory['global_resource'] = 'completed'
    result = {'status': 'completed'}
    study.final_memory_accounting(result, inventory)
    assert result['status'] == 'unresolved'
    assert inventory['global_resource'] == 'unresolved'
    assert inventory['decision'] == 'defer'
    result = {'status': 'failed', 'reason': 'known earlier failure'}
    study.final_memory_accounting(result, inventory)
    assert result['status'] == 'failed'
    assert result['reason'] == 'known earlier failure'


def test_plotting_cache_gate_rejects_empty_or_changed_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'CHECKS', tmp_path)
    with pytest.raises(RuntimeError, match='missing'):
        study.verify_plotting_cache({})
    cache = tmp_path / 'matplotlib-cache'
    cache.mkdir()
    path = cache / 'fontlist-test.json'
    path.write_text('{"frozen":true}')
    expected = {path.name: study.digest(path)}
    assert study.verify_plotting_cache(expected) == cache
    path.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match='hash changed'):
        study.verify_plotting_cache(expected)


def test_group_precision_is_visible_on_all_paired_rows_without_extra_replicates():
    inventory = completed_inventory()
    h1 = [r for r in inventory['rows'] if r['branch'] == 'H1']
    for row, error in zip(h1, [-.09, .09, -.09, .09]):
        row['reports']['0']['metrics']['mean_2']['value'] = error
    study.aggregate_inventory(inventory)
    assert inventory['aggregates']['H/0']['independent_groups'] == 4
    for row in [r for r in inventory['rows'] if r['case'] == 'H']:
        assert row['individual_assessment'] == 'pass'
        assert row['assessment'] == 'inconclusive'
        assert row['reports']['0']['group_assessment'] == 'inconclusive'
    assert inventory['decision'] == 'defer'


def test_second_launch_cannot_modify_an_existing_study(monkeypatch, tmp_path):
    checks, existing = tmp_path/'checks', tmp_path/'study'
    checks.mkdir()
    existing.mkdir()
    (checks/'gate.json').write_text('{}')
    previous = b'{"original":"completed evidence"}\n'
    (existing/'inventory.json').write_bytes(previous)
    monkeypatch.setattr(study, 'CHECKS', checks)
    monkeypatch.setattr(study, 'STUDY', existing)
    monkeypatch.setattr(study, 'verify_gate', lambda gate: None)
    monkeypatch.setattr(sys, 'argv', ['run_trajectory_mixture_study.py', 'study'])
    with pytest.raises(FileExistsError):
        study.main()
    assert (existing/'inventory.json').read_bytes() == previous
    assert sorted(p.name for p in existing.iterdir()) == ['inventory.json']


# Reporting-only checks: scalar tables and plotting spies, without study execution.
def out_of_order_plot_row():
    entries = [("750000", .75, 7.5), ("1000000", 1., 10.), ("0", 0., 0.),
               ("500000", .5, 5.), ("10000", .01, .1), ("250000", .25, 2.5)]
    reports = {timestamp: {"metrics": {
        "mean_0": {"value": -mean}, "mean_1": {"value": None},
        "cov_0_0": {"value": -covariance}, "log_evidence": {"value": 99.}}}
        for timestamp, mean, covariance in entries}
    row = {"case": "O", "branch": "O", "seed": 95004, "execution": "completed",
           "assessment": "pass", "events": [], "reports": reports}
    return json.loads(json.dumps(row, sort_keys=True))


def assert_plot_pairs(row, prefix, expected):
    times, values = study.observed_accuracy_series(row, prefix)
    assert list(zip(times, values)) == expected


@pytest.mark.parametrize("prefix,expected", [
    ("mean_", [(0., 0.), (.01, .01), (.25, .25), (.5, .5), (.75, .75), (1., 1.)]),
    ("cov_", [(0., 0.), (.01, .1), (.25, 2.5), (.5, 5.), (.75, 7.5), (1., 10.)]),
])
def test_plot_accuracy_pairs_survive_sorted_json_and_detect_old_order(monkeypatch, prefix, expected):
    row = out_of_order_plot_row()
    assert list(row["reports"]) == ["0", "10000", "1000000", "250000", "500000", "750000"]
    assert_plot_pairs(row, prefix, expected)

    def old_iteration_order(row, prefix):
        values = []
        for report in row["reports"].values():
            errors = [abs(metric["value"]) for name, metric in report["metrics"].items()
                      if name.startswith(prefix) and metric["value"] is not None]
            values.append(max(errors) if errors else math.nan)
        return [timestamp / 1e6 for timestamp in study.REPORTS["O"]], values

    with monkeypatch.context() as mutation:
        mutation.setattr(study, "observed_accuracy_series", old_iteration_order)
        with pytest.raises(AssertionError):
            assert_plot_pairs(row, prefix, expected)
    assert_plot_pairs(row, prefix, expected)


def test_plot_accuracy_unavailable_stays_at_its_timestamp():
    row = out_of_order_plot_row()
    row["reports"]["250000"]["metrics"]["mean_0"]["value"] = None
    times, values = study.observed_accuracy_series(row, "mean_")
    assert times == [0., .01, .25, .5, .75, 1.]
    assert math.isnan(values[2])
    assert values[:2] == [0., .01]
    assert values[3:] == [.5, .75, 1.]


def test_plot_accuracy_saved_seed95004_one_second():
    path = study.STUDY / "inventory.json"
    if not path.exists():
        pytest.skip("retained study inventory is not present in this checkout")
    inventory = json.loads(path.read_text())
    for row in (row for row in inventory["rows"] if row["case"] == "O"):
        for prefix in ("mean_", "cov_"):
            times, values = study.observed_accuracy_series(row, prefix)
            assert times == [0., .01, .25, .5, .75, 1.]
            for timestamp, value in zip(study.REPORTS["O"], values):
                metrics = row["reports"][str(timestamp)]["metrics"]
                expected = max(abs(metric["value"]) for name, metric in metrics.items()
                               if name.startswith(prefix) and metric["value"] is not None)
                assert value == expected
    row = next(row for row in inventory["rows"] if row["case"] == "O" and row["seed"] == 95004)
    times, values = study.observed_accuracy_series(row, "mean_")
    pairs = dict(zip(times, values))
    assert pairs[1.] == pytest.approx(.2369970003, rel=0., abs=5e-11)
    assert pairs[.25] == pytest.approx(.0979865070, rel=0., abs=5e-11)
    assert pairs[.25] != pytest.approx(.2369970003, rel=0., abs=5e-11)


@pytest.fixture
def reporting_plot_spy(monkeypatch):
    """Capture actual plot calls without a renderer, RNG, estimator, or oracle."""
    import types

    def forbidden(*args, **kwargs):
        raise AssertionError("reporting must use saved tables only")

    for name in ("fixture", "new_candidate", "oracle_inventory", "study_worker"):
        monkeypatch.setattr(study, name, forbidden)
    for name in ("PCG64", "SeedSequence", "default_rng"):
        monkeypatch.setattr(np.random, name, forbidden)

    class Axis:
        transAxes = None

        def __init__(self):
            self.lines = []

        def plot(self, times, values, *args, **kwargs):
            self.lines.append((list(times), list(values), kwargs.get("label")))

        def ignore(self, *args, **kwargs):
            pass

        text = annotate = axhline = set = ticklabel_format = legend = ignore

    class Figure:
        def savefig(self, buffer, **kwargs):
            buffer.write(b"reporting plot spy")

        def suptitle(self, *args, **kwargs):
            pass

    grids = []

    def subplots(rows, columns, **kwargs):
        axes = np.array([[Axis() for _ in range(columns)] for _ in range(rows)], dtype=object)
        if rows == 1:
            axes = axes[0]
        grids.append(axes)
        return Figure(), axes

    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots = subplots
    pyplot.close = lambda figure: None
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda backend: None
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return grids


def scalar_plot_inventory():
    rows = [{"case": "H", "branch": branch, "seed": seed, "assessment": "pass",
             "reports": {"100000": {"metrics": {"relative_var_x": {"value": 0.},
                                                "relative_var_y": {"value": 0.}}}}}
            for branch in ("H1", "H8") for seed in study.SEEDS]
    return {"rows": rows + [out_of_order_plot_row()]}


@pytest.mark.parametrize("explicit_destination", [False, True])
def test_plot_results_uses_ordered_series_and_destination(monkeypatch, tmp_path, reporting_plot_spy,
                                                        explicit_destination):
    original = tmp_path / "original"
    original.mkdir()
    (original / "inventory.json").write_bytes(b"preserved original")
    monkeypatch.setattr(study, "STUDY", original)
    output = tmp_path / "correction" if explicit_destination else original
    checked_roots = []
    monkeypatch.setattr(study, "output_bytes", lambda roots: checked_roots.append(roots) or 0)
    inventory = scalar_plot_inventory()
    before = copy.deepcopy(inventory)
    study.plot_results(inventory, destination=output if explicit_destination else None)
    assert inventory == before
    assert checked_roots == [[output], [output]]
    assert (output / "heading-variance.png").read_bytes() == b"reporting plot spy"
    assert (output / "observed-diagnostics.png").read_bytes() == b"reporting plot spy"
    assert (original / "inventory.json").read_bytes() == b"preserved original"
    if explicit_destination:
        assert sorted(path.name for path in original.iterdir()) == ["inventory.json"]
    lower = reporting_plot_spy[1][1]
    assert lower[0].lines == [([0., .01, .25, .5, .75, 1.], [0., .01, .25, .5, .75, 1.], "95004")]
    assert lower[1].lines == [([0., .01, .25, .5, .75, 1.], [0., .1, 2.5, 5., 7.5, 10.], "95004")]


@pytest.mark.parametrize("destination_kind", ["original", "descendant", "existing", "alias"])
def test_plot_results_refuses_preserved_destinations(monkeypatch, tmp_path, reporting_plot_spy,
                                                   destination_kind):
    original = tmp_path / "original"
    original.mkdir()
    sentinel = original / "inventory.json"
    sentinel.write_bytes(b"preserved original")
    monkeypatch.setattr(study, "STUDY", original)
    destination = original
    error = ValueError
    if destination_kind == "descendant":
        destination = original / "correction"
    elif destination_kind == "existing":
        destination = tmp_path / "existing"
        destination.mkdir()
        (destination / "saved.png").write_bytes(b"preserved correction")
        error = FileExistsError
    elif destination_kind == "alias":
        destination = tmp_path / "alias"
        destination.symlink_to(original, target_is_directory=True)
    with pytest.raises(error):
        study.plot_results(scalar_plot_inventory(), destination=destination)
    assert not reporting_plot_spy
    assert sentinel.read_bytes() == b"preserved original"
    assert sorted(path.name for path in original.iterdir()) == ["inventory.json"]
    if destination_kind == "existing":
        assert (destination / "saved.png").read_bytes() == b"preserved correction"
        assert sorted(path.name for path in destination.iterdir()) == ["saved.png"]
