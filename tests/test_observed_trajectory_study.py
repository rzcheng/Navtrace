"""Deterministic runner checks; no reserved streams or candidate study batches."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import observed_trajectory as candidate
from tools import run_observed_trajectory_study as study


def complete_inventory():
    inventory = study.planned_inventory()
    for row in inventory["rows"]:
        support = list(range(6)) if row["prefix_index"] < 2 else [2, 3, 4, 5]
        metrics = {f"mean_{j}": {"value": 0., "epsilon": .1} for j in support}
        metrics.update({f"cov_{j}_{k}": {"value": 0., "epsilon": .2}
                        for i, j in enumerate(support) for k in support[i:]})
        row.update(execution="completed", comparison_status="available", metrics=metrics,
                   whitened_covariance_error=np.zeros((len(support), len(support))).tolist(),
                   individual_assessment="pass", covariance_spectral_norm=0.)
    inventory["global_resource"] = "completed"
    return inventory


def test_inventory_has_all_groups_prefixes_and_110_scalar_targets():
    inventory = study.planned_inventory()
    assert len(inventory["rows"]) == 24
    assert [(row["group"], row["timestamp_us"]) for row in inventory["rows"]] == [
        (group, timestamp) for group in (96001, 96002, 96003, 96004)
        for timestamp in (0, 10000, 250000, 500000, 750000, 1000000)]
    study.aggregate_inventory(inventory)
    assert sum(len(entry["metrics"]) for entry in inventory["aggregates"].values()) == 110
    assert len(inventory["aggregates"]) == 6
    assert inventory["decision"] == "defer"
    assert all(row["execution"] == "not-started" for row in inventory["rows"])


def test_perfect_complete_comparisons_pass_without_pooled_covariance():
    inventory = complete_inventory()
    # Arbitrary differing full means must not add between-group covariance.
    for index, row in enumerate(inventory["rows"]):
        row["mean"] = [float(index)]*6
    study.aggregate_inventory(inventory)
    assert inventory["decision"] == "pass"
    assert all(entry["covariance_spectral"]["average_norm"] == 0
               for entry in inventory["aggregates"].values())


@pytest.mark.parametrize("kind", ["missing-row", "missing-comparison", "unresolved", "not-started"])
def test_missing_groups_never_form_smaller_replicate_set(kind):
    inventory = complete_inventory()
    if kind == "missing-row":
        inventory["rows"].pop()
    elif kind == "missing-comparison":
        inventory["rows"][-1]["comparison_status"] = "unavailable"
    else:
        inventory["rows"][-1].update(execution=kind, individual_assessment="inconclusive")
    study.aggregate_inventory(inventory)
    assert inventory["decision"] == "defer"
    assert not inventory["aggregates"]["1000000"]["available"]


@pytest.mark.parametrize("failure", ["individual", "scalar-average", "spectral-individual",
                                     "spectral-average", "resource", "exact"])
def test_no_go_precedes_unresolved(failure):
    inventory = complete_inventory()
    inventory["global_resource"] = "unresolved"
    if failure == "individual":
        inventory["rows"][0]["metrics"]["mean_0"]["value"] = .21
    elif failure == "scalar-average":
        for row in inventory["rows"]:
            if row["timestamp_us"] == 0:
                row["metrics"]["mean_0"]["value"] = .11
    elif failure.startswith("spectral"):
        for index, row in enumerate(inventory["rows"]):
            if row["timestamp_us"] == 0 and (failure == "spectral-average" or index == 0):
                row["whitened_covariance_error"][0][0] = .31 if failure == "spectral-average" else .51
    elif failure == "resource":
        study.stop_inventory(inventory, "mock resource failure")
    else:
        inventory["rows"][0].update(execution="failed", individual_assessment="fail")
    study.aggregate_inventory(inventory)
    assert inventory["decision"] == "no-go"


def test_four_group_precision_is_not_counted_as_24_trials():
    inventory = complete_inventory()
    for index, row in enumerate(inventory["rows"]):
        if row["timestamp_us"] == 0:
            row["metrics"]["mean_0"]["value"] = [-.04, .04, -.04, .04][index//6]
    study.aggregate_inventory(inventory)
    entry = inventory["aggregates"]["0"]["metrics"]["mean_0"]
    assert entry["assessment"] == "inconclusive"
    assert entry["four_se"] == pytest.approx(4*np.sqrt(4*.04**2/12))
    assert inventory["decision"] == "defer"


def test_stop_preserves_completed_rows_and_marks_current_and_remaining():
    inventory = study.planned_inventory()
    inventory["rows"][0].update(execution="completed", individual_assessment="pass")
    inventory["rows"][1]["execution"] = "running"
    study.stop_inventory(inventory, "mock cap")
    assert inventory["rows"][0]["execution"] == "completed"
    assert inventory["rows"][1]["execution"] == "failed"
    assert all(row["execution"] == "not-started" and row["reason"] == "mock cap"
               for row in inventory["rows"][2:])
    assert inventory["decision"] == "no-go"


def test_fixed_inputs_oracle_support_and_same_time_wheel():
    inputs = study.frozen_inputs()
    assert len(inputs["events"]) == 32
    assert [event[1] for event in inputs["events"][:2]] == ["gyro", "wheel"]
    oracles = study.independent_oracles(inputs)
    assert set(oracles) == set(study.PREFIXES)
    assert oracles[0]["support"] == list(range(6))
    assert oracles[10000]["support"] == list(range(6))
    assert all(oracles[timestamp]["support"] == [2, 3, 4, 5] for timestamp in study.PREFIXES[2:])
    for timestamp in (10000, 500000, 1000000):
        assert [event for event in inputs["events"] if event[0] <= timestamp][-1][1] == "wheel"
    for timestamp in (250000, 750000):
        assert [event for event in inputs["events"] if event[0] <= timestamp][-1][1] == "gyro"
    assert oracles[0]["log_evidence"] == 0


def test_only_observed_oracles_are_called(monkeypatch):
    monkeypatch.setattr(study.old, "oracle_inventory", lambda: pytest.fail("broad oracle inventory called"))
    monkeypatch.setattr(study.old, "linear_batch_oracle", lambda: pytest.fail("L outside scope"))
    monkeypatch.setattr(study.old, "heading_oracle", lambda _: pytest.fail("H outside scope"))
    assert set(study.independent_oracles(study.frozen_inputs())) == set(study.PREFIXES)


def test_coordinate_digest_has_fixed_order_and_endianness():
    values = np.array([[1., 2.], [3., 4.]])
    expected = hashlib.sha256(bytes.fromhex(
        "000000000000f03f000000000000004000000000000008400000000000001040")).hexdigest()
    assert study.coordinate_digest(values) == expected
    assert study.coordinate_digest(np.asarray(values, dtype=">f8", order="F")) == expected


def test_endpoint_audit_preserves_conditional_and_between_correlations():
    snapshot = {"angular": np.array([[.2, .1, -.02], [-.1, .04, .01]]),
        "conditional_mean": np.array([[1., 2., 3.], [4., -1., 2.]]),
        "conditional_covariance": np.array([[[2., .2, .1], [.2, 1., -.1], [.1, -.1, .4]],
                                              [[1., -.3, .1], [-.3, 2., .2], [.1, .2, .5]]]),
        "logweights": np.log([.5, .5])}
    components = np.array([[1., 2., .2, 3., .1, -.02], [4., -1., -.1, 2., .04, .01]])
    snapshot["mean"] = (components[0]+components[1])/2
    delta = components[0]-components[1]
    covariance = np.outer(delta, delta)/4
    covariance[np.ix_([0, 1, 3], [0, 1, 3])] += (
        snapshot["conditional_covariance"][0]+snapshot["conditional_covariance"][1])/2
    snapshot["covariance"] = covariance
    assert max(study.endpoint_audit(snapshot).values()) < 1e-14
    snapshot["covariance"] = covariance.copy()
    snapshot["covariance"][0, 2] = snapshot["covariance"][2, 0] = 0
    assert study.endpoint_audit(snapshot)["covariance_max_abs"] > .1


@pytest.mark.parametrize("timestamp", [0, 10000])
def test_small_fixed_coordinate_path_retains_and_checks_complete_snapshot(timestamp):
    inputs = study.frozen_inputs()
    events = [candidate.Observation(*event) for event in inputs["events"] if event[0] <= timestamp]
    posterior = candidate.condition_prefix(inputs["mean"], inputs["covariance"], inputs["diffusion"], events)
    # At most one step and two deterministic supplied rows; no random generator.
    coordinates = np.stack([np.zeros(posterior.angular_dimension), np.ones(posterior.angular_dimension)])
    report = posterior.evaluate(coordinates)
    snapshot = study.retained_snapshot(report, posterior.density_terms(report["angular_coordinates"]))
    assert set(snapshot) == {"angular", "conditional_mean", "conditional_covariance", "logweights",
        "mean", "covariance", "log_prior", "log_likelihood", "log_proposal", "importance_residual"}
    row = study.planned_inventory()["rows"][0]
    study.inspect_row(row, snapshot, report, study.independent_oracles(inputs)[timestamp])
    assert row["comparison_status"] == "available"
    assert row["exact"]["importance_max_abs"] < 1e-10
    assert row["exact"]["evidence_abs_error"] < 1e-10
    assert row["exact"]["endpoint_audit"]["covariance_max_abs"] < 1e-10
    assert snapshot["angular"].shape == (2, 3)
    assert snapshot["conditional_covariance"].shape == (2, 3, 3)


def test_exact_failure_is_retained_and_not_reclassified_as_accuracy_success():
    inputs = study.frozen_inputs()
    posterior = candidate.condition_prefix(inputs["mean"], inputs["covariance"], inputs["diffusion"], [])
    report = posterior.evaluate(np.zeros((1, posterior.angular_dimension)))
    snapshot = study.retained_snapshot(report, posterior.density_terms(report["angular_coordinates"]))
    snapshot["importance_residual"] = np.array([1e-6])
    row = study.planned_inventory()["rows"][0]
    with pytest.raises(ArithmeticError, match="exact identity"):
        study.inspect_row(row, snapshot, report, study.independent_oracles(inputs)[0])
    assert row["exact"]["importance_max_abs"] == 1e-6
    assert row["individual_assessment"] == "fail"
    assert row["mean"] is not None


@pytest.mark.parametrize("matrix", [np.array([[1., .01], [0., 1.]]), np.diag([1., -1e-3]),
                                     np.array([[np.nan]])])
def test_covariance_checks_reject_material_failures(matrix):
    with pytest.raises(ArithmeticError):
        study.covariance_check(matrix)


def test_unresolved_oracle_is_not_pseudoinverted():
    inputs = study.frozen_inputs()
    posterior = candidate.condition_prefix(inputs["mean"], inputs["covariance"], inputs["diffusion"], [])
    report = posterior.evaluate(np.zeros((1, posterior.angular_dimension)))
    snapshot = study.retained_snapshot(report, posterior.density_terms(report["angular_coordinates"]))
    oracle = {"mean": np.zeros(6), "covariance": np.zeros((6, 6)), "support": list(range(6)), "log_evidence": 0.}
    row = study.planned_inventory()["rows"][0]
    study.inspect_row(row, snapshot, report, oracle)
    assert row["comparison_status"] == "unavailable"
    assert row["individual_assessment"] == "inconclusive"


def test_output_reserve_checked_before_snapshot_write(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study, "OUTPUT", 100)
    monkeypatch.setattr(study, "RESERVE", 40)
    (tmp_path / "existing").write_bytes(b"x"*50)
    with pytest.raises(study.old.OutputLimit):
        study.write_bytes(tmp_path / "snapshot", b"x"*11)
    assert not (tmp_path / "snapshot").exists()
    study.write_bytes(tmp_path / "status", b"x"*11, metadata=True)
    assert (tmp_path / "existing").read_bytes() == b"x"*50


def test_atomic_replacement_charges_pending_and_previous_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study, "OUTPUT", 100)
    monkeypatch.setattr(study, "RESERVE", 0)
    path = tmp_path / "old"
    path.write_bytes(b"x"*60)
    with pytest.raises(study.old.OutputLimit):
        study.write_bytes(path, b"y"*60)
    assert path.read_bytes() == b"x"*60


def gate_fixture(tmp_path, monkeypatch):
    checks = tmp_path / "checks"
    checks.mkdir()
    cache = checks / "matplotlib-cache"
    cache.mkdir()
    (cache / "fontlist-test.json").write_text("{}")
    (tmp_path / "new.py").write_text("# fixed test source\n")
    initialization = checks / "initialization.json"
    initialization.write_text(json.dumps({"roots_absent": True, "groups_unopened": True}))
    attempts = [{"label": label, "attempt_path": f"attempt-{index}-{label}", "status": "completed",
                 "charged_s": 1., "assessed_peak_bytes": 100} for index, label in enumerate(sorted(study.REQUIRED_CHECKS))]
    ledger = checks / "validation.json"
    ledger.write_text(json.dumps({"monitor_review_accepted": True, "attempts": attempts}))
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "CHECKS", checks)
    monkeypatch.setattr(study, "DEMO", tmp_path / "demo")
    monkeypatch.setattr(study, "STUDY", tmp_path / "study")
    monkeypatch.setattr(study, "FROZEN", {})
    monkeypatch.setattr(study, "REQUIRED_SOURCE", {"new.py"})
    monkeypatch.setattr(study.subprocess, "check_output", lambda command, **kwargs: "abc\n" if command[1] == "rev-parse" else "")
    gate = {"decision": "eligible", "independent_review_accepted": True,
        "final_required_checks_passed": True, "clean_committed_source": True, "git_commit": "abc",
        "source_sha256": {"new.py": study.old.digest(tmp_path / "new.py")},
        "initialization_sha256": study.old.digest(initialization),
        "validation_ledger_sha256": study.old.digest(ledger),
        "required_attempts": {attempt["label"]: attempt["attempt_path"] for attempt in attempts},
        "plotting_cache_sha256": {"fontlist-test.json": study.old.digest(cache / "fontlist-test.json")}}
    return gate, checks, attempts


def test_gate_accepts_only_reviewed_unchanged_source_and_validation(tmp_path, monkeypatch):
    gate, _, _ = gate_fixture(tmp_path, monkeypatch)
    study.verify_gate(gate)
    (tmp_path / "new.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="source changed"):
        study.verify_gate(gate)


@pytest.mark.parametrize("kind", ["review", "missing-source", "initialization", "ledger", "missing-check",
                                 "failed-check", "memory", "time", "dirty", "font-cache"])
def test_gate_refuses_missing_changed_or_failed_evidence(tmp_path, monkeypatch, kind):
    gate, checks, attempts = gate_fixture(tmp_path, monkeypatch)
    if kind == "review":
        gate["independent_review_accepted"] = False
    elif kind == "missing-source":
        gate["source_sha256"] = {}
    elif kind == "initialization":
        gate["initialization_sha256"] = "wrong"
    elif kind == "ledger":
        gate["validation_ledger_sha256"] = "wrong"
    elif kind == "missing-check":
        gate["required_attempts"].pop("demo")
    elif kind in ("failed-check", "memory", "time"):
        if kind == "failed-check":
            attempts[0]["status"] = "failed"
        elif kind == "memory":
            attempts[0]["assessed_peak_bytes"] = None
        else:
            attempts[0]["charged_s"] = 901
        path = checks / "validation.json"
        path.write_text(json.dumps({"monitor_review_accepted": True, "attempts": attempts}))
        gate["validation_ledger_sha256"] = study.old.digest(path)
    elif kind == "dirty":
        monkeypatch.setattr(study.subprocess, "check_output", lambda command, **kwargs: "abc\n" if command[1] == "rev-parse" else " M new.py\n")
    else:
        (checks / "matplotlib-cache" / "fontlist-test.json").write_text("changed")
    with pytest.raises(RuntimeError):
        study.verify_gate(gate)


def test_second_launch_never_touches_existing_study_root(tmp_path, monkeypatch):
    gate, checks, _ = gate_fixture(tmp_path, monkeypatch)
    (checks / "gate.json").write_text(json.dumps(gate))
    study.STUDY.mkdir()
    retained = study.STUDY / "retained"
    retained.write_text("original evidence")
    monkeypatch.setattr(study.old, "monitor_command", lambda *args, **kwargs: pytest.fail("must not launch worker"))
    with pytest.raises(FileExistsError):
        study.launch(0.)
    assert retained.read_text() == "original evidence"
    assert list(study.STUDY.iterdir()) == [retained]


def test_gate_block_records_all24_without_consuming_root(tmp_path, monkeypatch):
    gate, checks, _ = gate_fixture(tmp_path, monkeypatch)
    gate["independent_review_accepted"] = False
    (checks / "gate.json").write_text(json.dumps(gate))
    with pytest.raises(RuntimeError):
        study.launch(0.)
    assert not study.STUDY.exists()
    inventory = json.loads(next(checks.glob("blocked-inventory-*.json")).read_text())
    assert len(inventory["rows"]) == 24
    assert all(row["execution"] == "not-started" and "gate blocked" in row["reason"] for row in inventory["rows"])


@pytest.mark.parametrize("kind", ["memory", "missing", "time", "output"])
def test_final_accounting_checks_each_cap_and_missing_measurements(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study.time, "monotonic", lambda: 10.)
    monkeypatch.setattr(study.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=100))
    result = {"status": "completed", "worker_job_high_water_bytes": 100, "sampled_aggregate_peak_bytes": 100}
    inventory = complete_inventory()
    if kind == "memory":
        result["worker_job_high_water_bytes"] = study.MEMORY
    elif kind == "missing":
        result["worker_job_high_water_bytes"] = None
    elif kind == "time":
        monkeypatch.setattr(study.time, "monotonic", lambda: 121.)
    else:
        monkeypatch.setattr(study.old, "output_bytes", lambda _: study.OUTPUT+1)
    study.final_resource_check(result, inventory, 0.)
    assert result["status"] == ("unresolved" if kind == "missing" else "failed")
    assert inventory["decision"] == ("defer" if kind == "missing" else "no-go")


def test_worker_requires_wrapper_token_without_opening_rng(monkeypatch):
    monkeypatch.delenv("NAVTRACE_OBSERVED_WORKER_TOKEN", raising=False)
    monkeypatch.setattr(np.random, "PCG64", lambda *args: pytest.fail("reserved generator opened"))
    with pytest.raises(RuntimeError, match="owning one-launch"):
        study.worker()


def test_mocked_worker_uses_each_key_once_and_saves_before_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study, "COUNT", 2)
    monkeypatch.setenv("NAVTRACE_OBSERVED_WORKER_TOKEN", str(study.os.getppid()))
    study.save_inventory(study.planned_inventory())
    keys, draws, inspected = [], [], []

    def fake_seed_sequence(key):
        saved = json.loads((tmp_path / "inventory.json").read_text())
        assert len(saved["rows"]) == 24  # Durable before even the first mock key.
        keys.append(list(key))
        return list(key)

    class FakeBitGenerator:
        def __init__(self, key):
            self.key, self.draws = key, 0

        @property
        def state(self):
            return {"bit_generator": "mock", "key": self.key, "draws": self.draws}

    class FakeGenerator:
        def __init__(self, bit_generator):
            self.bit_generator = bit_generator

        def standard_normal(self, shape):
            self.bit_generator.draws += 1
            assert self.bit_generator.draws == 1
            draws.append((self.bit_generator.key, shape))
            return np.zeros(shape, dtype=np.float64)

    class FakePosterior:
        angular_dimension, residual_dimension = 3, 3
        coordinate_layout = (("checkpoint", 0), ("checkpoint", 1), ("checkpoint", 2))
        step_sizes_s = ()
        phase = "mock"

        def __init__(self, timestamp):
            self.timestamp_us = timestamp

        def evaluate(self, coordinates):
            saved = json.loads((tmp_path / "inventory.json").read_text())
            current = next(row for row in saved["rows"] if row["execution"] == "running")
            assert current["rng"]["initial_state"]["draws"] == 0
            assert current["rng"]["final_state"]["draws"] == 1
            assert current["coordinate_sha256"] == study.coordinate_digest(coordinates)
            return {"angular_paths": np.zeros((2, 1, 3)), "conditional_mean": np.zeros((2, 3)),
                "conditional_covariance": np.repeat(np.eye(3)[None], 2, axis=0),
                "mean": np.zeros(6), "covariance": np.eye(6), "logweights": np.full(2, -np.log(2)),
                "log_evidence": 0., "angular_coordinates": coordinates}

        def density_terms(self, coordinates):
            return {"log_prior": np.zeros(2), "log_likelihood": np.zeros(2),
                    "log_proposal": np.zeros(2), "log_importance": np.zeros(2)}

    monkeypatch.setattr(np.random, "SeedSequence", fake_seed_sequence)
    monkeypatch.setattr(np.random, "PCG64", FakeBitGenerator)
    monkeypatch.setattr(np.random, "Generator", FakeGenerator)
    expected_events = study.frozen_inputs()["events"]

    def fake_condition(mean, covariance, diffusion, events, **kwargs):
        assert all(isinstance(event, candidate.Observation) for event in events)
        timestamp = events[-1].timestamp_us if events else 0
        # The adapter must retain values, variances, prefix boundaries and gyro/wheel order.
        assert [(event.timestamp_us, event.sensor, event.value, event.variance) for event in events] == [
            event for event in expected_events if event[0] <= timestamp]
        assert kwargs == {"checkpoint_us": 0, "max_step_s": .01, "max_gap_s": 2.}
        return FakePosterior(timestamp)

    monkeypatch.setattr(candidate, "condition_prefix", fake_condition)
    monkeypatch.setattr(study, "independent_oracles", lambda _: {timestamp: {} for timestamp in study.PREFIXES})
    template = {(row["group"], row["timestamp_us"]): row for row in complete_inventory()["rows"]}

    def fake_inspect(row, snapshot, report, oracle):
        path = tmp_path / row["snapshot"]
        assert path.exists()
        durable = json.loads((tmp_path / "inventory.json").read_text())
        saved_row = next(item for item in durable["rows"] if item["group"] == row["group"]
                         and item["timestamp_us"] == row["timestamp_us"])
        assert saved_row["snapshot"] == row["snapshot"]
        assert saved_row["snapshot_sha256"] == study.old.digest(path)
        with np.load(path) as saved:
            for name in ("log_prior", "log_likelihood", "log_proposal", "importance_residual"):
                np.testing.assert_array_equal(saved[name], snapshot[name])
        reference = template[(row["group"], row["timestamp_us"])]
        for name in ("metrics", "whitened_covariance_error", "comparison_status", "individual_assessment"):
            row[name] = reference[name]
        inspected.append((row["group"], row["timestamp_us"]))
        # A known finite failure must not suppress later rows.
        if len(inspected) == 1:
            row["individual_assessment"] = "fail"
            raise ArithmeticError("mock exact failure")

    monkeypatch.setattr(study, "inspect_row", fake_inspect)
    monkeypatch.setattr(study, "plot_results", lambda _: None)
    assert study.worker() == 0
    expected = [[group, 9, index] for group in study.GROUPS for index in range(6)]
    assert keys == expected
    assert draws == [(key, (2, 3)) for key in expected]
    assert len(inspected) == 24
    saved = json.loads((tmp_path / "inventory.json").read_text())
    assert saved["rows"][0]["execution"] == "failed"
    assert all(row["execution"] == "completed" for row in saved["rows"][1:])
    assert saved["decision"] == "no-go"


def test_plot_spies_keep_chronological_keys_and_all_statuses(monkeypatch):
    import sys
    import types
    inventory = complete_inventory()
    for row in inventory["rows"]:
        index = row["prefix_index"]
        mean_key = "mean_0" if index < 2 else "mean_2"
        row["metrics"][mean_key]["value"] = index+.01*(row["group"]-study.GROUPS[0])
        row.update(resources={"row_elapsed_s": index+.1}, dimensions={"angular": index+3},
                   structural_weights={"ess": 2048., "maximum_weight": 1/2048})
    inventory["rows"][2].update(comparison_status="unavailable", execution="unresolved",
                                individual_assessment="inconclusive")
    inventory["rows"][5].update(execution="failed", individual_assessment="fail")
    study.aggregate_inventory(inventory)
    inventory["rows"] = list(reversed(inventory["rows"]))
    # JSON key sorting reproduces the storage round trip that once reordered times.
    inventory = json.loads(json.dumps(inventory, sort_keys=True))
    figures, writes, tables = [], [], []

    class Axis:
        def __init__(self):
            self.plots = []
            self.ticks = None

        def plot(self, x, y, *args, **kwargs):
            self.plots.append((list(x), list(y), kwargs))

        def set_xticks(self, x, labels):
            self.ticks = (list(x), labels)

        def table(self, **kwargs):
            tables.append(kwargs)
            return SimpleNamespace(auto_set_font_size=lambda _: None, set_fontsize=lambda _: None)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    class Figure:
        def __init__(self, count):
            self.axes = [Axis() for _ in range(count)]

        def add_axes(self, *args):
            return Axis()

        def tight_layout(self):
            pass

        def subplots_adjust(self, **kwargs):
            pass

        def savefig(self, buffer, **kwargs):
            buffer.write(b"mock PNG")

    def subplots(count, *args, **kwargs):
        figure = Figure(count)
        figures.append(figure)
        return figure, figure.axes

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda _: None
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots, pyplot.close = subplots, lambda _: None
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setattr(study, "write_bytes", lambda path, data: writes.append((path.name, data)))
    study.plot_results(inventory)
    series = figures[0].axes[0].plots[0][1]
    np.testing.assert_allclose(series, [0., 1., np.nan, 3., 4., 5.], equal_nan=True)
    assert figures[1].axes[0].plots[0][1] == [.1, 1.1, 2.1, 3.1, 4.1, 5.1]
    assert tables[0]["colLabels"] == ["0", "0.01", "0.25", "0.5", "0.75", "1"]
    assert tables[0]["cellText"][0][2] == "U/I"
    assert tables[0]["cellText"][0][5] == "F/F"
    assert [name for name, _ in writes] == ["supported-errors.png", "cost-and-weights.png"]


@pytest.mark.parametrize("late", ["memory", "time", "output", "missing"])
def test_finalize_catches_accounting_failure_after_last_planned_flush(tmp_path, monkeypatch, late):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    flushes = {"count": 0}
    real_save = study.emergency_json

    def save(path, value):
        real_save(path, value)
        if path.name == "manifest.json":
            flushes["count"] += 1

    monkeypatch.setattr(study, "emergency_json", save)
    monkeypatch.setattr(study.time, "monotonic", lambda: 121. if late == "time" and flushes["count"] >= 2 else 10.)

    def usage(_):
        if flushes["count"] >= 2:
            if late == "missing":
                raise OSError("mock terminal accounting unavailable")
            if late == "memory":
                return SimpleNamespace(ru_maxrss=study.MEMORY)
        return SimpleNamespace(ru_maxrss=100)

    monkeypatch.setattr(study.resource, "getrusage", usage)
    real_bytes = study.old.output_bytes
    monkeypatch.setattr(study.old, "output_bytes", lambda roots:
        study.OUTPUT+1 if late == "output" and flushes["count"] >= 2 else real_bytes(roots))
    inventory = complete_inventory()
    result = {"status": "completed", "worker_job_high_water_bytes": 100, "sampled_aggregate_peak_bytes": 100}
    assert study.finalize(result, inventory, {}, 0.) == 1
    expected = "unresolved" if late == "missing" else "failed"
    assert result["status"] == expected
    resources = json.loads((tmp_path / "resources.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    saved = json.loads((tmp_path / "inventory.json").read_text())
    assert resources["status"] == manifest["resource_status"] == expected
    assert saved["decision"] == manifest["decision"] == ("defer" if late == "missing" else "no-go")
    assert all(study.old.digest(tmp_path / name) == digest for name, digest in manifest["artifact_sha256"].items())


def test_failure_during_manifest_preparation_retains_full_inventory(tmp_path, monkeypatch):
    gate, checks, _ = gate_fixture(tmp_path, monkeypatch)
    (checks / "gate.json").write_text(json.dumps(gate))
    monkeypatch.setattr(study, "verify_gate", lambda _: None)
    real_digest = study.old.digest

    def broken_digest(path):
        if Path(path) == checks / "gate.json":
            raise OSError("mock manifest digest failure")
        return real_digest(path)

    monkeypatch.setattr(study.old, "digest", broken_digest)
    monkeypatch.setattr(study.time, "monotonic", lambda: 10.)
    monkeypatch.setattr(study.old, "monitor_command", lambda *args, **kwargs: pytest.fail("worker must not launch"))
    assert study.launch(0.) == 1
    saved = json.loads((study.STUDY / "inventory.json").read_text())
    assert len(saved["rows"]) == 24
    assert saved["decision"] == "no-go"
    assert all(row["execution"] == "not-started" and "manifest digest" in row["reason"] for row in saved["rows"])
