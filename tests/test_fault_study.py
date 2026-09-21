import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.generate_scenario import ROOT, SENSOR_HEADER
from tools.fault_scenarios import generate_suite, validate_protocol
from tools.fault_metrics import gps_decisions, gps_return, position_settling, summarize, audit_event_outputs
from tools.run_fault_study import run_study


@pytest.fixture
def protocol():
    return json.loads((ROOT / "scenarios/gps_faults.json").read_text())


def test_fault_variants_preserve_noise_covariance_and_other_sensors(tmp_path, protocol):
    output = tmp_path / "inputs"
    annotations = generate_suite(output, protocol, protocol["development_seed"], "development")
    with (output / "base/nominal.csv").open() as file:
        nominal = list(csv.DictReader(file))
    original = {(r["timestamp_us"], r["sensor"]): r for r in nominal}
    expected_counts = {"nominal": (0, 0), "dropout": (0, 100), "spikes": (3, 0), "offset": (100, 0), "dropout_offset": (25, 190)}
    for scenario in protocol["scenarios"]:
        name = scenario["id"]
        annotation = annotations["scenarios"][name]
        corrupted = annotation["corrupted_gps_timestamps_us"]
        removed = annotation["removed_gps_timestamps_us"]
        assert (len(corrupted), len(removed)) == expected_counts[name]
        with (output / f"{name}.csv").open() as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == SENSOR_HEADER  # No evaluator-only columns.
            rows = list(reader)
        assert len(rows) == len(nominal) - len(removed)
        assert [r for r in rows if r["sensor"] != "gps"] == [r for r in nominal if r["sensor"] != "gps"]
        for row in rows:
            before = original[(row["timestamp_us"], row["sensor"])]
            time = int(row["timestamp_us"])
            if row["sensor"] == "gps" and time in corrupted:
                fault = next(f for f in scenario["offsets"] if f["start_us"] <= time < f["end_us"])
                np.testing.assert_allclose([float(row["value1"]) - float(before["value1"]),
                                            float(row["value2"]) - float(before["value2"])], fault["offset_m"], atol=1e-12, rtol=0)
                assert all(row[key] == before[key] for key in ("cov11", "cov12", "cov22"))
            else:
                assert row == before
    assert (output / "nominal.csv").read_bytes() == (output / "base/nominal.csv").read_bytes()
    assert (output / "dropout.csv").read_bytes() == (output / "base/dropout.csv").read_bytes()
    with pytest.raises(FileExistsError):
        generate_suite(output, protocol, protocol["development_seed"], "development")


def test_protocol_rejects_overlap_and_faulty_recovery_window(protocol):
    validate_protocol(protocol)
    bad = copy.deepcopy(protocol)
    bad["scenarios"][-1]["offsets"][0]["start_us"] = 39_000_000
    with pytest.raises(ValueError, match="overlap"):
        validate_protocol(bad)
    bad = copy.deepcopy(protocol)
    bad["scenarios"][-1]["recovery_windows"][0]["reference_us"] = 44_000_000
    with pytest.raises(ValueError, match="no faults"):
        validate_protocol(bad)
    bad = copy.deepcopy(protocol)
    bad["evaluation_seeds"][0] = bad["development_seed"]
    with pytest.raises(ValueError, match="disjoint"):
        validate_protocol(bad)


def settling(errors):
    times = np.arange(len(errors), dtype=np.int64) * 10_000
    window = {"reference_us": 0, "observation_end_us": int(times[-1])}
    return position_settling(times, errors, window, 1.0, 2_000_000, 10_000)


def test_settling_inclusive_threshold_and_endpoints():
    result = settling(np.ones(201))
    assert result["settling_status"] == "within_limit_at_return"
    assert result["settling_start_delay_s"] == 0
    assert result["settling_confirmation_delay_s"] == 2


def test_settling_requires_full_two_seconds_not_200_samples():
    # 200 good samples span 1.99 s; the sample at t=2.00 is above the limit.
    errors = np.r_[np.zeros(200), [2.0, 2.0]]
    result = settling(errors)
    assert result["settling_status"] == "not_observed"
    assert result["settling_start_delay_s"] is None
    assert result["settling_confirmation_delay_s"] is None


def test_settling_resets_dwell_and_records_later_recovery():
    errors = np.zeros(401)
    errors[100] = 1.01  # Break an otherwise qualifying run at one second.
    result = settling(errors)
    assert result["settling_status"] == "recovered"
    assert result["settling_start_delay_s"] == pytest.approx(1.01)
    assert result["settling_confirmation_delay_s"] == pytest.approx(3.01)


def test_settling_insufficient_followup_is_not_success():
    result = settling(np.zeros(200))
    assert result["settling_status"] == "insufficient_followup"
    assert result["settling_start_delay_s"] is None


def test_settling_missing_samples_and_nonfinite_errors_rejected():
    window = {"reference_us": 0, "observation_end_us": 2_000_000}
    times = np.arange(201, dtype=np.int64) * 10_000
    with pytest.raises(ValueError, match="complete uniform"):
        position_settling(np.delete(times, 100), np.zeros(200), window, 1, 2_000_000, 10_000)
    errors = np.zeros(201)
    errors[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        position_settling(times, errors, window, 1, 2_000_000, 10_000)


def test_gps_fault_counts_have_correct_denominators_and_nulls():
    annotation = {"gps_timestamps_us": [0, 200_000], "corrupted_gps_timestamps_us": [200_000]}
    rows = [{"timestamp_us": "0", "status": "accepted", "nis": "1", "threshold": "9"},
            {"timestamp_us": "200000", "status": "rejected", "nis": "10", "threshold": "9"}]
    result = gps_decisions(rows, annotation, "gated")
    assert result["clean_gps_rejection_fraction"] == 0
    assert result["fault_gps_rejection_fraction"] == 1
    assert result["fault_gps_considered"] == 1
    with pytest.raises(ValueError, match="timestamps"):
        gps_decisions(rows[:1], annotation, "gated")
    wrong = copy.deepcopy(rows)
    wrong[1]["status"] = "accepted"
    with pytest.raises(ValueError, match="squared-distance"):
        gps_decisions(wrong, annotation, "gated")
    for row in rows:
        row["status"] = "ignored"
    ignored = gps_decisions(rows, annotation, "dr")
    assert ignored["clean_gps_rejection_fraction"] is None
    assert ignored["fault_gps_rejection_fraction"] is None
    assert ignored["fault_gps_ignored"] == 1


def test_gps_acceptance_is_separate_from_pose_settling():
    annotation = {"corrupted_gps_timestamps_us": [0]}
    rows = [{"timestamp_us": "0", "status": "accepted"},
            {"timestamp_us": "200000", "status": "rejected"},
            {"timestamp_us": "400000", "status": "accepted"}]
    window = {"reference_us": 0, "observation_end_us": 2_000_000}
    result = gps_return(rows, annotation, window, "gated")
    assert result["first_clean_gps_accepted_delay_s"] == .4  # Corrupted acceptance is excluded.
    assert settling(np.full(201, 3.0))["settling_status"] == "not_observed"
    assert gps_return(rows[:2], annotation, window, "gated")["gps_return_status"] == "not_observed"
    assert gps_return(rows, annotation, window, "dr")["gps_return_status"] == "not_applicable"
    assert gps_return(rows, annotation, window, "dr")["first_clean_gps_accepted_delay_s"] is None


def test_summary_uses_median_and_keeps_unrecovered_episodes():
    metrics = []
    for method, adjustment in (("dr", 0), ("ekf", 1), ("gated", 2)):
        for seed, rmse in enumerate((1, 2, 100)):
            row = {"scenario": "nominal", "method": method, "seed": seed, "position_rmse_m": rmse + adjustment}
            row.update({f"{kind}_gps_{status}": 0 for kind in ("clean", "fault") for status in ("accepted", "rejected", "ignored")})
            metrics.append(row)
    episodes = [{"scenario": "nominal", "method": method, "seed": seed, "episode": "return",
                 "settling_status": "not_observed" if method == "dr" else "within_limit_at_return"}
                for seed in range(3) for method in ("dr", "ekf", "gated")]
    protocol = {"scenarios": [{"id": "nominal", "recovery_windows": [{"id": "return"}]}]}
    summary, paired = summarize(metrics, episodes, protocol, [0, 1, 2])
    assert summary[0]["position_rmse_median_m"] == 2
    assert summary[0]["position_rmse_max_m"] == 100
    assert summary[0]["settling_not_observed"] == 3
    assert all(r["gated_minus_basic_rmse_m"] == 1 for r in paired)
    with pytest.raises(ValueError, match="incomplete"):
        summarize([r for r in metrics if r["seed"] != 2], episodes, protocol, [0, 1, 2])
    with pytest.raises(ValueError, match="recovery episodes"):
        summarize(metrics, episodes[:-1], protocol, [0, 1, 2])


def test_runner_refuses_changed_filter_before_creating_output(tmp_path):
    changed = tmp_path / "filter.conf"
    changed.write_text((ROOT / "scenarios/filter.conf").read_text() + "# modified\n")
    with pytest.raises(ValueError, match="frozen"):
        run_study("unused-binary", tmp_path / "output", ROOT / "scenarios/gps_faults.json", changed)
    assert not (tmp_path / "output").exists()


def test_small_study_uses_actual_cpp_and_separate_annotations(tmp_path, protocol):
    binary = Path(os.environ.get("NAVTRACE_BINARY", ROOT / "build/navtrace_replay")).resolve()
    if not binary.exists():
        pytest.skip("build C++ executable first for integration test")
    protocol["generator_settings"].update(duration_s=3, outage_start_s=1, outage_end_s=2)
    protocol["reference_window_us"] = [1_000_000, 2_000_000]
    spike = {"id": "spike", "label": "Test spike", "outage_us": None,
             "offsets": [{"start_us": 400_000, "end_us": 600_000, "offset_m": [12, -8]}],
             "recovery_windows": [{"id": "return", "reference_us": 600_000, "observation_end_us": 3_000_000}]}
    protocol["scenarios"] = [protocol["scenarios"][0], spike]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    output = tmp_path / "study"
    metrics, episodes = run_study(binary, output, path, ROOT / "scenarios/filter.conf", development_smoke=True, make_plots=False)
    assert len(metrics) == 6 and len(episodes) == 3
    gated = next(r for r in metrics if r["scenario"] == "spike" and r["method"] == "gated")
    assert gated["fault_gps_rejected"] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["seed_role"] == "development"
    assert manifest["seeds"] == [20260921]
    assert len(manifest["commands"]) == 6
    for command in manifest["commands"]:
        assert len(command) == 6
        assert command[2] == str(output / "filter.conf")
        assert all("annotations.json" not in arg and "truth.csv" not in arg for arg in command)
    with pytest.raises(FileExistsError):
        run_study(binary, output, path, ROOT / "scenarios/filter.conf", development_smoke=True, make_plots=False)
    # Independent replay audit must catch an accidentally dropped output row.
    seed_dir = output / "seed-20260921"
    estimate = seed_dir / "nominal-dr.csv"
    lines = estimate.read_text().splitlines()
    estimate.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="exactly one"):
        audit_event_outputs(seed_dir / "inputs/nominal.csv", estimate, "dr")
