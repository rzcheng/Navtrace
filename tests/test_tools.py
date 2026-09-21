import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.evaluate import aligned_poses, compute_metrics, wrapped_angle_difference
from tools.generate_scenario import ROOT, generate, truth_at, validate_settings


@pytest.fixture
def settings():
    return json.loads((ROOT / "scenarios/development.json").read_text())


def test_metrics_hand_computed():
    times = np.arange(4) * 1_000_000
    truth = np.zeros((4, 3))
    estimates = np.array([[3, 4, 0], [0, 0, 0], [0, 2, 0], [0, 0, 0]])
    metrics = compute_metrics(times, estimates, truth, 1_000_000, 3_000_000)
    assert metrics["position_rmse_m"] == pytest.approx(np.sqrt(29 / 4))
    assert metrics["outage_position_rmse_m"] == pytest.approx(np.sqrt(2))
    assert metrics["outage_end_position_error_m"] == 2
    assert metrics["outage_end_evaluation_timestamp_us"] == 2_000_000
    assert metrics["final_position_error_m"] == 0
    assert metrics["heading_rmse_rad"] == 0


def test_heading_metric_wraps_and_rejects_invalid():
    assert wrapped_angle_difference(np.pi - .01, -np.pi + .01) == pytest.approx(-.02)
    true = np.array([[0, 0, -np.pi + .01], [0, 0, -np.pi + .01]])
    est = np.array([[0, 0, np.pi - .01], [0, 0, np.pi - .01]])
    result = compute_metrics([0, 1], est, true, 0, 2)
    assert result["heading_rmse_rad"] == pytest.approx(.02)
    assert result["final_wrapped_heading_error_rad"] == pytest.approx(-.02)
    with pytest.raises(ValueError):
        compute_metrics([0, 0], est, true, 0, 2)
    est[0, 0] = np.nan
    with pytest.raises(ValueError):
        compute_metrics([0, 1], est, true, 0, 2)


def test_analytical_truth_at_segment_boundaries():
    states = truth_at([0, 10, 20, 30])
    assert states[1, 0] == pytest.approx(15)
    assert states[1, 1] == pytest.approx(0)
    assert states[1, 4] == pytest.approx(.12)
    assert states[2, 0] == pytest.approx(15 + 12.5 * np.sin(1.2))
    assert states[2, 1] == pytest.approx(12.5 * (1 - np.cos(1.2)))
    assert states[2, 2] == pytest.approx(1.2)
    np.testing.assert_allclose(states[3, :2] - states[2, :2], 15 * np.array([np.cos(1.2), np.sin(1.2)]))


def test_generator_pair_reproducible_and_only_removes_gps(tmp_path, settings):
    first, second = tmp_path / "first", tmp_path / "second"
    generate(first, settings)
    generate(second, settings)
    for name in ("nominal.csv", "dropout.csv", "truth.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with (first / "nominal.csv").open() as file:
        nominal = list(csv.DictReader(file))
    with (first / "dropout.csv").open() as file:
        dropout = list(csv.DictReader(file))
    expected = [r for r in nominal if not (r["sensor"] == "gps" and 20_000_000 <= int(r["timestamp_us"]) < 40_000_000)]
    assert dropout == expected
    assert len(nominal) - len(dropout) == 100
    assert set(nominal[0]) == {"timestamp_us", "sensor", "value1", "value2", "cov11", "cov12", "cov22"}
    assert not any("truth" in field or "bias" in field or "fault" in field for field in nominal[0])
    with pytest.raises(FileExistsError):
        generate(first, settings)


def test_random_sensor_streams_are_independent(tmp_path, settings):
    first = generate(tmp_path / "first", settings)
    settings["gps_hz"] = 10
    second = generate(tmp_path / "second", settings)
    def non_gps(path):
        return [line for line in (path / "nominal.csv").read_text().splitlines() if ",gps," not in line]
    assert non_gps(first) == non_gps(second)


def test_bad_generator_settings(settings):
    settings["gyro_hz"] = 3
    with pytest.raises(ValueError):
        validate_settings(settings)


def test_evaluation_last_simultaneous_row_and_missing_timestamps(tmp_path):
    header = "timestamp_us,px_m,py_m,yaw_rad\n"
    truth, estimate = tmp_path / "truth.csv", tmp_path / "estimate.csv"
    truth.write_text(header + "0,0,0,0\n1,1,0,0\n")
    estimate.write_text(header + "0,3,0,0\n0,0,0,0\n1,1,0,0\n")
    _, est, true = aligned_poses(truth, estimate)
    np.testing.assert_array_equal(est, true)
    estimate.write_text(header + "0,0,0,0\n")
    with pytest.raises(ValueError, match="missing"):
        aligned_poses(truth, estimate)


@pytest.fixture
def binary():
    path = Path(os.environ.get("NAVTRACE_BINARY", ROOT / "build/navtrace_replay"))
    if not path.exists():
        pytest.skip("build C++ executable first for integration tests")
    return path.resolve()


def test_cli_actual_replay_is_deterministic_and_rejects_malformed(tmp_path, binary):
    sensors = tmp_path / "sensors.csv"
    sensors.write_text("timestamp_us,sensor,value1,value2,cov11,cov12,cov22\n"
                       "0,gyro,0.015,0,0.000064,0,0\n0,wheel,1.5,0,0.0025,0,0\n"
                       "0,gps,0,0,0.36,0,0.36\n10000,gyro,0.02,0,0.000064,0,0\n")
    config = ROOT / "scenarios/filter.conf"
    def replay(name):
        return subprocess.run([str(binary), str(sensors), str(config), "ekf", str(tmp_path / f"{name}.csv"),
                               str(tmp_path / f"{name}-gps.csv")], text=True, capture_output=True)
    assert replay("a").returncode == 0
    assert replay("b").returncode == 0
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert replay("a").returncode != 0  # Refuse overwrite.
    sensors.write_text(sensors.read_text() + "20000,gyro,nan,0,1,0,0\n")
    failed = replay("bad")
    assert failed.returncode != 0
    assert "sensor line 6" in failed.stderr


def test_demo_refuses_existing_results_before_writing(tmp_path, binary):
    from tools.run_demo import run_demo
    marker = tmp_path / "precious.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run_demo(binary, tmp_path, ROOT / "scenarios/development.json", ROOT / "scenarios/filter.conf")
    with pytest.raises(ValueError, match="marked"):
        run_demo(binary, tmp_path, ROOT / "scenarios/development.json", ROOT / "scenarios/filter.conf", overwrite=True)
    assert marker.read_text() == "keep"


def test_cli_rejects_output_path_aliases(tmp_path, binary):
    # Two spellings of one nonexistent file must not be opened as separate outputs.
    output = str(tmp_path / "output.csv")
    alias = str(tmp_path) + "/./output.csv"
    sensors = tmp_path / "input.csv"
    sensors.write_text("timestamp_us,sensor,value1,value2,cov11,cov12,cov22\n0,gps,0,0,1,0,1\n")
    result = subprocess.run([str(binary), str(sensors), str(ROOT / "scenarios/filter.conf"), "ekf", output, alias],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "distinct" in result.stderr
    assert not Path(output).exists()
