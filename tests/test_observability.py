import csv
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.generate_scenario import ROOT
from tools.observability import (GYRO_H, STATE_NAMES, STATE_SCALES,
                                 analytical_nominal,
                                 measurement_schedule, run_observability,
                                 sensitivity_matrix, summarize_sensitivity)


def test_process_product_order_and_measurement_timestamp():
    transitions = np.repeat(np.eye(6)[None], 3, axis=0)
    transitions[1, 2, 4] = 2.0
    transitions[2, 0, 2] = 3.0
    times = np.array([0, 10_000, 20_000])
    matrix = sensitivity_matrix(times, transitions, [(int(t), "gps") for t in times],
                                state_scales=np.ones(6),
                                measurements={"gps": (np.eye(6)[:2], np.eye(2))})
    # At t0 there is no propagation. F2 F1 creates a px/omega coupling of 6;
    # reversed multiplication or applying the next transition at t1 would fail.
    expected = np.zeros((6, 6))
    expected[::2, 0] = 1
    expected[1::2, 1] = 1
    expected[4, 2], expected[4, 4] = 3, 6
    np.testing.assert_array_equal(matrix, expected)


def test_consistent_state_unit_rescaling_preserves_whitened_sensitivity():
    times = np.array([0, 10_000, 20_000])
    transitions = np.repeat(np.eye(6)[None], 3, axis=0)
    transitions[1, 2, 4] = .01
    transitions[2, 1, 2] = .015
    h = np.eye(6)[:2]
    covariance = np.array([[4., 1.], [1., 9.]])
    events = [(int(time), "gps") for time in times]
    original = sensitivity_matrix(times, transitions, events,
                                  measurements={"gps": (h, covariance)})
    # Coordinate change x_new=T x: centimeters and degrees, consistently
    # changing F, H and D. Noise stays in the original measurement units.
    units = np.array([100., 100., 180 / np.pi, 100., 180 / np.pi, 180 / np.pi])
    transformed_f = transitions * units[None, :, None] / units[None, None, :]
    transformed = sensitivity_matrix(times, transformed_f, events,
                                     state_scales=STATE_SCALES * units,
                                     measurements={"gps": (h / units, covariance)})
    np.testing.assert_allclose(original, transformed, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(summarize_sensitivity(original)["singular_values"],
                               summarize_sensitivity(transformed)["singular_values"],
                               rtol=1e-14, atol=1e-14)
    # Direct correlated-R check: ||L^-1 H D eta||² = z^T solve(R,z).
    eta = np.array([1., 2., 0., 0., 0., 0.])
    z = h @ (STATE_SCALES * eta)
    assert np.sum((original[:2] @ eta)**2) == pytest.approx(z @ np.linalg.solve(covariance, z))


def test_gyro_only_cannot_distinguish_constant_yaw_rate_and_bias():
    times = np.arange(0, 2_000_001, 10_000, dtype=np.int64)
    transitions = np.repeat(np.eye(6)[None], len(times), axis=0)
    transitions[1:, 2, 4] = .01
    matrix = sensitivity_matrix(times, transitions, [(int(time), "gyro") for time in times])
    assert summarize_sensitivity(matrix)["numerical_rank"] == 1
    physical_ambiguity = np.array([0., 0., 0., 0., 1., -1.])
    np.testing.assert_allclose(np.sum(matrix * (physical_ambiguity / STATE_SCALES), axis=1),
                               0, atol=1e-12)
    # All rows are unchanged by process integration: gyro still sees omega+b.
    np.testing.assert_array_equal(matrix, np.repeat((GYRO_H * STATE_SCALES / .008), len(times), axis=0))


def test_invalid_sensitivities_and_schedules_are_rejected():
    times = np.array([0, 10_000])
    transitions = np.repeat(np.eye(6)[None], 2, axis=0)
    for events in ([(0, "gyro"), (0, "gyro")], [(1, "gyro")],
                   [(10_000, "gyro"), (0, "wheel")], [(0, "unknown")]):
        with pytest.raises(ValueError, match="measurement event"):
            sensitivity_matrix(times, transitions, events)
    with pytest.raises(ValueError, match="integer"):
        sensitivity_matrix(times.astype(float), transitions, [(0, "gyro")])
    bad = transitions.copy()
    bad[0, 0, 0] = .5
    with pytest.raises(ValueError, match="identity"):
        sensitivity_matrix(times, bad, [(0, "gyro")])
    with pytest.raises(ValueError, match="positive definite"):
        sensitivity_matrix(times, transitions, [(0, "gyro")],
                           measurements={"gyro": (GYRO_H, np.zeros((1, 1)))})


def test_nominal_paths_and_physical_measurement_schedules():
    times, states = analytical_nominal("turning", 2)
    assert len(times) == 201
    np.testing.assert_allclose(states[-1, :3], [12.5 * np.sin(.24), 12.5 * (1 - np.cos(.24)), .24])
    schedule = measurement_schedule(times)
    assert sum(sensor == "gyro" for _, sensor in schedule) == 201
    assert sum(sensor == "wheel" for _, sensor in schedule) == 41
    assert sum(sensor == "gps" for _, sensor in schedule) == 11
    assert not any(sensor == "gps" for _, sensor in measurement_schedule(times, gps_available=False))
    assert schedule[:3] == [(0, "gyro"), (0, "wheel"), (0, "gps")]
    _, stationary = analytical_nominal("stationary", 2)
    np.testing.assert_array_equal(stationary, np.zeros((201, 6)))


@pytest.fixture
def linearize_binary():
    replay = Path(os.environ.get("NAVTRACE_BINARY", ROOT / "build/navtrace_replay"))
    path = Path(os.environ.get("NAVTRACE_LINEARIZE_BINARY", replay.with_name("navtrace_linearize")))
    if not path.exists():
        pytest.skip("build navtrace_linearize for C++ observability integration tests")
    return path.resolve()


def test_compiled_helper_uses_previous_nominal_state_and_validates_input(tmp_path, linearize_binary):
    nominal, output = tmp_path / "nominal.csv", tmp_path / "f.csv"
    with nominal.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["timestamp_us", *STATE_NAMES])
        writer.writerows([[0, 0, 0, 0, 1.5, 0, 0], [10_000, 0, 0, np.pi / 2, 2, 0, 0],
                          [20_000, 0, 0, 1, 3, 0, 0]])
    def run(path):
        return subprocess.run([str(linearize_binary), str(nominal), str(path)],
                              capture_output=True, text=True)
    assert run(output).returncode == 0
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    first = np.array([float(rows[0][f"f{i}{j}"]) for i in range(6) for j in range(6)]).reshape(6, 6)
    np.testing.assert_array_equal(first, np.eye(6))
    assert float(rows[1]["f12"]) == pytest.approx(.015)
    assert float(rows[1]["f02"]) == 0
    assert float(rows[2]["f02"]) == pytest.approx(-.02)
    assert float(rows[2]["f12"]) == pytest.approx(0, abs=1e-16)
    assert run(output).returncode != 0
    original = nominal.read_text()
    nominal.write_text(original.replace("20000,", "21000,"))
    bad_output = tmp_path / "bad.csv"
    failed = run(bad_output)
    assert failed.returncode != 0
    assert "nominal line 4" in failed.stderr
    assert not bad_output.exists()
    # Validation precedes output creation for nonfinite and malformed values too.
    for malformed in (original.replace("1.5", "nan"), original.replace("1.5", "1.5junk"),
                      original.replace("0,0,0,0,1.5,0,0", "0,0,0,0,1.5,0,0,7")):
        nominal.write_text(malformed)
        failed = run(bad_output)
        assert failed.returncode != 0
        assert "nominal line 2" in failed.stderr
        assert not bad_output.exists()


def test_actual_cpp_observability_summary_and_gps_absence(tmp_path, linearize_binary):
    summaries = run_observability(linearize_binary, tmp_path / "study")
    assert len(summaries) == 12
    for summary in summaries:
        expected_rank = 2 if summary["sensor_availability"] == "gyro_wheel" else (
            4 if summary["trajectory"] == "stationary" else 6)
        assert summary["numerical_rank"] == expected_rank
        np.testing.assert_array_equal(summary["state_scales"], STATE_SCALES)
        for direction in summary["weak_directions"]:
            np.testing.assert_allclose(direction["physical_direction"],
                                       STATE_SCALES * direction["normalized_direction"])
    with pytest.raises(FileExistsError):
        run_observability(linearize_binary, tmp_path / "study")
