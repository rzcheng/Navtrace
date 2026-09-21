import csv
import json

import numpy as np
import pytest

from tools.consistency_scenarios import (TRUTH_HEADER, euler_maruyama_step,
                                         generate_matched, generate_stress, weak_start_truth)
from tools.generate_scenario import ROOT, SENSOR_HEADER, generate


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_stress_original_exactly_reproduces_existing_generator(tmp_path):
    protocol = json.loads((ROOT / "scenarios/consistency.json").read_text())
    settings = json.loads((ROOT / "scenarios/development.json").read_text())
    old = generate(tmp_path / "old", settings)
    new = tmp_path / "new"
    annotations = generate_stress(new, settings["seed"], protocol)
    assert (new / "nominal.csv").read_bytes() == (old / "nominal.csv").read_bytes()
    assert annotations["early_offset"]["removed_gps_timestamps_us"] == list(range(2_000_000, 22_000_000, 200_000))
    assert annotations["early_offset"]["corrupted_gps_timestamps_us"] == list(range(22_000_000, 27_000_000, 200_000))
    for family in ("original", "weak_start"):
        nominal = read_rows(new / f"{family}-nominal.csv")
        for case in (s for s in protocol["scenarios"] if s["family"] == family):
            variant = read_rows(new / f"{case['id']}.csv")
            assert [r for r in variant if r["sensor"] != "gps"] == [r for r in nominal if r["sensor"] != "gps"]
            by_event = {(r["timestamp_us"], r["sensor"]): r for r in nominal}
            for row in variant:
                baseline = by_event[(row["timestamp_us"], row["sensor"])]
                assert [row[k] for k in ("cov11", "cov12", "cov22")] == [baseline[k] for k in ("cov11", "cov12", "cov22")]
    with pytest.raises(FileExistsError):
        generate_stress(new, settings["seed"], protocol)


def test_weak_start_has_independent_continuous_pose_and_paired_sensor_noise(tmp_path):
    states = weak_start_truth([0, .2, 19.99, 20, 30, 45])
    np.testing.assert_allclose(states[:4, 0], [0, .3, .3, .3], atol=1e-14)
    np.testing.assert_allclose(states[:4, 1:3], 0, atol=1e-14)
    np.testing.assert_array_equal(states[:4, 3], [1.5, 0, 0, 1.5])
    assert states[4, 0] == pytest.approx(15.3)
    assert states[4, 4] == pytest.approx(-.1)
    assert states[5, 0] == pytest.approx(15.3 + 15 * np.sin(1.5))
    assert states[5, 1] == pytest.approx(-15 * (1 - np.cos(1.5)))
    protocol = json.loads((ROOT / "scenarios/consistency.json").read_text())
    generate_stress(tmp_path / "stress", protocol["development_seed"], protocol)
    noise = {}
    for family in ("original", "weak_start"):
        truth = {r["timestamp_us"]: r for r in read_rows(tmp_path / "stress" / f"{family}-truth.csv")}
        residuals = []
        for row in read_rows(tmp_path / "stress" / f"{family}-nominal.csv"):
            t = truth[row["timestamp_us"]]
            expected = {"gyro": float(t["yaw_rate_rad_per_s"]) + float(t["gyro_bias_rad_per_s"]),
                        "wheel": float(t["speed_m_per_s"]), "gps": float(t["px_m"])}[row["sensor"]]
            residuals.append(float(row["value1"]) - expected)
            if row["sensor"] == "gps":
                residuals.append(float(row["value2"]) - float(t["py_m"]))
        noise[family] = residuals
    np.testing.assert_allclose(noise["original"], noise["weak_start"], atol=1e-14, rtol=0)


def test_matched_step_uses_start_state_then_end_noise():
    state = np.array([10, -2, np.pi / 2, 2, .5, .01])
    updated = euler_maruyama_step(state, .2, [1, -.1, .002])
    np.testing.assert_allclose(updated, [10, -1.6, np.pi / 2 + .1, 3, .4, .012], atol=1e-14)
    np.testing.assert_array_equal(state, [10, -2, np.pi / 2, 2, .5, .01])
    crossing = euler_maruyama_step([0, 0, np.pi - .01, 0, .2, 0], .1, [0, 0, 0])
    assert crossing[2] == pytest.approx(-np.pi + .01)
    with pytest.raises(ValueError):
        euler_maruyama_step(state, .01, [0, np.nan, 0])


def test_matched_deterministic_sampled_prior_and_separate_truth(tmp_path):
    seed = 20260921
    config = ROOT / "scenarios/filter.conf"
    first = generate_matched(tmp_path / "first", seed, config, .2)
    second = generate_matched(tmp_path / "second", seed, config, .2)
    for name in ("nominal.csv", "truth.csv", "generator.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    metadata = json.loads((first / "generator.json").read_text())
    assert metadata["initial_mean"] == [0, 0, 0, 1.5, 0, 0]
    assert metadata["initial_covariance_diagonal"] == [.04, .04, .007615435494667714, .25, .01, .0009]
    assert metadata["process_diffusion_speed_rate_bias"] == [.25, .01, 1e-7]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed).spawn(5)[0]))
    expected_initial = np.array(metadata["initial_mean"]) + np.sqrt(metadata["initial_covariance_diagonal"]) * rng.standard_normal(6)
    expected_initial[2] = (expected_initial[2] + np.pi) % (2 * np.pi) - np.pi
    truth = read_rows(first / "truth.csv")
    np.testing.assert_array_equal([float(truth[0][key]) for key in TRUTH_HEADER[1:]], expected_initial)
    process_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed).spawn(5)[1]))
    increments = process_rng.standard_normal(3) * np.sqrt(np.array([.25, .01, 1e-7]) * .01)
    expected_next = expected_initial.copy()
    expected_next[0] += .01 * expected_initial[3] * np.cos(expected_initial[2])
    expected_next[1] += .01 * expected_initial[3] * np.sin(expected_initial[2])
    expected_next[2] = (expected_initial[2] + .01 * expected_initial[4] + np.pi) % (2 * np.pi) - np.pi
    expected_next[3:] += increments
    np.testing.assert_array_equal([float(truth[1][key]) for key in TRUTH_HEADER[1:]], expected_next)
    assert len(truth) == 21
    sensors = read_rows(first / "nominal.csv")
    assert list(sensors[0]) == SENSOR_HEADER
    assert not any("true" in key or "fault" in key or "bias" in key for key in sensors[0])
    assert [r["sensor"] for r in sensors[:3]] == ["gyro", "wheel", "gps"]
    assert [r["timestamp_us"] for r in sensors if r["sensor"] == "gps"] == ["0", "200000"]
    gyro_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed).spawn(5)[2]))
    assert float(sensors[0]["value1"]) == expected_initial[4] + expected_initial[5] + gyro_rng.normal(0, .008)
    with pytest.raises(FileExistsError):
        generate_matched(first, seed, config)
    with pytest.raises(ValueError):
        generate_matched(tmp_path / "invalid", seed, config, .15)
