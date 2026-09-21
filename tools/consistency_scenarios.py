"""Independent M3 truth and paired observations; truth never enters replay inputs."""
import json
from pathlib import Path

import numpy as np

from tools.fault_scenarios import apply_faults
from tools.generate_scenario import SEGMENTS, SENSOR_HEADER, truth_at, write_csv

TRUTH_HEADER = ["timestamp_us", "px_m", "py_m", "yaw_rad", "speed_m_per_s",
                "yaw_rate_rad_per_s", "gyro_bias_rad_per_s"]
SENSOR_RATES = {"gyro": 100, "wheel": 20, "gps": 5}
SENSOR_STD = {"gyro": .008, "wheel": .05, "gps": .6}
WEAK_SEGMENTS = [(.2, 1.5, 0.), (20., 0., 0.), (30., 1.5, 0.),
                 (45., 1.5, -.10), (60., 1.5, 0.)]


def weak_start_truth(times_s):
    """Analytical arcs: stop at .2 s, restart at 20 s from the stopped pose."""
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any((times < 0) | (times > 60)):
        raise ValueError("truth requires a finite one-dimensional time array in [0,60]")
    states = np.zeros((len(times), 5))
    x = y = yaw = start = 0.
    for end, speed, omega in WEAK_SEGMENTS:
        mask = (times >= start) & (times < end if end < 60 else times <= end)
        dt = times[mask] - start
        if omega == 0:
            states[mask, 0] = x + speed * dt * np.cos(yaw)
            states[mask, 1] = y + speed * dt * np.sin(yaw)
            x += speed * (end - start) * np.cos(yaw)
            y += speed * (end - start) * np.sin(yaw)
        else:
            states[mask, 0] = x + speed / omega * (np.sin(yaw + omega * dt) - np.sin(yaw))
            states[mask, 1] = y - speed / omega * (np.cos(yaw + omega * dt) - np.cos(yaw))
            x += speed / omega * (np.sin(yaw + omega * (end - start)) - np.sin(yaw))
            y -= speed / omega * (np.cos(yaw + omega * (end - start)) - np.cos(yaw))
        states[mask, 2] = (yaw + omega * dt + np.pi) % (2 * np.pi) - np.pi
        states[mask, 3] = speed
        states[mask, 4] = omega
        yaw += omega * (end - start)
        start = end
    return states


def _duration_us(duration_s):
    if not np.isfinite(duration_s) or not 0 < duration_s <= 60:
        raise ValueError("duration must be finite and in (0,60] seconds")
    duration_us = round(duration_s * 1_000_000)
    if duration_us != duration_s * 1_000_000 or duration_us % 200_000:
        raise ValueError("duration must end on the 5 Hz GPS grid")
    return duration_us


def _streams(seed, count):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return [np.random.Generator(np.random.PCG64(s))
            for s in np.random.SeedSequence(seed).spawn(count)]


def _sensor_rows(times_us, truth, sensor_rngs):
    """Sample only observation fields on fixed sensor grids; truth stays separate."""
    rows = []
    for sensor, rng in zip(SENSOR_RATES, sensor_rngs):
        stride = 100 // SENSOR_RATES[sensor]
        sample_times, sample_truth = times_us[::stride], truth[::stride]
        std = SENSOR_STD[sensor]
        if sensor == "gps":
            observations = sample_truth[:, :2] + rng.normal(0, std, (len(sample_times), 2))
            rows.extend((int(t), sensor, x, y, std**2, 0, std**2)
                        for t, (x, y) in zip(sample_times, observations))
        else:
            values = sample_truth[:, 4 if sensor == "gyro" else 3].copy()
            if sensor == "gyro":
                values += sample_truth[:, 5]
            values += rng.normal(0, std, len(sample_times))
            rows.extend((int(t), sensor, z, 0, std**2, 0, 0)
                        for t, z in zip(sample_times, values))
    rank = {"gyro": 0, "wheel": 1, "gps": 2}
    rows.sort(key=lambda row: (row[0], rank[row[1]]))
    return rows


def _write_truth(path, times, truth):
    write_csv(path, TRUTH_HEADER, [(int(t), *state) for t, state in zip(times, truth)])


def generate_stress(output, seed, protocol):
    """Create all paired protocol cases; return annotations indexed by case ID."""
    duration_us = _duration_us(protocol["stress_duration_s"])
    _streams(seed, 3)  # Validate seed before creating any files.
    scenarios = protocol["scenarios"]
    if (not scenarios or len({s["id"] for s in scenarios}) != len(scenarios)
            or any(s["family"] not in ("original", "weak_start") for s in scenarios)):
        raise ValueError("expected unique scenario IDs and known trajectory families")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    times = np.arange(0, duration_us + 1, 10_000, dtype=np.int64)
    rows_by_family = {}
    for family, truth_function in (("original", truth_at), ("weak_start", weak_start_truth)):
        states = np.column_stack((truth_function(times / 1e6), np.full(len(times), .015)))
        _write_truth(output / f"{family}-truth.csv", times, states)
        # Restart identical sensor streams: changing the trajectory cannot change noise draws.
        rows = _sensor_rows(times, states, _streams(seed, 3))
        rows_by_family[family] = rows
        write_csv(output / f"{family}-nominal.csv", SENSOR_HEADER, rows)
    annotations = {}
    for scenario in scenarios:
        rows, annotation = apply_faults(rows_by_family[scenario["family"]], scenario)
        write_csv(output / f"{scenario['id']}.csv", SENSOR_HEADER, rows)
        annotation.update(family=scenario["family"], truth_file=f"{scenario['family']}-truth.csv",
                          outage_us=scenario["outage_us"])
        annotations[scenario["id"]] = annotation
    provenance = {
        "seed": seed, "protocol_id": protocol["protocol_id"],
        "seed_role": "development" if seed == protocol["development_seed"] else "evaluation",
        "classification": "deterministic-prior analytical model-mismatch stress",
        "duration_s": protocol["stress_duration_s"], "true_gyro_bias_rad_per_s": .015,
        "truth_method": "analytical straight/circular segments, independent of C++",
        "original_segments_end_s_speed_yaw_rate": SEGMENTS,
        "weak_start_segments_end_s_speed_yaw_rate": WEAK_SEGMENTS,
        "sensor_hz": SENSOR_RATES, "sensor_noise_std": SENSOR_STD,
        "rng": "NumPy PCG64; SeedSequence.spawn(3): gyro=0, wheel=1, GPS=2; shared draws across families",
        "numpy_version": np.__version__, "timestamp_unit": "microseconds",
    }
    (output / "generator.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "annotations.json").write_text(json.dumps(annotations, indent=2) + "\n")
    return annotations


def euler_maruyama_step(state, dt_s, process_increments):
    """Independent truth step; increments are already sqrt(q*dt)-scaled draws."""
    state = np.asarray(state, dtype=float)
    increments = np.asarray(process_increments, dtype=float)
    if (state.shape != (6,) or increments.shape != (3,) or not np.all(np.isfinite(state))
            or not np.all(np.isfinite(increments)) or not np.isfinite(dt_s) or dt_s <= 0):
        raise ValueError("invalid state, increments, or positive truth step")
    next_state = state.copy()
    next_state[0] += dt_s * state[3] * np.cos(state[2])
    next_state[1] += dt_s * state[3] * np.sin(state[2])
    next_state[2] = (state[2] + dt_s * state[4] + np.pi) % (2 * np.pi) - np.pi
    next_state[3:] += increments
    return next_state


def _prior_and_diffusion(filter_path):
    entries = {}
    for line in Path(filter_path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key in entries:
            raise ValueError("duplicate filter entry")
        entries[key] = float(value)
    names = ("px", "py", "yaw", "speed", "yaw_rate", "gyro_bias")
    mean = np.array([entries[f"initial_{n}"] for n in names])
    variance = np.array([entries[f"variance_{n}"] for n in names])
    q = np.array([entries[f"{n}_diffusion_{unit}"] for n, unit in
                  (("speed", "m2_per_s3"), ("yaw_rate", "rad2_per_s3"), ("bias", "rad2_per_s3"))])
    if (not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance))
            or not np.all(np.isfinite(q)) or np.any(variance < 0) or np.any(q < 0)):
        raise ValueError("nonfinite prior or negative variance/diffusion")
    return mean, variance, q


def generate_matched(output, seed, filter_path, duration_s=2):
    """Sample the discrete stochastic model, using independent initial/process/sensor streams."""
    duration_us = _duration_us(duration_s)
    rngs = _streams(seed, 5)
    mean, variance, q = _prior_and_diffusion(filter_path)
    times = np.arange(0, duration_us + 1, 10_000, dtype=np.int64)
    truth = np.empty((len(times), 6))
    truth[0] = mean + np.sqrt(variance) * rngs[0].standard_normal(6)
    truth[0, 2] = (truth[0, 2] + np.pi) % (2 * np.pi) - np.pi
    increments = rngs[1].standard_normal((len(times) - 1, 3)) * np.sqrt(q * .01)
    for i, increment in enumerate(increments, start=1):
        truth[i] = euler_maruyama_step(truth[i - 1], .01, increment)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_truth(output / "truth.csv", times, truth)
    write_csv(output / "nominal.csv", SENSOR_HEADER, _sensor_rows(times, truth, rngs[2:]))
    provenance = {
        "seed": seed, "classification": "sampled-prior discrete-model-matched nonlinear sanity",
        "duration_s": duration_s, "truth_step_s": .01,
        "truth_method": "independent Python Euler-Maruyama using start-of-step state, end-of-step diffusion",
        "initial_mean": mean.tolist(), "initial_covariance_diagonal": variance.tolist(),
        "process_diffusion_speed_rate_bias": q.tolist(),
        "sensor_hz": SENSOR_RATES, "sensor_noise_std": SENSOR_STD,
        "rng": "NumPy PCG64; SeedSequence.spawn(5): initial=0, process=1, gyro=2, wheel=3, GPS=4",
        "numpy_version": np.__version__, "timestamp_unit": "microseconds",
        "caution": "Matches the discrete stochastic model; nonlinear EKF Gaussian assumptions remain approximate.",
    }
    (output / "generator.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output
