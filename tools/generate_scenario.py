"""Analytical piecewise circular/straight truth; never calls estimator code."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENSOR_HEADER = ["timestamp_us", "sensor", "value1", "value2", "cov11", "cov12", "cov22"]
# End time, forward speed (m/s), yaw rate (rad/s). Boundary values use new segment.
SEGMENTS = [(10.0, 1.5, 0.0), (20.0, 1.5, 0.12), (30.0, 1.5, 0.0),
            (45.0, 1.5, -0.10), (60.0, 1.5, 0.0)]


def truth_at(times_s):
    """Exact integration of each constant-v, constant-omega segment.

    Returns px, py, yaw, speed, yaw_rate. Heading is wrapped independently in
    Python; no C++ functions or generated estimates participate.
    """
    times = np.asarray(times_s, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(times > 60):
        raise ValueError("truth times must be finite and in [0, 60] seconds")
    result = np.zeros((len(times), 5))
    x = y = yaw = start = 0.0
    for end, speed, omega in SEGMENTS:
        mask = (times >= start) & (times < end if end < 60 else times <= end)
        dt = times[mask] - start
        if omega == 0.0:
            result[mask, 0] = x + speed * dt * np.cos(yaw)
            result[mask, 1] = y + speed * dt * np.sin(yaw)
            x += speed * (end - start) * np.cos(yaw)
            y += speed * (end - start) * np.sin(yaw)
        else:
            result[mask, 0] = x + speed / omega * (np.sin(yaw + omega * dt) - np.sin(yaw))
            result[mask, 1] = y - speed / omega * (np.cos(yaw + omega * dt) - np.cos(yaw))
            x += speed / omega * (np.sin(yaw + omega * (end - start)) - np.sin(yaw))
            y -= speed / omega * (np.cos(yaw + omega * (end - start)) - np.cos(yaw))
        result[mask, 2] = (yaw + omega * dt + np.pi) % (2 * np.pi) - np.pi
        result[mask, 3] = speed
        result[mask, 4] = omega
        yaw += omega * (end - start)
        start = end
    return result


def validate_settings(settings):
    duration = settings["duration_s"]
    if not np.isfinite(duration) or not 0 < duration <= 60:
        raise ValueError("duration_s must be in (0, 60]")
    duration_us = round(duration * 1_000_000)
    if duration_us != duration * 1_000_000:
        raise ValueError("duration must be an integer number of microseconds")
    for sensor in ("gyro", "wheel", "gps"):
        hz = settings[f"{sensor}_hz"]
        if not isinstance(hz, int) or hz <= 0 or 1_000_000 % hz:
            raise ValueError("sensor rates must divide 1,000,000 Hz exactly")
        if duration_us % (1_000_000 // hz):
            raise ValueError("duration must end on each sensor's grid")
    for key in ("gyro_noise_std_rad_per_s", "wheel_noise_std_m_per_s", "gps_noise_std_m"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not np.isfinite(settings["true_gyro_bias_rad_per_s"]):
        raise ValueError("bias must be finite")
    start, end = settings["outage_start_s"], settings["outage_end_s"]
    if not np.isfinite(start) or not np.isfinite(end) or not 0 <= start < end <= duration:
        raise ValueError("outage must be within recording")
    if not isinstance(settings["seed"], int) or settings["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")


def write_csv(path, header, rows):
    with Path(path).open("x", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def generate(output, settings):
    validate_settings(settings)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_streams = np.random.SeedSequence(settings["seed"]).spawn(3)
    rows = []
    grids = {}
    duration_us = round(settings["duration_s"] * 1_000_000)
    for rank, sensor in enumerate(("gyro", "wheel", "gps")):
        rng = np.random.Generator(np.random.PCG64(seed_streams[rank]))
        timestamps = np.arange(0, duration_us + 1, 1_000_000 // settings[f"{sensor}_hz"], dtype=np.int64)
        grids[sensor] = timestamps
        truth = truth_at(timestamps / 1e6)
        if sensor == "gps":
            std = settings["gps_noise_std_m"]
            observations = truth[:, :2] + rng.normal(0, std, (len(timestamps), 2))
            for time, (x, y) in zip(timestamps, observations):
                rows.append((int(time), sensor, x, y, std**2, 0, std**2))
        else:
            std = settings["gyro_noise_std_rad_per_s" if sensor == "gyro" else "wheel_noise_std_m_per_s"]
            values = truth[:, 4 if sensor == "gyro" else 3].copy()
            if sensor == "gyro":
                values += settings["true_gyro_bias_rad_per_s"]
            values += rng.normal(0, std, len(timestamps))
            rows.extend((int(t), sensor, z, 0, std**2, 0, 0) for t, z in zip(timestamps, values))
    rank = {"gyro": 0, "wheel": 1, "gps": 2}
    # Assemble a synthetic recording, not a repair of out-of-order replay input.
    rows.sort(key=lambda row: (row[0], rank[row[1]]))
    start_us = round(settings["outage_start_s"] * 1e6)
    end_us = round(settings["outage_end_s"] * 1e6)
    dropout = [r for r in rows if not (r[1] == "gps" and start_us <= r[0] < end_us)]
    write_csv(output / "nominal.csv", SENSOR_HEADER, rows)
    write_csv(output / "dropout.csv", SENSOR_HEADER, dropout)
    times = grids["gyro"]
    truth = truth_at(times / 1e6)
    write_csv(output / "truth.csv", ["timestamp_us", "px_m", "py_m", "yaw_rad", "speed_m_per_s", "yaw_rate_rad_per_s"],
              [(int(time), *state) for time, state in zip(times, truth)])
    provenance = {"settings": settings, "truth_method": "analytical piecewise constant speed/yaw-rate arcs",
                  "segments_end_s_speed_m_per_s_yaw_rate_rad_per_s": SEGMENTS,
                  "rng": "NumPy PCG64; SeedSequence.spawn(3); gyro=0 wheel=1 GPS=2",
                  "numpy_version": np.__version__, "seed_role": "development; no held-out claim",
                  "reserved_next_milestone_seeds": [70001, 70002, 70003],
                  "timestamp_unit": "microseconds", "evaluation_grid": "gyro timestamps, after all simultaneous updates"}
    (output / "generator.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "faults.json").write_text(json.dumps({"dropout": {"start_us": start_us, "end_us_exclusive": end_us},
                                                    "removed_gps_count": len(rows) - len(dropout)}, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=ROOT / "scenarios/development.json")
    args = parser.parse_args()
    generate(args.output, json.loads(args.settings.read_text()))
