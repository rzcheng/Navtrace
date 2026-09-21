"""Offline local process sensitivity, not an online filter or calibration proof.

For nominal samples x_k, the compiled helper exports production F_k evaluated
at x_(k-1). Thus Phi_0=I, Phi_k=F_k Phi_(k-1), with no Kalman correction.
At each physically available sensor event stack L_R^-1 H Phi_k D, where
R=L_R L_R^T and x_error=D eta. Fixed physical scales make eta dimensionless.
The deterministic calculation omits process noise and any gating selection.
"""
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

STATE_NAMES = ["px_m", "py_m", "yaw_rad", "speed_m_per_s",
               "yaw_rate_rad_per_s", "gyro_bias_rad_per_s"]
STATE_SCALES = np.array([1.0, 1.0, 0.1, 1.0, 0.1, 0.03])
RANK_RTOL = 1e-9
GPS_H = np.eye(6)[:2]
WHEEL_H = np.eye(6)[3:4]
GYRO_H = np.array([[0., 0., 0., 0., 1., 1.]])
MEASUREMENTS = {
    "gyro": (GYRO_H, np.array([[0.008**2]])),
    "wheel": (WHEEL_H, np.array([[0.05**2]])),
    "gps": (GPS_H, np.eye(2) * 0.6**2),
}


def measurement_schedule(timestamps_us, *, gps_available=True):
    """Available gyro100/wheel20/GPS5 Hz events, inclusive of both endpoints."""
    schedule = []
    for timestamp in timestamps_us:
        for name, period_us in (("gyro", 10_000), ("wheel", 50_000), ("gps", 200_000)):
            if timestamp % period_us == 0 and (gps_available or name != "gps"):
                schedule.append((int(timestamp), name))
    return schedule


def sensitivity_matrix(timestamps_us, transitions, schedule, *,
                       state_scales=STATE_SCALES, measurements=MEASUREMENTS):
    """Build whitened, scaled rows from process F only; no P, K, or updates.

    transitions[k] maps the preceding nominal sample to timestamps_us[k];
    transitions[0] must be identity. Each schedule entry is (timestamp, sensor).
    A sensor can appear once per timestamp. Every event must be on the grid.
    """
    timestamps = np.asarray(timestamps_us)
    transitions = np.asarray(transitions, dtype=float)
    scales = np.asarray(state_scales, dtype=float)
    if (timestamps.ndim != 1 or not len(timestamps) or
            not np.issubdtype(timestamps.dtype, np.integer) or timestamps[0] != 0 or
            np.any(np.diff(timestamps) <= 0)):
        raise ValueError("timestamps must be increasing integer microseconds starting at zero")
    if (transitions.shape != (len(timestamps), 6, 6) or not np.isfinite(transitions).all() or
            not np.array_equal(transitions[0], np.eye(6))):
        raise ValueError("transitions require finite 6x6 matrices and identity at t=0")
    if scales.shape != (6,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("state scales must be six finite positive physical magnitudes")
    whitened_h = {}
    for sensor, (h, covariance) in measurements.items():
        h, covariance = np.asarray(h, dtype=float), np.asarray(covariance, dtype=float)
        if (h.ndim != 2 or h.shape[1] != 6 or covariance.shape != (h.shape[0], h.shape[0]) or
                not np.isfinite(h).all() or not np.isfinite(covariance).all() or
                not np.allclose(covariance, covariance.T, rtol=0, atol=1e-14)):
            raise ValueError("invalid measurement Jacobian or covariance")
        try:
            whitened_h[sensor] = np.linalg.solve(np.linalg.cholesky(covariance), h)
        except np.linalg.LinAlgError as error:
            raise ValueError("measurement covariance must be positive definite") from error
    events = {}
    previous_time = -1
    seen = set()
    grid = set(timestamps.tolist())
    for timestamp, sensor in schedule:
        if (not isinstance(timestamp, (int, np.integer)) or timestamp not in grid or
                timestamp < previous_time or sensor not in whitened_h or (timestamp, sensor) in seen):
            raise ValueError("invalid, duplicate, out-of-order, or off-grid measurement event")
        events.setdefault(timestamp, []).append(sensor)
        seen.add((timestamp, sensor))
        previous_time = timestamp
    rows = []
    phi = np.eye(6)
    for index, timestamp in enumerate(timestamps):
        phi = transitions[index] @ phi
        for sensor in events.get(timestamp, []):
            rows.append((whitened_h[sensor] @ phi) * scales)
    if not rows:
        raise ValueError("observability requires at least one measurement row")
    return np.vstack(rows)


def summarize_sensitivity(matrix, *, state_scales=STATE_SCALES, rank_rtol=RANK_RTOL):
    """Return six singular values and weak directions (sign arbitrary).

    Repeated singular values define a subspace, not uniquely identifiable vectors.
    Physical directions are D*v for unit-length normalized right singular v.
    Their components carry the corresponding state units; they are not forecasts.
    Report at least three weakest directions and every numerical null direction.
    """
    matrix = np.asarray(matrix, dtype=float)
    scales = np.asarray(state_scales, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[1] != 6 or len(matrix) < 6 or
            not np.isfinite(matrix).all() or scales.shape != (6,) or
            not np.isfinite(scales).all() or np.any(scales <= 0) or
            not np.isfinite(rank_rtol) or rank_rtol <= 0):
        raise ValueError("invalid sensitivity matrix, state scales, or rank tolerance")
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    tolerance = rank_rtol * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    # Normalize signs only for readable reporting; zero-singular-value bases can rotate.
    weak_directions = []
    for index in range(5, min(2, rank - 1), -1):
        vector = vh[index].copy()
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
        weak_directions.append({"singular_value": float(singular_values[index]),
                                "normalized_direction": vector.tolist(),
                                "physical_direction": (scales * vector).tolist()})
    return {"measurement_rows": len(matrix), "state_names": STATE_NAMES,
            "state_scales": scales.tolist(), "singular_values": singular_values.tolist(),
            "rank_rtol": rank_rtol, "rank_absolute_tolerance": float(tolerance),
            "numerical_rank": rank, "nullspace_dimension": 6 - rank,
            "weak_directions": weak_directions}


def analytical_nominal(family, horizon_s):
    """Independent circular/straight nominal path; no C++ propagation is called."""
    if family not in ("stationary", "straight", "turning"):
        raise ValueError("unknown nominal trajectory family")
    if (not np.isfinite(horizon_s) or not 0 < horizon_s <= 60 or
            horizon_s * 100 != round(horizon_s * 100)):
        raise ValueError("horizon must end on the 10 ms grid in (0, 60] seconds")
    timestamps = np.arange(round(horizon_s * 1e6) + 1, step=10_000, dtype=np.int64)
    time_s = timestamps / 1e6
    states = np.zeros((len(timestamps), 6))
    speed = 0.0 if family == "stationary" else 1.5
    omega = 0.12 if family == "turning" else 0.0
    states[:, 3], states[:, 4] = speed, omega
    states[:, 2] = (omega * time_s + np.pi) % (2 * np.pi) - np.pi
    if omega == 0:
        states[:, 0] = speed * time_s
    else:
        states[:, 0] = speed / omega * np.sin(omega * time_s)
        states[:, 1] = speed / omega * (1 - np.cos(omega * time_s))
    return timestamps, states


def export_transitions(binary, output_dir, family, horizon_s):
    """Invoke an offline helper using the actual production transition Jacobian."""
    output_dir = Path(output_dir)
    timestamps, states = analytical_nominal(family, horizon_s)
    stem = f"{family}-{horizon_s:g}s"
    nominal_path = output_dir / f"{stem}-nominal.csv"
    jacobian_path = output_dir / f"{stem}-jacobians.csv"
    with nominal_path.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["timestamp_us", *STATE_NAMES])
        writer.writerows((int(time), *state) for time, state in zip(timestamps, states))
    subprocess.run([str(Path(binary).resolve()), str(nominal_path), str(jacobian_path)],
                   check=True, capture_output=True, text=True)
    with jacobian_path.open() as stream:
        reader = csv.DictReader(stream)
        expected_fields = ["timestamp_us", *(f"f{i}{j}" for i in range(6) for j in range(6))]
        if reader.fieldnames != expected_fields:
            raise ValueError("unexpected compiled Jacobian header")
        rows = list(reader)
    exported_times = np.array([int(row["timestamp_us"]) for row in rows], dtype=np.int64)
    if not np.array_equal(timestamps, exported_times):
        raise ValueError("compiled Jacobian timestamps differ from nominal grid")
    transitions = np.array([[float(row[f"f{i}{j}"]) for i in range(6) for j in range(6)]
                            for row in rows]).reshape(-1, 6, 6)
    return timestamps, transitions


def run_observability(binary, output_dir):
    """Write twelve predefined summaries and six actual-C++ Jacobian histories.

    output_dir must not exist. Full sensor availability and GPS absence are
    physical schedules; the accepted subset of a gate is deliberately not used.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for family in ("stationary", "straight", "turning"):
        for horizon_s in (2.0, 20.0):
            timestamps, transitions = export_transitions(binary, output_dir, family, horizon_s)
            for gps_available in (True, False):
                matrix = sensitivity_matrix(timestamps, transitions,
                                            measurement_schedule(timestamps, gps_available=gps_available))
                summary = summarize_sensitivity(matrix)
                summary.update(trajectory=family, horizon_s=horizon_s,
                               sensor_availability="gyro_wheel_gps" if gps_available else "gyro_wheel",
                               nominal_method="independent analytical constant speed/yaw-rate path",
                               transition_source="C++ navtrace::transition_jacobian at previous nominal sample",
                               state_scaling="physical error = diag(state_scales) * dimensionless perturbation",
                               limitations="local deterministic sensitivity; no process noise, prior, gate, or Gaussian calibration claim")
                summaries.append(summary)
    (output_dir / "observability.json").write_text(json.dumps(summaries, indent=2) + "\n")
    columns = ["trajectory", "horizon_s", "sensor_availability", "measurement_rows", "numerical_rank",
               "rank_rtol", "rank_absolute_tolerance", *(f"singular_value_{i}" for i in range(1, 7))]
    with (output_dir / "observability.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            row = {name: summary[name] for name in columns[:7]}
            row.update({f"singular_value_{i}": value for i, value in enumerate(summary["singular_values"], 1)})
            writer.writerow(row)
    return summaries
