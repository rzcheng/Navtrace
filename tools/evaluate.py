"""Evaluate saved C++ outputs at exact common timestamps, without interpolation."""
import csv
import json
from pathlib import Path

import numpy as np


def wrapped_angle_difference(estimated, true):
    return (np.asarray(estimated) - np.asarray(true) + np.pi) % (2 * np.pi) - np.pi


def compute_metrics(times_us, estimated, truth, outage_start_us, outage_end_us):
    times = np.asarray(times_us, dtype=np.int64)
    estimated, truth = np.asarray(estimated, dtype=float), np.asarray(truth, dtype=float)
    if len(times) == 0 or estimated.shape != (len(times), 3) or truth.shape != estimated.shape:
        raise ValueError("expected matching nonempty N x 3 pose arrays")
    if not np.all(np.isfinite(estimated)) or not np.all(np.isfinite(truth)) or np.any(np.diff(times) <= 0):
        raise ValueError("nonfinite poses or non-increasing evaluation times")
    position_squared = np.sum((estimated[:, :2] - truth[:, :2])**2, axis=1)
    heading = wrapped_angle_difference(estimated[:, 2], truth[:, 2])
    outage = (times >= outage_start_us) & (times < outage_end_us)
    indices = np.flatnonzero(outage)
    if not len(indices):
        raise ValueError("evaluation grid contains no outage samples")
    end = indices[-1]  # Last available instant BEFORE the returning GPS update.
    return {
        "samples": len(times), "outage_samples": int(outage.sum()),
        "position_rmse_m": float(np.sqrt(position_squared.mean())),
        "outage_position_rmse_m": float(np.sqrt(position_squared[outage].mean())),
        "outage_end_position_error_m": float(np.sqrt(position_squared[end])),
        "outage_end_evaluation_timestamp_us": int(times[end]),
        "final_position_error_m": float(np.sqrt(position_squared[-1])),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading**2))),
        "outage_heading_rmse_rad": float(np.sqrt(np.mean(heading[outage]**2))),
        "final_wrapped_heading_error_rad": float(heading[-1]),
    }


def load_pose(path, keep_last_per_timestamp=False):
    with Path(path).open(newline="") as file:
        records = list(csv.DictReader(file))
    selected = {}
    previous = -1
    for row in records:
        time = int(row["timestamp_us"])
        if time < previous or (time == previous and not keep_last_per_timestamp):
            raise ValueError("non-chronological or duplicate pose rows")
        previous = time
        selected[time] = [float(row[key]) for key in ("px_m", "py_m", "yaw_rad")]
    return selected


def aligned_poses(truth_path, estimate_path):
    truth = load_pose(truth_path)
    estimates = load_pose(estimate_path, keep_last_per_timestamp=True)
    times = np.array(list(truth), dtype=np.int64)
    if any(t not in estimates for t in times):
        raise ValueError("estimate is missing a required truth-grid timestamp")
    return times, np.array([estimates[t] for t in times]), np.array(list(truth.values()))


def evaluate_run(output, methods=("dr", "ekf", "gated")):
    output = Path(output)
    faults = json.loads((output / "inputs/faults.json").read_text())["dropout"]
    metrics = []
    errors = {}
    for scenario in ("nominal", "dropout"):
        for method in methods:
            stem = f"{scenario}-{method}"
            times, estimated, truth = aligned_poses(output / "inputs/truth.csv", output / f"{stem}.csv")
            result = compute_metrics(times, estimated, truth, faults["start_us"], faults["end_us_exclusive"])
            result.update(scenario=scenario, method=method)
            with (output / f"{stem}-gps.csv").open(newline="") as file:
                diagnostics = list(csv.DictReader(file))
            for status in ("accepted", "rejected", "ignored"):
                result[f"gps_{status}"] = sum(r["status"] == status for r in diagnostics)
                result[f"window_gps_{status}"] = sum(r["status"] == status and faults["start_us"] <= int(r["timestamp_us"]) < faults["end_us_exclusive"] for r in diagnostics)
            errors[stem] = (times / 1e6, np.linalg.norm(estimated[:, :2] - truth[:, :2], axis=1),
                            wrapped_angle_difference(estimated[:, 2], truth[:, 2]))
            metrics.append(result)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    return metrics, errors
