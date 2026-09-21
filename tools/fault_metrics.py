"""Independent fault/settling metrics; annotations never enter the C++ engine."""
import csv
from itertools import zip_longest
from pathlib import Path

import numpy as np

from tools.evaluate import aligned_poses, compute_metrics, wrapped_angle_difference


def position_settling(times_us, errors_m, window, limit_m, dwell_us, step_us):
    times, errors = np.asarray(times_us), np.asarray(errors_m, dtype=float)
    if (times.ndim != 1 or times.dtype.kind not in "iu" or errors.shape != times.shape or len(times) == 0
            or not np.all(np.isfinite(errors)) or np.any(errors < 0)
            or step_us <= 0 or np.any(np.diff(times) != step_us)):
        raise ValueError("settling requires a complete uniform integer-time grid and finite nonnegative errors")
    if not np.isfinite(limit_m) or limit_m <= 0 or dwell_us <= 0 or dwell_us % step_us:
        raise ValueError("invalid settling limit/dwell")
    start, end = window["reference_us"], window["observation_end_us"]
    if end < start or start not in times or end not in times:
        raise ValueError("settling window endpoints must be present in the evaluation grid")
    selected = (times >= start) & (times <= end)
    times, errors = times[selected], errors[selected]
    result = {"settling_status": "not_observed", "settling_start_delay_s": None,
              "settling_confirmation_delay_s": None, "followup_s": (end - start) / 1e6}
    if end - start < dwell_us:
        result["settling_status"] = "insufficient_followup"
        return result
    required_samples = dwell_us // step_us + 1  # Include BOTH endpoints of the dwell.
    consecutive = 0
    for i, error in enumerate(errors):
        consecutive = consecutive + 1 if error <= limit_m else 0
        if consecutive >= required_samples:
            first = int(times[i - required_samples + 1])
            result.update(settling_status="within_limit_at_return" if first == start else "recovered",
                          settling_start_delay_s=(first - start) / 1e6,
                          settling_confirmation_delay_s=(int(times[i]) - start) / 1e6)
            break
    return result


def gps_decisions(diagnostics, annotation, method):
    expected = annotation["gps_timestamps_us"]
    corrupted = annotation["corrupted_gps_timestamps_us"]
    actual = [int(row["timestamp_us"]) for row in diagnostics]
    if (actual != expected or len(set(actual)) != len(actual) or len(set(corrupted)) != len(corrupted)
            or not set(corrupted).issubset(expected)):
        raise ValueError("GPS diagnostics/annotations do not match actual sensor timestamps exactly")
    corrupted = set(corrupted)
    result = {f"{kind}_gps_{status}": 0 for kind in ("clean", "fault") for status in ("accepted", "rejected", "ignored")}
    for row, time in zip(diagnostics, actual):
        status = row["status"]
        if status not in (("ignored",) if method == "dr" else ("accepted", "rejected")):
            raise ValueError("invalid GPS status for method")
        if method != "dr":
            nis, threshold = float(row["nis"]), float(row["threshold"])
            if not np.isfinite(nis) or nis < 0 or not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("invalid NIS/threshold diagnostics")
            if method == "ekf" and status != "accepted":
                raise ValueError("ungated EKF unexpectedly rejected valid GPS")
            if method == "gated" and (nis > threshold) != (status == "rejected"):
                raise ValueError("GPS gate decision does not match squared-distance threshold")
        kind = "fault" if time in corrupted else "clean"
        result[f"{kind}_gps_{status}"] += 1
    for kind in ("clean", "fault"):
        denominator = result[f"{kind}_gps_accepted"] + result[f"{kind}_gps_rejected"]
        result[f"{kind}_gps_considered"] = denominator
        result[f"{kind}_gps_rejection_fraction"] = result[f"{kind}_gps_rejected"] / denominator if denominator else None
    return result


def gps_return(diagnostics, annotation, window, method):
    corrupted = set(annotation["corrupted_gps_timestamps_us"])
    rows = [r for r in diagnostics if window["reference_us"] <= int(r["timestamp_us"]) <= window["observation_end_us"]
            and int(r["timestamp_us"]) not in corrupted]
    accepted = [] if method == "dr" else [int(r["timestamp_us"]) for r in rows if r["status"] == "accepted"]
    return {"gps_return_status": "not_applicable" if method == "dr" else "accepted" if accepted else "not_observed",
            "first_clean_gps_accepted_delay_s": (accepted[0] - window["reference_us"]) / 1e6 if accepted else None,
            "return_window_clean_gps_seen": len(rows),
            "return_window_clean_gps_rejected": sum(r["status"] == "rejected" for r in rows)}


def audit_event_outputs(sensor_path, estimate_path, method):
    """Check every event, including equal timestamps, against compiled outputs."""
    minimum, asymmetry, count = float("inf"), 0.0, 0
    with Path(sensor_path).open(newline="") as source, Path(estimate_path).open(newline="") as estimates:
        for event, estimate in zip_longest(csv.DictReader(source), csv.DictReader(estimates)):
            if event is None or estimate is None or any(event[k] != estimate[k] for k in ("timestamp_us", "sensor")):
                raise ValueError("estimate must contain exactly one ordered row per input event")
            pose = [float(estimate[k]) for k in ("px_m", "py_m", "yaw_rad", "speed_m_per_s", "yaw_rate_rad_per_s", "gyro_bias_rad_per_s")]
            if not np.all(np.isfinite(pose)):
                raise ValueError("nonfinite estimated state")
            if method != "dr":
                eigenvalue, symmetry = float(estimate["covariance_min_eigenvalue"]), float(estimate["covariance_symmetry_error"])
                if not np.isfinite(eigenvalue) or not np.isfinite(symmetry) or eigenvalue < -1e-10 or not 0 <= symmetry <= 1e-10:
                    raise ValueError("invalid covariance diagnostics")
                minimum, asymmetry = min(minimum, eigenvalue), max(asymmetry, symmetry)
            count += 1
    if count == 0:
        raise ValueError("empty replay")
    return {"sensor_events": count, "minimum_covariance_eigenvalue": minimum if method != "dr" else None,
            "maximum_covariance_symmetry_error": asymmetry if method != "dr" else None}


def evaluate_case(seed_dir, scenario, method, protocol, seed, annotation):
    seed_dir = Path(seed_dir)
    estimate_path = seed_dir / f"{scenario['id']}-{method}.csv"
    times, estimated, truth = aligned_poses(seed_dir / "inputs/base/truth.csv", estimate_path)
    expected_times = np.arange(0, round(protocol["generator_settings"]["duration_s"] * 1e6) + 1,
                               1_000_000 // protocol["generator_settings"]["gyro_hz"], dtype=np.int64)
    if not np.array_equal(times, expected_times):
        raise ValueError("truth grid does not match the frozen protocol")
    base = compute_metrics(times, estimated, truth, *protocol["reference_window_us"])
    metrics = {key.replace("outage_", "reference_window_"): value for key, value in base.items()}
    metrics.update(seed=seed, scenario=scenario["id"], method=method)
    position_error = np.linalg.norm(estimated[:, :2] - truth[:, :2], axis=1)
    metrics["maximum_position_error_m"] = float(position_error.max())
    metrics.update(gps_outage_position_rmse_m=None, gps_outage_end_position_error_m=None,
                   gps_outage_end_evaluation_timestamp_us=None)
    if scenario["outage_us"] is not None:
        start, end = scenario["outage_us"]
        mask = (times >= start) & (times < end)
        metrics.update(gps_outage_position_rmse_m=float(np.sqrt(np.mean(position_error[mask] ** 2))),
                       gps_outage_end_position_error_m=float(position_error[mask][-1]),
                       gps_outage_end_evaluation_timestamp_us=int(times[mask][-1]))
    metrics.update(audit_event_outputs(seed_dir / f"inputs/{scenario['id']}.csv", estimate_path, method))
    with (seed_dir / f"{scenario['id']}-{method}-gps.csv").open(newline="") as file:
        diagnostics = list(csv.DictReader(file))
    metrics.update(gps_decisions(diagnostics, annotation, method))
    episodes = []
    for window in annotation["recovery_windows"]:
        episode = dict(seed=seed, scenario=scenario["id"], method=method, episode=window["id"],
                       reference_us=window["reference_us"], observation_end_us=window["observation_end_us"])
        episode.update(position_settling(times, position_error, window, protocol["settling"]["position_limit_m"],
                                        protocol["settling"]["dwell_us"], 1_000_000 // protocol["generator_settings"]["gyro_hz"]))
        episode.update(gps_return(diagnostics, annotation, window, method))
        episodes.append(episode)
    series = {"time_s": times / 1e6, "position_error_m": position_error,
              "heading_error_rad": wrapped_angle_difference(estimated[:, 2], truth[:, 2]), "gps": diagnostics}
    return metrics, episodes, series


def summarize(metrics, episodes, protocol, expected_seeds):
    expected = {(seed, s["id"], method) for seed in expected_seeds for s in protocol["scenarios"] for method in ("dr", "ekf", "gated")}
    actual = [(r["seed"], r["scenario"], r["method"]) for r in metrics]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("incomplete or duplicate seed/scenario/method matrix")
    expected_episodes = {(seed, s["id"], method, w["id"]) for seed in expected_seeds for s in protocol["scenarios"]
                         for method in ("dr", "ekf", "gated") for w in s["recovery_windows"]}
    actual_episodes = [(r["seed"], r["scenario"], r["method"], r["episode"]) for r in episodes]
    if len(actual_episodes) != len(set(actual_episodes)) or set(actual_episodes) != expected_episodes:
        raise ValueError("incomplete or duplicate recovery episodes")
    summaries, paired = [], []
    for scenario in protocol["scenarios"]:
        name = scenario["id"]
        for method in ("dr", "ekf", "gated"):
            rows = [r for r in metrics if r["scenario"] == name and r["method"] == method]
            if not rows or len({r["seed"] for r in rows}) != len(rows):
                raise ValueError("missing or duplicate seed/method results")
            values = [r["position_rmse_m"] for r in rows]
            result = {"scenario": name, "method": method, "seeds": len(rows),
                      "position_rmse_median_m": float(np.median(values)),
                      "position_rmse_min_m": min(values), "position_rmse_max_m": max(values)}
            for kind in ("clean", "fault"):
                for status in ("accepted", "rejected", "ignored"):
                    key = f"{kind}_gps_{status}"
                    result[key] = sum(r[key] for r in rows)
                considered = result[f"{kind}_gps_accepted"] + result[f"{kind}_gps_rejected"]
                result[f"{kind}_gps_considered"] = considered
                result[f"{kind}_gps_rejection_fraction"] = result[f"{kind}_gps_rejected"] / considered if considered else None
            matching = [r for r in episodes if r["scenario"] == name and r["method"] == method]
            for status in ("within_limit_at_return", "recovered", "not_observed", "insufficient_followup"):
                result[f"settling_{status}"] = sum(r["settling_status"] == status for r in matching)
            summaries.append(result)
        seeds = sorted({r["seed"] for r in metrics if r["scenario"] == name})
        for seed in seeds:
            matching = {r["method"]: r for r in metrics if r["scenario"] == name and r["seed"] == seed}
            paired.append({"seed": seed, "scenario": name,
                           "gated_minus_basic_rmse_m": matching["gated"]["position_rmse_m"] - matching["ekf"]["position_rmse_m"]})
    return summaries, paired
