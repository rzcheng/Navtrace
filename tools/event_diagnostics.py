"""Independent, offline audits of the C++ GPS pre/post event snapshots."""
import csv
import gzip
from pathlib import Path

import numpy as np

from tools.consistency import innovation_directions


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _close(actual, expected, description):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-10):
        raise ValueError(f"GPS audit mismatch: {description}")
    return float(np.max(np.abs(actual - expected)))


def _covariance(matrix, name, strictly_positive=False):
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"nonfinite {name}")
    tolerance = 1e-10 * max(1., float(np.max(np.abs(matrix))))
    if np.max(np.abs(matrix - matrix.T)) > tolerance:
        raise ValueError(f"asymmetric {name}")
    values = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    if values[0] < -tolerance or (strictly_positive and values[0] <= 0):
        raise ValueError(f"invalid {name}")


def audit_gps_events(path):
    """Audit every EKF GPS event and return compact, scalar-only diagnostic rows.

    Full pre/post states and covariances stay in the input CSV. The snapshots
    must bracket the GPS correction itself, after same-time gyro/wheel updates.
    This checks the logged equations, not whether the assumed model describes
    reality. Accepted high NIS is permitted for the ungated baseline. Rejected
    measurements must have high NIS and exactly unchanged GPS snapshots.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as file:
        rows = list(csv.DictReader(file))
    result = []
    last_available = last_accepted = rejection_start = None
    for line_number, row in enumerate(rows, start=2):
        try:
            time = int(row["timestamp_us"])
            if time < 0 or (last_available is not None and time <= last_available):
                raise ValueError("GPS timestamps must be nonnegative and strictly increasing")
            status = row["status"]
            if status not in ("accepted", "rejected"):
                raise ValueError("GPS audit requires EKF accepted/rejected events")
            pre = np.array([float(row[f"pre_x{i}"]) for i in range(6)])
            post = np.array([float(row[f"post_x{i}"]) for i in range(6)])
            pre_p = np.array([[float(row[f"pre_p{i}{j}"]) for j in range(6)] for i in range(6)])
            post_p = np.array([[float(row[f"post_p{i}{j}"]) for j in range(6)] for i in range(6)])
            measurement = np.array([float(row["z0"]), float(row["z1"])])
            measurement_covariance = np.array([[float(row["r00"]), float(row["r01"])],
                                               [float(row["r01"]), float(row["r11"])]])
            residual = np.array([float(row["residual_x_m"]), float(row["residual_y_m"])])
            innovation = np.array([[float(row["s00"]), float(row["s01"])],
                                   [float(row["s01"]), float(row["s11"])]])
            nis, threshold = float(row["nis"]), float(row["threshold"])
            delta = np.array([float(row[f"delta_x{i}"]) for i in range(6)])
            if (not np.all(np.isfinite(np.r_[pre, post, measurement, residual, delta, nis, threshold]))
                    or nis < 0 or threshold <= 0):
                raise ValueError("nonfinite or invalid GPS event values")
            _covariance(pre_p, "pre-update covariance")
            _covariance(post_p, "post-update covariance")
            _covariance(measurement_covariance, "measurement covariance", strictly_positive=True)
            residual_error = _close(residual, measurement - pre[:2], "pre-update residual")
            innovation_error = _close(innovation, pre_p[:2, :2] + measurement_covariance, "pre-update innovation covariance")
            directions = innovation_directions(residual, innovation)
            nis_error = _close(nis, directions["nis"], "pre-update NIS")
            _close(sum(directions["contributions"]), directions["nis"], "NIS directional sum")
            factor = np.linalg.cholesky(innovation)
            gain = np.linalg.solve(factor.T, np.linalg.solve(factor, pre_p[:, :2].T)).T
            expected = pre.copy()
            expected_p = pre_p.copy()
            if status == "accepted":
                expected += gain @ residual
                expected[2] = _wrap(expected[2])
                h = np.eye(6)[:2]
                a = np.eye(6) - gain @ h
                expected_p = a @ pre_p @ a.T + gain @ measurement_covariance @ gain.T
            else:
                if nis <= threshold:
                    raise ValueError("rejected GPS NIS does not exceed threshold")
                if not np.array_equal(pre, post) or not np.array_equal(pre_p, post_p):
                    raise ValueError("rejected GPS applied a state or covariance correction")
            mean_difference = post - expected
            mean_difference[2] = _wrap(mean_difference[2])
            mean_error = _close(mean_difference, np.zeros(6), "posterior mean")
            joseph_error = _close(post_p, expected_p, "Joseph posterior covariance")
            expected_delta = post - pre
            expected_delta[2] = _wrap(expected_delta[2])
            delta_error = _close(delta, expected_delta, "wrapped state correction")

            elapsed = {}
            for key, previous in (("since_available_s", last_available), ("since_accepted_s", last_accepted)):
                actual = None if row[key] == "" else float(row[key])
                expected_time = None if previous is None else (time - previous) / 1e6
                if (actual is None) != (expected_time is None):
                    raise ValueError(f"GPS audit mismatch: {key} availability")
                if actual is not None:
                    _close(actual, expected_time, key)
                elapsed[key] = actual
            if status == "accepted":
                rejection_start = None
                duration = 0.
                last_accepted = time
            else:
                if rejection_start is None:
                    rejection_start = time
                duration = (time - rejection_start) / 1e6
            _close(float(row["rejection_duration_s"]), duration, "consecutive rejection duration")
            last_available = time

            compact = {"timestamp_us": time, "status": status, "nis": nis, "threshold": threshold,
                       "reason": row["reason"], "residual_x_m": float(residual[0]), "residual_y_m": float(residual[1]),
                       "s00": float(innovation[0, 0]), "s01": float(innovation[0, 1]), "s11": float(innovation[1, 1]),
                       **elapsed, "rejection_duration_s": duration,
                       "residual_max_abs_error": residual_error, "innovation_max_abs_error": innovation_error,
                       "nis_abs_error": nis_error, "mean_max_abs_error": mean_error,
                       "joseph_max_abs_error": joseph_error, "delta_max_abs_error": delta_error}
            for column, name in ((0, "px"), (1, "py"), (2, "yaw"), (3, "speed"), (4, "rate"), (5, "bias")):
                compact[f"delta_{name}"] = float(delta[column])
                compact[f"pre_{name}"] = float(pre[column])
                compact[f"post_{name}"] = float(post[column])
            for i in range(2):
                compact[f"eigenvalue_{i}"] = float(directions["eigenvalues"][i])
                compact[f"projection_{i}"] = float(directions["projections"][i])
                compact[f"contribution_{i}"] = float(directions["contributions"][i])
                for j in range(2):
                    compact[f"u{j}{i}"] = float(directions["eigenvectors"][j, i])
            result.append(compact)
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
            raise ValueError(f"{path.name} GPS row {line_number}: {error}") from error
    return result
