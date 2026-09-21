"""Offline uncertainty diagnostics, with no covariance repair or estimator access.

The caller supplies independently generated truth and labels the experiment's
assumptions. Chi-square values here are Gaussian reference values, not evidence
that nonlinear, faulted, or gated estimates satisfy those assumptions.
"""
import numpy as np
from scipy.stats import chi2


def _cholesky(matrix):
    """Factor an SPD covariance, accepting only roundoff-level asymmetry.

    Averaging the two triangles below removes roundoff only; it never changes
    eigenvalues deliberately, adds jitter, or makes a singular matrix usable.
    """
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))):
        raise ValueError("covariance must be a finite square matrix")
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(matrix))))
    if np.max(np.abs(matrix - matrix.T)) > tolerance:
        raise ValueError("covariance is not symmetric")
    symmetric = (matrix + matrix.T) / 2
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance is singular or not positive definite") from error
    return symmetric, factor


def _quadratic(error, covariance):
    if not np.all(np.isfinite(error)):
        raise ValueError("error must be finite")
    _, factor = _cholesky(covariance)
    whitened = np.linalg.solve(factor, error)
    value = float(whitened @ whitened)
    if not np.isfinite(value):
        raise ValueError("quadratic statistic is not finite")
    return value


def state_statistics(estimated, truth, covariance):
    """Return row-wise six-state NEES and marginal uncertainty diagnostics.

    State order is px, py, yaw, speed, yaw_rate, gyro_bias. Position NEES uses
    the 2x2 *marginal covariance*, never a block of the full precision matrix.
    Undefined statistics are NaN with separate validity flags. Coverage is
    0.0/1.0 when defined, NaN otherwise, for the nominal 95% two-dimensional
    Gaussian position ellipse. Standard deviations are marginal quantities;
    their availability alone does not certify the full covariance as valid.
    """
    estimated, truth = np.asarray(estimated, dtype=float), np.asarray(truth, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if (estimated.ndim != 2 or estimated.shape[1] != 6 or truth.shape != estimated.shape
            or covariance.shape != (len(estimated), 6, 6)):
        raise ValueError("expected estimated/truth N x 6 and covariance N x 6 x 6")
    error = estimated - truth
    error[:, 2] = (error[:, 2] + np.pi) % (2 * np.pi) - np.pi
    size = len(estimated)
    result = {key: np.full(size, np.nan) for key in (
        "full_nees", "position_nees", "heading_std", "bias_std",
        "rate_bias_correlation", "position_covered_95")}
    result.update(heading_error=error[:, 2], bias_error=error[:, 5],
                  valid_full=np.zeros(size, dtype=bool), valid_position=np.zeros(size, dtype=bool))
    position_threshold = float(chi2.ppf(.95, 2))
    for row, (difference, matrix) in enumerate(zip(error, covariance)):
        for name, indices in (("full", slice(None)), ("position", slice(0, 2))):
            try:
                value = _quadratic(difference[indices], matrix[indices, indices])
            except ValueError:
                continue
            result[f"{name}_nees"][row] = value
            result[f"valid_{name}"][row] = True
        if result["valid_position"][row]:
            result["position_covered_95"][row] = float(result["position_nees"][row] <= position_threshold)
        for column, name in ((2, "heading_std"), (5, "bias_std")):
            variance = matrix[column, column]
            if np.isfinite(variance) and variance >= 0:
                result[name][row] = np.sqrt(variance)
        pair = matrix[4:6, 4:6]
        if (np.all(np.isfinite(pair)) and pair[0, 0] > 0 and pair[1, 1] > 0
                and abs(pair[0, 1] - pair[1, 0]) <= 1e-10 * max(1.0, float(np.max(np.abs(pair))))):
            correlation = (pair[0, 1] + pair[1, 0]) / (2 * np.sqrt(pair[0, 0] * pair[1, 1]))
            if np.isfinite(correlation) and abs(correlation) <= 1:
                result["rate_bias_correlation"][row] = correlation
    return result


def innovation_directions(residual, innovation_covariance):
    """Resolve a GPS innovation into S's orthogonal uncertainty directions.

    Eigenvalues are ascending; eigenvectors are columns. Eigenvector signs are
    arbitrary, while squared contributions and their sum are invariant.
    NIS itself is independently obtained by a Cholesky solve.
    """
    residual = np.asarray(residual, dtype=float)
    matrix = np.asarray(innovation_covariance, dtype=float)
    if residual.shape != (2,) or matrix.shape != (2, 2) or not np.all(np.isfinite(residual)):
        raise ValueError("GPS residual and innovation covariance must have shapes 2 and 2 x 2 and be finite")
    symmetric, factor = _cholesky(matrix)
    values, vectors = np.linalg.eigh(symmetric)
    if np.any(values <= 0):
        raise ValueError("innovation covariance has an unresolved nonpositive eigenvalue")
    projections = vectors.T @ residual
    contributions = projections**2 / values
    whitened = np.linalg.solve(factor, residual)
    nis = float(whitened @ whitened)
    if not np.isfinite(nis) or not np.all(np.isfinite(contributions)):
        raise ValueError("innovation statistic is not finite")
    return {"eigenvalues": values, "eigenvectors": vectors, "projections": projections,
            "contributions": contributions, "nis": nis}


def ensemble_nees(values, dimension, alpha=.05):
    """Summarize one value per independent run at ONE fixed timestamp.

    Run independence and model assumptions are the caller's responsibility.
    A time-by-run matrix is rejected rather than silently pooled. Failures
    remain counted; a survivor-only mean is descriptive, and the complete
    ensemble mean/reference comparison is undefined if any run failed.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("supply a nonempty one-dimensional independent-run sample at a fixed timestamp")
    if not isinstance(dimension, (int, np.integer)) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError("state dimension must be a positive integer")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    finite = np.isfinite(values)
    if np.any(values[finite] < 0):
        raise ValueError("NEES cannot be negative")
    count, valid = len(values), int(finite.sum())
    result = {"independent_runs": count, "finite_runs": valid, "failed_runs": count - valid,
              "dimension": int(dimension), "alpha": float(alpha),
              "mean_nees": float(values.mean()) if valid == count else None,
              "finite_mean_nees": float(values[finite].mean()) if valid else None,
              "reference_lower": None, "reference_upper": None,
              "reference_assumption": "independent Gaussian errors with correct covariance; nonlinear EKF use is approximate"}
    if valid == count:
        bounds = chi2.ppf([alpha / 2, 1 - alpha / 2], count * dimension) / count
        result.update(reference_lower=float(bounds[0]), reference_upper=float(bounds[1]))
    return result


def elapsed_settling(times_us, errors, reference_us, end_us, limit=1.0,
                     dwell_us=2_000_000, max_gap_us=10_000):
    """Find a below-limit sampled dwell using elapsed time, not sample count.

    Timestamps may be irregular. An interval above max_gap_us breaks the dwell;
    no interpolation across absent samples is evidence of recovery. This is a
    sampled criterion with a maximum tolerated observation gap, not a claim
    about unobserved continuous-time error. All endpoints are inclusive.
    """
    times, errors = np.asarray(times_us), np.asarray(errors, dtype=float)
    integer_arguments = (reference_us, end_us, dwell_us, max_gap_us)
    if (times.ndim != 1 or times.dtype.kind not in "iu" or len(times) == 0
            or errors.shape != times.shape or np.any(times[1:] <= times[:-1])
            or np.any(times < 0) or np.any(times > np.iinfo(np.int64).max)
            or not np.all(np.isfinite(errors)) or np.any(errors < 0)):
        raise ValueError("settling requires increasing integer timestamps and finite nonnegative errors")
    if (any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) for value in integer_arguments)
            or dwell_us <= 0 or max_gap_us <= 0 or reference_us < 0 or end_us < reference_us
            or not np.isfinite(limit) or limit <= 0):
        raise ValueError("invalid settling window, limit, dwell, or maximum gap")
    if times[0] > reference_us or times[-1] < end_us:
        raise ValueError("observations must span the requested follow-up window")
    times = times.astype(np.int64)
    selected = (times >= reference_us) & (times <= end_us)
    times, errors = times[selected], errors[selected]
    gaps = np.diff(np.r_[reference_us, times, end_us])
    result = {"settling_status": "not_observed", "settling_start_delay_s": None,
              "settling_confirmation_delay_s": None, "followup_s": (end_us - reference_us) / 1e6,
              "gaps_exceeding_limit": int(np.sum(gaps > max_gap_us)),
              "maximum_observed_gap_us": int(np.max(gaps)) if len(gaps) else 0}
    if end_us - reference_us < dwell_us:
        result["settling_status"] = "insufficient_followup"
        return result
    first = previous = None
    for time, error in zip(times, errors):
        time = int(time)
        if previous is not None and time - previous > max_gap_us:
            first = None
        if error > limit:
            first = None
        elif first is None:
            first = time
        if first is not None and time - first >= dwell_us:
            result.update(settling_status="within_limit_at_return" if first == reference_us else "recovered",
                          settling_start_delay_s=(first - reference_us) / 1e6,
                          settling_confirmation_delay_s=(time - reference_us) / 1e6)
            return result
        previous = time
    return result
