"""Compare independent position errors with the covariances that described them."""
import numpy as np

from tools.consistency import innovation_directions


def position_geometry(errors, covariances):
    """One sample per independent run at one time, in a shared world frame.

    Keep the mean error, centered covariance and second moment separate. The
    aggregate ratios use mean P; they must not replace the per-run NEES mean.
    Missing/invalid samples keep their place and make ensemble geometry null.
    """
    errors = np.asarray(errors, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if (errors.ndim != 2 or errors.shape[1] != 2 or len(errors) == 0
            or covariances.shape != (len(errors), 2, 2)):
        raise ValueError('expected errors N x 2 and covariances N x 2 x 2')
    valid = []
    for error, covariance in zip(errors, covariances):
        try:
            innovation_directions(error, covariance)
            valid.append(True)
        except ValueError:
            valid.append(False)
    result = dict(planned_runs=len(errors), valid_runs=sum(valid),
                  invalid_runs=len(errors)-sum(valid), mean_error=None,
                  centered_sample_covariance=None, error_second_moment=None,
                  mean_reported_covariance=None, aggregate_ratio_eigenvalues=None)
    if not all(valid):
        return result
    mean_error = errors.mean(axis=0)
    centered = errors - mean_error
    second_moment = errors.T @ errors / len(errors)
    mean_p = covariances.mean(axis=0)
    factor = np.linalg.cholesky((mean_p + mean_p.T) / 2)
    left = np.linalg.solve(factor, second_moment)
    whitened = np.linalg.solve(factor, left.T).T
    result.update(mean_error=mean_error.tolist(),
                  centered_sample_covariance=(centered.T @ centered / (len(errors)-1)).tolist()
                  if len(errors) > 1 else None,
                  error_second_moment=second_moment.tolist(),
                  mean_reported_covariance=mean_p.tolist(),
                  aggregate_ratio_eigenvalues=np.linalg.eigvalsh((whitened + whitened.T)/2).tolist())
    return result
