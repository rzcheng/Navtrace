"""Independent conditional path sampling for NavTrace's GPS-free model.

Only the four linear latent states are filtered. Position is integrated along
sampled paths; it is never corrected by a Gaussian approximation here.
"""
from dataclasses import dataclass
import math

import numpy as np


def multiply_samples(matrix, samples):
    # These matrices have only 2–4 rows. Direct contraction also avoids the
    # local BLAS path that warns on finite, modest matrix products.
    return np.einsum('ij,jn->in', matrix, samples, optimize=False)


def factor_covariance(matrix, size):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError('covariance must have the expected shape and finite values')
    if np.max(np.abs(matrix-matrix.T)) > 1e-10*max(1., float(np.abs(matrix).max())):
        raise ValueError('covariance is asymmetric')
    matrix = (matrix+matrix.T)/2
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError('covariance must be positive definite; no repair is applied') from error
    return matrix, factor


def normal_log_density(residual, variance):
    if not np.isfinite(residual) or not np.isfinite(variance) or variance <= 0:
        raise ValueError('invalid Gaussian residual or variance')
    with np.errstate(over='ignore'):
        value = -.5*(math.log(2*math.pi)+math.log(variance)+(np.float64(residual)/np.sqrt(variance))**2)
    if not np.isfinite(value):
        raise ValueError('nonfinite Gaussian log likelihood')
    return float(value)


def observe(mean, covariance, sensor, value, variance):
    h = {'wheel': np.array([0., 1., 0., 0.]), 'gyro': np.array([0., 0., 1., 1.])}.get(sensor)
    mean = np.asarray(mean, dtype=float)
    if h is None or mean.shape != (4,) or not np.isfinite(mean).all():
        raise ValueError('expected a finite four-state mean and wheel/gyro sensor')
    covariance, _ = factor_covariance(covariance, 4)
    normal_log_density(0., variance)
    residual = value-h@mean
    innovation_variance = float(h@covariance@h+variance)
    likelihood = normal_log_density(residual, innovation_variance)
    gain = covariance@h/innovation_variance
    updated = mean+gain*residual
    a = np.eye(4)-np.outer(gain, h)
    posterior, _ = factor_covariance(a@covariance@a.T+variance*np.outer(gain, gain), 4)
    if not np.isfinite(updated).all():
        raise ValueError('nonfinite latent posterior')
    return updated, posterior, likelihood


@dataclass(frozen=True)
class Observation:
    timestamp_us: int
    sensor: str
    value: float
    variance: float


@dataclass
class History:
    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    means: list
    covariances: list
    steps: list
    diffusion: np.ndarray
    indices: dict
    observation_counts: dict
    log_likelihoods: dict
    backward: list


def backward_parameters(mean, covariance, step, diffusion):
    """Condition three process-noise coordinates, avoiding singular 4D Q."""
    a = np.eye(4); a[0, 2] = step
    b = np.zeros((4, 3)); b[1:, :] = np.diag(np.sqrt(diffusion*step))
    _, factor = factor_covariance(a@covariance@a.T, 4)
    solved_b = np.linalg.solve(factor.T, np.linalg.solve(factor, b))
    _, precision_factor = factor_covariance(np.eye(3)+b.T@solved_b, 3)
    noise_gain = np.linalg.solve(precision_factor.T, np.linalg.solve(precision_factor, solved_b.T))
    noise_factor = np.linalg.solve(precision_factor.T, np.eye(3))
    return a@mean, noise_gain, noise_factor, np.sqrt(diffusion*step)


def build_history(mean, covariance, checkpoint_us, events, end_us, diffusion,
                  max_step_s=.01, max_gap_s=2.):
    mean = np.asarray(mean, dtype=float)
    covariance, _ = factor_covariance(covariance, 6)
    diffusion = np.asarray(diffusion, dtype=float)
    if mean.shape != (6,) or not np.isfinite(mean).all():
        raise ValueError('expected finite six-state checkpoint mean')
    if diffusion.shape != (3,) or not np.isfinite(diffusion).all() or np.any(diffusion < 0):
        raise ValueError('invalid diffusion intensities')
    if (type(checkpoint_us) is not int or type(end_us) is not int or not 0 <= checkpoint_us <= end_us
            or not np.isfinite(max_step_s) or not 1e-5 <= max_step_s <= .1
            or not np.isfinite(max_gap_s) or not max_step_s <= max_gap_s <= 60):
        raise ValueError('invalid timestamps or propagation bounds')
    previous = (checkpoint_us, 2)
    for event in events:
        if type(event.timestamp_us) is not int or event.sensor not in ['gyro', 'wheel']:
            raise ValueError('invalid observation timestamp or sensor')
        key = (event.timestamp_us, ['gyro', 'wheel'].index(event.sensor))
        if key <= previous or event.timestamp_us > end_us:
            raise ValueError('duplicate, out-of-order, or out-of-window observation')
        normal_log_density(event.value, event.variance)
        previous = key
    history = History(mean.copy(), covariance.copy(), [mean[2:].copy()], [covariance[2:, 2:].copy()],
                      [], diffusion.copy(), {checkpoint_us: 0}, {checkpoint_us: 0}, {checkpoint_us: 0.}, [])
    timestamp, count, likelihood = checkpoint_us, 0, 0.
    def advance(target):
        nonlocal timestamp
        dt = (target-timestamp)*1e-6
        if dt > max_gap_s:
            raise ValueError('observation gap exceeds configured maximum')
        steps = math.ceil(dt/max_step_s)
        if steps:
            step = dt/steps
            a = np.eye(4); a[0, 2] = step
            q = np.diag(np.r_[0., diffusion*step])
            for _ in range(steps):
                next_mean = a@history.means[-1]
                next_covariance, _ = factor_covariance(a@history.covariances[-1]@a.T+q, 4)
                history.means.append(next_mean); history.covariances.append(next_covariance)
                history.steps.append(step)
        timestamp = target
    for event in events:
        advance(event.timestamp_us)
        m, c, log_density = observe(history.means[-1], history.covariances[-1],
                                    event.sensor, event.value, event.variance)
        history.means[-1], history.covariances[-1] = m, c
        count += 1; likelihood += log_density
        history.indices[timestamp] = len(history.means)-1
        history.observation_counts[timestamp] = count
        history.log_likelihoods[timestamp] = likelihood
    advance(end_us)
    history.indices[end_us] = len(history.means)-1
    history.observation_counts[end_us] = count; history.log_likelihoods[end_us] = likelihood
    history.backward = [backward_parameters(m, c, h, diffusion)
                        for m, c, h in zip(history.means, history.covariances, history.steps)]
    return history


def sample_endpoint(history, timestamp_us, count, reference_seed, recording_seed=0, check_budget=None):
    """Independent paths conditioned only through timestamp_us, returned N x 6."""
    if timestamp_us not in history.indices or type(count) is not int or count < 2:
        raise ValueError('unknown report time or invalid sample count')
    if any(type(s) is not int or s < 0 for s in [reference_seed, recording_seed]):
        raise ValueError('random seeds must be nonnegative integers')
    if check_budget:
        check_budget()
    streams = np.random.SeedSequence([reference_seed, recording_seed, count, timestamp_us]).spawn(3)
    terminal_rng, backward_rng, position_rng = [np.random.Generator(np.random.PCG64(s)) for s in streams]
    index = history.indices[timestamp_us]
    _, factor = factor_covariance(history.covariances[index], 4)
    current = history.means[index][:, None]+multiply_samples(factor, terminal_rng.standard_normal((4, count)))
    terminal = current.copy()
    displacement = np.zeros((2, count))
    for k in range(index-1, -1, -1):
        if check_budget and k % 128 == 0:
            check_budget()
        step = history.steps[k]
        predicted, gain, noise_factor, noise_std = history.backward[k]
        noise = (multiply_samples(gain, current-predicted[:, None])
                 +multiply_samples(noise_factor, backward_rng.standard_normal((3, count))))
        previous = current.copy()
        previous[1:] -= noise_std[:, None]*noise
        previous[0] -= step*previous[2]
        # Euler position uses the PRE-transition speed and angle.
        displacement[0] += step*previous[1]*np.cos(previous[0])
        displacement[1] += step*previous[1]*np.sin(previous[0])
        current = previous
    initial = history.initial_covariance
    _, factor = factor_covariance(initial[2:, 2:], 4)
    position_gain = np.linalg.solve(factor.T, np.linalg.solve(factor, initial[2:, :2])).T
    conditional = initial[:2, :2]-position_gain@initial[2:, :2]
    _, position_factor = factor_covariance(conditional, 2)
    position = (history.initial_mean[:2, None]+multiply_samples(position_gain, current-history.initial_mean[2:, None])
                +multiply_samples(position_factor, position_rng.standard_normal((2, count)))+displacement)
    samples = np.vstack([position, terminal]).T
    samples[:, 2] = (samples[:, 2]+np.pi) % (2*np.pi)-np.pi
    if not np.isfinite(samples).all():
        raise ValueError('nonfinite reference samples')
    return samples


def circular_summary(angles):
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError('expected finite angles')
    cosine, sine = float(np.cos(angles).mean()), float(np.sin(angles).mean())
    resultant = math.hypot(cosine, sine)
    # A direction is undefined for near-uniform or balanced opposing samples.
    return dict(mean_rad=math.atan2(sine, cosine) if resultant > 1e-8 else None,
                resultant=resultant, mean_defined=resultant > 1e-8)
