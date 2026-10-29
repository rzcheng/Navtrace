"""Offline Gaussian moment recursion for the fixed GPS-free Euler model.

Prediction has exact moments for a Gaussian input. Replacing each nonlinear
joint law by a Gaussian before the next prediction or observation is an
approximation, including when the linear latent marginal remains exact.
Yaw stays on a real, unwrapped lift. This module accepts no reference samples.
"""
import math

import numpy as np


def _finite_scalar(value, name):
    if np.ndim(value) != 0:
        raise ValueError(f'{name} must be a finite scalar')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite scalar') from error
    if not math.isfinite(value):
        raise ValueError(f'{name} must be a finite scalar')
    return value


def _checked_covariance(covariance):
    if covariance.shape != (6, 6) or not np.isfinite(covariance).all():
        raise ValueError('expected a finite 6x6 covariance')
    tolerance = 1e-10*max(1., float(np.abs(covariance).max()))
    try:
        with np.errstate(over='raise', invalid='raise'):
            if np.abs(covariance-covariance.T).max() > tolerance:
                raise ValueError('covariance is asymmetric')
            # Only roundoff symmetrization; never add noise or clip eigenvalues.
            symmetric = .5*covariance+.5*covariance.T
            if np.linalg.eigvalsh(symmetric).min() < -tolerance:
                raise ValueError('covariance is not positive semidefinite')
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError('covariance validation failed') from error
    return symmetric


def _state(mean, covariance):
    mean = np.array(mean, dtype=float, copy=True)
    covariance = np.array(covariance, dtype=float, copy=True)
    if mean.shape != (6,) or not np.isfinite(mean).all():
        raise ValueError('expected a finite six-state mean')
    return mean, _checked_covariance(covariance)


def _configuration(diffusion, max_step_s, max_gap_s):
    diffusion = np.array(diffusion, dtype=float, copy=True)
    if diffusion.shape != (3,) or not np.isfinite(diffusion).all() or np.any(diffusion < 0):
        raise ValueError('diffusion must contain three finite nonnegative intensities')
    max_step_s = _finite_scalar(max_step_s, 'max_step_s')
    max_gap_s = _finite_scalar(max_gap_s, 'max_gap_s')
    if not 1e-5 <= max_step_s <= .1 or not max_step_s <= max_gap_s <= 60:
        raise ValueError('invalid propagation bounds')
    return diffusion, max_step_s, max_gap_s


def _displacement_moments(mean, covariance):
    """Moments of g = speed * [cos(yaw), sin(yaw)], before multiplying by h."""
    angle, speed = mean[2:4]
    yaw_variance, speed_variance = covariance[2, 2], covariance[3, 3]
    correlation = covariance[3, 2]
    decay = np.exp(-yaw_variance)
    decay_twice = np.exp(-2*yaw_variance)
    decay_half = np.exp(-.5*yaw_variance)
    one_minus_decay = -np.expm1(-yaw_variance)
    one_minus_decay_twice = -np.expm1(-2*yaw_variance)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    local_mean = decay_half*np.array([speed, correlation])
    local_cross = decay_half*np.column_stack([
        covariance[:, 3]-correlation*covariance[:, 2],
        speed*covariance[:, 2]])
    # Centered covariance in mean-heading axes. These identities avoid
    # subtracting raw O(speed^2) moments to recover O(yaw_variance^2).
    mixed_factor = 2*decay_twice-decay
    local_covariance = np.array([
        [.5*speed_variance*(1+decay_twice)
         +.5*speed**2*one_minus_decay**2-2*correlation**2*decay_twice,
         speed*correlation*mixed_factor],
        [speed*correlation*mixed_factor,
         .5*(speed**2+speed_variance)*one_minus_decay_twice
         +correlation**2*mixed_factor]])
    return (rotation@local_mean, local_cross@rotation.T,
            rotation@local_covariance@rotation.T)


def _predict(mean, covariance, dt_s, diffusion, max_step_s, max_gap_s, check_budget=None):
    dt_s = _finite_scalar(dt_s, 'dt_s')
    if dt_s < 0 or dt_s > max_gap_s:
        raise ValueError('dt_s must be nonnegative and no greater than max_gap_s')
    if dt_s == 0:
        return mean, covariance
    steps = math.ceil(dt_s/max_step_s)
    step = dt_s/steps
    a = np.eye(6)
    a[2, 4] = step
    b = np.zeros((6, 2))
    b[:2] = step*np.eye(2)
    noise = np.diag(np.r_[np.zeros(3), diffusion*step])
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for _ in range(steps):
                if check_budget is not None:
                    check_budget()
                increment, cross, spread = _displacement_moments(mean, covariance)
                # All moments use the old state; a@cross includes next-yaw/rate
                # correlation. Independent diffusion enters after the increment.
                next_mean = a@mean+b@increment
                propagated_cross = a@cross@b.T
                next_covariance = (a@covariance@a.T+propagated_cross
                                   +propagated_cross.T+b@spread@b.T+noise)
                if not np.isfinite(next_mean).all():
                    raise ValueError('nonfinite predicted mean')
                covariance = _checked_covariance(next_covariance)
                mean = next_mean
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('nonfinite prediction arithmetic') from error
    return mean, covariance


def predict(mean, covariance, dt_s, diffusion, max_step_s=.01, max_gap_s=2.):
    """Return copied mean/P after ceil(dt/max_step_s) equal Euler steps."""
    mean, covariance = _state(mean, covariance)
    diffusion, max_step_s, max_gap_s = _configuration(diffusion, max_step_s, max_gap_s)
    return _predict(mean, covariance, dt_s, diffusion, max_step_s, max_gap_s)


def observe(mean, covariance, sensor, value, variance):
    """Condition the projected joint Gaussian on one wheel or gyro scalar."""
    mean, covariance = _state(mean, covariance)
    if sensor not in ('wheel', 'gyro'):
        raise ValueError('only wheel and gyro observations are supported')
    value = _finite_scalar(value, 'observation value')
    variance = _finite_scalar(variance, 'observation variance')
    if variance <= 0:
        raise ValueError('observation variance must be positive')
    h = np.zeros(6)
    if sensor == 'wheel':
        h[3] = 1
    else:
        h[4:] = 1
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            residual = value-h@mean
            innovation_variance = float(h@covariance@h+variance)
            if not np.isfinite(innovation_variance) or innovation_variance <= 0:
                raise ValueError('innovation variance must be finite and positive')
            gain = covariance@h/innovation_variance
            updated = mean+gain*residual
            a = np.eye(6)-np.outer(gain, h)
            posterior = _checked_covariance(a@covariance@a.T+variance*np.outer(gain, gain))
            if not np.isfinite(updated).all():
                raise ValueError('nonfinite posterior mean')
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('nonfinite observation arithmetic') from error
    return updated, posterior


def run_candidate(mean, covariance, checkpoint_us, events, report_times_us,
                  diffusion, max_step_s=.01, max_gap_s=2., check_budget=None):
    """Causal replay with reports after every same-time gyro/wheel observation.

    Events must be strictly after the checkpoint and at or before the final
    report. A report is the checkpoint or an actual observation timestamp;
    requesting a report never adds a propagation step. Events are never sorted.
    """
    mean, covariance = _state(mean, covariance)
    diffusion, max_step_s, max_gap_s = _configuration(diffusion, max_step_s, max_gap_s)
    reports = list(report_times_us)
    if (type(checkpoint_us) is not int or checkpoint_us < 0 or not reports
            or any(type(t) is not int or t < checkpoint_us for t in reports)
            or any(left >= right for left, right in zip(reports, reports[1:]))):
        raise ValueError('invalid checkpoint or report timestamps/order')
    if check_budget is not None and not callable(check_budget):
        raise ValueError('check_budget must be callable')
    requested = set(reports)
    snapshots = {}
    count = 0

    def save(timestamp):
        snapshots[timestamp] = {'mean': mean.copy(), 'covariance': covariance.copy(),
                                'observations': count}

    if checkpoint_us in requested:
        save(checkpoint_us)
    previous = (checkpoint_us, 2)
    timestamp = checkpoint_us
    for event in events:
        if check_budget is not None:
            check_budget()
        if type(event.timestamp_us) is not int or event.sensor not in ('gyro', 'wheel'):
            raise ValueError('invalid observation timestamp or sensor')
        key = (event.timestamp_us, ('gyro', 'wheel').index(event.sensor))
        if key <= previous or event.timestamp_us > reports[-1]:
            raise ValueError('duplicate, out-of-order, or out-of-window observation')
        mean, covariance = _predict(mean, covariance, (event.timestamp_us-timestamp)*1e-6,
                                    diffusion, max_step_s, max_gap_s, check_budget)
        mean, covariance = observe(mean, covariance, event.sensor, event.value, event.variance)
        timestamp, previous = event.timestamp_us, key
        count += 1
        if timestamp in requested:
            # A later same-time wheel replaces the post-gyro snapshot.
            save(timestamp)
    if set(snapshots) != requested:
        raise ValueError('every report must be the checkpoint or an observation timestamp')
    return snapshots
