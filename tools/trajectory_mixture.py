"""Persistent angular paths with conditional Gaussian position and speed.

This is the fixed offline prior-proposal variant in trajectory_mixture_proposal.
It never resamples, projects the joint law to a Gaussian, or accepts truth.
The angular order is theta, omega, bias; conditional order is px, py, speed.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np


ANGULAR = np.array([2, 4, 5])
LINEAR = np.array([0, 1, 3])
ROUND_OFF = 1e-10
SUPPORT_RESOLUTION = 1e-12


class InvalidInput(ValueError):
    """An input violates the finite, covariance, sensor, or time contract."""


class UnresolvedInference(ValueError):
    """A valid covariance has unsupported or insufficiently resolved support."""


class NumericalFailure(ValueError):
    """Produced inference is materially invalid or nonfinite."""


@dataclass(frozen=True)
class Observation:
    timestamp_us: int
    sensor: str
    value: float
    variance: float


@dataclass(frozen=True)
class CheckpointSplit:
    angular_mean: np.ndarray
    angular_covariance: np.ndarray
    linear_mean: np.ndarray
    conditional_gain: np.ndarray
    conditional_covariance: np.ndarray
    angular_active: np.ndarray
    conditional_active: np.ndarray

    def conditional_means(self, angular):
        angular = np.asarray(angular, dtype=float)
        if angular.ndim != 2 or angular.shape[1] != 3 or not np.isfinite(angular).all():
            raise InvalidInput('angular points must be a finite M by 3 array')
        inactive = np.ones(3, dtype=bool)
        inactive[self.angular_active] = False
        if np.any(angular[:, inactive] != self.angular_mean[inactive]):
            raise InvalidInput('angular point is outside deterministic checkpoint support')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                result = self.linear_mean + np.einsum(
                    'ij,nj->ni', self.conditional_gain, angular-self.angular_mean,
                    optimize=False)
        except FloatingPointError as error:
            raise NumericalFailure('nonfinite conditional checkpoint mean') from error
        if not np.isfinite(result).all():
            raise NumericalFailure('nonfinite conditional checkpoint mean')
        return result


def _scalar(value, name):
    if isinstance(value, (bool, str, bytes)) or np.ndim(value) != 0:
        raise InvalidInput(f'{name} must be a finite scalar')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidInput(f'{name} must be a finite scalar') from error
    if not math.isfinite(result):
        raise InvalidInput(f'{name} must be a finite scalar')
    return result


def _covariance(matrix, size, name, produced=False):
    """Only roundoff symmetrization, in the frozen unit scales (all numeric 1)."""
    error_type = NumericalFailure if produced else InvalidInput
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim < 2 or matrix.shape[-2:] != (size, size) or not np.isfinite(matrix).all():
        raise error_type(f'{name} must contain finite {size} by {size} matrices')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            scale = np.maximum(1., np.max(np.abs(matrix), axis=(-2, -1)))
            tolerance = ROUND_OFF*scale
            asymmetry = np.max(np.abs(matrix-np.swapaxes(matrix, -1, -2)), axis=(-2, -1))
            if np.any(asymmetry > tolerance):
                raise error_type(f'{name} is asymmetric')
            symmetric = .5*matrix+.5*np.swapaxes(matrix, -1, -2)
            eigenvalues = np.linalg.eigvalsh(symmetric)
            if np.any(eigenvalues[..., 0] < -tolerance):
                raise error_type(f'{name} is not positive semidefinite')
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise error_type(f'{name} covariance validation failed') from error
    return symmetric


def _coordinate_support(matrix, name, produced=False):
    """Reject rotated singular active support instead of choosing a numerical rank."""
    error_type = NumericalFailure if produced else InvalidInput
    diagonal = np.diag(matrix)
    zero = diagonal == 0
    if np.any(matrix[zero, :] != 0) or np.any(matrix[:, zero] != 0):
        raise error_type(f'{name} correlates a deterministic coordinate')
    if np.any(diagonal < 0):
        raise UnresolvedInference(f'{name} has unresolved nonpositive active support')
    active = np.flatnonzero(diagonal > 0)
    if len(active):
        block = matrix[np.ix_(active, active)]
        try:
            eigenvalues = np.linalg.eigvalsh(block)
        except np.linalg.LinAlgError as error:
            raise NumericalFailure(f'{name} support decomposition failed') from error
        threshold = SUPPORT_RESOLUTION*max(1., float(eigenvalues[-1]))
        if eigenvalues[0] <= threshold:
            raise UnresolvedInference(f'{name} active support is singular or insufficiently resolved')
    return active


def _conditional_covariances(covariances, produced=False):
    covariances = _covariance(covariances, 3, 'conditional covariance', produced)
    if covariances.ndim != 3:
        raise InvalidInput('conditional covariance must have shape M by 3 by 3')
    error_type = NumericalFailure if produced else InvalidInput
    diagonal = np.diagonal(covariances, axis1=1, axis2=2)
    zero = diagonal == 0
    if np.any((covariances != 0) & (zero[:, :, None] | zero[:, None, :])):
        raise error_type('conditional covariance correlates a deterministic coordinate')
    if np.any(diagonal < 0):
        raise UnresolvedInference('conditional covariance has unresolved nonpositive support')
    masks = np.sum((diagonal > 0)*np.array([1, 2, 4]), axis=1)
    for mask in np.unique(masks):
        active = np.flatnonzero(int(mask) & np.array([1, 2, 4]))
        if not len(active):
            continue
        selected = covariances[masks == mask][:, active][:, :, active]
        try:
            eigenvalues = np.linalg.eigvalsh(selected)
        except np.linalg.LinAlgError as error:
            raise NumericalFailure('conditional support decomposition failed') from error
        threshold = SUPPORT_RESOLUTION*np.maximum(1., eigenvalues[:, -1])
        if np.any(eigenvalues[:, 0] <= threshold):
            raise UnresolvedInference('conditional active support is singular or insufficiently resolved')
    return covariances


def split_checkpoint(mean, covariance):
    """Derive the full correlated Gaussian split, using only resolved active blocks."""
    mean = np.array(mean, dtype=float, copy=True)
    raw = np.array(covariance, dtype=float, copy=True)
    if mean.shape != (6,) or not np.isfinite(mean).all() or raw.shape != (6, 6):
        raise InvalidInput('checkpoint must have a finite six-state mean and 6 by 6 covariance')
    # Check exact deterministic rows before roundoff symmetrization can hide one.
    if not np.isfinite(raw).all():
        raise InvalidInput('checkpoint covariance must be finite')
    zero = np.diag(raw) == 0
    if np.any(raw[zero, :] != 0) or np.any(raw[:, zero] != 0):
        raise InvalidInput('checkpoint correlates a deterministic coordinate')
    covariance = _covariance(raw, 6, 'checkpoint')
    angular = covariance[np.ix_(ANGULAR, ANGULAR)]
    linear = covariance[np.ix_(LINEAR, LINEAR)]
    cross = covariance[np.ix_(LINEAR, ANGULAR)]
    active = _coordinate_support(angular, 'angular checkpoint')
    gain = np.zeros((3, 3))
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if len(active):
                factor = np.linalg.cholesky(angular[np.ix_(active, active)])
                gain[:, active] = np.linalg.solve(
                    factor.T, np.linalg.solve(factor, cross[:, active].T)).T
            conditional = linear-np.einsum('ij,kj->ik', gain, cross, optimize=False)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise NumericalFailure('checkpoint conditioning failed') from error
    conditional = _covariance(conditional, 3, 'checkpoint conditional covariance')
    conditional_active = _coordinate_support(conditional, 'checkpoint conditional covariance')
    return CheckpointSplit(mean[ANGULAR].copy(), angular.copy(), mean[LINEAR].copy(),
                           gain, conditional, active, conditional_active)


def _normal_log_density(residual, variance):
    residual, variance = np.broadcast_arrays(np.asarray(residual, dtype=float),
                                             np.asarray(variance, dtype=float))
    if not np.isfinite(residual).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise NumericalFailure('likelihood requires finite residual and positive variance')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            result = -.5*(math.log(2*math.pi)+np.log(variance)+(residual/np.sqrt(variance))**2)
    except FloatingPointError as error:
        raise NumericalFailure('Gaussian log likelihood arithmetic failed') from error
    if not np.isfinite(result).all():
        raise NumericalFailure('nonfinite Gaussian log likelihood')
    return result


def _exponential(values):
    """Count zero/subnormal exponentials; finite log values remain authoritative."""
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            result = np.exp(values)
    except FloatingPointError as error:
        raise NumericalFailure('weight exponential failed') from error
    if not np.isfinite(result).all():
        raise NumericalFailure('nonfinite exponential weight')
    underflow = int(np.count_nonzero(result < np.finfo(float).tiny))
    return result, underflow


def _normalize_logweights(logweights):
    logweights = np.asarray(logweights, dtype=float)
    if logweights.ndim != 1 or not len(logweights) or not np.isfinite(logweights).all():
        raise NumericalFailure('every retained log weight must be finite')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            maximum = float(logweights.max())
            shifted, underflow = _exponential(logweights-maximum)
            total = math.fsum(float(value) for value in shifted)
            if not math.isfinite(total) or total <= 0:
                raise NumericalFailure('weight normalizer is invalid')
            log_increment = maximum+math.log(total)
            normalized = logweights-log_increment
    except (FloatingPointError, OverflowError) as error:
        raise NumericalFailure('log normalization arithmetic failed') from error
    if not math.isfinite(log_increment) or not np.isfinite(normalized).all():
        raise NumericalFailure('nonfinite normalized log weights')
    return normalized, log_increment, underflow


def _conditional_transition(angular, mean, covariance, step, speed_diffusion):
    """All displacement coefficients use the old angle and old speed."""
    transform = np.broadcast_to(np.eye(3), covariance.shape).copy()
    transform[:, 0, 2] = step*np.cos(angular[:, 0])
    transform[:, 1, 2] = step*np.sin(angular[:, 0])
    updated_mean = np.einsum('nij,nj->ni', transform, mean, optimize=False)
    updated_covariance = np.einsum('nij,njk,nlk->nil', transform, covariance,
                                   transform, optimize=False)
    updated_covariance[:, 2, 2] += speed_diffusion*step
    return updated_mean, updated_covariance


def _wheel_update(mean, covariance, value, variance):
    residual = value-mean[:, 2]
    innovation = covariance[:, 2, 2]+variance
    likelihood = _normal_log_density(residual, innovation)
    gain = covariance[:, :, 2]/innovation[:, None]
    updated_mean = mean+gain*residual[:, None]
    transform = np.broadcast_to(np.eye(3), covariance.shape).copy()
    transform[:, :, 2] -= gain
    updated_covariance = (np.einsum('nij,njk,nlk->nil', transform, covariance,
                                    transform, optimize=False)
                          +variance*np.einsum('ni,nj->nij', gain, gain, optimize=False))
    return updated_mean, updated_covariance, likelihood


def _configuration(diffusion, max_step_s, max_gap_s):
    diffusion = np.array(diffusion, dtype=float, copy=True)
    if diffusion.shape != (3,) or not np.isfinite(diffusion).all() or np.any(diffusion < 0):
        raise InvalidInput('diffusion must be three finite nonnegative intensities')
    max_step_s = _scalar(max_step_s, 'max_step_s')
    max_gap_s = _scalar(max_gap_s, 'max_gap_s')
    if not 1e-5 <= max_step_s <= .1 or not max_step_s <= max_gap_s <= 60:
        raise InvalidInput('invalid propagation bounds')
    return diffusion, max_step_s, max_gap_s


def _event(event):
    try:
        timestamp, sensor = event.timestamp_us, event.sensor
        value, variance = event.value, event.variance
    except AttributeError as error:
        raise InvalidInput('observation needs timestamp_us, sensor, value, variance') from error
    if type(timestamp) is not int or sensor not in ('gyro', 'wheel'):
        raise InvalidInput('observation needs an integer timestamp and gyro or wheel sensor')
    value = _scalar(value, 'observation value')
    variance = _scalar(variance, 'observation variance')
    if variance <= 0:
        raise InvalidInput('observation variance must be positive')
    return Observation(timestamp, sensor, value, variance)


class TrajectoryMixture:
    """Fixed-count forward inference; all mutating operations are transactional."""

    @classmethod
    def initialize(cls, mean, covariance, count, initial_rng, transition_rng, diffusion,
                   checkpoint_us=0, max_step_s=.01, max_gap_s=2.):
        if type(count) is not int or count <= 0:
            raise InvalidInput('component count must be a positive integer')
        if not isinstance(initial_rng, np.random.Generator):
            raise InvalidInput('initial_rng must be a NumPy Generator')
        split = split_checkpoint(mean, covariance)
        saved_rng = deepcopy(initial_rng.bit_generator.state)
        try:
            angular = np.broadcast_to(split.angular_mean, (count, 3)).copy()
            if len(split.angular_active):
                factor = np.linalg.cholesky(split.angular_covariance[
                    np.ix_(split.angular_active, split.angular_active)])
                standard = initial_rng.standard_normal((count, len(split.angular_active)))
                angular[:, split.angular_active] += np.einsum(
                    'ni,ji->nj', standard, factor, optimize=False)
            result = cls.from_components(
                angular, split.conditional_means(angular),
                np.broadcast_to(split.conditional_covariance, (count, 3, 3)),
                np.full(count, -math.log(count)), np.arange(count, dtype=np.int64),
                transition_rng, diffusion, checkpoint_us, max_step_s, max_gap_s)
        except Exception:
            initial_rng.bit_generator.state = saved_rng
            raise
        return result

    @classmethod
    def from_components(cls, angular, conditional_mean, conditional_covariance, logweights,
                        ids, transition_rng, diffusion, checkpoint_us=0,
                        max_step_s=.01, max_gap_s=2.):
        """Construct a declared mixture for deterministic algebra checks, without sampling."""
        if not isinstance(transition_rng, np.random.Generator):
            raise InvalidInput('transition_rng must be a NumPy Generator')
        if type(checkpoint_us) is not int or checkpoint_us < 0:
            raise InvalidInput('checkpoint timestamp must be a nonnegative integer')
        diffusion, max_step_s, max_gap_s = _configuration(diffusion, max_step_s, max_gap_s)
        angular = np.array(angular, dtype=float, copy=True)
        mean = np.array(conditional_mean, dtype=float, copy=True)
        covariance = np.array(conditional_covariance, dtype=float, copy=True)
        logweights = np.array(logweights, dtype=float, copy=True)
        ids = np.asarray(ids)
        if angular.ndim != 2 or angular.shape[1] != 3 or not len(angular):
            raise InvalidInput('angular states must have shape M by 3')
        count = len(angular)
        if mean.shape != (count, 3) or covariance.shape != (count, 3, 3):
            raise InvalidInput('conditional means/covariances have incorrect component shapes')
        if not np.isfinite(angular).all() or not np.isfinite(mean).all():
            raise InvalidInput('component states must be finite')
        if logweights.shape != (count,):
            raise InvalidInput('log weights must have one entry per component')
        if ids.shape != (count,) or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != count:
            raise InvalidInput('component identities must be unique integers')
        covariance = _conditional_covariances(covariance)
        normalized, _, underflow = _normalize_logweights(logweights)
        self = cls()
        self.angular = angular
        self.conditional_mean = mean
        self.conditional_covariance = covariance
        self.logweights = normalized
        self.ids = ids.astype(np.int64, copy=True)
        self.transition_rng = transition_rng
        self.diffusion = diffusion
        self.max_step_s, self.max_gap_s = max_step_s, max_gap_s
        self.timestamp_us = checkpoint_us
        self.phase = 'checkpoint'
        self.observation_count = 0
        self.substeps = 0
        self.primitive_elapsed_s = 0.
        self.log_evidence = 0.
        self.normalization_underflow_count = underflow
        self.reporting_underflow_count = 0
        self.moment_underflow_count = 0
        self._last_reporting_revision = None
        self._last_moment_revision = None
        self._revision = 0
        self._last_event_key = (checkpoint_us, 2)
        self._primitive_mode = False
        return self

    def clone(self):
        """Copy all arrays and the independent proposal stream, for a paired H branch."""
        return deepcopy(self)

    def _saved(self):
        return (deepcopy({key: value for key, value in self.__dict__.items()
                          if key != 'transition_rng'}),
                deepcopy(self.transition_rng.bit_generator.state))

    def _restore(self, saved):
        state, rng_state = saved
        self.__dict__.update(state)
        self.transition_rng.bit_generator.state = rng_state

    def _advance(self, step, innovation=None):
        if innovation is None:
            innovation = np.zeros_like(self.angular)
            active = np.flatnonzero(self.diffusion[1:] > 0)
            if len(active):
                standard = self.transition_rng.standard_normal((len(self.angular), len(active)))
                innovation[:, active+1] = standard*np.sqrt(self.diffusion[active+1]*step)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                mean, covariance = _conditional_transition(
                    self.angular, self.conditional_mean, self.conditional_covariance,
                    step, self.diffusion[0])
                angular = self.angular.copy()
                angular[:, 0] += step*self.angular[:, 1]
                angular += innovation
                if not np.isfinite(angular).all() or not np.isfinite(mean).all():
                    raise NumericalFailure('nonfinite predicted component mean')
                covariance = _conditional_covariances(covariance, produced=True)
        except (FloatingPointError, OverflowError) as error:
            raise NumericalFailure('conditional prediction arithmetic failed') from error
        self.angular, self.conditional_mean, self.conditional_covariance = angular, mean, covariance
        self.substeps += 1
        self.primitive_elapsed_s += step
        self._revision += 1

    def predict_step(self, h, angular_innovation=None):
        """One prescribed primitive step, independent of the event-runner partition."""
        step = _scalar(h, 'step')
        if not 0 <= step <= .1:
            raise InvalidInput('primitive step must be between zero and 0.1 seconds')
        innovation = None
        if angular_innovation is not None:
            innovation = np.asarray(angular_innovation, dtype=float)
            if innovation.shape == (3,):
                innovation = np.broadcast_to(innovation, self.angular.shape).copy()
            if innovation.shape != self.angular.shape or not np.isfinite(innovation).all():
                raise InvalidInput('angular innovation must be finite with shape 3 or M by 3')
            deterministic = np.r_[True, self.diffusion[1:] == 0]
            if np.any(innovation[:, deterministic] != 0) or (step == 0 and np.any(innovation != 0)):
                raise InvalidInput('angular innovation is outside deterministic transition support')
        if step == 0:
            return self.diagnostics()
        saved = self._saved()
        try:
            self._advance(step, innovation)
            self._primitive_mode = True
            self.phase = 'prediction'
            return self.diagnostics()
        except Exception:
            self._restore(saved)
            raise

    def _apply_log_likelihood(self, likelihood):
        likelihood = np.asarray(likelihood, dtype=float)
        if likelihood.shape != self.logweights.shape or not np.isfinite(likelihood).all():
            raise NumericalFailure('every component likelihood must be finite in log space')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                normalized, increment, underflow = _normalize_logweights(self.logweights+likelihood)
                evidence = self.log_evidence+increment
        except FloatingPointError as error:
            raise NumericalFailure('likelihood accumulation failed') from error
        if not math.isfinite(evidence):
            raise NumericalFailure('nonfinite cumulative log evidence')
        self.logweights = normalized
        self.log_evidence = evidence
        self.normalization_underflow_count += underflow
        self._revision += 1

    def apply_log_likelihood(self, likelihood):
        """Normalization primitive for the frozen deterministic log-weight check."""
        saved = self._saved()
        try:
            self._apply_log_likelihood(likelihood)
            self._primitive_mode = True
            return self.diagnostics()
        except Exception:
            self._restore(saved)
            raise

    def _observe(self, sensor, value, variance):
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                if sensor == 'gyro':
                    likelihood = _normal_log_density(value-self.angular[:, 1]-self.angular[:, 2], variance)
                else:
                    mean, covariance, likelihood = _wheel_update(
                        self.conditional_mean, self.conditional_covariance, value, variance)
                    if not np.isfinite(mean).all():
                        raise NumericalFailure('nonfinite conditional wheel mean')
                    covariance = _conditional_covariances(covariance, produced=True)
                    self.conditional_mean, self.conditional_covariance = mean, covariance
                self._apply_log_likelihood(likelihood)
        except (FloatingPointError, OverflowError) as error:
            raise NumericalFailure('observation arithmetic failed') from error
        self.observation_count += 1
        self.phase = sensor

    def observe(self, sensor, value, variance):
        """Observation primitive for declared-mixture checks, without event-time changes."""
        event = _event(Observation(0, sensor, value, variance))
        saved = self._saved()
        try:
            self._observe(event.sensor, event.value, event.variance)
            self._primitive_mode = True
            return self.diagnostics()
        except Exception:
            self._restore(saved)
            raise

    def process(self, event):
        if self._primitive_mode:
            raise InvalidInput('a primitive-mode state cannot enter the timestamped event runner')
        event = _event(event)
        key = (event.timestamp_us, ('gyro', 'wheel').index(event.sensor))
        if key <= self._last_event_key:
            raise InvalidInput('duplicate, delayed, or out-of-order observation')
        interval = (event.timestamp_us-self.timestamp_us)*1e-6
        if interval < 0 or interval > self.max_gap_s:
            raise InvalidInput('observation gap exceeds the fixed bounds')
        saved = self._saved()
        try:
            steps = math.ceil(interval/self.max_step_s)
            if steps:
                step = interval/steps
                for _ in range(steps):
                    self._advance(step)
            self._observe(event.sensor, event.value, event.variance)
            self.timestamp_us = event.timestamp_us
            self._last_event_key = key
            return self.diagnostics()
        except Exception:
            self._restore(saved)
            raise

    def _weights(self):
        if not np.isfinite(self.logweights).all():
            raise NumericalFailure('every retained log weight must remain finite')
        weights, underflow = _exponential(self.logweights)
        if underflow and self._last_reporting_revision != self._revision:
            self.reporting_underflow_count += underflow
            self._last_reporting_revision = self._revision
        return weights

    def diagnostics(self):
        weights = self._weights()
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                weight_sum = math.fsum(float(value) for value in weights)
                square_sum = math.fsum(float(value*value) for value in weights)
                if abs(weight_sum-1.) > ROUND_OFF or square_sum <= 0:
                    raise NumericalFailure('represented weights are not normalized')
                ess = 1./square_sum
                log_range = float(self.logweights.max()-self.logweights.min())
        except (FloatingPointError, OverflowError) as error:
            raise NumericalFailure('weight diagnostics failed') from error
        return dict(timestamp_us=self.timestamp_us, phase=self.phase,
                    observation_count=self.observation_count, substeps=self.substeps,
                    primitive_elapsed_s=self.primitive_elapsed_s, log_evidence=self.log_evidence,
                    ess=ess, max_weight=float(weights.max()), logweight_range=log_range,
                    normalization_underflow_count=self.normalization_underflow_count,
                    reporting_underflow_count=self.reporting_underflow_count,
                    moment_underflow_count=self.moment_underflow_count,
                    underflow_count=(self.normalization_underflow_count
                                     +self.reporting_underflow_count+self.moment_underflow_count))

    def full_moments(self):
        weights = self._weights()
        component_means = np.empty((len(weights), 6))
        component_means[:, ANGULAR] = self.angular
        component_means[:, LINEAR] = self.conditional_mean

        def reconstruct():
            # Anchoring preserves exactly deterministic coordinates under roundoff.
            mean = component_means[0]+np.sum(
                weights[:, None]*(component_means-component_means[0]), axis=0)
            centered = component_means-mean
            weighted_centered = weights[:, None]*centered
            covariance = np.sum(weighted_centered[:, :, None]*centered[:, None, :], axis=0)
            covariance[np.ix_(LINEAR, LINEAR)] += np.sum(
                weights[:, None, None]*self.conditional_covariance, axis=0)
            return mean, covariance

        underflow = False
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
                mean, covariance = reconstruct()
        except FloatingPointError as error:
            if 'underflow' not in str(error).lower():
                raise NumericalFailure('full moment reconstruction failed') from error
            underflow = True
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                    mean, covariance = reconstruct()
            except FloatingPointError as retry_error:
                raise NumericalFailure('full moment reconstruction failed') from retry_error
        if not np.isfinite(mean).all():
            raise NumericalFailure('nonfinite full mixture mean')
        covariance = _covariance(covariance, 6, 'full mixture covariance', produced=True)
        if underflow and self._last_moment_revision != self._revision:
            self.moment_underflow_count += 1
            self._last_moment_revision = self._revision
        return mean, covariance

    def snapshot(self):
        mean, covariance = self.full_moments()
        result = self.diagnostics()
        result.update(angular=self.angular.copy(), conditional_mean=self.conditional_mean.copy(),
                      conditional_covariance=self.conditional_covariance.copy(),
                      logweights=self.logweights.copy(), ids=self.ids.copy(),
                      mean=mean, covariance=covariance)
        return result


def run_events(mixture, events, report_times_us):
    """Validate the schedule, then filter causally and report after same-time events."""
    if not isinstance(mixture, TrajectoryMixture) or mixture._primitive_mode:
        raise InvalidInput('event runner needs a timestamped trajectory mixture')
    events = [_event(event) for event in events]
    reports = list(report_times_us)
    if (not reports or any(type(t) is not int or t < mixture.timestamp_us for t in reports)
            or any(left >= right for left, right in zip(reports, reports[1:]))):
        raise InvalidInput('invalid report timestamp sequence')
    previous_key = mixture._last_event_key
    previous_time = mixture.timestamp_us
    available = {mixture.timestamp_us}
    for event in events:
        key = (event.timestamp_us, ('gyro', 'wheel').index(event.sensor))
        if key <= previous_key or event.timestamp_us > reports[-1]:
            raise InvalidInput('duplicate, delayed, out-of-order, or out-of-window observation')
        if (event.timestamp_us-previous_time)*1e-6 > mixture.max_gap_s:
            raise InvalidInput('observation gap exceeds the fixed bound')
        previous_key, previous_time = key, event.timestamp_us
        available.add(event.timestamp_us)
    if any(timestamp not in available for timestamp in reports):
        raise InvalidInput('every report must be the initial timestamp or a received event timestamp')
    snapshots, diagnostics = {}, []
    requested = set(reports)
    if mixture.timestamp_us in requested:
        snapshots[mixture.timestamp_us] = mixture.snapshot()
    for index, event in enumerate(events):
        diagnostics.append(mixture.process(event))
        last_at_time = index+1 == len(events) or events[index+1].timestamp_us != event.timestamp_us
        if last_at_time and event.timestamp_us in requested:
            snapshots[event.timestamp_us] = mixture.snapshot()
    return snapshots, diagnostics
