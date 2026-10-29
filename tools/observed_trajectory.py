"""Exact received-prefix Gaussian proposal, evaluated at supplied coordinates.

This bounded batch diagnostic never draws samples or accepts previous weights.
Angular and conditional-linear primitive coordinates have independent standard
normal priors. The physical checkpoint split retains every cross-covariance.
"""
from dataclasses import dataclass, fields
import math
from types import MappingProxyType

import numpy as np

if __package__:
    from .trajectory_mixture import (
        ANGULAR, LINEAR, InvalidInput, NumericalFailure, UnresolvedInference,
        Observation, ROUND_OFF, SUPPORT_RESOLUTION, _configuration, _covariance,
        _event, split_checkpoint)
else:
    from trajectory_mixture import (
        ANGULAR, LINEAR, InvalidInput, NumericalFailure, UnresolvedInference,
        Observation, ROUND_OFF, SUPPORT_RESOLUTION, _configuration, _covariance,
        _event, split_checkpoint)

MAX_STEPS = 128
MAX_EVENTS = 64
MAX_ROWS = 2048


def _mm(left, right):
    # Direct contraction avoids backend-dependent matrix-product warnings.
    return np.einsum('ij,jk->ik', left, right, optimize=False)


def _mv(matrix, vectors):
    return np.einsum('ij,...j->...i', matrix, vectors, optimize=False)


def _readonly(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _report(**values):
    return MappingProxyType({name: _readonly(value) if isinstance(value, np.ndarray) else value
                             for name, value in values.items()})


def _factor(matrix, name):
    """Factor resolved SPD matrices, never assign rank from a small eigenvalue."""
    size = matrix.shape[0]
    if not size:
        return np.empty((0, 0))
    matrix = _covariance(matrix, size, name, produced=True)
    eigenvalues = np.linalg.eigvalsh(matrix)
    threshold = SUPPORT_RESOLUTION*max(1., float(eigenvalues[-1]))
    if eigenvalues[0] <= threshold:
        raise UnresolvedInference(f'{name} is insufficiently resolved')
    return np.linalg.cholesky(matrix)


def _solve(factor, rhs):
    if not factor.shape[0]:
        return np.array(rhs, copy=True)
    return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))


def _active_factor(covariance, active):
    result = np.zeros((3, len(active)))
    if len(active):
        result[active, :] = np.linalg.cholesky(covariance[np.ix_(active, active)])
    return result


def _checkpoint_linear_map(split, angular_factor):
    return _mm(split.conditional_gain, angular_factor)


def _coordinates(value, dimension, name):
    # Inspect dimensions before dtype conversion can allocate an oversized array.
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1] != dimension:
            raise InvalidInput(f'{name} must have shape M by {dimension}')
        count = value.shape[0]
    elif isinstance(value, (list, tuple)):
        count = len(value)
        if not 1 <= count <= MAX_ROWS:
            raise InvalidInput(f'{name} exceeds the 1-to-{MAX_ROWS} row bound')
        for row in value:
            if isinstance(row, np.ndarray):
                valid = row.ndim == 1 and len(row) == dimension
            else:
                valid = isinstance(row, (list, tuple)) and len(row) == dimension
            if not valid:
                raise InvalidInput(f'{name} must contain rows of width {dimension}')
    else:
        raise InvalidInput(f'{name} must be an array or a sequence of bounded rows')
    if not 1 <= count <= MAX_ROWS:
        raise InvalidInput(f'{name} exceeds the 1-to-{MAX_ROWS} row bound')
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidInput(f'{name} must be a finite matrix') from error
    if array.ndim != 2 or array.shape != (count, dimension) or not np.isfinite(array).all():
        raise InvalidInput(f'{name} must be a finite M by {dimension} matrix')
    return array


def _standard_log_density(points):
    return -.5*(points.shape[1]*math.log(2*math.pi)
                +np.einsum('ni,ni->n', points, points, optimize=False))


def _gaussian_log_density(points, mean, factor):
    dimension = factor.shape[0]
    if dimension == 0:
        return np.zeros(len(points))
    standardized = np.linalg.solve(factor, (points-mean).T).T
    log_determinant = 2*math.fsum(math.log(float(value)) for value in np.diag(factor))
    return -.5*(dimension*math.log(2*math.pi)+log_determinant
                +np.einsum('ni,ni->n', standardized, standardized, optimize=False))


def _importance_ratio(prior, likelihood, proposal):
    return prior+likelihood-proposal


def _residual_covariance(gain, design, variances):
    residual = np.eye(design.shape[1])-_mm(gain, design)
    return _mm(residual, residual.T)+_mm(gain*variances, gain.T)


def _old_directions(angular_paths):
    angles = angular_paths[:, :-1, 0]
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1)


@dataclass(frozen=True)
class PrefixPosterior:
    """Immutable law for one received prefix; there is no continuing ensemble."""
    timestamp_us: int
    phase: str
    step_sizes_s: tuple
    coordinate_layout: tuple
    angular_mean: np.ndarray
    angular_covariance: np.ndarray
    residual_covariance: np.ndarray
    log_evidence: float
    _angular_factor: np.ndarray
    _residual_factor: np.ndarray
    _angular_offsets: np.ndarray
    _angular_maps: np.ndarray
    _linear_offset: np.ndarray
    _linear_angular_map: np.ndarray
    _linear_residual_map: np.ndarray
    _speed_maps: np.ndarray
    _observation_residual: np.ndarray
    _angular_design: np.ndarray
    _residual_design: np.ndarray
    _variances: np.ndarray
    _marginal_factor: np.ndarray
    _residual_gain: np.ndarray

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, field.name, _readonly(value))

    @property
    def angular_dimension(self):
        return len(self.angular_mean)

    @property
    def residual_dimension(self):
        return len(self.residual_covariance)

    def _conditional_residual_mean(self, angular_coordinates):
        residual = self._observation_residual-_mv(self._angular_design, angular_coordinates)
        return _mv(self._residual_gain, residual)

    def path_maps(self, angular_coordinates):
        """Physical paths and ell=b(U)+F(U)Z, without any observation likelihood."""
        points = _coordinates(angular_coordinates, self.angular_dimension, 'angular coordinates')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                paths = self._angular_offsets+np.einsum(
                    'sij,nj->nsi', self._angular_maps, points, optimize=False)
                offsets = self._linear_offset+_mv(self._linear_angular_map, points)
                maps = np.broadcast_to(self._linear_residual_map,
                                       (len(points), 3, self.residual_dimension)).copy()
                directions = _old_directions(paths)
                for index, step in enumerate(self.step_sizes_s):
                    # Speed offset is constant; its process noises are in the Z map.
                    offsets[:, :2] += step*directions[:, index]*offsets[:, 2, None]
                    maps[:, :2, :] += step*directions[:, index, :, None]*self._speed_maps[index]
                    maps[:, 2, :] = self._speed_maps[index+1]
                for value in (paths, offsets, maps):
                    if not np.isfinite(value).all():
                        raise NumericalFailure('nonfinite physical path map')
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            raise NumericalFailure('physical path map failed') from error
        return _report(angular_paths=paths, offset=offsets, residual_map=maps)

    def density_terms(self, angular_coordinates, residual_coordinates=None):
        """Pointwise densities on active primitives, including all normalizers."""
        points = _coordinates(angular_coordinates, self.angular_dimension, 'angular coordinates')
        residual_points = None
        if residual_coordinates is not None:
            residual_points = _coordinates(residual_coordinates, self.residual_dimension,
                                           'residual coordinates')
            if len(residual_points) != len(points):
                raise InvalidInput('angular and residual coordinates need equal row counts')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                likelihood_residual = self._observation_residual-_mv(self._angular_design, points)
                prior = _standard_log_density(points)
                likelihood = _gaussian_log_density(likelihood_residual, 0., self._marginal_factor)
                proposal = _gaussian_log_density(points, self.angular_mean, self._angular_factor)
                values = dict(log_prior=prior, log_likelihood=likelihood, log_proposal=proposal,
                              log_evidence=self.log_evidence,
                              log_importance=_importance_ratio(prior, likelihood, proposal))
                if residual_points is not None:
                    conditional_mean = self._conditional_residual_mean(points)
                    values['log_residual_prior'] = _standard_log_density(residual_points)
                    noise_residual = likelihood_residual-_mv(self._residual_design, residual_points)
                    values['log_observation_likelihood'] = _gaussian_log_density(
                        noise_residual, 0., np.diag(np.sqrt(self._variances)))
                    values['log_residual_posterior'] = _gaussian_log_density(
                        residual_points, conditional_mean, self._residual_factor)
                    values['log_joint_importance'] = (prior+values['log_residual_prior']
                        +values['log_observation_likelihood']-proposal-values['log_residual_posterior'])
                if any(not np.isfinite(value).all() for value in values.values()):
                    raise NumericalFailure('nonfinite density or importance ratio')
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            raise NumericalFailure('Gaussian density arithmetic failed') from error
        return _report(**values)

    def evaluate(self, standard_coordinates):
        """Describe the finite supplied mixture, without a calibration claim."""
        coordinates = _coordinates(standard_coordinates, self.angular_dimension, 'standard coordinates')
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                points = self.angular_mean+_mv(self._angular_factor, coordinates)
                maps = self.path_maps(points)
                residual_mean = self._conditional_residual_mean(points)
                means = maps['offset']+np.einsum('nij,nj->ni', maps['residual_map'],
                                                 residual_mean, optimize=False)
                covariances = np.einsum('nij,jk,nlk->nil', maps['residual_map'],
                                       self.residual_covariance, maps['residual_map'], optimize=False)
                covariances = _covariance(covariances, 3, 'conditional endpoint', produced=True)
                component_means = np.empty((len(points), 6))
                component_means[:, ANGULAR] = maps['angular_paths'][:, -1]
                component_means[:, LINEAR] = means
                mean = component_means[0]+np.mean(component_means-component_means[0], axis=0)
                centered = component_means-mean
                covariance = np.einsum('ni,nj->ij', centered, centered, optimize=False)/len(points)
                covariance[np.ix_(LINEAR, LINEAR)] += np.mean(covariances, axis=0)
                covariance = _covariance(covariance, 6, 'represented covariance', produced=True)
                terms = self.density_terms(points)
                if not all(np.isfinite(value).all() for value in (points, means, residual_mean, mean)):
                    raise NumericalFailure('nonfinite evaluated mixture')
                return _report(timestamp_us=self.timestamp_us, phase=self.phase,
                    angular_coordinates=points, angular_paths=maps['angular_paths'],
                    conditional_mean=means, conditional_covariance=covariances,
                    residual_mean=residual_mean, mean=mean, covariance=covariance,
                    logweights=np.full(len(points), -math.log(len(points))),
                    log_importance=terms['log_importance'], log_evidence=self.log_evidence)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            raise NumericalFailure('mixture evaluation failed') from error


def _received_schedule(events, checkpoint_us, max_step_s, max_gap_s):
    if type(checkpoint_us) is not int:
        raise InvalidInput('checkpoint timestamp must be an integer')
    received, sizes, observation_steps = [], [], []
    last_time, last_key = checkpoint_us, (checkpoint_us, 1)
    try:
        iterator = iter(events)
    except TypeError as error:
        raise InvalidInput('events must be an ordered received prefix') from error
    for index, raw in enumerate(iterator):
        if index >= MAX_EVENTS:
            raise InvalidInput(f'prefix exceeds the {MAX_EVENTS}-event bound')
        event = _event(raw)
        key = (event.timestamp_us, ('gyro', 'wheel').index(event.sensor))
        if event.timestamp_us <= checkpoint_us or key <= last_key:
            raise InvalidInput('checkpoint-time, duplicate, delayed, or reordered event')
        try:
            interval = (event.timestamp_us-last_time)*1e-6
        except OverflowError as error:
            raise InvalidInput('event gap exceeds the fixed bounds') from error
        if interval < 0 or interval > max_gap_s:
            raise InvalidInput('event gap exceeds the fixed bounds')
        count = math.ceil(interval/max_step_s)
        if len(sizes)+count > MAX_STEPS:
            raise InvalidInput(f'prefix exceeds the {MAX_STEPS}-substep bound')
        if count:
            sizes.extend([interval/count]*count)
        observation_steps.append(len(sizes))
        received.append(event)
        last_time, last_key = event.timestamp_us, key
    return tuple(received), tuple(sizes), tuple(observation_steps)


def condition_prefix(mean, covariance, diffusion, events, *, checkpoint_us=0,
                     max_step_s=.01, max_gap_s=2.):
    """Condition only the supplied prefix; callers retain earlier reports themselves."""
    diffusion, max_step_s, max_gap_s = _configuration(diffusion, max_step_s, max_gap_s)
    events, steps, event_steps = _received_schedule(events, checkpoint_us, max_step_s, max_gap_s)
    split = split_checkpoint(mean, covariance)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            return _build_prefix(split, diffusion, events, steps, event_steps, checkpoint_us)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        raise NumericalFailure('prefix Gaussian construction failed') from error


def _build_prefix(split, diffusion, events, steps, event_steps, checkpoint_us):
    initial_a = _active_factor(split.angular_covariance, split.angular_active)
    initial_z = _active_factor(split.conditional_covariance, split.conditional_active)
    angular_layout = [('checkpoint', int(index)) for index in split.angular_active]
    residual_layout = [('checkpoint', int(index)) for index in split.conditional_active]
    angular_columns, speed_columns = [], []
    for index, step in enumerate(steps):
        columns = []
        for coordinate, intensity in zip((1, 2), diffusion[1:]):
            columns.append(len(angular_layout) if intensity > 0 else -1)
            if intensity > 0:
                angular_layout.append(('diffusion', index, coordinate))
        angular_columns.append(columns)
        speed_columns.append(len(residual_layout) if diffusion[0] > 0 else -1)
        if diffusion[0] > 0:
            residual_layout.append(('diffusion', index, 2))
    dimension_u, dimension_z = len(angular_layout), len(residual_layout)
    offsets = np.zeros((len(steps)+1, 3))
    angular_maps = np.zeros((len(steps)+1, 3, dimension_u))
    offsets[0] = split.angular_mean
    angular_maps[0, :, :initial_a.shape[1]] = initial_a
    linear_u = np.zeros((3, dimension_u))
    linear_u[:, :initial_a.shape[1]] = _checkpoint_linear_map(split, initial_a)
    linear_z = np.zeros((3, dimension_z))
    linear_z[:, :initial_z.shape[1]] = initial_z
    speed_maps = np.zeros((len(steps)+1, dimension_z))
    speed_maps[0] = linear_z[2]
    for index, step in enumerate(steps):
        offsets[index+1] = offsets[index]
        offsets[index+1, 0] += step*offsets[index, 1]
        angular_maps[index+1] = angular_maps[index]
        angular_maps[index+1, 0] += step*angular_maps[index, 1]
        speed_maps[index+1] = speed_maps[index]
        for coordinate, column in zip((1, 2), angular_columns[index]):
            if column >= 0:
                scale = math.sqrt(float(diffusion[coordinate]*step))
                if scale == 0:
                    raise UnresolvedInference('positive angular diffusion scale underflowed')
                angular_maps[index+1, coordinate, column] = scale
        if speed_columns[index] >= 0:
            scale = math.sqrt(float(diffusion[0]*step))
            if scale == 0:
                raise UnresolvedInference('positive speed diffusion scale underflowed')
            speed_maps[index+1, speed_columns[index]] = scale
    design_u = np.zeros((len(events), dimension_u))
    design_z = np.zeros((len(events), dimension_z))
    residual = np.zeros(len(events))
    variances = np.array([event.variance for event in events])
    for index, (event, step_index) in enumerate(zip(events, event_steps)):
        if event.sensor == 'gyro':
            design_u[index] = angular_maps[step_index, 1]+angular_maps[step_index, 2]
            predicted = offsets[step_index, 1]+offsets[step_index, 2]
        else:
            design_u[index] = linear_u[2]
            design_z[index] = speed_maps[step_index]
            predicted = split.linear_mean[2]
        residual[index] = event.value-predicted
    marginal = np.diag(variances)+_mm(design_z, design_z.T)
    marginal_factor = _factor(marginal, 'marginal observation covariance')
    solved_u = _solve(marginal_factor, design_u)
    information = np.eye(dimension_u)+_mm(design_u.T, solved_u)
    information_factor = _factor(information, 'angular information')
    angular_covariance = _solve(information_factor, np.eye(dimension_u))
    angular_covariance = .5*angular_covariance+.5*angular_covariance.T
    angular_factor = _factor(angular_covariance, 'angular proposal covariance')
    angular_mean = _mv(angular_covariance, _mv(solved_u.T, residual))
    gain = _solve(marginal_factor, design_z).T
    residual_covariance = _residual_covariance(gain, design_z, variances)
    residual_covariance = .5*residual_covariance+.5*residual_covariance.T
    residual_factor = _factor(residual_covariance, 'conditional residual covariance')
    evidence_covariance = marginal+_mm(design_u, design_u.T)
    evidence_factor = _factor(evidence_covariance, 'evidence covariance')
    evidence = float(_gaussian_log_density(residual[None, :], 0., evidence_factor)[0])
    if not math.isfinite(evidence) or not np.isfinite(angular_mean).all():
        raise NumericalFailure('nonfinite posterior mean or evidence')
    return PrefixPosterior(
        timestamp_us=events[-1].timestamp_us if events else checkpoint_us,
        phase=events[-1].sensor if events else 'checkpoint', step_sizes_s=steps,
        coordinate_layout=(tuple(angular_layout), tuple(residual_layout)),
        angular_mean=angular_mean, angular_covariance=angular_covariance,
        residual_covariance=residual_covariance, log_evidence=evidence,
        _angular_factor=angular_factor, _residual_factor=residual_factor,
        _angular_offsets=offsets, _angular_maps=angular_maps,
        _linear_offset=split.linear_mean, _linear_angular_map=linear_u,
        _linear_residual_map=linear_z, _speed_maps=speed_maps,
        _observation_residual=residual, _angular_design=design_u,
        _residual_design=design_z, _variances=variances,
        _marginal_factor=marginal_factor, _residual_gain=gain)
