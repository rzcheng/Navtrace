"""Independent GPS-free position moments from a conditioned physical Gaussian.

The declared checkpoint factor defines the law. This module does not import an
estimator, draw coordinates, read artifacts, or run a comparison. Returned
moments are not a Gaussian approximation to the complete position distribution.
"""
from dataclasses import dataclass, fields
import math
from types import MappingProxyType

import numpy as np

MAX_STEPS = 100
MAX_EVENTS = 32
MAX_PHYSICAL = 306
MAX_FACTOR_COLUMNS = 338
MAX_STEP_S = .01
MAX_GAP_S = 2.
ROUND_OFF = 1e-10
SUPPORT_RESOLUTION = 1e-12


def _freeze(value):
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, dict):
        return MappingProxyType({name: _freeze(item) for name, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class OracleError(ValueError):
    status = "failed"

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = _freeze({} if diagnostics is None else diagnostics)


class InvalidInput(OracleError):
    status = "invalid"


class NumericalFailure(OracleError):
    status = "failed"


class UnresolvedInference(OracleError):
    status = "unresolved"


@dataclass(frozen=True)
class Observation:
    timestamp_us: int
    sensor: str
    value: float
    variance: float


def _shape_matches(value, shape):
    # Reject dimensions before conversion can invoke a large array allocation.
    if isinstance(value, np.ndarray):
        return value.shape == shape
    if not shape:
        return not isinstance(value, (list, tuple, np.ndarray))
    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        return False
    return all(_shape_matches(item, shape[1:]) for item in value)


def _array(value, shape, name):
    if not _shape_matches(value, shape):
        raise InvalidInput(f"{name} must have shape {shape}")
    try:
        if np.iscomplexobj(value):
            raise InvalidInput(f"{name} must be real")
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidInput(f"{name} must contain finite real values") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise InvalidInput(f"{name} must contain finite real values")
    return result


def _scalar(value, name):
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidInput(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise InvalidInput(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise InvalidInput(f"{name} must be a finite real scalar")
    return result


def _factor_width(value):
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[0] != 6 or value.shape[1] > 6:
            raise InvalidInput("checkpoint factor must have shape 6 by r, 0 <= r <= 6")
        return value.shape[1]
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise InvalidInput("checkpoint factor must contain six bounded rows")
    row = value[0]
    if not isinstance(row, (list, tuple, np.ndarray)):
        raise InvalidInput("checkpoint factor must contain bounded rows")
    if (isinstance(row, np.ndarray) and row.ndim != 1) or len(row) > 6:
        raise InvalidInput("checkpoint factor width must be at most six")
    return len(row)


def _event(raw):
    if isinstance(raw, (list, tuple)):
        if len(raw) != 4:
            raise InvalidInput("event sequence must have four fields")
        timestamp, sensor, value, variance = raw
    else:
        try:
            timestamp, sensor = raw.timestamp_us, raw.sensor
            value, variance = raw.value, raw.variance
        except AttributeError as exc:
            raise InvalidInput("event needs timestamp_us, sensor, value, variance") from exc
    if type(timestamp) is not int or not isinstance(sensor, str) or sensor not in ("gyro", "wheel"):
        raise InvalidInput("event requires an integer timestamp and gyro or wheel sensor")
    value, variance = _scalar(value, "observation"), _scalar(variance, "sensor variance")
    if variance <= 0:
        raise InvalidInput("sensor variance must be positive")
    return Observation(timestamp, sensor, value, variance)


def _schedule(events, checkpoint_us):
    if type(checkpoint_us) is not int:
        raise InvalidInput("checkpoint timestamp must be an integer")
    if isinstance(events, (list, tuple)) and len(events) > MAX_EVENTS:
        raise InvalidInput("received prefix exceeds the 32-event bound")
    try:
        iterator = iter(events)
    except TypeError as exc:
        raise InvalidInput("events must be a bounded ordered iterable") from exc
    received, sizes, event_steps = [], [], []
    last_time, last_key = checkpoint_us, (checkpoint_us, 1)
    # One bounded lookahead detects an overlong generator without consuming it.
    for index, raw in enumerate(iterator):
        if index >= MAX_EVENTS:
            raise InvalidInput("received prefix exceeds the 32-event bound")
        event = _event(raw)
        key = (event.timestamp_us, 0 if event.sensor == "gyro" else 1)
        if event.timestamp_us <= checkpoint_us or key <= last_key:
            raise InvalidInput("checkpoint-time, duplicate, delayed or reordered event")
        try:
            interval = (event.timestamp_us-last_time)*1e-6
        except OverflowError as exc:
            raise InvalidInput("event interval exceeds the fixed gap bound") from exc
        if interval < 0 or interval > MAX_GAP_S:
            raise InvalidInput("event interval exceeds the fixed gap bound")
        count = math.ceil(interval/MAX_STEP_S)
        if len(sizes)+count > MAX_STEPS:
            raise InvalidInput("received prefix exceeds the 100-step bound")
        if count:
            sizes.extend([interval/count]*count)
        event_steps.append(len(sizes))
        received.append(event)
        last_time, last_key = event.timestamp_us, key
    return tuple(received), tuple(sizes), tuple(event_steps)


def _mm(left, right):
    return np.einsum('ij,jk->ik', left, right, optimize=False)


def _mv(matrix, vector):
    return np.einsum('ij,j->i', matrix, vector, optimize=False)


def _dot(left, right):
    # The multiplication is a ufunc, so underflow is recorded by the caller's
    # error callback; fsum retains a fixed compensated summation order.
    return math.fsum(np.multiply(left, right).tolist())


def _covariance(matrix, name, *, produced):
    error = NumericalFailure if produced else InvalidInput
    if not np.isfinite(matrix).all():
        raise error(f"{name} contains nonfinite values")
    scale = max(1., float(np.max(np.abs(matrix))))
    asymmetry = float(np.max(np.abs(matrix-matrix.T)))
    if asymmetry > ROUND_OFF*scale:
        raise error(f"{name} is materially asymmetric", {"asymmetry": asymmetry})
    symmetric = .5*matrix+.5*matrix.T
    spectrum = np.linalg.eigvalsh(symmetric)
    if spectrum[0] < -ROUND_OFF*scale:
        raise error(f"{name} is materially indefinite", {"eigenvalues": spectrum})
    return symmetric, {"asymmetry_before_symmetrization": asymmetry,
                       "eigenvalues": spectrum, "tolerance": ROUND_OFF*scale}


def _count(diagnostics, name):
    counts = diagnostics.setdefault("underflow_counts", {})
    counts[name] = counts.get(name, 0)+1


def _attenuation(exponent, diagnostics):
    result = math.exp(exponent)
    if exponent < 0 and result == 0:
        _count(diagnostics, "damping_to_zero")
    elif 0 < result < np.finfo(np.float64).tiny:
        _count(diagnostics, "damping_subnormal")
    return result


def _phase(mean, variance, diagnostics):
    damping = _attenuation(-.5*variance, diagnostics)
    return complex(math.cos(mean), math.sin(mean))*damping


def gaussian_first(mu_a, mu_t, var_t, cov_at):
    """E[A exp(iT)] for real jointly Gaussian A,T, including singular laws."""
    mu_a, mu_t, var_t, cov_at = [_scalar(value, name) for value, name in (
        (mu_a, "mean A"), (mu_t, "mean T"), (var_t, "variance T"), (cov_at, "covariance AT"))]
    if var_t < 0:
        raise InvalidInput("phase variance must be nonnegative")
    result = complex(mu_a, cov_at)*_phase(mu_t, var_t, {})
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise NumericalFailure("nonfinite first characteristic moment")
    return result


def gaussian_second(mu_a, mu_b, cov_ab, mu_t, var_t, cov_at, cov_bt):
    """E[A B exp(iT)] with all Gaussian cross terms retained."""
    values = [_scalar(value, name) for value, name in zip(
        (mu_a, mu_b, cov_ab, mu_t, var_t, cov_at, cov_bt),
        ("mean A", "mean B", "covariance AB", "mean T", "variance T", "covariance AT", "covariance BT"))]
    mu_a, mu_b, cov_ab, mu_t, var_t, cov_at, cov_bt = values
    if var_t < 0:
        raise InvalidInput("phase variance must be nonnegative")
    result = (cov_ab+complex(mu_a, cov_at)*complex(mu_b, cov_bt))*_phase(mu_t, var_t, {})
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise NumericalFailure("nonfinite second characteristic moment")
    return result


def _centered_pair(left_mean, right_mean, left_factor, right_factor, sign, diagnostics):
    # Factor rows are theta, v in one shared Gaussian coordinate system.
    theta_j, speed_j = left_factor
    theta_k, speed_k = right_factor
    var_j, var_k = _dot(theta_j, theta_j), _dot(theta_k, theta_k)
    phase_row = theta_j+sign*theta_k
    exponent_pair = -.5*_dot(phase_row, phase_row)
    exponent_product = -.5*(var_j+var_k)
    a = complex(left_mean[1], _dot(speed_j, theta_j))
    b = complex(right_mean[1], sign*_dot(speed_k, theta_k))
    d, e = _dot(speed_j, theta_k), _dot(speed_k, theta_j)
    pair_damping = _attenuation(exponent_pair, diagnostics)
    if exponent_pair >= exponent_product:
        difference = pair_damping*(-math.expm1(exponent_product-exponent_pair))
    else:
        difference = _attenuation(exponent_product, diagnostics)*math.expm1(exponent_pair-exponent_product)
    bracket = _dot(speed_j, speed_k)+1j*e*a+1j*sign*d*b-sign*d*e
    phase = left_mean[0]+sign*right_mean[0]
    result = complex(math.cos(phase), math.sin(phase))*(difference*a*b+pair_damping*bracket)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise NumericalFailure("nonfinite centered displacement pair")
    return result


def _moment_shape(means, factors):
    if isinstance(means, np.ndarray):
        if means.ndim != 2 or means.shape[1] != 2:
            raise InvalidInput("displacement means need shape N by 2")
        count = means.shape[0]
    elif isinstance(means, (list, tuple)):
        count = len(means)
    else:
        raise InvalidInput("displacement means must be a bounded matrix")
    if count > MAX_STEPS:
        raise InvalidInput("displacement kernel exceeds the 100-step bound")
    if not isinstance(factors, np.ndarray) or factors.ndim != 3 or factors.shape[:2] != (count, 2):
        raise InvalidInput("displacement factors need shape N by 2 by r")
    width = factors.shape[2]
    if width > MAX_FACTOR_COLUMNS:
        raise InvalidInput("displacement factor width exceeds the 338-column bound")
    return count, width


def displacement_moments(means, factors):
    """Analytic moments for old-step (theta,v) rows of one physical Gaussian.

    `means` has shape (N,2), `factors` shape (N,2,r). All rows use the same
    Gaussian factor coordinates; individual step factors are not independent.
    """
    count, width = _moment_shape(means, factors)
    means = _array(means, (count, 2), "displacement means")
    factors = _array(factors, (count, 2, width), "displacement factors")
    diagnostics = {"underflow_counts": {}}
    previous_callback = np.geterrcall()

    def record_underflow(message, flag):
        _count(diagnostics, "numpy_"+message.replace(" ", "_"))

    np.seterrcall(record_underflow)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="call"):
            return _displacement_moments(means, factors, diagnostics)
    except OracleError:
        raise
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise NumericalFailure("displacement moment arithmetic failed", diagnostics) from exc
    finally:
        np.seterrcall(previous_callback)


def _displacement_moments(means, factors, diagnostics):
    count = len(means)
    means_complex = np.empty(count, dtype=np.complex128)
    plus, minus = np.empty((count, count), dtype=np.complex128), np.empty((count, count), dtype=np.complex128)
    phase_variances = np.empty((count, count, 2))  # last index: plus, minus
    blocks = np.empty((count, count, 2, 2))
    for j in range(count):
        theta, speed = factors[j]
        variance = _dot(theta, theta)
        phase = _phase(float(means[j, 0]), variance, diagnostics)
        means_complex[j] = complex(means[j, 1], _dot(speed, theta))*phase
        for k in range(count):
            for index, sign in enumerate((1, -1)):
                row = theta+sign*factors[k, 0]
                phase_variances[j, k, index] = _dot(row, row)
            plus[j, k] = _centered_pair(means[j], means[k], factors[j], factors[k], 1, diagnostics)
            minus[j, k] = _centered_pair(means[j], means[k], factors[j], factors[k], -1, diagnostics)
            cp, cm = plus[j, k], minus[j, k]
            blocks[j, k] = .5*np.array([[(cp+cm).real, (cp-cm).imag],
                                       [(cp+cm).imag, (cm-cp).real]])
    if any(not np.isfinite(value).all() for value in (means_complex, plus, minus, blocks, phase_variances)):
        raise NumericalFailure("nonfinite displacement moments", diagnostics)
    diagnostics["plus_symmetry_residual"] = float(np.max(np.abs(plus-plus.T))) if count else 0.
    diagnostics["minus_conjugate_residual"] = float(np.max(np.abs(minus-minus.T.conjugate()))) if count else 0.
    diagnostics["block_transpose_residual"] = float(np.max(np.abs(blocks-blocks.transpose(1, 0, 3, 2)))) if count else 0.
    return _freeze({"mean_complex": means_complex, "centered_plus": plus, "centered_minus": minus,
                    "covariance": blocks, "phase_variances": phase_variances, "diagnostics": diagnostics})


def _physical_maps(step_sizes):
    count, dimension = len(step_sizes), 6+3*len(step_sizes)
    times = np.array([math.fsum(step_sizes[:j]) for j in range(count+1)])
    maps = {name: np.zeros((count+1, dimension)) for name in ("theta", "speed", "rate", "bias")}
    for j in range(count+1):
        maps["theta"][j, 2] = 1.
        maps["theta"][j, 4] = times[j]
        maps["speed"][j, 3] = maps["rate"][j, 4] = maps["bias"][j, 5] = 1.
        for k in range(j):
            maps["theta"][j, 6+3*k+1] = times[j]-times[k+1]
            maps["speed"][j, 6+3*k] = 1.
            maps["rate"][j, 6+3*k+1] = 1.
            maps["bias"][j, 6+3*k+2] = 1.
    maps["times_s"] = times
    return maps


def _old_step_rows(physical):
    return np.stack((physical["theta"][:-1], physical["speed"][:-1]), axis=1)


def _condition_physical(prior_mean, prior_factor, observation_rows, values, variances):
    count = len(values)
    if not count:
        return prior_mean.copy(), prior_factor.copy(), 0., {
            "innovation_covariance": np.empty((0, 0)), "innovation_eigenvalues": np.empty(0),
            "innovation_asymmetry": 0., "solve_residual_max_abs": 0., "innovation": np.empty(0)}
    observed_factor = _mm(observation_rows, prior_factor)
    innovation_covariance = _mm(observed_factor, observed_factor.T)+np.diag(variances)
    innovation_covariance, diagnostic = _covariance(innovation_covariance, "observation covariance", produced=True)
    spectrum = diagnostic["eigenvalues"]
    threshold = SUPPORT_RESOLUTION*max(1., float(spectrum[-1]))
    if spectrum[0] <= threshold:
        raise UnresolvedInference("observation covariance is insufficiently resolved",
                                  {"innovation_eigenvalues": spectrum, "resolution_threshold": threshold})
    factor = np.linalg.cholesky(innovation_covariance)
    cross = _mm(prior_factor, observed_factor.T)
    gain = np.linalg.solve(factor.T, np.linalg.solve(factor, cross.T)).T
    innovation = values-_mv(observation_rows, prior_mean)
    posterior_mean = prior_mean+_mv(gain, innovation)
    posterior_factor = np.concatenate((prior_factor-_mm(gain, observed_factor),
                                       gain*np.sqrt(variances)), axis=1)
    whitened = np.linalg.solve(factor, innovation)
    log_evidence = -.5*(count*math.log(2*math.pi)+2*math.fsum(np.log(np.diag(factor)).tolist())
                       +_dot(whitened, whitened))
    residual = _mm(innovation_covariance, gain.T)-cross.T
    return posterior_mean, posterior_factor, log_evidence, {
        "innovation_covariance": innovation_covariance, "innovation_eigenvalues": spectrum,
        "innovation_asymmetry": diagnostic["asymmetry_before_symmetrization"],
        "solve_residual_max_abs": float(np.max(np.abs(residual))), "innovation": innovation,
        "resolution_threshold": threshold}


def _cross_moment(step_mean, step_factor, x_factor, diagnostics):
    # This is E[(X-mu_X)(z-mu_z)], without conjugating z.
    theta, speed = step_factor
    variance, speed_theta = _dot(theta, theta), _dot(speed, theta)
    phase = _phase(float(step_mean[0]), variance, diagnostics)
    speed_mean = complex(step_mean[1], speed_theta)
    return np.array([(_dot(row, speed)+1j*_dot(row, theta)*speed_mean)*phase
                     for row in x_factor], dtype=np.complex128)


def _entry_terms(base, position_cross, pair_terms, latent_cross, row, column):
    terms = [float(base[row, column])]
    if row < 2 and column < 2:
        for cross in position_cross:
            terms.extend((float(cross[row, column]), float(cross[column, row])))
        for pair in pair_terms.reshape((-1, 2, 2)):
            terms.append(float(pair[row, column]))
    elif row < 2 <= column:
        terms.extend(float(cross[row, column-2]) for cross in latent_cross)
    elif column < 2 <= row:
        terms.extend(float(cross[column, row-2]) for cross in latent_cross)
    return terms


@dataclass(frozen=True)
class MomentResult:
    mean: np.ndarray
    covariance: np.ndarray
    log_evidence: float
    timestamp_us: int
    phase: str
    diagnostics: object
    physical: object
    _base_factor: np.ndarray
    _position_cross: np.ndarray
    _pair_terms: np.ndarray
    _latent_cross: np.ndarray
    status: str = "completed"

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))

    def directional_variance(self, direction):
        """Project retained assembly terms independently of the final 6x6 sum."""
        direction = _array(direction, (6,), "direction")
        position, latent = direction[:2], direction[2:]
        diagnostic = {"underflow_counts": {}}
        previous_callback = np.geterrcall()

        def record_underflow(message, flag):
            _count(diagnostic, "numpy_"+message.replace(" ", "_"))

        np.seterrcall(record_underflow)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="call"):
                projected_factor = np.einsum('i,ij->j', direction, self._base_factor, optimize=False)
                terms = [_dot(projected_factor, projected_factor)]
                for cross in self._position_cross:
                    term = _dot(position, _mv(cross, position))
                    terms.extend((term, term))
                for pair in self._pair_terms.reshape((-1, 2, 2)):
                    terms.append(_dot(position, _mv(pair, position)))
                for cross in self._latent_cross:
                    term = _dot(position, _mv(cross, latent))
                    terms.extend((term, term))
                direct = math.fsum(terms)
                matrix = _dot(direction, _mv(self.covariance, direction))
                absolute = math.fsum(abs(term) for term in terms)
                if not all(math.isfinite(value) for value in (direct, matrix, absolute)):
                    raise NumericalFailure("nonfinite directional variance")
        except (FloatingPointError, OverflowError) as exc:
            raise NumericalFailure("directional projection arithmetic failed", diagnostic) from exc
        finally:
            np.seterrcall(previous_callback)
        return _freeze({"direction": direction, "direct_term_sum": direct,
                        "matrix_projection": matrix, "absolute_term_sum": absolute,
                        "difference": direct-matrix, "precision_certified": False,
                        "underflow_counts": diagnostic["underflow_counts"]})


def _assemble(mean, covariance, factor, diffusion, received, sizes, event_steps, checkpoint_us, diagnostic):
    count, dimension = len(sizes), 6+3*len(sizes)
    positive = [index for index, value in enumerate(diffusion) if value > 0]
    prior_factor = np.zeros((dimension, factor.shape[1]+len(positive)*count))
    prior_factor[:6, :factor.shape[1]] = factor
    prior_mean = np.zeros(dimension)
    prior_mean[:6] = mean
    column = factor.shape[1]
    process_layout = []
    for step, size in enumerate(sizes):
        for coordinate in positive:
            variance = float(diffusion[coordinate])*size
            if variance == 0:
                _count(diagnostic, "process_variance_to_zero")
                raise UnresolvedInference("positive process variance underflowed to zero",
                                          {"step": step, "coordinate": coordinate})
            if 0 < variance < np.finfo(np.float64).tiny:
                _count(diagnostic, "process_variance_subnormal")
            scale = math.sqrt(variance)
            if scale == 0:
                raise UnresolvedInference("positive process scale underflowed to zero")
            prior_factor[6+3*step+coordinate, column] = scale
            process_layout.append((step, coordinate, column))
            column += 1
    physical = _physical_maps(sizes)
    observations = np.zeros((len(received), dimension))
    for index, (event, step) in enumerate(zip(received, event_steps)):
        observations[index] = (physical["rate"][step]+physical["bias"][step]
                               if event.sensor == "gyro" else physical["speed"][step])
    values = np.array([event.value for event in received])
    variances = np.array([event.variance for event in received])
    post_mean, post_factor, evidence, conditioning = _condition_physical(
        prior_mean, prior_factor, observations, values, variances)
    if not np.isfinite(post_mean).all() or not np.isfinite(post_factor).all() or not math.isfinite(evidence):
        raise NumericalFailure("nonfinite conditioned physical law")
    diagnostic["conditioning"] = conditioning
    old_rows = _old_step_rows(physical)
    step_mean = np.einsum('nij,j->ni', old_rows, post_mean, optimize=False)
    step_factor = np.einsum('nij,jk->nik', old_rows, post_factor, optimize=False)
    if not np.isfinite(step_mean).all() or not np.isfinite(step_factor).all():
        raise NumericalFailure("nonfinite physical displacement map", diagnostic)
    moments = displacement_moments(step_mean, step_factor)
    diagnostic["displacement"] = moments["diagnostics"]
    diagnostic["phase_variances"] = moments["phase_variances"]
    x_rows = np.zeros((6, dimension))
    x_rows[0, 0] = x_rows[1, 1] = 1.
    for row, name in enumerate(("theta", "speed", "rate", "bias"), start=2):
        x_rows[row] = physical[name][-1]
    x_mean, x_factor = _mv(x_rows, post_mean), _mm(x_rows, post_factor)
    base = _mm(x_factor, x_factor.T)
    endpoint_mean = x_mean.copy()
    rates = np.column_stack((moments["mean_complex"].real, moments["mean_complex"].imag))
    for coordinate in range(2):
        endpoint_mean[coordinate] = math.fsum([float(x_mean[coordinate])]+
            [size*float(rate[coordinate]) for size, rate in zip(sizes, rates)])
    position_cross = np.empty((count, 2, 2))
    latent_cross = np.empty((count, 2, 4))
    for j, size in enumerate(sizes):
        cross = _cross_moment(step_mean[j], step_factor[j], x_factor, diagnostic)
        real_cross = np.column_stack((cross.real, cross.imag))
        position_cross[j] = size*real_cross[:2]
        latent_cross[j] = size*real_cross[2:].T
    step_array = np.array(sizes)
    pair_terms = moments["covariance"]*step_array[:, None, None, None]*step_array[None, :, None, None]
    endpoint_covariance, absolute_terms = np.empty((6, 6)), np.empty((6, 6))
    for row in range(6):
        for column in range(6):
            terms = _entry_terms(base, position_cross, pair_terms, latent_cross, row, column)
            endpoint_covariance[row, column] = math.fsum(terms)
            absolute_terms[row, column] = math.fsum(abs(value) for value in terms)
    endpoint_covariance, covariance_checks = _covariance(endpoint_covariance, "endpoint covariance", produced=True)
    if not np.isfinite(endpoint_mean).all() or not np.isfinite(absolute_terms).all():
        raise NumericalFailure("nonfinite endpoint mean or cancellation diagnostic")
    diagnostic.update(endpoint_covariance=covariance_checks, covariance_absolute_term_sums=absolute_terms,
                      precision_certified=False, precision_status="not certified",
                      interpretation="completed algebraic moments do not certify directional precision or calibration",
                      dimensions={"steps": count, "events": len(received), "physical": dimension,
                                  "prior_factor_columns": prior_factor.shape[1], "posterior_factor_columns": post_factor.shape[1]})
    physical.update(prior_mean=prior_mean, prior_factor=prior_factor, posterior_mean=post_mean,
        posterior_factor=post_factor, observation_rows=observations, observation_values=values,
        observation_variances=variances, observation_step_indices=np.array(event_steps, dtype=int),
        step_sizes_s=np.array(sizes), process_layout=tuple(process_layout),
        checkpoint_covariance_supplied=covariance, checkpoint_factor=factor)
    return MomentResult(endpoint_mean, endpoint_covariance, float(evidence),
        received[-1].timestamp_us if received else checkpoint_us,
        received[-1].sensor if received else "checkpoint", diagnostic, physical,
        x_factor, position_cross, pair_terms, latent_cross)


def condition_prefix(mean, covariance, factor, diffusion, events, *, checkpoint_us=0):
    """Return full six-state moments for one causal received prefix.

    The input factor B0 has six physical rows and at most six declared columns.
    P0 is a redundant consistency check. Dimensions are checked before dense
    construction; positive diffusion is never truncated to choose a rank.
    """
    received, sizes, event_steps = _schedule(events, checkpoint_us)
    width = _factor_width(factor)
    mean = _array(mean, (6,), "checkpoint mean")
    covariance = _array(covariance, (6, 6), "checkpoint covariance")
    factor = _array(factor, (6, width), "checkpoint factor")
    diffusion = _array(diffusion, (3,), "diffusion")
    if np.any(diffusion < 0):
        raise InvalidInput("diffusion must be nonnegative")
    diagnostic = {"underflow_counts": {}}
    previous_callback = np.geterrcall()

    def record_underflow(message, flag):
        _count(diagnostic, "numpy_"+message.replace(" ", "_"))

    np.seterrcall(record_underflow)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="call"):
            _, covariance_checks = _covariance(covariance, "checkpoint covariance", produced=False)
            factor_covariance = _mm(factor, factor.T)
            if not np.isfinite(factor_covariance).all():
                raise NumericalFailure("checkpoint factor Gram arithmetic overflowed", diagnostic)
            residual = factor_covariance-covariance
            maximum = float(np.max(np.abs(residual)))
            diagnostic.update(checkpoint_covariance=covariance_checks,
                factor_covariance=factor_covariance, factor_covariance_residual=residual,
                factor_covariance_residual_max_abs=maximum,
                factor_law="all calculations, including the empty prefix, use B0 B0^T")
            if maximum > ROUND_OFF*max(1., float(np.max(np.abs(covariance)))):
                raise InvalidInput("declared factor is inconsistent with covariance", diagnostic)
            return _assemble(mean, covariance, factor, diffusion, received, sizes,
                             event_steps, checkpoint_us, diagnostic)
    except OracleError:
        raise
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError, ValueError) as exc:
        raise NumericalFailure(f"physical Gaussian moment arithmetic failed: {exc}", diagnostic) from exc
    finally:
        np.seterrcall(previous_callback)
