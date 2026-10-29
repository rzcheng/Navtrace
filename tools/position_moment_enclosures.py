"""Independent physical raw-moment enclosures for GPS-free prefixes.

The canonical law uses the documented binary64 symmetrization of P10. The
separate factor law uses exact supplied B0 entries. Both use exact q*h and real
old-state Euler propagation. No candidate or binary64 oracle kernel is imported.
"""
from dataclasses import dataclass
from fractions import Fraction
import math

try:
    from . import moment_precision as mp
except ImportError:
    import moment_precision as mp

arb, acb, arb_mat = mp.arb, mp.acb, mp.arb_mat
InvalidInput, UnresolvedArithmetic = mp.InvalidInput, mp.UnresolvedArithmetic
MAX_STEPS = 100
MAX_EVENTS = 32
MAX_PHYSICAL = 306
MAX_MARGINALS = 206
STATE_ORDER = ("px", "py", "yaw", "speed", "yaw_rate", "gyro_bias")


@dataclass(frozen=True)
class Observation:
    timestamp_us: int
    sensor: str
    value: float
    variance: float


@dataclass(frozen=True)
class EnclosureResult:
    mean: tuple
    covariance: tuple
    log_evidence: object
    timestamp_us: int
    model_law: str
    support: str
    whitening_allowed: bool
    diagnostics: object
    physical: object
    status: str = "completed"


def _binary_input(value, name):
    try:
        q = mp.ratio(value)
        result = mp.binary64_round(q)
    except (ValueError, OverflowError, TypeError) as exc:
        raise InvalidInput(f"{name} requires a finite binary64 scalar") from exc
    # Rational inputs are useful for explicit fixture construction only when
    # exactly representable as a binary64 input; otherwise no silent law change.
    if mp.ratio(result) != q:
        raise InvalidInput(f"{name} is not an exact binary64 input")
    return result


def _vector(value, length, name):
    return tuple(_binary_input(item, name) for item in mp.sequence(value, length, name))


def _matrix(value, rows, columns, name):
    shape = getattr(value, "shape", None)
    if shape is not None:
        if tuple(shape) != (rows, columns):
            raise InvalidInput(f"{name} must have shape {(rows, columns)}")
    elif not isinstance(value, (tuple, list)) or len(value) != rows:
        raise InvalidInput(f"{name} must contain {rows} bounded rows")
    return tuple(_vector(row, columns, name) for row in value)


def _factor(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 2 or shape[0] != 6 or shape[1] > 6:
            raise InvalidInput("factor must have shape 6 by r, 0 <= r <= 6")
        columns = shape[1]
    else:
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise InvalidInput("factor must have six bounded rows")
        if not isinstance(value[0], (list, tuple)) or len(value[0]) > 6:
            raise InvalidInput("factor must have at most six columns")
        columns = len(value[0])
    return _matrix(value, 6, columns, "factor")


def checkpoint_factor(j, la, k):
    """Freeze B0 from the published J/La/K definitions in scalar RN order."""
    j, la, k = (_matrix(value, 3, 3, name) for value, name in ((j, "J"), (la, "La"), (k, "K")))
    product = []
    for row in range(3):
        result = []
        for column in range(3):
            total = 0.0
            for inner in range(3):
                term = mp.binary64_round(mp.ratio(k[row][inner])*mp.ratio(la[inner][column]))
                total = mp.binary64_round(mp.ratio(total)+mp.ratio(term))
            result.append(total)
        product.append(tuple(result))
    factor = [[0.0]*6 for _ in range(6)]
    for index, physical_row in enumerate((2, 4, 5)):
        factor[physical_row][:3] = la[index]
    for index, physical_row in enumerate((0, 1, 3)):
        factor[physical_row][:3] = product[index]
        factor[physical_row][3:] = j[index]
    return tuple(tuple(row) for row in factor)


def _event(raw):
    if isinstance(raw, (tuple, list)):
        if len(raw) != 4:
            raise InvalidInput("observation needs exactly four fields")
        timestamp, sensor, value, variance = raw
    else:
        try:
            timestamp, sensor, value, variance = raw.timestamp_us, raw.sensor, raw.value, raw.variance
        except AttributeError as exc:
            raise InvalidInput("observation needs timestamp_us, sensor, value and variance") from exc
    if type(timestamp) is not int or sensor not in ("gyro", "wheel"):
        raise InvalidInput("observation needs an integer timestamp and gyro or wheel sensor")
    value, variance = _binary_input(value, "observation"), _binary_input(variance, "variance")
    if variance <= 0:
        raise InvalidInput("observation variance must be positive")
    return Observation(timestamp, sensor, value, variance)


def schedule(events, checkpoint_us=0):
    """Independent scalar binary64 schedule, with no sorting or report steps."""
    if type(checkpoint_us) is not int:
        raise InvalidInput("checkpoint timestamp must be an integer")
    if isinstance(events, (tuple, list)) and len(events) > MAX_EVENTS:
        raise InvalidInput("prefix exceeds 32 events")
    try:
        iterator = iter(events)
    except TypeError as exc:
        raise InvalidInput("events must be an ordered bounded iterable") from exc
    received, sizes, event_steps = [], [], []
    last_time, last_key = checkpoint_us, (checkpoint_us, 1)
    for index, raw in enumerate(iterator):
        if index >= MAX_EVENTS:
            raise InvalidInput("prefix exceeds 32 events")
        event = _event(raw)
        key = (event.timestamp_us, 0 if event.sensor == "gyro" else 1)
        if event.timestamp_us <= checkpoint_us or key <= last_key:
            raise InvalidInput("checkpoint-time, delayed, duplicate or reordered event")
        delta = event.timestamp_us-last_time
        # int->binary64 is an explicit operation in delta_us*1e-6.
        delta64 = mp.binary64_round(Fraction(delta))
        interval = mp.binary64_round(mp.ratio(delta64)*mp.ratio(1e-6))
        if interval < 0 or interval > 2.0:
            raise InvalidInput("interval exceeds the two-second bound")
        quotient = mp.binary64_round(mp.ratio(interval)/mp.ratio(.01))
        count = math.ceil(quotient)
        if len(sizes)+count > MAX_STEPS:
            raise InvalidInput("prefix exceeds 100 substeps")
        if count:
            h = mp.binary64_round(mp.ratio(interval)/count)
            sizes.extend([h]*count)
        event_steps.append(len(sizes))
        received.append(event)
        last_time, last_key = event.timestamp_us, key
    return mp.freeze({"events": received, "step_sizes_s": sizes, "event_steps": event_steps,
                      "timestamp_us": last_time, "checkpoint_us": checkpoint_us})


def _rows(schedule_data, mean, covariance, factor, diffusion, support):
    sizes = tuple(mp.ratio(h) for h in schedule_data["step_sizes_s"])
    steps = len(sizes)
    dimension = 6+3*steps
    zero = Fraction(0)
    current = [[zero]*dimension for _ in range(4)]
    for row, coordinate in enumerate((2, 3, 4, 5)):
        current[row][coordinate] = Fraction(1)
    theta, speed, rate, bias = [], [], [], []
    for index in range(steps+1):
        theta.append(tuple(current[0]))
        speed.append(tuple(current[1]))
        rate.append(tuple(current[2]))
        bias.append(tuple(current[3]))
        if index < steps:
            h = sizes[index]
            # Real arithmetic: heading uses old rate, then independent end noise.
            current[0] = [a+h*b for a, b in zip(current[0], current[2])]
            current[1][6+3*index] = Fraction(1)
            current[2][7+3*index] = Fraction(1)
            current[3][8+3*index] = Fraction(1)
    observation_rows = []
    for event, index in zip(schedule_data["events"], schedule_data["event_steps"]):
        observation_rows.append(tuple(a+b for a, b in zip(rate[index], bias[index]))
                                if event.sensor == "gyro" else speed[index])
    physical_mean = tuple(mp.ratio(item) for item in mean)+(zero,)*(3*steps)
    prior = [[zero]*dimension for _ in range(dimension)]
    for i in range(6):
        for j in range(6):
            prior[i][j] = covariance[i][j]
    variances = []
    rounding = []
    for index, h in enumerate(sizes):
        row = []
        for channel, q in enumerate(diffusion):
            variance = mp.ratio(q)*h
            prior[6+3*index+channel][6+3*index+channel] = variance
            row.append(variance)
            product = mp.binary64_round(variance)
            root = mp.binary64_sqrt(product)
            effective = mp.ratio(root)**2
            rounding.append({"step": index, "channel": channel, "exact_qh": variance,
                             "rounded_qh": product, "rounded_root": root,
                             "effective_factor_variance": effective,
                             "difference": effective-variance})
        variances.append(tuple(row))
    times = [zero]
    for h in sizes:
        times.append(times[-1]+h)
    map_rounding = []
    # The old point oracle used rounded fsum values and rounded subtractions.
    rounded_times = tuple(mp.binary64_round(item) for item in times)
    for j in range(steps+1):
        map_rounding.append({"step": j, "exact_time": times[j], "rounded_time": rounded_times[j],
                             "time_difference": mp.ratio(rounded_times[j])-times[j],
                             "coefficient_differences": tuple(
                                 mp.ratio(mp.binary64_round(mp.ratio(rounded_times[j])-mp.ratio(rounded_times[k+1])))
                                 -(times[j]-times[k+1]) for k in range(j))})
    return mp.freeze({"prior_mean": physical_mean, "prior_covariance": prior,
                      "theta_rows": theta, "speed_rows": speed, "rate_rows": rate, "bias_rows": bias,
                      "observation_rows": observation_rows, "checkpoint_factor": factor,
                      "checkpoint_covariance": covariance, "process_variances": variances,
                      "step_sizes_s": schedule_data["step_sizes_s"], "event_steps": schedule_data["event_steps"],
                      "support": support, "diffusion_rounding": rounding, "map_rounding": map_rounding,
                      "dimension": dimension})


def _old_step_rows(physical):
    return tuple(zip(physical["theta_rows"][:-1], physical["speed_rows"][:-1]))


def _constant_row(row, physical):
    """Algebraic support proof. A ball containing zero is never such a proof."""
    for index, variance_row in enumerate(physical["process_variances"]):
        if any(row[6+3*index+channel] and variance_row[channel] for channel in range(3)):
            return False
    support = physical["support"]
    if support == "full":
        return not any(row[:6])  # checkpoint SPD was independently proved.
    if support in ("diagonal", "deterministic"):
        covariance = physical["checkpoint_covariance"]
        return not any(row[i] and covariance[i][i] for i in range(6))
    factor = physical["checkpoint_factor"]
    return all(sum((row[i]*factor[i][column] for i in range(6)), Fraction(0)) == 0
               for column in range(len(factor[0])))


def _symmetric_hull(value):
    """Use the analytically symmetric target to retain both computed bounds."""
    size = len(value)
    result = [[arb(0)]*size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            joined = value[i][j] if i == j else value[i][j].union(value[j][i])
            result[i][j] = result[j][i] = joined
    return tuple(tuple(row) for row in result)


def _condition_marginals(prior_mean, prior_covariance, selector, observation_rows, values, variances):
    """One verified physical Gaussian conditioning, using a joint RHS solve."""
    dimension, selected, count = len(prior_mean), len(selector), len(values)
    p = arb_mat([[mp.exact(item) for item in row] for row in prior_covariance])
    f = arb_mat([[mp.exact(item) for item in row] for row in selector])
    mu = arb_mat([[mp.exact(item)] for item in prior_mean])
    f_mu, f_p = f*mu, f*p
    covariance = f_p*f.transpose()
    if not count:
        mean = tuple(f_mu[i, 0] for i in range(selected))
        covariance = _symmetric_hull(tuple(tuple(covariance[i, j] for j in range(selected)) for i in range(selected)))
        return mean, covariance, arb(0), mp.freeze({"observation_count": 0, "log_determinant": arb(0),
                                                   "solve_residual": (), "conditioning": None})
    h = arb_mat([[mp.exact(item) for item in row] for row in observation_rows])
    h_p = h*p
    innovation = h_p*h.transpose()
    for i in range(count):
        innovation[i, i] += mp.exact(variances[i])
    symmetric_s = _symmetric_hull(tuple(tuple(innovation[i, j] for j in range(count)) for i in range(count)))
    conditioning = mp.verified_cholesky(symmetric_s, require_resolution=True)
    innovation = arb_mat([list(row) for row in symmetric_s])
    cross = f_p*h.transpose()
    prior_y = h*mu
    rhs = arb_mat([[mp.exact(values[i])-prior_y[i, 0]]+[cross[j, i] for j in range(selected)]
                   for i in range(count)])
    try:
        solved = innovation.solve(rhs, algorithm="lu")
    except (ZeroDivisionError, ValueError) as exc:
        raise UnresolvedArithmetic("verified observation solve did not resolve",
                                   {"conditioning": conditioning}) from exc
    residual = innovation*solved-rhs
    residual_rows = tuple(tuple(mp.exact(residual[i, j]) for j in range(selected+1)) for i in range(count))
    if any(not item.contains(0) for row in residual_rows for item in row):
        raise UnresolvedArithmetic("verified solve residual excludes zero", {"solve_residual": residual_rows})
    delta = cross*solved
    mean = tuple(mp.exact(f_mu[i, 0]+delta[i, 0]) for i in range(selected))
    posterior = _symmetric_hull(tuple(tuple(mp.exact(covariance[i, j]-delta[i, j+1])
                                           for j in range(selected)) for i in range(selected)))
    log_determinant = 2*sum((conditioning["lower"][i][i].log() for i in range(count)), arb(0))
    quadratic = sum((rhs[i, 0]*solved[i, 0] for i in range(count)), arb(0))
    log_evidence = -(count*(2*arb.pi()).log()+log_determinant+quadratic)/2
    return mean, posterior, mp.exact(log_evidence), mp.freeze({
        "observation_count": count, "innovation_covariance": symmetric_s,
        "log_determinant": log_determinant, "quadratic": quadratic,
        "solve_residual": residual_rows, "conditioning": conditioning,
        "solve_algorithm": "lu", "physical_dimension": dimension, "selected_dimension": selected})


def _phase(mean, variance):
    mean, variance = mp.exact(mean), mp.exact(variance)
    if mp.bounds(variance)[1] < 0:
        raise UnresolvedArithmetic("phase variance is proved negative", {"variance": variance})
    # exp is entire. Retain a straddling ball, including its negative lower
    # endpoint; no clipping or zero-support conclusion follows from it.
    result = acb(-variance/2, mean).exp()
    if not result.is_finite():
        raise UnresolvedArithmetic("nonfinite characteristic function")
    return result


def raw_gaussian_first(mean_a, cov_a_phase, mean_phase, var_phase):
    """Raw E[A exp(iT)], with exact input and outward complex arithmetic."""
    result = acb(mp.exact(mean_a), mp.exact(cov_a_phase))*_phase(mean_phase, var_phase)
    if not result.is_finite():
        raise UnresolvedArithmetic("nonfinite first raw moment")
    return result


def raw_gaussian_second(mean_a, mean_b, cov_ab, cov_a_phase, cov_b_phase, mean_phase, var_phase):
    """Raw E[A B exp(iT)], without a centered/expm1 pair kernel."""
    result = (mp.exact(cov_ab)+acb(mp.exact(mean_a), mp.exact(cov_a_phase))
              *acb(mp.exact(mean_b), mp.exact(cov_b_phase)))*_phase(mean_phase, var_phase)
    if not result.is_finite():
        raise UnresolvedArithmetic("nonfinite second raw moment")
    return result


def _raw_pair(left_mean, right_mean, left_covariance, right_covariance, cross_covariance,
              sign, *, phase_constant=False):
    # Pair order is theta,v. Cross covariance rows are left, columns right.
    if sign not in (-1, 1):
        raise InvalidInput("pair sign must be plus or minus one")
    phase_mean = left_mean[0]+sign*right_mean[0]
    if phase_constant:
        phase_variance = left_cross = right_cross = arb(0)
    else:
        phase_variance = left_covariance[0][0]+right_covariance[0][0]+2*sign*cross_covariance[0][0]
        left_cross = left_covariance[1][0]+sign*cross_covariance[1][0]
        right_cross = cross_covariance[0][1]+sign*right_covariance[1][0]
    return raw_gaussian_second(left_mean[1], right_mean[1], cross_covariance[1][1],
                               left_cross, right_cross, phase_mean, phase_variance)


def raw_displacement_pair(left_mean, right_mean, left_covariance, right_covariance, cross_covariance, sign):
    left_mean = tuple(mp.exact(item) for item in mp.sequence(left_mean, 2, "left pair mean"))
    right_mean = tuple(mp.exact(item) for item in mp.sequence(right_mean, 2, "right pair mean"))
    return _raw_pair(left_mean, right_mean, mp.matrix(left_covariance, 2, symmetric=True),
                     mp.matrix(right_covariance, 2, symmetric=True), mp.matrix(cross_covariance, 2), sign)


def _raw_cross(mu_x, mu_theta, mu_v, cov_x_theta, cov_x_v, cov_v_theta, var_theta, *, output_index=None):
    return raw_gaussian_second(mu_x, mu_v, cov_x_v, cov_x_theta, cov_v_theta, mu_theta, var_theta)


def _endpoint_moments(mean, covariance, physical, selector):
    steps = len(physical["step_sizes_s"])
    final_offset = 2+2*steps
    pair_means = [tuple(mean[2+2*j:4+2*j]) for j in range(steps)]
    pair_covariances = [tuple(tuple(covariance[2+2*j+a][2+2*j+b] for b in range(2)) for a in range(2))
                        for j in range(steps)]
    phase_constant = [_constant_row(selector[2+2*j], physical) for j in range(steps)]
    first = [raw_gaussian_first(pair_means[j][1], arb(0) if phase_constant[j] else pair_covariances[j][1][0],
                               pair_means[j][0], arb(0) if phase_constant[j] else pair_covariances[j][0][0])
             for j in range(steps)]
    displacement_mean = [(value.real, value.imag) for value in first]
    h = tuple(mp.exact(value) for value in physical["step_sizes_s"])
    endpoint_mean = [mean[i]+sum((h[j]*displacement_mean[j][i] for j in range(steps)), arb(0)) for i in range(2)]
    endpoint_mean.extend(mean[final_offset:final_offset+4])
    raw = [[arb(0) for _ in range(6)] for _ in range(6)]
    indices = (0, 1)+tuple(range(final_offset, final_offset+4))
    for i in range(6):
        for k in range(6):
            raw[i][k] = covariance[indices[i]][indices[k]]+mean[indices[i]]*mean[indices[k]]
    # Cross terms with conditioned p0 and every final latent component.
    for j in range(steps):
        theta_index, speed_index = 2+2*j, 3+2*j
        for output_index, gaussian_index in enumerate(indices):
            cross = _raw_cross(mean[gaussian_index], pair_means[j][0], pair_means[j][1],
                arb(0) if phase_constant[j] else covariance[gaussian_index][theta_index],
                covariance[gaussian_index][speed_index],
                arb(0) if phase_constant[j] else pair_covariances[j][1][0],
                arb(0) if phase_constant[j] else pair_covariances[j][0][0], output_index=output_index)
            for component, value in enumerate((cross.real, cross.imag)):
                raw[output_index][component] += h[j]*value
                raw[component][output_index] += h[j]*value
    constant_pairs = 0
    # Every ordered pair is included. H+ and H- have different imaginary signs.
    for j in range(steps):
        for k in range(steps):
            cross_covariance = tuple(tuple(covariance[2+2*j+a][2+2*k+b] for b in range(2)) for a in range(2))
            moments = []
            for sign in (1, -1):
                phase_row = tuple(a+sign*b for a, b in zip(selector[2+2*j], selector[2+2*k]))
                constant = _constant_row(phase_row, physical)
                constant_pairs += int(constant)
                moments.append(_raw_pair(pair_means[j], pair_means[k], pair_covariances[j],
                               pair_covariances[k], cross_covariance, sign, phase_constant=constant))
            plus, minus = moments
            block = (((plus.real+minus.real)/2, (plus.imag-minus.imag)/2),
                     ((plus.imag+minus.imag)/2, (minus.real-plus.real)/2))
            for a in range(2):
                for b in range(2):
                    raw[a][b] += h[j]*h[k]*block[a][b]
    result = _symmetric_hull(tuple(tuple(mp.exact(raw[i][j]-endpoint_mean[i]*endpoint_mean[j])
                                        for j in range(6)) for i in range(6)))
    constant_law = all(_constant_row(tuple(Fraction(int(i == j)) for i in range(physical["dimension"])), physical)
                       for j in range(physical["dimension"]))
    before_proof = result
    if constant_law:
        # The entire primitive g is deterministic, hence this endpoint is a
        # constant function. This algebraic proof, not a tolerance, gives zero.
        result = tuple(tuple(arb(0) for _ in range(6)) for _ in range(6))
    return tuple(mp.exact(item) for item in endpoint_mean), result, mp.freeze({
        "ordered_pair_count": steps*steps, "constant_phase_count": constant_pairs,
        "deterministic_endpoint_proved": constant_law,
        "raw_second_moment": raw,
        "covariance_before_constant_proof": before_proof if constant_law else None})


def condition_prefix(mean, covariance, factor, diffusion, events, checkpoint_us=0,
                     *, law="canonical", support="full"):
    """Enclose the same bounded physical prefix under one explicitly named law."""
    mp.check_context()
    if law not in ("canonical", "factor") or support not in ("full", "diagonal", "deterministic", "declared_factor"):
        raise InvalidInput("unknown input law or support declaration")
    if (support == "declared_factor" and law != "factor") or (support == "diagonal" and law != "canonical"):
        raise InvalidInput("support declaration does not match the input law")
    mean = _vector(mean, 6, "checkpoint mean")
    covariance = _matrix(covariance, 6, 6, "checkpoint covariance")
    factor = _factor(factor)
    diffusion = _vector(diffusion, 3, "diffusion")
    if any(item < 0 for item in diffusion):
        raise InvalidInput("diffusion must be nonnegative")
    schedule_data = schedule(events, checkpoint_us)
    input_ratios = tuple(tuple(mp.ratio(item) for item in row) for row in covariance)
    input_scale = max(Fraction(1), max(abs(item) for row in input_ratios for item in row))
    input_asymmetry = max(abs(input_ratios[i][j]-input_ratios[j][i]) for i in range(6) for j in range(6))
    if law == "canonical" and input_asymmetry > mp.ratio(1e-10)*input_scale:
        raise InvalidInput("checkpoint covariance is materially asymmetric",
                           {"asymmetry": input_asymmetry, "scale": input_scale})
    canonical = tuple(tuple(mp.ratio(item) for item in row) for row in mp.symmetrize_binary64(covariance))
    factor_ratios = tuple(tuple(mp.ratio(item) for item in row) for row in factor)
    gram = tuple(tuple(sum((factor_ratios[i][k]*factor_ratios[j][k] for k in range(len(factor[0]))), Fraction(0))
                       for j in range(6)) for i in range(6))
    checkpoint = canonical if law == "canonical" else gram
    checkpoint_certificate = None
    if support == "full":
        checkpoint_certificate = mp.verified_cholesky(checkpoint)
    elif support == "diagonal":
        if any(checkpoint[i][j] != 0 for i in range(6) for j in range(6) if i != j) or any(checkpoint[i][i] < 0 for i in range(6)):
            raise InvalidInput("diagonal support requires exact diagonal nonnegative covariance")
    elif support == "deterministic":
        if any(item != 0 for row in checkpoint for item in row):
            raise InvalidInput("deterministic support requires exact zero checkpoint covariance")
    physical = _rows(schedule_data, mean, checkpoint, factor_ratios, diffusion, support)
    dimension = physical["dimension"]
    p0 = [tuple(Fraction(int(index == coordinate)) for index in range(dimension)) for coordinate in (0, 1)]
    pairs = _old_step_rows(physical)
    selector = tuple(p0+[row for pair in pairs for row in pair]+
                     [physical[name][-1] for name in ("theta_rows", "speed_rows", "rate_rows", "bias_rows")])
    if dimension > MAX_PHYSICAL or len(selector) > MAX_MARGINALS:
        raise InvalidInput("physical or selected marginal dimension exceeds the frozen bound")
    received = schedule_data["events"]
    marginal_mean, marginal_covariance, evidence, conditioning = _condition_marginals(
        physical["prior_mean"], physical["prior_covariance"], selector, physical["observation_rows"],
        tuple(event.value for event in received), tuple(event.variance for event in received))
    endpoint_mean, endpoint_covariance, raw_diagnostics = _endpoint_moments(marginal_mean, marginal_covariance, physical, selector)
    diagnostics = mp.freeze({"backend": mp.BACKEND_VERSION, "precision_bits": mp.PRECISION_BITS,
        "state_order": STATE_ORDER, "support": support, "checkpoint_certificate": checkpoint_certificate,
        "input_asymmetry": input_asymmetry, "input_covariance_scale": input_scale,
        "checkpoint_law_difference": tuple(tuple(gram[i][j]-canonical[i][j] for j in range(6)) for i in range(6)),
        "conditioning": conditioning, "moments": raw_diagnostics,
        "diffusion_rounding": physical["diffusion_rounding"], "map_rounding": physical["map_rounding"],
        "step_count": len(schedule_data["step_sizes_s"]), "event_count": len(received),
        "marginal_count": len(selector), "coordinate_count": dimension})
    returned_physical = dict(physical)
    returned_physical.update({"selector": selector, "marginal_mean": marginal_mean,
                             "marginal_covariance": marginal_covariance,
                             "events": received, "checkpoint_us": checkpoint_us,
                             "timestamp_us": schedule_data["timestamp_us"]})
    return EnclosureResult(endpoint_mean, endpoint_covariance, evidence, schedule_data["timestamp_us"],
                           law, support, support == "full", diagnostics, mp.freeze(returned_physical))
