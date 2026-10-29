"""Fixed-precision interval arithmetic and the frozen moment eligibility rules.

This module has no estimator or artifact dependency. The only numerical backend
is python-flint 0.9.0, at 256 bits. A ball is never replaced by its midpoint for a
decision; midpoint binary64 arithmetic only constructs the tentative scale T.
"""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from fractions import Fraction
import math
import os
import struct
from types import MappingProxyType

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import flint
flint.ctx.threads = 1
flint.ctx.prec = 256

arb = flint.arb
acb = flint.acb
arb_mat = flint.arb_mat
PRECISION_BITS = 256
BACKEND_VERSION = "0.9.0"
MAX_DYADIC_EXPONENT = 1_000_000


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


class InvalidInput(ValueError):
    status = "invalid"

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = freeze({} if diagnostics is None else diagnostics)


class UnresolvedArithmetic(ArithmeticError):
    status = "unresolved"

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = freeze({} if diagnostics is None else diagnostics)


def check_context():
    if (flint.__version__ != BACKEND_VERSION or flint.ctx.prec != PRECISION_BITS
            or flint.ctx.threads != 1):
        raise UnresolvedArithmetic("the pinned backend context changed")


def ratio(value):
    """Read an exact rational or a finite binary64 value, without decimal text."""
    if isinstance(value, bool):
        raise InvalidInput("boolean is not a numerical input")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    # numpy float64 inputs provide as_integer_ratio; complex values do not.
    if isinstance(value, float) or type(value).__name__ in ("float64", "float32"):
        try:
            numerator, denominator = value.as_integer_ratio()
        except (ValueError, OverflowError) as exc:
            raise InvalidInput("a finite real input is required") from exc
        return Fraction(numerator, denominator)
    raise InvalidInput("expected an integer, binary64 value or exact Fraction")


def exact(value):
    """Enclose the exact input (binary64 inputs retain their integer ratio)."""
    check_context()
    if isinstance(value, arb):
        if not value.is_finite():
            raise UnresolvedArithmetic("nonfinite ball")
        return value
    q = ratio(value)
    result = arb(q.numerator) / arb(q.denominator)
    if not result.is_finite():
        raise UnresolvedArithmetic("input conversion is not finite")
    return result


def _dyadic(value):
    # This resource limit bounds exact Fraction allocation, not Arb precision.
    # Refuse before shifting; never replace a tiny value by zero.
    mantissa, exponent = value.man_exp()
    mantissa, exponent = int(mantissa), int(exponent)
    if abs(exponent) > MAX_DYADIC_EXPONENT:
        raise UnresolvedArithmetic("dyadic endpoint exceeds the fixed representation bound",
                                   {"mantissa": str(mantissa), "exponent": exponent,
                                    "maximum_absolute_exponent": MAX_DYADIC_EXPONENT})
    return (Fraction(mantissa << exponent) if exponent >= 0
            else Fraction(mantissa, 1 << -exponent))


def bounds(value):
    """Exact rational endpoints of the stored midpoint/radius, not floats."""
    value = exact(value)
    midpoint, radius = _dyadic(value.mid()), _dyadic(value.rad())
    return midpoint-radius, midpoint+radius


def interval(lower, upper):
    check_context()
    lower, upper = ratio(lower), ratio(upper)
    if lower > upper:
        raise InvalidInput("interval endpoints are reversed")
    midpoint, radius = (lower+upper)/2, (upper-lower)/2
    # Both constructor arguments use rational text, never a rounded float.
    result = arb(f"{midpoint.numerator}/{midpoint.denominator}",
                 f"{radius.numerator}/{radius.denominator}")
    lo, hi = bounds(result)
    if lo > lower or hi < upper:
        raise UnresolvedArithmetic("interval constructor lost an endpoint")
    return result


def width(value):
    lo, hi = bounds(value)
    return hi-lo


def _square_bounds(value):
    lo, hi = bounds(value)
    lower = Fraction(0) if lo <= 0 <= hi else min(lo*lo, hi*hi)
    return lower, max(lo*lo, hi*hi)


def square_range(value):
    """Enclose the mathematical square range; sums use its exact endpoints."""
    return interval(*_square_bounds(value))


def abs_range(value):
    lo, hi = bounds(value)
    lower = 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
    return interval(lower, max(abs(lo), abs(hi)))


def sqrt_range(value):
    """Square root of a proved nonnegative range, using endpoint roots."""
    lo, hi = bounds(value)
    if lo < 0:
        raise UnresolvedArithmetic("square-root domain is not proved")
    low_root, high_root = exact(lo).sqrt(), exact(hi).sqrt()
    return interval(bounds(low_root)[0], bounds(high_root)[1])


def _norm(values):
    # Sum exact endpoint bounds to avoid a radius rounding below known zero.
    ranges = [_square_bounds(value) for value in values]
    lo = sum((item[0] for item in ranges), Fraction(0))
    hi = sum((item[1] for item in ranges), Fraction(0))
    low_root, high_root = exact(lo).sqrt(), exact(hi).sqrt()
    return interval(bounds(low_root)[0], bounds(high_root)[1])


def serialize_ball(value):
    """JSON-ready exact dyadic endpoints; the radius is never discarded."""
    lower, upper = bounds(value)
    def endpoint(q):
        exponent = -(q.denominator.bit_length()-1)
        if q.denominator != 1 << -exponent:
            raise UnresolvedArithmetic("stored ball endpoint is not dyadic")
        return {"mantissa": str(q.numerator), "exponent": exponent}
    return {"lower": endpoint(lower), "upper": endpoint(upper)}


def deserialize_ball(data):
    if not isinstance(data, Mapping) or set(data) != {"lower", "upper"}:
        raise InvalidInput("invalid serialized ball schema")
    def endpoint(item):
        if not isinstance(item, Mapping) or set(item) != {"mantissa", "exponent"}:
            raise InvalidInput("invalid dyadic endpoint schema")
        if not isinstance(item["mantissa"], str) or type(item["exponent"]) is not int:
            raise InvalidInput("dyadic endpoint requires integer text and exponent")
        try:
            mantissa = int(item["mantissa"])
        except ValueError as exc:
            raise InvalidInput("invalid dyadic mantissa") from exc
        exponent = item["exponent"]
        if abs(exponent) > MAX_DYADIC_EXPONENT:
            raise UnresolvedArithmetic("serialized endpoint exceeds the fixed representation bound",
                                       {"mantissa": item["mantissa"], "exponent": exponent})
        return (Fraction(mantissa << exponent) if exponent >= 0
                else Fraction(mantissa, 1 << -exponent))
    return interval(endpoint(data["lower"]), endpoint(data["upper"]))


def serialize(value):
    """Recursively convert immutable numerical evidence to JSON-ready data."""
    if isinstance(value, arb):
        return serialize_ball(value)
    if isinstance(value, acb):
        return {"real": serialize_ball(value.real), "imag": serialize_ball(value.imag)}
    if isinstance(value, Fraction):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    return value


def _power_two(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _floor_log2(q):
    exponent = q.numerator.bit_length()-q.denominator.bit_length()
    if q < _power_two(exponent):
        exponent -= 1
    return exponent


def binary64_round(value):
    """Independent rational round-to-nearest, ties-to-even, including subnormals."""
    q = ratio(value)
    sign = 1 if q < 0 else 0
    q = abs(q)
    if not q:
        return 0.0
    exponent = _floor_log2(q)
    quantum = max(-1074, exponent-52)
    scaled = q/_power_two(quantum)
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2*remainder
    if twice > scaled.denominator or (twice == scaled.denominator and integer % 2):
        integer += 1
    if not integer:
        return struct.unpack(">d", struct.pack(">Q", sign << 63))[0]
    if quantum == -1074 and integer < 1 << 52:
        bits = (sign << 63) | integer
    else:
        if integer == 1 << 53:
            integer >>= 1
            quantum += 1
        exponent = quantum+52
        if exponent > 1023:
            raise UnresolvedArithmetic("binary64 rounding overflow")
        bits = (sign << 63) | ((exponent+1023) << 52) | (integer-(1 << 52))
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def binary64_sqrt(value):
    """Correctly rounded root, verified with integer midpoint-square comparisons."""
    q = ratio(value)
    if q < 0:
        raise InvalidInput("negative root input")
    if not q:
        return 0.0
    exponent = _floor_log2(q)//2
    quantum = max(-1074, exponent-52)
    scaled = q/_power_two(2*quantum)
    integer = math.isqrt(scaled.numerator//scaled.denominator)
    difference = 4*scaled.numerator-scaled.denominator*(2*integer+1)**2
    if difference > 0 or (difference == 0 and integer % 2):
        integer += 1
    return binary64_round(integer*_power_two(quantum))


def sequence(value, length, name):
    """Validate a bounded container before converting any entries."""
    shape = getattr(value, "shape", None)
    if shape is not None and tuple(shape) != (length,):
        raise InvalidInput(f"{name} must have length {length}")
    if not isinstance(value, (list, tuple)) and shape is None:
        raise InvalidInput(f"{name} must be a bounded sequence")
    if len(value) != length:
        raise InvalidInput(f"{name} must have length {length}")
    return tuple(value)


def matrix(value, rows, columns=None, name="matrix", *, symmetric=False):
    columns = rows if columns is None else columns
    shape = getattr(value, "shape", None)
    if shape is not None:
        if tuple(shape) != (rows, columns):
            raise InvalidInput(f"{name} must have shape {(rows, columns)}")
    elif not isinstance(value, (list, tuple)) or len(value) != rows:
        raise InvalidInput(f"{name} must have {rows} rows")
    result = tuple(tuple(exact(item) for item in sequence(row, columns, name))
                   for row in value)
    if symmetric:
        if rows != columns or any(bounds(result[i][j]) != bounds(result[j][i])
                                  for i in range(rows) for j in range(i)):
            raise InvalidInput(f"{name} must be exactly symmetric")
    return result


def symmetrize_binary64(value):
    size = len(value)
    if size < 1 or size > 6:
        raise InvalidInput("checkpoint matrix dimension must be at most six")
    matrix(value, size, name="checkpoint covariance")
    half = Fraction(1, 2)
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            left = ratio(binary64_round(half*ratio(value[i][j])))
            right = ratio(binary64_round(half*ratio(value[j][i])))
            row.append(binary64_round(left+right))
        result.append(tuple(row))
    return tuple(result)


def _dimension(value, maximum=6):
    try:
        size = len(value)
    except TypeError as exc:
        raise InvalidInput("expected a bounded matrix") from exc
    if size < 1 or size > maximum:
        raise InvalidInput(f"matrix dimension must be between one and {maximum}")
    return size


def _cholesky(value):
    size = len(value)
    lower = [[arb(0) for _ in range(size)] for _ in range(size)]
    pivots = []
    for i in range(size):
        for j in range(i+1):
            residual = value[i][j]-sum((lower[i][k]*lower[j][k] for k in range(j)), arb(0))
            if i == j:
                if bounds(residual)[0] <= 0:
                    raise UnresolvedArithmetic("positive Cholesky pivot is not proved",
                                               {"pivot_index": i, "pivot": residual})
                pivots.append(residual)
                lower[i][j] = residual.sqrt()
            else:
                if bounds(lower[j][j])[0] <= 0:
                    raise UnresolvedArithmetic("Cholesky division domain is not proved")
                lower[i][j] = residual/lower[j][j]
    return tuple(tuple(row) for row in lower), tuple(pivots)


def verified_cholesky(value, *, require_resolution=False):
    check_context()
    size = _dimension(value, 32)
    covariance = matrix(value, size, symmetric=True)
    lower, pivots = _cholesky(covariance)
    upper = bounds(_norm(item for row in covariance for item in row))[1]
    shift = ratio(1e-12)*max(Fraction(1), upper)
    shifted_pivots = ()
    if require_resolution:
        shifted = tuple(tuple(covariance[i][j]-(exact(shift) if i == j else arb(0))
                              for j in range(size)) for i in range(size))
        _, shifted_pivots = _cholesky(shifted)
    return freeze({"lower": lower, "pivots": pivots,
                   "frobenius_upper": exact(upper), "resolution_shift": exact(shift),
                   "shifted_pivots": shifted_pivots,
                   "resolution_proved": bool(require_resolution)})


def triangular_solve(lower, right):
    """Verified forward solve; right is a vector or a matrix of columns."""
    size = _dimension(lower, 32)
    lower = matrix(lower, size, name="triangular factor")
    if any(not lower[i][j].is_zero() for i in range(size) for j in range(i+1, size)):
        raise InvalidInput("factor must be lower triangular")
    if not isinstance(right, (list, tuple)) or len(right) != size:
        raise InvalidInput("triangular right-hand side must be a bounded sequence")
    vector = not isinstance(right[0], (list, tuple))
    rhs = tuple((exact(item),) for item in right) if vector else tuple(tuple(row) for row in right)
    if len(rhs) != size or not rhs or not rhs[0]:
        raise InvalidInput("triangular right-hand side shape mismatch")
    columns = len(rhs[0])
    if columns > 206:
        raise InvalidInput("triangular right-hand side exceeds 206 columns")
    if any(len(row) != columns for row in rhs):
        raise InvalidInput("ragged triangular right-hand side")
    result = [[arb(0) for _ in range(columns)] for _ in range(size)]
    for i in range(size):
        if bounds(lower[i][i])[0] <= 0:
            raise UnresolvedArithmetic("positive triangular divisor is not proved")
        for column in range(columns):
            result[i][column] = (rhs[i][column]-sum(
                (lower[i][j]*result[j][column] for j in range(i)), arb(0)))/lower[i][i]
    return tuple(row[0] for row in result) if vector else tuple(tuple(row) for row in result)


def _transpose(value):
    return tuple(zip(*value))


def _scaled_covariance(lower, value):
    return _transpose(triangular_solve(lower, _transpose(triangular_solve(lower, value))))


def _binary64_cholesky(covariance):
    # Only a tentative coordinate scale. Exact binary64 entries are re-embedded
    # and independently certified below; this computation establishes no bound.
    size = len(covariance)
    midpoint = [[binary64_round(_dyadic(covariance[i][j].mid()))
                 for j in range(size)] for i in range(size)]
    lower = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i+1):
            residual = midpoint[i][j]
            for k in range(j):
                product = binary64_round(ratio(lower[i][k])*ratio(lower[j][k]))
                residual = binary64_round(ratio(residual)-ratio(product))
            if i == j:
                if residual <= 0:
                    raise UnresolvedArithmetic("tentative binary64 scale is not positive")
                lower[i][j] = binary64_sqrt(residual)
            else:
                lower[i][j] = binary64_round(ratio(residual)/ratio(lower[j][j]))
    return tuple(tuple(exact(item) for item in row) for row in lower)


def _scale_context(reference_mean, reference_covariance):
    size = _dimension(reference_covariance)
    covariance = matrix(reference_covariance, size, symmetric=True)
    mean = tuple(exact(item) for item in sequence(reference_mean, size, "reference mean"))
    conditioning = verified_cholesky(covariance, require_resolution=True)
    scale = _binary64_cholesky(covariance)
    normalized = _scaled_covariance(scale, covariance)
    residual = tuple(tuple(normalized[i][j]-(1 if i == j else 0)
                           for j in range(size)) for i in range(size))
    alpha = Fraction(1)-bounds(_norm(item for row in residual for item in row))[1]
    if alpha <= 0:
        raise UnresolvedArithmetic("positive all-direction denominator is not proved")
    return mean, covariance, scale, exact(alpha), conditioning


def _difference_bounds(mean_difference, covariance_difference, scale, alpha):
    scaled_mean = triangular_solve(scale, mean_difference)
    scaled_covariance = _scaled_covariance(scale, covariance_difference)
    gamma = exact(bounds(_norm(scaled_mean))[1])
    beta = exact(bounds(_norm(item for row in scaled_covariance for item in row))[1])
    mean_bound = exact(bounds(gamma/alpha.sqrt())[1])
    covariance_bound = exact(bounds(beta/alpha)[1])
    return freeze({"mean_bound": mean_bound, "covariance_bound": covariance_bound,
                   "beta": beta, "gamma": gamma})


def _subtract_vectors(left, right):
    return tuple(a-b for a, b in zip(left, right))


def _subtract_matrices(left, right):
    return tuple(_subtract_vectors(a, b) for a, b in zip(left, right))


def certificate(point_mean, point_covariance, reference_mean, reference_covariance):
    """Standalone all-direction certificate for dimensions one through six."""
    check_context()
    size = _dimension(reference_covariance)
    point_mean = tuple(exact(item) for item in sequence(point_mean, size, "point mean"))
    point_covariance = matrix(point_covariance, size, symmetric=True)
    try:
        mean, covariance, scale, alpha, conditioning = _scale_context(reference_mean, reference_covariance)
        error = _difference_bounds(_subtract_vectors(point_mean, mean),
                                   _subtract_matrices(point_covariance, covariance), scale, alpha)
        uncertainty = _difference_bounds(tuple(item.mid()-item for item in mean),
            tuple(tuple(item.mid()-item for item in row) for row in covariance), scale, alpha)
        good_width = (bounds(uncertainty["mean_bound"])[1] <= ratio(1e-10)
                      and bounds(uncertainty["covariance_bound"])[1] <= ratio(1e-10))
        good_error = (bounds(error["mean_bound"])[1] <= ratio(1e-7)
                      and bounds(error["covariance_bound"])[1] <= ratio(1e-7))
        return freeze({"status": "eligible" if good_width and good_error else "withheld_precision",
                       "reason": "proved" if good_width and good_error else
                       ("reference_width" if not good_width else "point_error_bound"),
                       "T": scale, "alpha": alpha, "conditioning": conditioning, **error,
                       "reference_mean_bound": uncertainty["mean_bound"],
                       "reference_covariance_bound": uncertainty["covariance_bound"]})
    except UnresolvedArithmetic as exc:
        return freeze({"status": "withheld_precision", "reason": str(exc),
                       "diagnostics": exc.diagnostics})


def certify_oracle(point_mean, point_covariance, point_log_evidence,
                   canonical, factor_reference, *, point_status="completed", point_diagnostics=None):
    """Keep arithmetic, prior-law and direct total errors on one canonical scale."""
    check_context()
    if point_status != "completed":
        return freeze({"status": "withheld_precision", "reason": "point_oracle_refused",
                       "point_status": point_status, "point_diagnostics": point_diagnostics})
    if canonical.model_law != "canonical" or factor_reference.model_law != "factor":
        raise InvalidInput("eligibility requires the canonical and declared-factor references")
    if canonical.physical["checkpoint_factor"] != factor_reference.physical["checkpoint_factor"]:
        raise InvalidInput("reference declared checkpoint factors differ")
    if canonical.diagnostics["checkpoint_law_difference"] != factor_reference.diagnostics["checkpoint_law_difference"]:
        raise InvalidInput("reference canonical checkpoint covariances differ")
    for name in ("step_sizes_s", "events", "checkpoint_us"):
        if canonical.physical[name] != factor_reference.physical[name]:
            raise InvalidInput("reference prefixes differ")
    if canonical.physical["prior_mean"] != factor_reference.physical["prior_mean"]:
        raise InvalidInput("reference checkpoint means differ")
    if canonical.physical["process_variances"] != factor_reference.physical["process_variances"]:
        raise InvalidInput("reference diffusion laws differ")
    if not canonical.whitening_allowed:
        return freeze({"status": "withheld_precision", "reason": "declared_support_not_full"})
    size = 6
    point_mean = tuple(exact(item) for item in sequence(point_mean, size, "point mean"))
    point_covariance = matrix(point_covariance, size, symmetric=True)
    try:
        mean, covariance, scale, alpha, conditioning = _scale_context(canonical.mean, canonical.covariance)
        uncertainty = _difference_bounds(tuple(item.mid()-item for item in mean),
            tuple(tuple(item.mid()-item for item in row) for row in covariance), scale, alpha)
        def contribution(left_mean, left_covariance, left_evidence, right_mean, right_covariance, right_evidence):
            result = dict(_difference_bounds(_subtract_vectors(left_mean, right_mean),
                          _subtract_matrices(left_covariance, right_covariance), scale, alpha))
            result["evidence_bound"] = exact(bounds(abs_range(exact(left_evidence)-exact(right_evidence)))[1])
            return freeze(result)
        arithmetic = contribution(point_mean, point_covariance, point_log_evidence,
                                   factor_reference.mean, factor_reference.covariance, factor_reference.log_evidence)
        bridge = contribution(factor_reference.mean, factor_reference.covariance, factor_reference.log_evidence,
                              mean, covariance, canonical.log_evidence)
        direct = contribution(point_mean, point_covariance, point_log_evidence,
                              mean, covariance, canonical.log_evidence)
        summed = {name: exact(bounds(arithmetic[name]+bridge[name])[1])
                  for name in ("mean_bound", "covariance_bound", "evidence_bound")}
        total = {name: exact(max(bounds(direct[name])[1], bounds(summed[name])[1])) for name in summed}
        widths_ok = all(bounds(uncertainty[name])[1] <= ratio(1e-10)
                        for name in ("mean_bound", "covariance_bound"))
        errors_ok = (bounds(total["mean_bound"])[1] <= ratio(1e-7)
                     and bounds(total["covariance_bound"])[1] <= ratio(1e-7)
                     and bounds(total["evidence_bound"])[1] <= ratio(1e-10))
        return freeze({"status": "eligible" if widths_ok and errors_ok else "withheld_precision",
                       "reason": "proved" if widths_ok and errors_ok else
                       ("reference_width" if not widths_ok else "total_error_bound"),
                       "T": scale, "alpha": alpha, "conditioning": conditioning,
                       "arithmetic": arithmetic, "law_bridge": bridge, "direct_total": direct,
                       "summed_mean_bound": summed["mean_bound"],
                       "summed_covariance_bound": summed["covariance_bound"],
                       "summed_evidence_bound": summed["evidence_bound"], **total,
                       "reference_mean_bound": uncertainty["mean_bound"],
                       "reference_covariance_bound": uncertainty["covariance_bound"],
                       "point_status": point_status, "point_diagnostics": point_diagnostics})
    except UnresolvedArithmetic as exc:
        return freeze({"status": "withheld_precision", "reason": str(exc),
                       "diagnostics": exc.diagnostics, "point_status": point_status,
                       "point_diagnostics": point_diagnostics})


def normalized_errors(saved_mean, saved_covariance, reference, *,
                      state_order=("px", "py", "yaw", "speed", "yaw_rate", "gyro_bias")):
    """Pure six-state scoring helper; no snapshot/file access or reconstruction."""
    if sequence(state_order, 6, "saved state order") != ("px", "py", "yaw", "speed", "yaw_rate", "gyro_bias"):
        raise InvalidInput("saved state order does not match the physical six-state order")
    if not reference.whitening_allowed:
        raise UnresolvedArithmetic("full-six support is not declared")
    mean = tuple(exact(item) for item in sequence(saved_mean, 6, "saved mean"))
    covariance = matrix(saved_covariance, 6, symmetric=True, name="saved covariance")
    lower = verified_cholesky(reference.covariance, require_resolution=True)["lower"]
    original_mean = _subtract_vectors(mean, reference.mean)
    original_covariance = _subtract_matrices(covariance, reference.covariance)
    normalized = _scaled_covariance(lower, covariance)
    # A verified G encloses the canonical Cholesky. Apply the stated G^-1 C G^-T-I.
    error = tuple(tuple(normalized[i][j]-(1 if i == j else 0) for j in range(6)) for i in range(6))
    # Equivalent transposed entries may have different ball radii. Enclose both
    # expressions in their union using the exact symmetry of the target law.
    symmetric_error = [[arb(0)]*6 for _ in range(6)]
    for i in range(6):
        for j in range(i, 6):
            joined = error[i][j] if i == j else error[i][j].union(error[j][i])
            symmetric_error[i][j] = symmetric_error[j][i] = joined
    error = tuple(tuple(row) for row in symmetric_error)
    return freeze({"mean": triangular_solve(lower, original_mean), "covariance": error,
                   "original_mean": original_mean, "original_covariance": original_covariance, "G": lower})


def group_assessment(errors, epsilon):
    check_context()
    if not isinstance(errors, (tuple, list)) or len(errors) != 4 or any(item is None for item in errors):
        return freeze({"status": "withheld_missing", "reason": "exactly_four_groups_required"})
    epsilon = ratio(epsilon)
    if epsilon not in (ratio(.10), ratio(.20)):
        raise InvalidInput("scalar epsilon must be the frozen mean or covariance limit")
    values = tuple(exact(item) for item in errors)
    average = sum(values, arb(0))/4
    squared = [_square_bounds(item-average) for item in values]
    sum_lo = sum((item[0] for item in squared), Fraction(0))/12
    sum_hi = sum((item[1] for item in squared), Fraction(0))/12
    se = interval(bounds(exact(sum_lo).sqrt())[0], bounds(exact(sum_hi).sqrt())[1])
    four_se = 4*se
    absolute = tuple(bounds(abs_range(item)) for item in values)
    average_abs = bounds(abs_range(average))
    se_lo, se_hi = bounds(four_se)
    widths = tuple(width(item) for item in values)
    width_ok = all(item <= ratio(1e-7) for item in widths) and width(four_se) <= ratio(1e-7)
    failure = average_abs[0]-se_hi > epsilon or any(item[0] > 2*epsilon for item in absolute)
    passing = (se_hi <= epsilon/2 and average_abs[1]+se_hi <= epsilon
               and all(item[1] <= 2*epsilon for item in absolute))
    if failure:
        status, reason = "fail", "certified_failure"
    elif not width_ok:
        status, reason = "withheld_precision", "interval_width"
    elif passing:
        status, reason = "pass", "all_inequalities_proved"
    else:
        status = "inconclusive"
        # A proved precision-limit violation is statistical; a boundary overlap
        # is numerical, even when the intervals happen to be very narrow.
        reason = "statistical_precision" if se_lo > epsilon/2 else "threshold_overlap"
    return freeze({"status": status, "reason": reason, "mean": average, "se": se,
                   "four_se": four_se, "widths": widths, "four_se_width": width(four_se),
                   "epsilon": exact(epsilon), "errors": values, "width_ok": width_ok})


def norm_assessment(value, threshold):
    check_context()
    size = _dimension(value)
    value = matrix(value, size, symmetric=True)
    threshold = ratio(threshold)
    if threshold not in (ratio(.5), ratio(.3)):
        raise InvalidInput("norm threshold must be the frozen individual or group limit")
    column_lower = max(bounds(_norm(value[i][j] for i in range(size)))[0] for j in range(size))
    frobenius_upper = bounds(_norm(item for row in value for item in row))[1]
    row_upper = max(sum((bounds(abs_range(item))[1] for item in row), Fraction(0)) for row in value)
    upper = min(frobenius_upper, row_upper)
    status = "pass" if upper <= threshold else "fail" if column_lower > threshold else "inconclusive"
    return freeze({"status": status, "lower": exact(column_lower), "upper": exact(upper),
                   "threshold": exact(threshold), "frobenius_upper": exact(frobenius_upper),
                   "row_sum_upper": exact(row_upper)})
