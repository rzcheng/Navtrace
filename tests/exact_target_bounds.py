"""Independent exact-rational certificates for the frozen deterministic targets.

The 2^-640 grid and every series length are fixed in the repair protocol. This
module uses no oracle or numerical backend. A certificate proves membership only
when all its exact rational endpoints lie inside the supplied oracle endpoints.
"""
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
import math
from types import MappingProxyType

GRID_BITS = 640
INPUT_BITS = 65_536
INTERMEDIATE_BITS = 131_072
OPERATION_LIMIT = 10_000_000
CACHE_LIMIT = 4_096
SERIES_TERMS = 256
GRID = 1 << GRID_BITS
_ACTIVE = ContextVar("exact_target_proof_context", default=None)


class InvalidProofInput(ValueError):
    """Malformed input, rather than an unresolved numerical proof."""


class ProofUnresolved(ArithmeticError):
    """A fixed domain, integer-size or operation budget did not resolve."""


class ProofContext:
    """One fixed-budget context per parametrized pytest test invocation."""

    def __init__(self):
        self._operations = 0
        self._caches = {}
        self._token = None

    @property
    def operations(self):
        return self._operations

    def __enter__(self):
        if _ACTIVE.get() is not None or self._token is not None:
            raise InvalidProofInput("proof contexts must not nest or be reused")
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exception):
        _ACTIVE.reset(self._token)
        # Keep the token as a marker: a spent context cannot reset its budget.
        return False

    def _tick(self):
        if self._operations >= OPERATION_LIMIT:
            raise ProofUnresolved("fixed proof operation budget exhausted")
        self._operations += 1

    def _lookup(self, primitive, key):
        return self._caches.get(primitive, {}).get(key)

    def _remember(self, primitive, key, value):
        cache = self._caches.setdefault(primitive, {})
        # Cache capacity limits storage only. Further fixed requests are
        # computed without storage; domain, bit and operation limits still apply.
        if len(cache) < CACHE_LIMIT:
            cache[key] = value
        return value


def current_context():
    context = _ACTIVE.get()
    if context is None:
        raise ProofUnresolved("a fresh active proof context is required")
    return context


def _tick():
    current_context()._tick()


def _bits(integer):
    return abs(integer).bit_length()


def _size(bits):
    if bits > INTERMEDIATE_BITS:
        raise ProofUnresolved("fixed intermediate integer bit limit exceeded")


def _check(value, limit):
    if max(_bits(value.numerator), _bits(value.denominator)) > limit:
        raise ProofUnresolved("fixed rational input or result bit limit exceeded")
    return value


def q(value):
    """Interpret a finite float as its exact binary64 ratio, never decimal text."""
    if isinstance(value, bool):
        raise InvalidProofInput("booleans are not proof numbers")
    if isinstance(value, Fraction):
        return _check(value, INPUT_BITS)
    if isinstance(value, int):
        if _bits(value) > INPUT_BITS:
            raise ProofUnresolved("fixed rational input bit limit exceeded")
        return Fraction(value)
    if isinstance(value, float):
        try:
            numerator, denominator = value.as_integer_ratio()
        except (ValueError, OverflowError) as error:
            raise InvalidProofInput("proof inputs must be finite") from error
        return _check(Fraction(numerator, denominator), INPUT_BITS)
    raise InvalidProofInput("expected an integer, float or Fraction")


def _compare(left, right):
    _size(_bits(left.numerator)+_bits(right.denominator))
    _size(_bits(right.numerator)+_bits(left.denominator))
    return (left > right)-(left < right)


def _add(left, right):
    _size(max(_bits(left.numerator)+_bits(right.denominator),
              _bits(right.numerator)+_bits(left.denominator))+1)
    _size(_bits(left.denominator)+_bits(right.denominator))
    return _check(left+right, INTERMEDIATE_BITS)


def _subtract(left, right):
    return _add(left, -right)


def _multiply(left, right):
    _size(_bits(left.numerator)+_bits(right.numerator))
    _size(_bits(left.denominator)+_bits(right.denominator))
    return _check(left*right, INTERMEDIATE_BITS)


def _divide(left, right):
    if not right:
        raise ProofUnresolved("division domain contains zero")
    _size(_bits(left.numerator)+_bits(right.denominator))
    _size(_bits(left.denominator)+_bits(right.numerator))
    return _check(left/right, INTERMEDIATE_BITS)


def _shift(integer, count):
    if count < 0:
        raise InvalidProofInput("negative left shift")
    _size(_bits(integer)+count)
    return integer << count


def _integer_power(base, exponent):
    if exponent < 0:
        raise InvalidProofInput("negative integer power")
    _size(_bits(base)*exponent)
    return base**exponent


def _rounded(value, upward):
    _tick()  # one charge before each endpoint rounding, including negative ones
    scaled = _shift(value.numerator, GRID_BITS)
    integer, remainder = divmod(scaled, value.denominator)
    if upward and remainder:
        _size(_bits(integer)+1)
        integer += 1
    return _check(Fraction(integer, GRID), INTERMEDIATE_BITS)


def _minimum(values):
    result = values[0]
    for value in values[1:]:
        if _compare(value, result) < 0:
            result = value
    return result


def _maximum(values):
    result = values[0]
    for value in values[1:]:
        if _compare(value, result) > 0:
            result = value
    return result


@dataclass(frozen=True, init=False)
class Real:
    lo: Fraction
    hi: Fraction

    def __init__(self, lo=0, hi=None):
        current_context()
        if isinstance(lo, Real) and hi is None:
            lower, upper = lo.lo, lo.hi
        else:
            lower, upper = q(lo), q(lo if hi is None else hi)
        if _compare(lower, upper) > 0:
            raise InvalidProofInput("reversed real interval")
        object.__setattr__(self, "lo", lower)
        object.__setattr__(self, "hi", upper)

    @classmethod
    def _bounds(cls, lo, hi, *, rounded):
        if rounded:
            lo, hi = _rounded(lo, False), _rounded(hi, True)
        _check(lo, INTERMEDIATE_BITS)
        _check(hi, INTERMEDIATE_BITS)
        if _compare(lo, hi) > 0:
            raise InvalidProofInput("reversed internal real interval")
        result = object.__new__(cls)
        object.__setattr__(result, "lo", lo)
        object.__setattr__(result, "hi", hi)
        return result

    @property
    def lower(self):
        return self.lo

    @property
    def upper(self):
        return self.hi

    def is_zero(self):
        return self.lo == 0 and self.hi == 0

    def is_exact(self):
        return self.lo == self.hi

    def width(self):
        _tick()
        return _subtract(self.hi, self.lo)

    def contains(self, value):
        value = q(value)
        return _compare(self.lo, value) <= 0 and _compare(value, self.hi) <= 0

    def abs_bounds(self):
        _tick()
        upper = _maximum((abs(self.lo), abs(self.hi)))
        lower = Fraction(0) if self.lo <= 0 <= self.hi else _minimum((abs(self.lo), abs(self.hi)))
        return lower, upper

    def _plus(self, other, *, force_grid=False):
        other = _real(other)
        _tick()
        lower, upper = _add(self.lo, other.lo), _add(self.hi, other.hi)
        exact_sum = self.is_exact() and other.is_exact() and not force_grid
        return Real._bounds(lower, upper, rounded=not exact_sum)

    def __add__(self, other):
        if isinstance(other, Complex):
            return NotImplemented
        return self._plus(other)

    __radd__ = __add__

    def __neg__(self):
        _tick()
        return Real._bounds(-self.hi, -self.lo, rounded=False)

    def __sub__(self, other):
        if isinstance(other, Complex):
            return NotImplemented
        return self+(-_real(other))

    def __rsub__(self, other):
        return _real(other)+(-self)

    def __mul__(self, other):
        if isinstance(other, Complex):
            return NotImplemented
        other = _real(other)
        _tick()
        if self.is_zero() or other.is_zero():
            return Real._bounds(Fraction(0), Fraction(0), rounded=False)
        products = tuple(_multiply(left, right) for left in (self.lo, self.hi)
                         for right in (other.lo, other.hi))
        return Real._bounds(_minimum(products), _maximum(products), rounded=True)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Complex):
            return NotImplemented
        other = _real(other)
        _tick()
        if other.lo <= 0 <= other.hi:
            raise ProofUnresolved("division domain contains zero")
        ratios = tuple(_divide(left, right) for left in (self.lo, self.hi)
                       for right in (other.lo, other.hi))
        return Real._bounds(_minimum(ratios), _maximum(ratios), rounded=True)

    def __rtruediv__(self, other):
        return _real(other)/self

    def __pow__(self, exponent):
        _tick()
        if type(exponent) is not int:
            raise InvalidProofInput("real powers require an integer exponent")
        if _bits(exponent) > INPUT_BITS:
            raise ProofUnresolved("power exponent exceeds input bit limit")
        if exponent < 0:
            return point(1)/(self**(-exponent))
        result, base = point(1), self
        while exponent:
            _tick()
            if exponent & 1:
                result = result*base
            exponent >>= 1
            if exponent:
                base = base*base
        return result

    def exp(self):
        return exp(self)

    def expm1(self):
        return expm1(self)

    def sin(self):
        return sin(self)

    def cos(self):
        return cos(self)

    def log(self):
        return log(self)

    def sqrt(self):
        return sqrt(self)


def _real(value):
    current_context()
    return value if isinstance(value, Real) else point(value)


def point(value=0):
    return Real(value)


def interval(lower, upper):
    return Real(lower, upper)


@dataclass(frozen=True, init=False)
class Complex:
    real: Real
    imag: Real

    def __init__(self, real=0, imag=0):
        current_context()
        if isinstance(real, Complex):
            if not _real(imag).is_zero():
                raise InvalidProofInput("cannot add an imaginary constructor value to a complex certificate")
            real, imag = real.real, real.imag
        object.__setattr__(self, "real", _real(real))
        object.__setattr__(self, "imag", _real(imag))

    def is_zero(self):
        return self.real.is_zero() and self.imag.is_zero()

    def _plus(self, other, *, force_grid=False):
        other = _complex(other)
        return Complex(self.real._plus(other.real, force_grid=force_grid),
                       self.imag._plus(other.imag, force_grid=force_grid))

    def __add__(self, other):
        return self._plus(other)

    __radd__ = __add__

    def __neg__(self):
        return Complex(-self.real, -self.imag)

    def __sub__(self, other):
        return self+(-_complex(other))

    def __rsub__(self, other):
        return _complex(other)+(-self)

    def __mul__(self, other):
        other = _complex(other)
        return Complex(self.real*other.real-self.imag*other.imag,
                       self.real*other.imag+self.imag*other.real)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _complex(other)
        if other.imag.is_zero():
            return Complex(self.real/other.real, self.imag/other.real)
        denominator = other.real*other.real+other.imag*other.imag
        numerator = self*other.conjugate()
        return Complex(numerator.real/denominator, numerator.imag/denominator)

    def __rtruediv__(self, other):
        return _complex(other)/self

    def conjugate(self):
        return Complex(self.real, -self.imag)

    def exp(self):
        return exp(self)


def _complex(value):
    return value if isinstance(value, Complex) else Complex(value)


def _cached(name, key, calculate):
    context = current_context()
    found = context._lookup(name, key)
    if found is not None:
        return found
    return context._remember(name, key, calculate())


def _exp_tail():
    # Every factorial partial product is below 257^257, so this preflight
    # covers the whole fixed integer factorial calculation before it starts.
    _size(257*(257).bit_length())
    _tick()
    factorial = math.factorial(257)
    first = Fraction(_integer_power(16, 257), factorial)
    return _divide(first, Fraction(1)-Fraction(16, 258))


def _log_tail():
    _tick()
    denominator = _multiply(Fraction(_integer_power(3, 513)), Fraction(513))
    first = _divide(Fraction(2), denominator)
    return _divide(first, Fraction(1)-Fraction(1, 9))


def _exponential(value, omit_constant):
    key = (value, omit_constant)
    def calculate():
        if value.is_zero():
            return Complex(0 if omit_constant else 1)
        radius = _add(value.real.abs_bounds()[1], value.imag.abs_bounds()[1])
        if _compare(radius, Fraction(16)) > 0:
            raise ProofUnresolved("complex exponential rectangle exceeds L1 radius 16")
        term = Complex(1)
        total = Complex(0 if omit_constant else 1)
        if not omit_constant:
            _tick()  # the k=0 summand
        for index in range(1, SERIES_TERMS+1):
            _tick()
            term = term*value/index
            total = total._plus(term, force_grid=True)
        tail = _exp_tail()
        uncertainty = interval(-tail, tail)
        real = total.real._plus(uncertainty, force_grid=True)
        # A real input has an exactly real exponential. This follows from its
        # input support; no observed oracle zero enters this reduction.
        imag = point(0) if value.imag.is_zero() else total.imag._plus(uncertainty, force_grid=True)
        return Complex(real, imag)
    return _cached("expm1" if omit_constant else "exp", key, calculate)


def exp(value):
    complex_input = isinstance(value, Complex)
    result = _exponential(_complex(value), False)
    return result if complex_input else result.real


def expm1(value):
    complex_input = isinstance(value, Complex)
    result = _exponential(_complex(value), True)
    return result if complex_input else result.real


def sin(value):
    value = _real(value)
    return _cached("sin", value, lambda: point(0) if value.is_zero() else exp(Complex(0, value)).imag)


def cos(value):
    value = _real(value)
    return _cached("cos", value, lambda: point(1) if value.is_zero() else exp(Complex(0, value)).real)


def _power_of_two(exponent):
    return Fraction(_shift(1, exponent)) if exponent >= 0 else Fraction(1, _shift(1, -exponent))


def _log_series(u):
    if u < 0 or _compare(u, Fraction(1, 3)) > 0:
        raise ProofUnresolved("logarithm reduction did not prove 0 <= u <= 1/3")
    u_value = Real._bounds(u, u, rounded=False)
    square = u_value*u_value
    power, total = u_value, point(0)
    for index in range(SERIES_TERMS):
        _tick()
        total = total._plus(power/(2*index+1), force_grid=True)
        power = power*square
    total = 2*total
    return total._plus(interval(0, _log_tail()), force_grid=True)


def _log_point(value):
    if value == 1:
        return point(0)
    exponent = _bits(value.numerator)-_bits(value.denominator)
    if _compare(value, _power_of_two(exponent)) < 0:
        exponent -= 1
    mantissa = _divide(value, _power_of_two(exponent))
    u = _divide(_subtract(mantissa, Fraction(1)), _add(mantissa, Fraction(1)))
    log_mantissa = point(0) if not u else _log_series(u)
    log_two = _cached("log_two", (), lambda: _log_series(Fraction(1, 3)))
    return exponent*log_two+log_mantissa


def log(value):
    value = _real(value)
    def calculate():
        if _compare(value.lo, _power_of_two(-128)) < 0 or _compare(value.hi, _power_of_two(128)) > 0:
            raise ProofUnresolved("logarithm domain must lie in [2^-128, 2^128]")
        lower, upper = _log_point(value.lo), _log_point(value.hi)
        return Real._bounds(lower.lo, upper.hi, rounded=True)
    return _cached("log", value, calculate)


def _atan_inverse(denominator):
    total = point(0)
    for index in range(SERIES_TERMS):
        _tick()
        exponent = 2*index+1
        power = _integer_power(denominator, exponent)
        divisor = _multiply(Fraction(power), Fraction(exponent))
        term = point(_divide(Fraction(1 if index % 2 == 0 else -1), divisor))
        total = total._plus(term, force_grid=True)
    divisor = _multiply(Fraction(_integer_power(denominator, 513)), Fraction(513))
    remainder = _divide(Fraction(1), divisor)
    # 256 terms end negative, so the first omitted term is positive.
    return total._plus(interval(0, remainder), force_grid=True)


def pi():
    return _cached("pi", (), lambda: 16*_atan_inverse(5)-4*_atan_inverse(239))


def _sqrt_point(value):
    if value < 0:
        raise ProofUnresolved("square-root domain contains negative values")
    scaled_numerator = _shift(value.numerator, 2*GRID_BITS)
    integer = math.isqrt(scaled_numerator//value.denominator)
    _tick()
    lower = Fraction(integer, GRID)
    square = _multiply(lower, lower)
    _tick()
    _size(_bits(integer)+1)
    upper = lower if square == value else Fraction(integer+1, GRID)
    return lower, upper


def sqrt(value):
    value = _real(value)
    def calculate():
        _tick()
        if value.lo < 0:
            raise ProofUnresolved("square-root domain contains negative values")
        lower, upper = _sqrt_point(value.lo), _sqrt_point(value.hi)
        return Real._bounds(lower[0], upper[1], rounded=False)
    return _cached("sqrt", value, calculate)


def _actual_bounds(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise InvalidProofInput("actual interval needs exactly two rational endpoints")
    lower, upper = q(value[0]), q(value[1])
    if _compare(lower, upper) > 0:
        raise InvalidProofInput("actual interval endpoints are reversed")
    return lower, upper


def mutation_error(actual_bounds, target):
    """Exact difference bounds, with no target conversion to the oracle backend."""
    _tick()
    lower, upper = _actual_bounds(actual_bounds)
    target = _real(target)
    return Real._bounds(_subtract(lower, target.hi), _subtract(upper, target.lo), rounded=False)


def verify(actual_bounds, target, width_limit=None):
    """Return a receipt; unresolved overlap is never a membership proof."""
    _tick()
    lower, upper = _actual_bounds(actual_bounds)
    target = _real(target)
    lower_margin = _subtract(target.lo, lower)
    upper_margin = _subtract(upper, target.hi)
    actual_width = _subtract(upper, lower)
    if lower_margin >= 0 and upper_margin >= 0:
        membership = "proved"
    elif _compare(upper, target.lo) < 0 or _compare(target.hi, lower) < 0:
        membership = "disproved"
    else:
        membership = "unresolved"
    if width_limit is not None:
        width_limit = q(width_limit)
        if width_limit < 0:
            raise InvalidProofInput("width limit must be nonnegative")
    width_ok = width_limit is None or _compare(actual_width, width_limit) <= 0
    status = "disproved" if membership == "disproved" else "failed_width" if not width_ok else membership
    difference = mutation_error((lower, upper), target)
    return MappingProxyType({"status": status, "membership": membership,
        "actual": (lower, upper), "target": target, "lower_margin": lower_margin,
        "upper_margin": upper_margin, "actual_width": actual_width, "width": actual_width,
        "width_limit": width_limit, "width_ok": width_ok, "error_bounds": difference.abs_bounds(),
        "operations": current_context().operations})


def serialize(value):
    """JSON-ready exact evidence. No floating-point endpoints are produced."""
    if isinstance(value, Fraction):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    if isinstance(value, Real):
        return {"lo": serialize(value.lo), "hi": serialize(value.hi)}
    if isinstance(value, Complex):
        return {"real": serialize(value.real), "imag": serialize(value.imag)}
    if isinstance(value, Mapping):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize(item) for item in value]
    return value
