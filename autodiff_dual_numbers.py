"""
HistoGrad — Automatic Differentiation via Dual Numbers
=======================================================
Role:   Algorithm Engineer (Numerical Derivative Methods)
Method: Forward-Mode Automatic Differentiation using the Dual Number algebra

CORE IDEA
---------
A dual number extends the real line with an infinitesimal unit ε (epsilon)
satisfying the algebraic rule:

         ε² = 0    (but ε ≠ 0)

Every dual number has the form:

         a + b·ε

where:
  • a  is the "real" part  — holds the function value  f(x)
  • b  is the "dual" part  — holds the derivative     f′(x)

The magic: if we seed the input as x = (x + 1·ε) and evaluate f with
dual arithmetic, the ε-coefficient of the result is exactly f′(x),
computed in a single forward pass with NO approximation error.

This is fundamentally different from finite differences:
  • Finite difference:  [f(x+h) − f(x)] / h   ← approximation, h-sensitive
  • Dual numbers:        ε-part of f(x+ε)       ← exact, h-free
"""

import math
from dataclasses import dataclass
from typing import Callable, NamedTuple


# ─── 1. The Dual Number Class ────────────────────────────────────────────────

@dataclass
class Dual:
    """
    Represents a dual number of the form (real + eps·ε).

    Python's dunder (double-underscore) methods let us overload the standard
    arithmetic operators (+, -, *, /) so that dual-arithmetic expressions
    look identical to ordinary Python math — no .add() call syntax needed.

    Attributes
    ----------
    real : float
        The ordinary (ε⁰) component. After differentiation this holds f(x).
    eps : float
        The infinitesimal (ε¹) component. After differentiation this holds f′(x).
        Defaults to 0 so that plain numeric literals behave as constants
        whose derivative is zero — mathematically correct.
    """

    real: float          # function value carrier
    eps:  float = 0.0    # derivative carrier; default 0 → constant behaviour


    # ── Arithmetic operations ─────────────────────────────────────────────────
    #
    # Each dunder method extends the matching real operation to dual numbers.
    # They are derived by substituting (a + bε) and (c + dε), expanding,
    # then dropping every term that contains ε² (because ε² = 0 by definition).

    def __add__(self, other):
        """
        Addition: (a + bε) + (c + dε) = (a+c) + (b+d)ε

        The sum rule of derivatives falls out automatically:
            (f + g)′ = f′ + g′   ← the ε-parts simply add together

        'other' may be a Dual or a plain float/int (scalar).
        The scalar branch handles expressions like  d + 5  or  5 + d.
        """
        if isinstance(other, Dual):
            return Dual(
                self.real + other.real,   # real parts add normally
                self.eps  + other.eps     # dual parts add normally → sum rule
            )
        # Scalar: (a + bε) + k = (a+k) + bε
        # Adding a constant shifts the value but does not change the derivative.
        return Dual(self.real + other, self.eps)

    def __radd__(self, other):
        """Right-hand addition: k + d  is the same as  d + k for scalars."""
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction: (a + bε) − (c + dε) = (a−c) + (b−d)ε

            (f − g)′ = f′ − g′   ← difference rule
        """
        if isinstance(other, Dual):
            return Dual(
                self.real - other.real,
                self.eps  - other.eps
            )
        # Scalar: (a + bε) − k = (a−k) + bε
        return Dual(self.real - other, self.eps)

    def __rsub__(self, other):
        """Right-hand subtraction: k − d = −(d − k)."""
        return Dual(other - self.real, -self.eps)

    def __mul__(self, other):
        """
        Multiplication: (a + bε)(c + dε) = ac + (ad + bc)ε + bd·ε²
                                          = ac + (ad + bc)ε     [since ε²=0]

        The product rule of derivatives appears automatically:
            (f · g)′ = f·g′ + f′·g    →   a·d + b·c
        """
        if isinstance(other, Dual):
            return Dual(
                self.real * other.real,                             # f(x)·g(x)
                self.real * other.eps + self.eps * other.real       # f·g′ + f′·g
            )
        # Scalar: k·(a + bε) = ka + kb·ε   (constant-multiple rule: (k·f)′ = k·f′)
        return Dual(self.real * other, self.eps * other)

    def __rmul__(self, other):
        """Right-hand multiplication: k * d  is the same as  d * k."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Division: (a + bε) / (c + dε)

        Derived by multiplying numerator and denominator by (c − dε):
            numerator:   (a + bε)(c − dε) = ac + (bc − ad)ε   [ε² dropped]
            denominator: (c + dε)(c − dε) = c²                [ε² dropped]

            result = a/c  +  (bc − ad)/c² · ε

        This is exactly the quotient rule: (f/g)′ = (f′g − fg′) / g²
        """
        if isinstance(other, Dual):
            real_part = self.real / other.real
            eps_part  = (self.eps * other.real - self.real * other.eps) / (other.real ** 2)
            return Dual(real_part, eps_part)
        # Scalar: (a + bε) / k = a/k + (b/k)·ε
        return Dual(self.real / other, self.eps / other)

    def __rtruediv__(self, other):
        """
        Right-hand division: k / (a + bε)

        Treat k as Dual(k, 0) and apply the quotient rule:
            result = k/a + (−k·b/a²)·ε    (derivative of k/f is −k·f′/f²)
        """
        return Dual(other, 0.0).__truediv__(self)

    def __neg__(self):
        """Unary negation: −(a + bε) = (−a) + (−b)ε"""
        return Dual(-self.real, -self.eps)

    def __pow__(self, n):
        """
        Power: (a + bε)^n = a^n + n·b·a^(n−1)·ε

        g(x) = x^n, g′(x) = n·x^(n−1)  — the power rule.

        Works for any real exponent n (integer, fractional, negative).
        Verification for n=2: (a+bε)² = a² + 2ab·ε + b²·ε² = a² + 2ab·ε  ✓
        """
        return Dual(
            self.real ** n,                         # a^n
            n * self.eps * (self.real ** (n - 1))   # n · b · a^(n-1) — power rule
        )

    def __repr__(self):
        """Human-readable form: shows both components for debugging."""
        sign = "+" if self.eps >= 0 else "-"
        return f"Dual({self.real:.6g} {sign} {abs(self.eps):.6g}·ε)"


# ─── 2. Elementary function lifts (module-level, mirrors math.*) ─────────────
#
# For any smooth function g, the first-order Taylor expansion gives:
#
#    g(a + bε) = g(a) + b·g′(a)·ε
#
# This is the generalised chain rule for dual numbers.
# Every function below is implemented by this identity.
#
# Usage mirrors the standard math module:
#    import autodiff_dual_numbers as ad
#    result = ad.sin(ad.seed(x))

def sin(d: Dual) -> Dual:
    """
    sin(a + bε) = sin(a) + b·cos(a)·ε

    Proof: g = sin, g′ = cos  →  g(a+bε) = sin(a) + b·cos(a)·ε

    Chain rule in action: if the incoming ε-part b already encodes f′(x)
    from a prior operation, the output ε-part is cos(a)·f′(x) — exactly
    the chain rule result d/dx sin(f(x)) = cos(f(x))·f′(x).
    """
    return Dual(
        math.sin(d.real),           # sin(a)
        d.eps * math.cos(d.real)    # b · cos(a)  — chain rule factor
    )


def cos(d: Dual) -> Dual:
    """
    cos(a + bε) = cos(a) − b·sin(a)·ε

    g = cos, g′ = −sin  →  g(a+bε) = cos(a) − b·sin(a)·ε
    Note the minus sign — it comes from d/dx cos(x) = −sin(x).
    """
    return Dual(
        math.cos(d.real),
        -d.eps * math.sin(d.real)   # minus sign from g′ = −sin
    )


def exp(d: Dual) -> Dual:
    """
    exp(a + bε) = e^a + b·e^a·ε

    g = exp, g′ = exp — the exponential is its own derivative.
    We compute e^a once and reuse it for both parts to save a call.
    """
    ea = math.exp(d.real)   # compute once; reused for both components
    return Dual(
        ea,                  # e^a
        d.eps * ea           # b · e^a
    )


def log(d: Dual) -> Dual:
    """
    log(a + bε) = ln(a) + b·(1/a)·ε

    g = ln, g′ = 1/x  →  g(a+bε) = ln(a) + b/a · ε
    Domain: a > 0  (raises ValueError for a ≤ 0, consistent with math.log).
    """
    return Dual(
        math.log(d.real),    # ln(a)
        d.eps / d.real       # b / a  — chain rule with g′ = 1/x
    )


def sqrt(d: Dual) -> Dual:
    """
    sqrt(a + bε) = √a + b·(1 / (2√a))·ε

    Equivalent to d ** 0.5, written explicitly for readability.
    g = √x, g′ = 1/(2√x).
    """
    sqrt_a = math.sqrt(d.real)
    return Dual(
        sqrt_a,
        d.eps / (2 * sqrt_a)   # b · 1/(2√a)
    )


def tan(d: Dual) -> Dual:
    """
    tan(a + bε) = tan(a) + b·sec²(a)·ε

    Implemented as sin(d) / cos(d) so the chain rule cascades automatically
    through the dual division — same result as directly computing sec²(a).
    """
    return sin(d) / cos(d)
    # Equivalent direct form:
    # return Dual(math.tan(d.real), d.eps / math.cos(d.real) ** 2)


def abs_dual(d: Dual) -> Dual:
    """
    abs(a + bε) = |a| + b·sign(a)·ε

    Non-differentiable at a = 0.  math.copysign(1, 0) = 1.0 in Python,
    so the dual part is set to b at a=0 — a convention, not a true derivative.
    """
    return Dual(
        abs(d.real),
        d.eps * math.copysign(1.0, d.real)   # b · sign(a)
    )


# ─── 3. Seeding — how to inject the derivative at a point ────────────────────

def seed(x: float) -> Dual:
    """
    Create the seeded input for differentiating with respect to x.

    To compute f′(x):
      1. Represent x as the dual number (x + 1·ε)  ← seed the derivative to 1
      2. Evaluate f using dual arithmetic
      3. Read f′(x) from the .eps attribute of the result

    The "1" in the ε-slot means ∂x/∂x = 1 (we are differentiating w.r.t. x).
    In a multivariate setting, seed only the i-th variable with ε=1 and pass
    Dual(xⱼ, 0) for all j ≠ i — that gives the i-th partial derivative ∂f/∂xᵢ.

    Parameters
    ----------
    x : float
        The point at which to evaluate f and f′.

    Returns
    -------
    Dual
        The dual number (x + 1·ε).
    """
    return Dual(x, 1.0)
    # real = x  (the numeric value of the input)
    # eps  = 1  (∂x/∂x = 1, by definition of the identity function)


def constant(c: float) -> Dual:
    """
    Wrap a plain numeric constant as a dual number with zero derivative.

    Any literal that appears inside f (like the "2" in f(x) = 2x + 5)
    is a constant: d(constant)/dx = 0, so its eps-part must be 0.

    Parameters
    ----------
    c : float
        The constant value.

    Returns
    -------
    Dual
        The dual number (c + 0·ε).
    """
    return Dual(c, 0.0)
    # eps = 0  → the derivative of a constant is zero


# ─── 4. Result container ─────────────────────────────────────────────────────

class DiffResult(NamedTuple):
    """
    Named tuple returned by differentiate().

    Using NamedTuple gives both attribute access (result.value) and
    tuple unpacking (value, derivative = differentiate(f, x)).
    """
    value:      float   # f(x)
    derivative: float   # f′(x)


# ─── 5. Differentiator — the public-facing API ────────────────────────────────

def differentiate(f: Callable[[Dual], Dual], x: float) -> DiffResult:
    """
    Differentiate a scalar function f at point x using dual numbers.

    The function f must be written purely in terms of Dual arithmetic —
    i.e. it should accept a Dual argument and return a Dual.  Any Python
    operator (+, -, *, /, **) or module-level function (sin, cos, exp …)
    from this module will work automatically.

    Parameters
    ----------
    f : Callable[[Dual], Dual]
        A function ℝ → ℝ expressed in dual arithmetic.
    x : float
        The point at which to evaluate f and f′.

    Returns
    -------
    DiffResult
        .value      = f(x)   (the function value)
        .derivative = f′(x)  (the exact derivative, no approximation)

    Example
    -------
    >>> g = lambda d: sin(d * d)          # f(x) = sin(x²)
    >>> r = differentiate(g, 1.5)
    >>> r.value                           # sin(2.25)  ≈  0.7781
    >>> r.derivative                      # 2·1.5·cos(2.25) ≈ −1.5886
    """
    dual_input  = seed(x)    # inject x as (x + 1·ε)
    dual_output = f(dual_input)  # propagate dual arithmetic through every operation in f

    # Because ε² = 0, dual arithmetic cannot accumulate higher-order terms
    # in the ε-slot.  The result is guaranteed to be exactly [f(x), f′(x)].
    return DiffResult(
        value=dual_output.real,   # f(x)
        derivative=dual_output.eps  # f′(x) — exact, zero truncation error
    )


# ─── 6. Pre-built function library for HistoGrad ─────────────────────────────
#
# Each lambda is a pure dual-arithmetic expression.
# The same lambda evaluates BOTH f(x) and f′(x) in one forward pass.

FUNCTIONS: dict[str, Callable[[Dual], Dual]] = {

    # f(x) = sin(x),    f′(x) = cos(x)
    "sin":      lambda d: sin(d),

    # f(x) = cos(x),    f′(x) = −sin(x)
    "cos":      lambda d: cos(d),

    # f(x) = e^x,       f′(x) = e^x
    "exp":      lambda d: exp(d),

    # f(x) = ln(x),     f′(x) = 1/x
    "log":      lambda d: log(d),

    # f(x) = x³ − 2x + 1,   f′(x) = 3x² − 2
    #
    # Step-by-step dual arithmetic:
    #   Step 1: d**3          →  Dual(x³,  3x²·1)      ← power rule with eps=1
    #   Step 2: 2*d           →  Dual(2x,  2·1)         ← scalar multiply
    #   Step 3: d**3 − 2*d + 1 →  Dual(x³−2x+1, 3x²−2) ← subtraction + scalar add
    "poly":     lambda d: d**3 - 2*d + 1,

    # f(x) = sin(x²),   f′(x) = 2x·cos(x²)   (chain rule)
    #
    #   Step 1: d**2    →  Dual(x², 2x)          ← power rule; eps=2x propagates
    #   Step 2: sin(↑)  →  Dual(sin(x²), 2x·cos(x²))  ← sin lift uses incoming eps
    #   The chain rule is implicit in how sin() reads the incoming eps.
    "comp":     lambda d: sin(d**2),

    # f(x) = x·eˣ,      f′(x) = eˣ + x·eˣ = (1+x)·eˣ   (product rule)
    "xexp":     lambda d: d * exp(d),

    # f(x) = ln(sin(x)),  f′(x) = cos(x)/sin(x) = cot(x)   (chain rule)
    "logsin":   lambda d: log(sin(d)),

    # f(x) = √(x² + 1),   f′(x) = x / √(x²+1)   (chain rule)
    "sqrtpoly": lambda d: sqrt(d**2 + 1),

    # f(x) = tan(x),    f′(x) = sec²(x) = 1/cos²(x)
    "tan":      lambda d: tan(d),
}


# ─── 7. Batch evaluation — differentiate multiple points at once ─────────────

def batch_differentiate(
    f:  Callable[[Dual], Dual],
    xs: list[float]
) -> list[dict]:
    """
    Evaluate f and f′ at each point in xs.

    All evaluations are independent (no shared state), making this
    trivially parallelisable with concurrent.futures if needed.

    Parameters
    ----------
    f  : Dual function as described in differentiate().
    xs : List of evaluation points.

    Returns
    -------
    List of dicts with keys 'x', 'value', 'derivative'.
    """
    results = []
    for x in xs:
        r = differentiate(f, x)
        results.append({"x": x, "value": r.value, "derivative": r.derivative})
        # Each call to differentiate() is entirely stateless —
        # the seed Dual is created fresh each iteration.
    return results


# ─── 8. Error analysis — dual numbers vs finite differences ──────────────────

def error_analysis(
    plain_f: Callable[[float], float],
    dual_f:  Callable[[Dual],  Dual],
    x:       float,
    steps:   list[float] = None
) -> list[dict]:
    """
    Compare the dual-number derivative to a central finite difference
    approximation  [f(x+h) − f(x−h)] / (2h)  for a range of step sizes h.

    This function is designed to demonstrate the key advantage of dual numbers:

      • Finite diff error ~ O(h²) for the central scheme — shrinks as h → 0,
        BUT catastrophic cancellation kicks in for very small h
        (subtracting nearly-equal floats destroys significant digits).
      • Dual numbers: machine-precision exact regardless of h.

    Parameters
    ----------
    plain_f : Plain Python function  f: float → float  (for finite differences).
    dual_f  : Same function in dual arithmetic          (for ground truth).
    x       : Evaluation point.
    steps   : List of h values to test.

    Returns
    -------
    List of dicts: { 'h', 'finite_diff', 'dual_deriv', 'abs_error' }
    """
    if steps is None:
        steps = [1e-1, 1e-3, 1e-5, 1e-7, 1e-10, 1e-14]

    exact = differentiate(dual_f, x).derivative   # ground truth from dual numbers

    rows = []
    for h in steps:
        # Central finite difference (2nd-order accurate):
        # cancels the O(h) leading-error term, leaving O(h²) truncation error.
        finite_diff = (plain_f(x + h) - plain_f(x - h)) / (2 * h)
        abs_error   = abs(finite_diff - exact)

        rows.append({
            "h":           h,
            "finite_diff": finite_diff,
            "dual_deriv":  exact,       # constant across all h — that's the point
            "abs_error":   abs_error
        })
    return rows


# ─── 9. Usage examples ───────────────────────────────────────────────────────

def run_examples():
    print("=== HistoGrad — Automatic Differentiation via Dual Numbers ===\n")

    # ── Example 1: Trigonometric function ────────────────────────────────────
    # f(x) = sin(x),  f′(x) = cos(x)
    x = math.pi / 4                          # 45°
    r = differentiate(FUNCTIONS["sin"], x)
    analytical = math.cos(x)                 # exact answer for comparison

    print(f"f(x) = sin(x),  x = π/4")
    print(f"  f(x)   = {r.value:.8f}    (expected {math.sin(x):.8f})")
    print(f"  f′(x)  = {r.derivative:.8f}    (expected {analytical:.8f})")
    print(f"  error  = {abs(r.derivative - analytical):.2e}\n")

    # ── Example 2: Polynomial ─────────────────────────────────────────────────
    # f(x) = x³ − 2x + 1,  f′(x) = 3x² − 2
    x = 2.0
    r = differentiate(FUNCTIONS["poly"], x)
    analytic_f  = x**3 - 2*x + 1    # 5
    analytic_df = 3*x**2 - 2         # 10

    print(f"f(x) = x³ − 2x + 1,  x = 2")
    print(f"  f(x)   = {r.value}          (expected {analytic_f})")
    print(f"  f′(x)  = {r.derivative}         (expected {analytic_df})")
    print(f"  error  = {abs(r.derivative - analytic_df):.2e}\n")

    # ── Example 3: Composition — multi-layer chain rule ───────────────────────
    # f(x) = sin(x²),  f′(x) = 2x·cos(x²)
    x = 1.5
    r = differentiate(FUNCTIONS["comp"], x)
    analytic_df = 2 * x * math.cos(x**2)

    print(f"f(x) = sin(x²),  x = 1.5")
    print(f"  f(x)   = {r.value:.8f}")
    print(f"  f′(x)  = {r.derivative:.8f}    (expected {analytic_df:.8f})")
    print(f"  error  = {abs(r.derivative - analytic_df):.2e}\n")

    # ── Example 4: Error analysis — dual vs finite differences ────────────────
    x = 1.0
    print(f"Error analysis: f(x) = sin(x),  x = {x}")
    print(f"  {'h':<12}  {'Finite diff':>14}  {'Abs error':>14}")

    rows = error_analysis(math.sin, FUNCTIONS["sin"], x)
    for row in rows:
        print(f"  h={row['h']:<8.0e}  {row['finite_diff']:>14.8f}  {row['abs_error']:>14.2e}")

    r = differentiate(FUNCTIONS["sin"], x)
    print(f"  dual (exact)  {r.derivative:>14.8f}  {'< 1e-15':>14}\n")

    # ── Example 5: Custom inline function ─────────────────────────────────────
    # f(x) = x·eˣ,  f′(x) = (1+x)·eˣ
    # Defined inline — no entry in FUNCTIONS needed.
    f_xexp = lambda d: d * exp(d)
    x = 1.0
    r = differentiate(f_xexp, x)
    analytic_df = (1 + x) * math.exp(x)

    print(f"f(x) = x·eˣ,  x = {x}")
    print(f"  f′(x)  = {r.derivative:.8f}    (expected {analytic_df:.8f})")
    print(f"  error  = {abs(r.derivative - analytic_df):.2e}\n")


if __name__ == "__main__":
    run_examples()
