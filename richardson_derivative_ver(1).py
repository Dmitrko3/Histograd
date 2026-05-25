"""
richardson_derivative.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Senior Numerical Analyst Implementation
Richardson Extrapolation — First Derivative Estimator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Standard library only. No numpy, no scipy.
# Why: Richardson Extrapolation is pure algebra. Pulling in a heavy
# library for three arithmetic operations would be dishonest to the
# simplicity of the method.
import math


def richardson_derivative(f, x, h=0.1, order=2):
    """
    ════════════════════════════════════════════════════════════════════
    HOW THIS METHOD RECEIVES ITS FUNCTION  (inputs)
    ════════════════════════════════════════════════════════════════════

    f  : callable  — the mathematical function whose derivative we want.
                     The caller passes in any Python function that accepts
                     a single float and returns a float.
                     Example:  f = lambda x: math.sin(x)
                               f = lambda x: x**3 - 2*x + 1

                     REQUIREMENT: f must be smooth (infinitely differentiable)
                     at and around the point x.  A function with a corner or
                     jump at x will silently return garbage — see the PDF you
                     shared, section "Types of functions this method can't solve."

    x  : float     — the point at which we want the derivative f'(x).

    h  : float     — the initial step size.  Default 0.1.
                     Think of this as the "resolution knob."  Larger h means
                     coarser (blurrier) approximations; smaller h means finer
                     ones.  We intentionally start coarse because Richardson
                     Extrapolation's entire job is to transcend the step size.

                     WARNING: Do NOT set h below ~1e-5 for float64 arithmetic.
                     Making h too tiny causes catastrophic cancellation
                     (round-off error dominates), which is the "Numerical
                     Stability" failure discussed in the PDF.

    order : int    — the known order of accuracy n of the base finite-difference
                     formula.  Default 2 (standard central difference).
                     This matches the exponent n in the PDF formula
                     A(h) = L − C·hⁿ.
                     If you switch to a different base method (forward difference,
                     4th-order stencil, etc.) you must update this value, or the
                     algebraic cancellation will be wrong.

    ════════════════════════════════════════════════════════════════════
    HOW THIS METHOD EXPORTS ITS SOLUTION  (outputs)
    ════════════════════════════════════════════════════════════════════

    Returns a dict with three keys so the caller gets not just the number
    but enough context to trust it.  A raw float return is a bad habit in
    numerical software — you should always know how good your answer is.

    {
        "derivative"  : float  — the Richardson-extrapolated estimate of f'(x).
        "A_h"         : float  — the raw (coarse) approximation using step h.
        "A_h2"        : float  — the raw (finer) approximation using step h/2.
        "improvement" : float  — how much Richardson improved over A_h alone,
                                 expressed as a percentage.
    }
    ════════════════════════════════════════════════════════════════════
    """

    # ── STEP 1: Compute the first (coarse) central-difference approximation.
    #
    # Formula:  A(h)  ≈  [ f(x+h) − f(x−h) ] / (2h)
    #
    # WHY central difference and not forward difference?
    # A forward difference  [ f(x+h) − f(x) ] / h  has error O(h¹).
    # A central difference  [ f(x+h) − f(x−h) ] / (2h)  has error O(h²).
    # Starting with a higher-order base formula means Richardson Extrapolation
    # will boost us from O(h²) → O(h⁴), which is a much bigger payoff
    # than boosting O(h¹) → O(h²).

    A_h = (f(x + h) - f(x - h)) / (2 * h)
    # ↑ We subtract the function value one step to the LEFT from the value one
    # step to the RIGHT, then divide by the full span (2h).  This is the
    # "first low-resolution approximation" described in the PDF.


    # ── STEP 2: Compute the second (finer) central-difference approximation.
    #
    # We simply halve the step size: h → h/2.
    # This is the "second low-resolution approximation" from the PDF.
    # It is slightly more accurate, but still imperfect.

    half_h = h / 2
    # ↑ We store h/2 in its own variable for clarity.  Inline arithmetic
    # like (h/2) used three times in one expression is a maintenance hazard.

    A_h2 = (f(x + half_h) - f(x - half_h)) / (2 * half_h)
    # ↑ Same central-difference formula as above, but with half_h instead of h.
    # The error here is C·(h/2)ⁿ, which is smaller than C·hⁿ,
    # exactly matching the PDF derivation: A(h/2) = L − C·hⁿ / 2ⁿ


    # ── STEP 3: Compute the scaling factor 2ⁿ.
    #
    # This is the core of the Richardson formula.
    # From the PDF:  L ≈ (2ⁿ · A(h/2) − A(h)) / (2ⁿ − 1)
    #
    # The factor 2ⁿ is what lets us scale the finer error term to perfectly
    # match the coarser error term so they cancel when we subtract.

    factor = 2 ** order
    # ↑ With order=2 (central difference), factor = 4.
    # This means the error in A(h/2) is exactly 1/4 the error in A(h),
    # so multiplying A(h/2) by 4 makes the error terms identical in magnitude.


    # ── STEP 4: Apply the Richardson Extrapolation formula.
    #
    # This is the single algebraic step that eliminates the dominant error.
    # It directly implements the formula derived step-by-step in the PDF:
    #
    #            2ⁿ · A(h/2)  −  A(h)
    #   L  ≈  ──────────────────────────
    #                  2ⁿ − 1
    #
    # WHY this works (in one sentence):
    # Both A(h) and A(h2) contain the same error SHAPE (C·hⁿ), just at
    # different volumes.  Scaling and subtracting makes those volumes match
    # perfectly — the error terms cancel, and only L remains.

    L = (factor * A_h2 - A_h) / (factor - 1)
    # ↑ 'L' mirrors the PDF's notation for the true exact value we are seeking.
    # After cancellation, this is no longer a crude O(h²) estimate —
    # it is an O(h⁴) estimate.  We doubled the order of accuracy for free.


    # ── STEP 5: Compute how much Richardson improved over the raw estimate.
    #
    # This is a diagnostic: it tells the caller how big a "leap" was made.
    # We compute it as the percentage change from the raw coarse value to
    # the extrapolated value.

    improvement = abs(L - A_h) / (abs(A_h) + 1e-30) * 100
    # ↑ We add a tiny epsilon (1e-30) in the denominator to guard against
    # division by zero if A_h happens to be exactly 0.0.
    # This is standard defensive arithmetic in numerical code.


    # ── STEP 6: Return a structured result dict, not a bare float.
    #
    # Returning a dict forces the caller to think about what they received.
    # In a production numerical pipeline, you never want a magic number
    # flying around with no context about how it was computed.

    return {
        "derivative" : L,      # The extrapolated, high-accuracy estimate.
        "A_h"        : A_h,    # The raw coarse estimate (step = h).
        "A_h2"       : A_h2,   # The raw finer estimate  (step = h/2).
        "improvement": improvement,  # How much the extrapolation moved the needle.
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DEMONSTRATION  —  run this file directly to see the method in action
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Test case 1: f(x) = sin(x),  f'(x) = cos(x),  at x = 1.0
    #    Ground truth: cos(1.0) = 0.5403023058681398
    f1          = math.sin
    x1          = 1.0
    true_answer = math.cos(x1)

    result = richardson_derivative(f=f1, x=x1, h=0.1, order=2)

    print("═" * 60)
    print("  TEST: f(x) = sin(x),  x = 1.0")
    print("═" * 60)
    print(f"  Ground truth  f'(x) = cos(1.0)  :  {true_answer:.16f}")
    print(f"  Raw A(h)   estimate  (coarse)    :  {result['A_h']:.16f}")
    print(f"  Raw A(h/2) estimate  (finer)     :  {result['A_h2']:.16f}")
    print(f"  Richardson extrapolated answer   :  {result['derivative']:.16f}")
    print(f"  Improvement over raw A(h)        :  {result['improvement']:.6f}%")
    print()

    abs_error_raw = abs(result['A_h']        - true_answer)
    abs_error_re  = abs(result['derivative'] - true_answer)

    print(f"  Absolute error  (raw A_h)        :  {abs_error_raw:.2e}")
    print(f"  Absolute error  (Richardson)     :  {abs_error_re:.2e}")
    print(f"  Error reduction factor           :  {abs_error_raw / abs_error_re:.1f}×")
    print("═" * 60)

    # ── Test case 2: f(x) = x³,  f'(x) = 3x²,  at x = 2.0
    #    Ground truth: 3 * (2.0)² = 12.0
    f2          = lambda x: x**3
    x2          = 2.0
    true_answer2 = 3 * x2**2   # = 12.0

    result2 = richardson_derivative(f=f2, x=x2, h=0.5, order=2)

    print()
    print("═" * 60)
    print("  TEST: f(x) = x³,  x = 2.0")
    print("═" * 60)
    print(f"  Ground truth  f'(x) = 3x²       :  {true_answer2:.16f}")
    print(f"  Raw A(h)   estimate  (coarse)    :  {result2['A_h']:.16f}")
    print(f"  Raw A(h/2) estimate  (finer)     :  {result2['A_h2']:.16f}")
    print(f"  Richardson extrapolated answer   :  {result2['derivative']:.16f}")
    print()

    abs_error_raw2 = abs(result2['A_h']        - true_answer2)
    abs_error_re2  = abs(result2['derivative'] - true_answer2)

    print(f"  Absolute error  (raw A_h)        :  {abs_error_raw2:.2e}")
    print(f"  Absolute error  (Richardson)     :  {abs_error_re2:.2e}")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#
#  HOW THE METHOD WORKS — END-TO-END WALKTHROUGH
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  PHASE 1 — TWO CHEAP APPROXIMATIONS                                     │
#  │                                                                         │
#  │  We call the function f exactly four times total:                       │
#  │    f(x+h),  f(x-h)      → used to build A(h)   [the coarse estimate]   │
#  │    f(x+h/2), f(x-h/2)   → used to build A(h/2) [the finer estimate]    │
#  │                                                                         │
#  │  Each of these is a central difference — a finite-difference formula    │
#  │  that approximates the slope of f at x by looking at two nearby points. │
#  │  It is cheap, simple, and has a known error of the form  C · h².        │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  PHASE 2 — ALGEBRAIC CANCELLATION                                       │
#  │                                                                         │
#  │  We know from Taylor Series theory that both approximations contain     │
#  │  the exact same error CONSTANT C.  The only difference is the step:     │
#  │                                                                         │
#  │    A(h)   =  L  −  C · h²                                              │
#  │    A(h/2) =  L  −  C · h² / 4                                          │
#  │                                                                         │
#  │  By multiplying the second equation by 4 and subtracting the first,    │
#  │  the C · h² terms cancel perfectly.  We are left with only L.          │
#  │                                                                         │
#  │  This is not magic. It is a system of two equations with two unknowns   │
#  │  (L and C), solved by simple elimination.                               │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  PHASE 3 — ORDER UPGRADE                                                │
#  │                                                                         │
#  │  Before:  A(h) has error O(h²).  Halving h gives a 4× improvement.     │
#  │  After:   Richardson's L has error O(h⁴). Halving h gives a 16× gain. │
#  │                                                                         │
#  │  We doubled the order of accuracy for the cost of four function        │
#  │  evaluations.  No extra brute-force computation.  No finer grid.       │
#  │  Pure algebra.                                                          │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  WHAT CAN BREAK IT                                                      │
#  │                                                                         │
#  │  1. Non-smooth functions (corners, jumps): the Taylor Series assumption │
#  │     breaks down. C is no longer constant. Garbage in, garbage out.     │
#  │                                                                         │
#  │  2. h too small (< 1e-7 for float64): round-off error from the         │
#  │     floating-point subtraction f(x+h) − f(x−h) starts to dominate.    │
#  │     The "noise" is larger than the "signal."                           │
#  │                                                                         │
#  │  3. Wrong 'order' parameter: if you pass order=2 but your base method  │
#  │     actually has O(h¹) error, the factor (2ⁿ) will be wrong and the   │
#  │     cancellation will be incomplete.                                   │
#  │                                                                         │
#  │  4. Singularities or asymptotes near x: the constant C (bundled        │
#  │     derivatives) will blow up, violating the core assumption.          │
#  └─────────────────────────────────────────────────────────────────────────┘
#
# ══════════════════════════════════════════════════════════════════════════════
