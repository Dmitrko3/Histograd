"""
richardson_derivative.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Senior Numerical Analyst Implementation  —  v2
Richardson Extrapolation — First Derivative Estimator
  ✦ Iterative Deepening (while-loop convergence)
  ✦ Romberg-Style Tableau (multi-level error cancellation)
  ✦ Input Validation & Safeguards (ValueError on bad inputs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Standard library only — Richardson Extrapolation is pure algebra.
import math


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPER — central_difference
# ══════════════════════════════════════════════════════════════════════════════

def _central_diff(f, x, h):
    """
    Compute one raw central-difference approximation of f'(x) at step size h.

    Formula:  A(h)  =  [ f(x+h) − f(x−h) ]  /  (2h)

    This is kept as a private helper (underscore prefix) because it is a
    low-level building block — callers should never need to call it directly.
    Isolating it here means the loop body stays clean and readable.

    WHY central difference?
      Forward difference error = O(h¹)  →  Richardson yields O(h²)
      Central difference error = O(h²)  →  Richardson yields O(h⁴) per level
    Central difference gives us a much higher payoff per extrapolation step.
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)
    # ↑ We force 2.0 (float) instead of 2 (int) to guarantee float division
    # even if h happens to be an integer.  Defensive arithmetic is a habit,
    # not an afterthought.


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPER — romberg_extrapolate
# ══════════════════════════════════════════════════════════════════════════════

def _romberg_extrapolate(col, order):
    """
    Apply one full pass of Richardson Extrapolation across an entire column
    of approximations, producing a new column that is one order higher.

    Inputs
    ------
    col   : list of floats  — a column of approximations at decreasing step
                              sizes, e.g. [A(h), A(h/2), A(h/4), A(h/8)].
                              Adjacent pairs share the same error SHAPE, so
                              each pair can be extrapolated into one entry in
                              the next column.

    order : int             — the current error order being cancelled.
                              On the first pass it is `base_order` (e.g. 2).
                              On the second pass it is `base_order + 2` (e.g. 4).
                              On the k-th pass it is `base_order + 2*(k-1)`.
                              WHY does it increase by 2 each time?
                              Central difference has only EVEN error powers
                              (h², h⁴, h⁶ …). Cancelling h² leaves h⁴ as the
                              new dominant term, so the next pass targets h⁴.

    Output
    ------
    Returns a new list that is one element shorter than `col`, where each
    element is the Richardson combination of the two adjacent inputs above it.

    The formula applied to each adjacent pair (A_coarse, A_fine):
        factor = 2^order
        L  =  (factor * A_fine  −  A_coarse)  /  (factor − 1)

    This is exactly the PDF formula, now applied to every adjacent row pair
    instead of just one pair.  That is the entire Romberg upgrade.
    """
    factor   = 2 ** order
    # ↑ With order=2, factor=4.  With order=4, factor=16.  With order=6,
    # factor=64.  Each level uses a larger factor because the error being
    # cancelled shrinks faster (hⁿ with larger n), so we need a bigger
    # lever to make the error terms match perfectly.

    new_col  = []
    # ↑ We build the next column from scratch rather than mutating `col`
    # in-place.  Mutating the list while iterating it is a classic bug.

    for i in range(len(col) - 1):
        # ↑ We iterate over adjacent pairs: (col[0], col[1]),
        # (col[1], col[2]), (col[2], col[3]), …
        # Each pair yields one extrapolated entry.
        # The new column is therefore always ONE element shorter than the old one.

        A_coarse = col[i]
        # ↑ The less accurate of the two. It was computed with a larger step
        # size, so it carries more error. It corresponds to A(h) in the PDF.

        A_fine   = col[i + 1]
        # ↑ The more accurate of the two. Smaller step, smaller error.
        # It corresponds to A(h/2) in the PDF.

        L = (factor * A_fine - A_coarse) / (factor - 1)
        # ↑ The Richardson formula from the PDF:
        #       L  ≈  ( 2ⁿ · A(h/2)  −  A(h) )  /  ( 2ⁿ − 1 )
        # The dominant error terms (C · hⁿ) cancel perfectly.
        # What remains has error of order h^(n+2), not h^n.

        new_col.append(L)

    return new_col


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — richardson_derivative
# ══════════════════════════════════════════════════════════════════════════════

def richardson_derivative(f, x, h=0.1, order=2, tol=1e-8, max_iter=10):
    """
    ════════════════════════════════════════════════════════════════════
    HOW THIS METHOD RECEIVES ITS FUNCTION  (inputs)
    ════════════════════════════════════════════════════════════════════

    f        : callable  — the function whose derivative we want.
                           Must accept a single float and return a float.
                           Must be smooth (infinitely differentiable) near x.
                           Example:  math.sin,  lambda x: x**3 - 2*x + 1

    x        : float     — the point at which we want f'(x).

    h        : float     — the INITIAL (coarsest) step size.  Default 0.1.
                           We intentionally start coarse. The iterative
                           deepening loop will halve this automatically.
                           HARD LOWER BOUND:  h >= 1e-5  is enforced below.
                           Setting h below 1e-5 triggers catastrophic
                           cancellation in float64 arithmetic.

    order    : int       — the base error order of the central-difference
                           formula.  Default 2 (O(h²) central difference).
                           Must be a positive integer.  Enforced below.

    tol      : float     — convergence tolerance.  The while-loop stops when
                           |L_current − L_previous| < tol.  Default 1e-8.

    max_iter : int       — safety ceiling on the number of halving steps.
                           Prevents an infinite loop if the function is
                           pathological and never converges.  Default 10.

    ════════════════════════════════════════════════════════════════════
    HOW THIS METHOD EXPORTS ITS SOLUTION  (outputs)
    ════════════════════════════════════════════════════════════════════

    Returns a dict — never a bare float.  A raw number with no metadata
    is a liability in numerical pipelines.

    {
        "derivative"      : float  — best Richardson estimate of f'(x).
        "iterations"      : int    — how many halving steps were taken.
        "converged"       : bool   — True if tol was met; False if max_iter hit.
        "final_h"         : float  — the step size used in the last iteration.
        "tableau"         : list   — the full Romberg tableau (list of columns)
                                     for inspection/debugging.
        "raw_column"      : list   — the first column: raw central-difference
                                     values at h, h/2, h/4, h/8, … for context.
    }
    ════════════════════════════════════════════════════════════════════
    """

    # ── GUARD 1: Validate that `order` is a positive integer.
    #
    # The formula uses 2^order as a scaling factor.  A float (e.g. 2.5) or
    # a non-positive integer (e.g. 0 or -2) would produce nonsensical factors
    # and silently corrupt the algebra.  We must blow up loudly.
    if not isinstance(order, int) or order < 1:
        raise ValueError(
            f"'order' must be a positive integer (got {order!r}).\n"
            f"For central difference use order=2. "
            f"If you are using a higher-order base stencil, pass its integer order."
        )
    # ↑ isinstance() check catches floats like 2.0 that look like integers
    # but are not.  This is intentionally strict — if the caller means 2,
    # they should write 2, not 2.0.

    # ── GUARD 2: Enforce the minimum step-size floor.
    #
    # In IEEE 754 double-precision (float64), the machine epsilon is ~2.2e-16.
    # When h < 1e-5, the two function values f(x+h) and f(x-h) become so close
    # that their difference loses 10+ significant digits to subtraction noise.
    # The round-off error then dominates the truncation error, and Richardson
    # Extrapolation amplifies that noise instead of cancelling it.
    if h < 1e-5:
        raise ValueError(
            f"Initial step size h={h} is below the safe floor of 1e-5.\n"
            f"For float64 arithmetic, h < 1e-5 causes catastrophic cancellation.\n"
            f"Use h >= 1e-5 (the default h=0.1 is a safe starting point)."
        )
    # ↑ This converts a silent wrong answer into a loud, actionable error message.
    # Junior engineers get bitten by this constantly.  Senior engineers make it
    # impossible to fall into.

    # ── GUARD 3: Validate that f is callable.
    #
    # Python will raise a TypeError eventually if f is not callable, but the
    # message ("'float' object is not callable") is cryptic inside a loop.
    # We intercept it here and give a clear message before anything runs.
    if not callable(f):
        raise ValueError(
            f"'f' must be a callable function (got {type(f).__name__!r}).\n"
            f"Example: pass f=math.sin, not f=math.sin(x)."
        )

    # ── INITIALISE the Romberg tableau.
    #
    # The tableau is a list of columns.  Column 0 will be the raw
    # central-difference values at decreasing step sizes.
    # Columns 1, 2, 3, … are successively extrapolated from the column before.
    #
    # Visual layout of the full 4-iteration tableau:
    #
    #   Col 0 (raw)   Col 1 (O(h⁴))   Col 2 (O(h⁶))   Col 3 (O(h⁸))
    #   A(h)
    #   A(h/2)        R(0,0)
    #   A(h/4)        R(1,0)           R(1,1)
    #   A(h/8)        R(2,0)           R(2,1)           R(2,2)
    #
    # Each diagonal entry is better than the one above it.  The bottom-right
    # corner is always our best current estimate.

    raw_col = []
    # ↑ This will hold the growing first column of the tableau.
    # We add one new raw A(h / 2^k) value each iteration.

    tableau = []
    # ↑ Will hold every column as a list of lists, for transparency and
    # debugging. The final answer is always tableau[-1][-1].

    current_h    = h
    # ↑ We track the step size as a live variable so we can report it
    # in the output and check it hasn't crossed dangerous thresholds.

    L_previous   = None
    # ↑ We need to compare consecutive best estimates to test convergence.
    # None signals "no estimate yet" on the first iteration.

    converged    = False
    iteration    = 0
    # ↑ Bookkeeping: how many halving steps did we take, and did we converge?


    # ══════════════════════════════════════════════════════════════════
    #  ITERATIVE DEEPENING LOOP
    #
    #  Each pass of this loop:
    #    1. Adds one new raw approximation at the current (halved) step size.
    #    2. Builds the entire new right edge of the Romberg tableau.
    #    3. Reads the best estimate from the bottom-right corner.
    #    4. Checks for convergence or max_iter.
    # ══════════════════════════════════════════════════════════════════

    while iteration < max_iter:

        # ── STEP A: Compute one new raw central-difference value.
        #
        # On iteration 0, step = h          → A(h)
        # On iteration 1, step = h/2        → A(h/2)
        # On iteration 2, step = h/4        → A(h/4)
        # On iteration k, step = h / 2^k   → A(h / 2^k)
        #
        # We append rather than recompute because every prior raw value
        # is still valid — we are only ever ADDING resolution, never
        # discarding work already done.

        new_raw = _central_diff(f, x, current_h)
        # ↑ One cheap function call to the central-difference helper.
        # We need f evaluated at two points: x + current_h and x - current_h.

        raw_col.append(new_raw)
        # ↑ Grow the first column of the tableau by one row.

        # ── STEP B: Guard against h falling below the catastrophic floor
        #           INSIDE the loop.
        #
        # The initial h was validated above, but each iteration halves h.
        # If the caller set h=2e-5 and max_iter=10, by iteration 5 the
        # step would be h/32 ≈ 6e-7, which is dangerously small.
        # We catch this mid-loop and stop cleanly rather than producing
        # corrupted results silently.

        if current_h < 1e-5:
            # We've crossed the floor inside the loop.  Stop here and
            # return what we have, marking as not converged.
            break
        # ↑ Note: we do NOT raise here (only warn by setting converged=False).
        # By this point we may already have a good estimate from prior
        # iterations; throwing that away with an exception would be wasteful.

        # ── STEP C: We need at least 2 raw values to extrapolate.
        #           On the very first iteration there is only one value,
        #           so skip the tableau build and go collect a second point.

        if len(raw_col) < 2:
            current_h /= 2.0
            # ↑ Halve h so the next iteration produces a finer approximation.
            iteration  += 1
            continue

        # ── STEP D: Build the new right edge of the Romberg tableau.
        #
        # We start from the raw column and repeatedly apply
        # _romberg_extrapolate(), each time passing an order that is
        # 2 higher than the last (because central difference only has even
        # error powers: h², h⁴, h⁶ …).
        #
        # VISUAL EXAMPLE after 3 iterations (raw_col has 3 entries):
        #
        #   raw_col = [A(h), A(h/2), A(h/4)]
        #
        #   Pass 1: _romberg_extrapolate(raw_col,     order=2)
        #           → col1 = [R(0,1), R(1,1)]    (eliminates O(h²) error)
        #
        #   Pass 2: _romberg_extrapolate(col1,         order=4)
        #           → col2 = [R(0,2)]             (eliminates O(h⁴) error)
        #
        #   The bottom-right corner col2[-1] = R(0,2) is our best estimate:
        #   it has error O(h⁶).

        tableau = [raw_col[:]]
        # ↑ Reset and rebuild the entire tableau from the full raw column.
        # We slice with [:] to store a COPY, not a reference.
        # Storing a reference would mean future .append() calls to raw_col
        # would silently mutate tableau[0] — a notoriously subtle Python bug.

        current_order = order
        # ↑ Track which order we are cancelling at each pass.
        # Starts at base_order (e.g. 2), then 4, then 6, …

        working_col = raw_col[:]
        # ↑ The column we will feed into the next extrapolation pass.
        # Again a copy, not a reference.

        while len(working_col) > 1:
            # ↑ Keep building new columns until the column is reduced to
            # a single entry (the best estimate at this depth).

            next_col = _romberg_extrapolate(working_col, current_order)
            # ↑ Apply one level of Richardson cancellation across every
            # adjacent pair in the current column.

            tableau.append(next_col)
            # ↑ Store the new column in the tableau for transparency.

            working_col   = next_col
            # ↑ The output of this pass becomes the input for the next pass.

            current_order += 2
            # ↑ Advance the order by 2 (h² → h⁴ → h⁶ → …).
            # Using += 2 (not += order) is correct here: we are always
            # climbing the even-power ladder, regardless of what the
            # base order was.

        # ── STEP E: Read the best estimate from the bottom-right corner
        #           of the tableau.
        #
        # After building all columns, working_col has exactly one element.
        # That element is the most extrapolated, highest-order estimate
        # we can compute from all the raw values collected so far.

        L_current = working_col[0]
        # ↑ This is the diagonal entry in the Romberg tableau —
        # the "most refined" number we have.

        # ── STEP F: Check convergence.
        #
        # We compare the current best estimate to the previous one.
        # If they agree to within `tol`, the extrapolation has stabilised
        # and further halving will not improve the answer meaningfully.

        if L_previous is not None and abs(L_current - L_previous) < tol:
            converged = True
            break
        # ↑ Note: we skip this check on the second iteration (L_previous is None
        # on the very first estimate). We always need at least two estimates
        # before convergence can be declared.

        # ── STEP G: Prepare for the next iteration.

        L_previous  = L_current
        # ↑ Slide the window: what is "current" now becomes "previous" next time.

        current_h  /= 2.0
        # ↑ Halve the step size. This is the "iterative deepening":
        # each iteration adds one more level of resolution to the tableau.

        iteration  += 1
        # ↑ Increment the iteration counter for bookkeeping and max_iter guard.


    # ── After the loop: package and return results.
    #
    # If the loop completed (max_iter hit) or broke early (h floor crossed),
    # L_current holds the best estimate we managed to compute.
    # If somehow we exited before even one full extrapolation, fall back
    # gracefully to the single raw value we have.

    if not tableau or len(tableau[-1]) == 0:
        # Edge case: max_iter=1 or h floor hit on first pass.
        # We have at least one raw value; return it with a warning.
        best_estimate = raw_col[-1] if raw_col else float('nan')
    else:
        best_estimate = tableau[-1][-1]
    # ↑ tableau[-1] is the rightmost (most extrapolated) column.
    # tableau[-1][-1] is its last (deepest) entry — the best single number.

    return {
        "derivative"  : best_estimate,
        # ↑ The primary output: the Richardson/Romberg estimate of f'(x).

        "iterations"  : iteration,
        # ↑ How many halving steps were actually taken.  Useful for profiling
        # and for understanding how "hard" the function was to converge.

        "converged"   : converged,
        # ↑ True = tol was met.  False = max_iter ceiling was hit, or h floor
        # was crossed.  The caller should ALWAYS check this flag.

        "final_h"     : current_h,
        # ↑ The step size at the final iteration.  If this is close to 1e-5,
        # the caller knows we are near the catastrophic cancellation boundary.

        "tableau"     : tableau,
        # ↑ The full Romberg tableau for inspection.  Each element is a list
        # (column), ordered from raw [column 0] to most extrapolated [last column].

        "raw_column"  : raw_col,
        # ↑ The first column: the unprocessed central-difference values.
        # Lets the caller see exactly what the base approximations looked like
        # before any extrapolation was applied.
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DEMONSTRATION — run this file directly to see all three upgrades in action
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    SEP = "═" * 65

    # ─────────────────────────────────────────────────────────────────
    #  TEST 1: f(x) = sin(x),  f'(x) = cos(x),  at x = 1.0
    #  Ground truth: cos(1.0) ≈ 0.5403023058681398
    # ─────────────────────────────────────────────────────────────────
    truth1 = math.cos(1.0)
    r1     = richardson_derivative(f=math.sin, x=1.0, h=0.5, order=2, tol=1e-10)

    print(SEP)
    print("  TEST 1:  f(x) = sin(x)  |  x = 1.0  |  h₀ = 0.5")
    print(SEP)
    print(f"  Ground truth cos(1.0)      :  {truth1:.16f}")
    print(f"  Richardson/Romberg answer  :  {r1['derivative']:.16f}")
    print(f"  Absolute error             :  {abs(r1['derivative'] - truth1):.2e}")
    print(f"  Converged                  :  {r1['converged']}")
    print(f"  Iterations taken           :  {r1['iterations']}")
    print(f"  Final step size            :  {r1['final_h']:.2e}")
    print()
    print("  ── Romberg Tableau (each column = one extrapolation level) ──")
    for col_idx, col in enumerate(r1['tableau']):
        order_label = 2 + col_idx * 2
        print(f"  Col {col_idx}  [O(h^{order_label:2d}) base → cancelled]:", end="")
        for val in col:
            print(f"  {val:.10f}", end="")
        print()
    print(SEP)

    # ─────────────────────────────────────────────────────────────────
    #  TEST 2: f(x) = x⁵,  f'(x) = 5x⁴,  at x = 2.0
    #  Ground truth: 5 * 16 = 80.0
    # ─────────────────────────────────────────────────────────────────
    truth2 = 5 * (2.0 ** 4)
    r2     = richardson_derivative(f=lambda x: x**5, x=2.0, h=0.5, order=2)

    print()
    print(SEP)
    print("  TEST 2:  f(x) = x⁵  |  x = 2.0  |  h₀ = 0.5")
    print(SEP)
    print(f"  Ground truth 5·x⁴ at x=2  :  {truth2:.16f}")
    print(f"  Richardson/Romberg answer  :  {r2['derivative']:.16f}")
    print(f"  Absolute error             :  {abs(r2['derivative'] - truth2):.2e}")
    print(f"  Converged                  :  {r2['converged']}")
    print(f"  Iterations taken           :  {r2['iterations']}")
    print(SEP)

    # ─────────────────────────────────────────────────────────────────
    #  TEST 3: Input validation — should raise ValueError
    # ─────────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  TEST 3:  Input validation safeguards")
    print(SEP)

    for label, kwargs in [
        ("order = 0 (non-positive)",     dict(f=math.sin, x=1.0, order=0)),
        ("order = 1.5 (float)",          dict(f=math.sin, x=1.0, order=1.5)),
        ("h = 1e-6  (below floor)",      dict(f=math.sin, x=1.0, h=1e-6)),
        ("f = 3.14  (not callable)",     dict(f=3.14,     x=1.0)),
    ]:
        try:
            richardson_derivative(**kwargs)
            print(f"  ✗  {label:40s}  — no error raised (BUG)")
        except ValueError as e:
            first_line = str(e).splitlines()[0]
            print(f"  ✓  {label:40s}  — ValueError: {first_line}")

    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
#
#  HOW THE METHOD WORKS — END-TO-END WALKTHROUGH  (v2, all three upgrades)
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  UPGRADE 1 — ITERATIVE DEEPENING (the while loop)                       │
#  │                                                                         │
#  │  v1 did a single extrapolation step: it computed A(h) and A(h/2) and   │
#  │  combined them once.  That was a fixed single-step calculation.         │
#  │                                                                         │
#  │  v2 wraps the entire process in a while loop.  Each iteration halves h  │
#  │  and adds one new raw value to the tableau.  The loop keeps going until  │
#  │  one of two stopping conditions is met:                                 │
#  │    (a) Convergence: |L_current − L_previous| < tol                     │
#  │    (b) Safety cap:  iteration >= max_iter                               │
#  │                                                                         │
#  │  This means the method is now adaptive: a well-behaved smooth function  │
#  │  will converge in 3–4 iterations, while a barely-smooth function might  │
#  │  need 8–9 before tol is met.  The method does exactly as much work as   │
#  │  is needed — no more, no less.                                          │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  UPGRADE 2 — ROMBERG-STYLE TABLEAU (multi-level extrapolation)          │
#  │                                                                         │
#  │  v1 eliminated only the O(h²) error, leaving an O(h⁴) result.          │
#  │                                                                         │
#  │  v2 applies Richardson Extrapolation recursively, column by column:     │
#  │                                                                         │
#  │   Column 0  — raw central differences          error O(h²)             │
#  │   Column 1  — extrapolate column 0, order 2    error O(h⁴)             │
#  │   Column 2  — extrapolate column 1, order 4    error O(h⁶)             │
#  │   Column 3  — extrapolate column 2, order 6    error O(h⁸)             │
#  │   …                                                                     │
#  │                                                                         │
#  │  Each pass doubles the order of accuracy.  After 4 iterations, the     │
#  │  answer has error O(h⁸).  This is the same structure used in           │
#  │  Romberg Integration — here applied to derivatives.                     │
#  │                                                                         │
#  │  The key insight: you are not just recycling the same trick once.       │
#  │  You are applying the same algebraic cancellation to the ALREADY-       │
#  │  EXTRAPOLATED values, eliminating a brand-new, higher-order error term  │
#  │  each time.                                                             │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  UPGRADE 3 — INPUT VALIDATION AND SAFEGUARDS                            │
#  │                                                                         │
#  │  v1 warned about bad inputs in comments but trusted the caller.         │
#  │                                                                         │
#  │  v2 raises ValueError immediately on three conditions:                  │
#  │    (a) order is not a positive integer                                  │
#  │        → The scaling factor 2^order would be meaningless.              │
#  │    (b) h < 1e-5 at entry                                                │
#  │        → Float64 subtraction noise dominates below this floor.          │
#  │    (c) f is not callable                                                │
#  │        → Prevents a cryptic TypeError deep inside the loop.             │
#  │                                                                         │
#  │  Additionally, a mid-loop guard checks if h drops below 1e-5 during    │
#  │  iterative deepening and exits the loop cleanly rather than             │
#  │  accumulating corrupt values into the tableau.                          │
#  └─────────────────────────────────────────────────────────────────────────┘
#
# ══════════════════════════════════════════════════════════════════════════════
