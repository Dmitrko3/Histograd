import math
import pytest


EPS = 1e-6


# Temporary placeholder implementation.
# Later replace this with:
# from richardson import richardson_derivative
def richardson_derivative(f, x, h):
    if not callable(f):
        raise TypeError("f must be callable")
    if not isinstance(x, (int, float)):
        raise TypeError("x must be numeric")
    if not isinstance(h, (int, float)):
        raise TypeError("h must be numeric")
    if h == 0:
        raise ValueError("h cannot be zero")
    if h < 0:
        raise ValueError("h cannot be negative")

    d_h = (f(x + h) - f(x - h)) / (2 * h)
    d_h2 = (f(x + h / 2) - f(x - h / 2)) / h

    return (4 * d_h2 - d_h) / 3


def assert_close(actual, expected, tol=EPS):
    assert abs(actual - expected) <= tol


def test_polynomial_derivative():
    result = richardson_derivative(lambda x: x**2, 3, 0.01)
    assert_close(result, 6)


def test_trigonometric_derivative():
    result = richardson_derivative(math.sin, 0, 0.01)
    assert_close(result, 1)


def test_exponential_derivative():
    result = richardson_derivative(math.exp, 1, 0.01)
    assert_close(result, math.e)


def test_logarithmic_derivative():
    result = richardson_derivative(math.log, 2, 0.01)
    assert_close(result, 0.5)


def test_accuracy_improvement():
    f = math.sin
    x = 1
    h = 0.1

    basic = (f(x + h) - f(x - h)) / (2 * h)
    richardson = richardson_derivative(f, x, h)
    expected = math.cos(x)

    assert abs(richardson - expected) < abs(basic - expected)


def test_output_consistency():
    r1 = richardson_derivative(lambda x: x**3, 2, 0.01)
    r2 = richardson_derivative(lambda x: x**3, 2, 0.01)
    assert r1 == r2


def test_very_small_h():
    result = richardson_derivative(math.sin, 1, 1e-8)
    assert math.isfinite(result)


def test_very_large_h():
    result = richardson_derivative(math.sin, 1, 1)
    assert math.isfinite(result)


def test_zero_h():
    with pytest.raises(ValueError):
        richardson_derivative(math.sin, 1, 0)


def test_negative_h():
    with pytest.raises(ValueError):
        richardson_derivative(math.sin, 1, -0.01)


def test_non_smooth_function():
    result = richardson_derivative(abs, 0, 0.01)
    assert math.isfinite(result)


def test_steep_curvature_function():
    result = richardson_derivative(lambda x: math.exp(10 * x), 0.1, 0.001)
    assert math.isfinite(result)


def test_floating_point_precision():
    result = richardson_derivative(lambda x: x**2, 1e-8, 1e-8)
    assert math.isfinite(result)


def test_boundary_values():
    result = richardson_derivative(lambda x: x**2, 0, 0.01)
    assert_close(result, 0)


def test_invalid_function_input():
    with pytest.raises(TypeError):
        richardson_derivative("not a function", 1, 0.01)


def test_empty_input():
    with pytest.raises(TypeError):
        richardson_derivative(None, 1, 0.01)


def test_invalid_numeric_values():
    with pytest.raises(TypeError):
        richardson_derivative(math.sin, "invalid", 0.01)


def test_unsupported_expression():
    def bad_function(x):
        return "invalid output"

    with pytest.raises(TypeError):
        richardson_derivative(bad_function, 1, 0.01)


def test_overflow_conditions():
    with pytest.raises(OverflowError):
        richardson_derivative(lambda x: math.exp(x), 1000, 0.01)


def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        richardson_derivative(lambda x: 1 / (x - 0.01), 0, 0.01)


def test_repeated_calculations():
    results = [richardson_derivative(lambda x: x**2, 3, 0.01) for _ in range(10)]
    assert all(result == results[0] for result in results)


def test_reproducibility():
    result = richardson_derivative(lambda x: x**3, 2, 0.01)
    assert_close(result, 12)


def test_precision_stability():
    result = richardson_derivative(math.cos, 0, 0.001)
    assert_close(result, 0)