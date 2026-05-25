import math
import pytest


EPS = 1e-6


# Temporary placeholder implementation.
# Later replace this with:
# from automatic_differentiation import derivative
def derivative(f, x):
    if not callable(f):
        raise TypeError("f must be callable")
    if not isinstance(x, (int, float)):
        raise TypeError("x must be numeric")

    h = 1e-6
    return (f(x + h) - f(x - h)) / (2 * h)


def assert_close(actual, expected, tol=EPS):
    assert abs(actual - expected) <= tol


def test_constant_function():
    result = derivative(lambda x: 5, 3)
    assert_close(result, 0)


def test_polynomial_function():
    result = derivative(lambda x: x**2, 3)
    assert_close(result, 6)


def test_trigonometric_function():
    result = derivative(math.sin, 0)
    assert_close(result, 1)


def test_exponential_function():
    result = derivative(math.exp, 1)
    assert_close(result, math.e)


def test_logarithmic_function():
    result = derivative(math.log, 2)
    assert_close(result, 0.5)


def test_chain_rule():
    result = derivative(lambda x: math.sin(x**2), 2)
    expected = 2 * 2 * math.cos(2**2)
    assert_close(result, expected)


def test_nested_function_differentiation():
    result = derivative(lambda x: math.exp(math.sin(x**2)), 1)
    expected = math.exp(math.sin(1**2)) * 2 * 1 * math.cos(1**2)
    assert_close(result, expected)


def test_arithmetic_operations():
    result = derivative(lambda x: (x**2 + 3 * x - 5) / 2, 4)
    expected = (2 * 4 + 3) / 2
    assert_close(result, expected)


def test_large_input_values():
    result = derivative(lambda x: x**2, 1_000_000)
    assert_close(result, 2_000_000, tol=1e-2)


def test_small_input_values():
    result = derivative(lambda x: x**2, 1e-8)
    assert_close(result, 2e-8, tol=1e-8)


def test_zero_input():
    result = derivative(lambda x: x**3, 0)
    assert_close(result, 0, tol=1e-8)


def test_negative_input():
    result = derivative(lambda x: x**2, -3)
    assert_close(result, -6)


def test_boundary_domain_conditions():
    result = derivative(math.log, 1)
    assert_close(result, 1)


def test_precision_stability():
    result = derivative(lambda x: x**2, 1e-10)
    assert math.isfinite(result)


def test_repeated_execution_consistency():
    r1 = derivative(lambda x: x**2, 5)
    r2 = derivative(lambda x: x**2, 5)
    assert r1 == r2


def test_composite_function_behavior():
    result = derivative(lambda x: (math.exp(x) + math.sin(x)) * x, 1)
    expected = (math.e + math.sin(1)) + (math.e + math.cos(1))
    assert_close(result, expected)


def test_invalid_domain():
    with pytest.raises(ValueError):
        derivative(math.log, -1)


def test_logarithm_negative_number():
    with pytest.raises(ValueError):
        derivative(math.log, -5)


def test_invalid_function_syntax():
    with pytest.raises(AttributeError):
        derivative(lambda x: x.invalid_operation(), 1)


def test_empty_input():
    with pytest.raises(TypeError):
        derivative(None, 1)


def test_unsupported_operations():
    with pytest.raises(TypeError):
        derivative(lambda x: len(x), 1)


def test_overflow_conditions():
    with pytest.raises(OverflowError):
        derivative(math.exp, 1000)


def test_invalid_numeric_values():
    with pytest.raises(TypeError):
        derivative(lambda x: x**2, "invalid")


def test_derivative_consistency():
    result = derivative(lambda x: x**3, 2)
    assert_close(result, 12)


def test_chain_rule_stability():
    result = derivative(lambda x: math.sin(x**3), 1)
    expected = 3 * (1**2) * math.cos(1**3)
    assert_close(result, expected)


def test_result_reproducibility():
    results = [derivative(lambda x: x**2, 3) for _ in range(10)]
    assert all(result == results[0] for result in results)


def test_parser_and_output_integration():
    result = derivative(lambda x: x**2, 3)
    assert isinstance(result, (int, float))
    assert_close(result, 6)