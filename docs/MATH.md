# Mathematical Background

This document explains the numerical differentiation methods used in the HistoGrad project.

---

# 1. Richardson Extrapolation

Richardson Extrapolation is a numerical method used to improve the accuracy of derivative approximations.

Instead of calculating the derivative using only one approximation, the method combines multiple approximations with different step sizes in order to reduce the numerical error.

The method is based on the idea that the error of a numerical approximation follows a predictable mathematical pattern.

By comparing approximations computed with smaller and smaller step sizes, Richardson Extrapolation can cancel part of the error and produce a more accurate result.

---

## Main Idea

Suppose we approximate a derivative using a step size `h`.

We then calculate another approximation using a smaller step size such as `h / 2`.

These approximations contain different amounts of error.

Richardson Extrapolation combines them using a mathematical formula to eliminate lower-order error terms.

The result is a more accurate derivative approximation.

---

## Advantages

- Improves numerical accuracy
- Reduces truncation error
- More advanced than simple finite difference methods
- Useful for convergence analysis

---

## Limitations

- Requires multiple derivative evaluations
- Can become unstable for extremely small step sizes
- More computationally expensive than basic methods

---

## Expected Input

The method usually receives:

- mathematical function `f(x)`
- evaluation point `x`
- initial step size `h`
- extrapolation depth or number of levels

---

## Expected Output

The method returns:

- estimated derivative value at point `x`

---

# 2. Automatic Differentiation

Automatic Differentiation is an advanced method for calculating derivatives accurately.

Unlike finite difference methods, it does not estimate the derivative using subtraction between nearby values.

Instead, the derivative is calculated automatically while evaluating the function itself.

In this project, Automatic Differentiation is implemented using Dual Numbers.

---

## Dual Numbers

A dual number has the form:

```txt
a + bε
```

Where:

```txt
ε² = 0
```

The first part stores the regular function value.

The second part stores derivative information.

When mathematical operations are performed on dual numbers, derivative values propagate automatically through the computation.

This allows accurate derivative calculation without numerical approximation errors caused by finite differences.

---

## Main Idea

The method evaluates the original mathematical function using dual numbers instead of regular numbers.

During the computation:

- the real component tracks the normal function value
- the dual component tracks the derivative

At the end of the calculation, the derivative can be extracted directly from the dual component.

---

## Advantages

- Very accurate derivative computation
- Avoids subtraction cancellation errors
- Does not require very small step sizes
- Efficient for many mathematical functions

---

## Limitations

- Requires custom implementation of dual number operations
- More complex than basic numerical methods
- Some functions may require additional operator support

---

## Expected Input

The method usually receives:

- mathematical function `f(x)`
- evaluation point `x`

---

## Expected Output

The method returns:

- derivative value at point `x`

---

# 3. Comparison Between Methods

The project compares Richardson Extrapolation and Automatic Differentiation using several criteria.

The comparison includes:

- numerical accuracy
- convergence behavior
- computational cost
- stability
- error relative to analytical derivative

---

# 4. Ground Truth

The analytical derivative is used as the ground truth.

Example:

```txt
f(x) = sin(x)
f'(x) = cos(x)
```

The numerical derivative can then be compared to the analytical derivative to calculate the error.

---

# 5. Error Calculation

The absolute error can be calculated using:

```txt
absolute_error = |numerical_derivative - analytical_derivative|
```

Smaller error values indicate higher accuracy.

---

# 6. Convergence Analysis

The project also analyzes convergence behavior.

As the step size becomes smaller:

- Richardson Extrapolation should converge toward the analytical derivative
- Automatic Differentiation should remain highly accurate and stable

Graphs may be used to visualize convergence rates and error behavior for both methods.