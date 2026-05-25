# Docstring Templates

This document defines the standard docstring format for all functions in the HistoGrad project.

---

# Basic Function Template

```python
def function_name(parameter1, parameter2):
    """
    Short description of the function.

    Parameters:
        parameter1: description of parameter1
        parameter2: description of parameter2

    Returns:
        Description of returned value.
    """
```

---

# Numerical Method Template

```python
def compute_derivative(function, x, h):
    """
    Computes the numerical derivative of a function.

    Parameters:
        function: mathematical function to evaluate
        x: evaluation point
        h: step size

    Returns:
        Estimated derivative value at point x.
    """
```

---

# Richardson Extrapolation Template

```python
def richardson_extrapolation(function, x, h, levels):
    """
    Computes derivative using Richardson Extrapolation.

    Parameters:
        function: mathematical function
        x: evaluation point
        h: initial step size
        levels: number of extrapolation levels

    Returns:
        Improved derivative approximation.
    """
```

---

# Automatic Differentiation Template

```python
def automatic_differentiation(function, x):
    """
    Computes derivative using Automatic Differentiation.

    Parameters:
        function: mathematical function
        x: evaluation point

    Returns:
        Derivative value at point x.
    """
```

---

# Testing Function Template

```python
def test_function_name():
    """
    Tests a specific function behavior.

    Verifies:
        - expected output
        - edge cases
        - numerical accuracy
    """
```

---

# Class Template

```python
class ClassName:
    """
    Short description of the class.

    Attributes:
        attribute_name: description
    """
```

---

# Notes

- Every public function must include a docstring.
- Descriptions should stay short and clear.
- Parameters and return values must always be documented.
- Complex mathematical logic should also be explained in MATH.md.