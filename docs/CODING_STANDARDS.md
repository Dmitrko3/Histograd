# Coding Standards

## General Rules

- Write simple and readable code.
- Every function should have one clear purpose.
- Do not upload code you do not understand.
- Keep naming consistent across the project.
- Always test code before pushing to GitHub.

---

## Python Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Variables | snake_case | step_size |
| Functions | snake_case | compute_derivative |
| Classes | PascalCase | RichardsonMethod |
| Constants | UPPER_CASE | MAX_ITERATIONS |

---

## Function Rules

- Every function must include a docstring.
- Functions should stay relatively short.
- Avoid duplicated code.
- Use meaningful parameter names.

Example:

```python
def compute_derivative(function, x, h):
    """
    Computes numerical derivative using finite difference.

    Parameters:
        function: mathematical function
        x: evaluation point
        h: step size
    """
```

---

## Documentation Rules

- Important logic should be documented.
- Mathematical formulas should be explained in MATH.md.
- AI-generated suggestions that were modified should be noted in prompt logs.

---

## GitHub Rules

- Pull before starting work.
- Push changes regularly.
- Do not push broken code.
- Write clear commit messages.

Good commit message examples:

```txt
Added meeting transcript template
Updated README placeholders
Created coding standards document
```

Bad examples:

```txt
fix
stuff
update
```

---

## Testing Rules

- Numerical methods must be tested against analytical derivatives.
- Edge cases should be tested.
- QA tests should remain separated from implementation code.

---

## AI Usage Rules

- AI is a tool, not the final decision maker.
- Team members are responsible for all code they submit.
- Conflicts between AI suggestions should be documented when relevant.