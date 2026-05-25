# HistoGrad

HistoGrad is a numerical differentiation project developed in a multi-agent AI software development workshop.

The system calculates numerical derivatives using advanced differentiation methods and compares their accuracy against analytical derivatives.

---

# Project Goals

The project aims to:

- Implement Richardson Extrapolation
- Implement Automatic Differentiation using Dual Numbers
- Compare numerical accuracy between methods
- Analyze convergence behavior
- Provide a CLI interface for derivative calculations
- Generate error and convergence comparisons

---

# Methods Used

## Richardson Extrapolation

Richardson Extrapolation improves derivative approximations by combining multiple numerical approximations with different step sizes in order to reduce error.

## Automatic Differentiation

Automatic Differentiation computes derivatives during function evaluation using Dual Numbers instead of finite difference approximations.

---

# Repository Structure

```txt
Histograd/
│
├── README.md
├── requirements.txt
│
├── docs/
├── src/
├── tests/
└── examples/
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/Histograd.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

TODO: Add execution command after implementation is completed.

Example:

```bash
python src/main.py
```

---


# Documentation

Project documentation can be found in the `docs/` folder.

Included documentation:

- MATH.md
- CODING_STANDARDS.md
- DOCSTRING_TEMPLATE.md
- ARCHITECTURE_CHECK.md

---

# AI Usage

This project was developed in a multi agent AI environment.

Each team member worked with their own AI assistant while remaining fully responsible for understanding and validating all produced code.

---

# Team Roles

| Role | Responsibility |
|---|---|
| Product Owner | Requirements and backlog |
| Algorithm Engineer 1 | Richardson Extrapolation |
| Algorithm Engineer 2 | Automatic Differentiation |
| Integration Engineer | System integration and CLI |
| QA Engineer | Testing and coverage |
| Documentation Lead | Documentation and standards |

---

# Current Status

Project currently in active development as part of Sprint 1.

---
