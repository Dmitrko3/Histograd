# Repository Structure

This document describes the planned folder structure for the HistoGrad project.

<pre><code>
HistoGrad_Plugin/
├── src/                               # Main source code directory
│   ├── main.py                        # Entry point & CLI setup
│   │
│   ├── core/                          # The Core Framework (Integration Eng)
│   │   ├── __init__.py
│   │   ├── interfaces.py              # Contains the IDerivativeMethod base class
│   │   ├── evaluator.py               # Calculates analytical derivative and error margins
│   │   └── visualizer.py              # Generates the Convergence Rate graph
│   │
│   └── plugins/                       # The Strategy Modules (Algorithm Engineers)
│       ├── __init__.py
│       ├── richardson.py              # Richardson Extrapolation module
│       └── advanced_method.py         # Second advanced method module
│
├── tests/                             # Automated Test Suite (QA Engineer)
│   ├── __init__.py
│   ├── test_richardson.py             # Unit tests for Algo 1
│   ├── test_advanced_method.py        # Unit tests for Algo 2
│   ├── test_core.py                   # Tests for the evaluator and visualizer
│   └── test_integration.py            # End-to-end tests for the whole CLI
│
├── docs/                              # Documentation (Documentation Lead)
│   └── MATH.md                        # Mathematical theory and summaries
│
├── requirements.txt                   # Project dependencies (e.g., numpy, matplotlib)
├── coding_standards.md                # Repo rules and AI prompt guidelines
└── README.md                          # Main project guide with placeholders
</code></pre>
