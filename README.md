# Dynamic Symbolic Manipulation and Reasoning (DSMR) Framework

## Overview

The Dynamic Symbolic Manipulation and Reasoning (DSMR) framework enhances mathematical comprehension and problem-solving capabilities in AI systems. It integrates symbolic mathematics with neural networks within a vision-enhanced Large Language Model (LLM) framework.

## Features

- Riemannian Manifold operations
- Lie Group operations
- Poisson Manifold and symplectic geometry
- Neural-Symbolic Mapping
- Stochastic Neural Updates
- Symbolic Manipulation
- LLM Integration (using GPT-2)
- Hamiltonian Flows

## Requirements

- Python 3.8+
- PyTorch 1.8+
- SymPy
- NumPy
- SciPy
- Transformers (Hugging Face)
- pytest

## Usage

Basic example of using the DSMR framework:

```python
from dsmr import DSMR

# Initialize the DSMR system
dsmr = DSMR(neural_dim=768, symbolic_dim=50)

# Set an initial equation
dsmr.set_equation("x**2 + 2*x + 1")

# Perform iterations of symbolic manipulation and reasoning
results = dsmr.iterate(n_steps=3)

# Print the results
print("Initial equation:", dsmr.symbolic_repr)
for i, result in enumerate(results, 1):
    print(f"Step {i}:", result)
```

## Testing

To run the test suite:

```
pytest dsmr_test.py
```

