# Eureka-EML

Discover interpretable equations from data using EML (Exp-Minus-Log) symbolic regression.

## Install

```bash
pip install -e .
pip install -e ".[web]"  # for Streamlit UI
```

## Quick Start

### Python API

```python
import numpy as np
from eureka import discover
from eureka.simplify import simplify_tree

x = np.linspace(-2, 2, 100).astype(np.float32)
y = np.exp(x).astype(np.float32)

result = discover(x, y, population=30, generations=50, timeout=60)
print(simplify_tree(result.tree))  # exp(x0)
print(f"MSE: {result.mse:.6e}")
```

### CLI

```bash
eureka data.csv --target price
eureka data.csv --target price --latex --timeout 120
```

### Web UI

```bash
streamlit run app.py
```

## How It Works

EML is a single binary operator: `eml(x, y) = exp(x) - ln(y)`. Together with the constant 1, it generates all elementary functions.

Eureka uses this uniformity to perform gradient-based symbolic regression:

1. Generate a population of random EML trees
2. Optimize each tree's parameters via Adam
3. Evolve tree structures (grow, prune, mutate)
4. Simplify the best tree into a readable equation

The result is an interpretable equation, not a black box.

## Project Structure

```
eureka/
  operator.py    # EML operator (differentiable)
  tree.py        # EML tree as PyTorch nn.Module
  regression.py  # Evolutionary + gradient search
  simplify.py    # EML tree -> readable equation (SymPy)
  cli.py         # Command-line interface
app.py           # Streamlit web UI
```
