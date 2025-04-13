# computational-capacity

Exploring the relationship between network topology and computational capacity in constrained environments.

## Installation

### Creation of the dev environment

Run these steps when creating the project for the first time.

1. Initialize environment

```bash
uv init
```

Creates `pyproject.toml`, `uv.lock`, and `main.py` files.

2. Add dependencies

```bash
uv add numpy matplotlib seaborn torch
uv add ipykernel --optional dev
```

### Installing the project

Run this step to install the project after `git clone`ing the repo.

```bash
uv sync --extra dev  # to include ipykernel
```

## Initial goals

1. Build out the basic framework for the optimization problem.
2. Test optimization on simple problems.
3. Implement large-scale optimization.

### Building out the basic framework

The framework should consist of at least 3 modules:

1. A representation module: this handles the storage and manipulation of the network topology.
2. An optimization module: this handles optimizing topology. Examples: evolutionary algorithms, gradient-based approaches, etc.
3. A simulation module: this handles running tasks used in the optimization module. Examples: classification, pattern completion, memory.

### Testing optimization on simple problems
