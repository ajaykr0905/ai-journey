# Day 6 — Pure-NumPy Forward Pass

**Date:** 2026-09-10

## Goal

Implement a `2 → 2 → 1` sigmoid network without a machine-learning framework.
`default_parameters()` supplies hardcoded weights and biases, making every run
reproducible. `forward()` returns `z₁`, `a₁`, `z₂`, and `a₂` so intermediate
values can be inspected.

The note focus is how activation functions compose simple operations into a
nonlinear curve instead of leaving the network as one linear transformation.

## Run

```bash
PYTHONPATH=src python -c \
  "import numpy as np; from ai_journey.neural_net import default_parameters, forward; print(forward(np.array([[1., 0.]]), default_parameters())['a2'])"
```

## Videos

- Primary — *Neural Networks Pt. 1: Inside the Black Box*: https://www.youtube.com/watch?v=CqOfi41LfDw
- Support — *Ch 5 — Three-dimensional linear transformations*: https://www.youtube.com/watch?v=rHLEWRxRGiM

## Evidence

The deterministic forward pass and shape tests are present. Lecture completion is
pending user confirmation.
