# Day 2 — Environment Diagnostics

**Date:** 2026-09-06

## Goal

Verify a Python 3.11 environment suitable for NumPy exercises and understand the
difference between local CPU, local GPU, Jupyter, and hosted Colab runtimes.

The lecture model is deliberately simple: a neuron computes a weighted sum,
adds a bias, and applies a nonlinearity. For one layer, write
`output = activation(inputs @ weights + bias)` by hand.

## Run

```bash
PYTHONPATH=src python -c \
  "from ai_journey.environment import environment_report; print(environment_report())"
```

The diagnostic calls `torch.cuda.is_available()` when PyTorch is installed and
returns a clear CPU-only message when it is not. It prints no sensitive local data.

## Videos

- Primary — *But what is a neural network? | Deep learning chapter 1*: https://www.youtube.com/watch?v=aircAruvnKk
- Support — *Ch 1 — Vectors, what even are they?*: https://www.youtube.com/watch?v=fNk_zzaMoSs

## Evidence

The local diagnostic is executable. A Jupyter or Colab run and screenshot remain
pending user completion.
