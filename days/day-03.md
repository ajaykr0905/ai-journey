# Day 3 — Gradient Descent on `f(x) = x²`

**Date:** 2026-09-07

For `f(x) = x²`, the derivative is `df/dx = 2x`. With learning rate `η`, one
gradient-descent update is:

```text
x_next = x - η(2x)
```

Starting at `x = 4` with `η = 0.1`, each step multiplies `x` by `0.8`, so both
the parameter magnitude and loss approach zero deterministically.

Interpret the cost as a landscape. The gradient points toward steepest ascent,
so the negative gradient points downhill; the learning rate controls step size.

## Run

```bash
python scripts/run_days_00_11.py --artifact-dir artifacts
```

If matplotlib is installed, the runner saves `day-03-gradient-descent.png`.

## Videos

- Primary — *Gradient descent, how neural networks learn | Chapter 2*: https://www.youtube.com/watch?v=IHZwWFHWa-w
- Support — *Ch 2 — Linear combinations, span, and basis vectors*: https://www.youtube.com/watch?v=k7RM-ot2NWY

## Evidence

The implementation and convergence test are present. Lecture completion remains
pending user confirmation.
