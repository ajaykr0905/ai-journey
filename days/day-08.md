# Day 8 — Manual Output-Weight Gradients

**Date:** 2026-09-12

Each output weight connects one hidden activation to the output neuron. In matrix
form, all output-layer weight gradients are:

```text
dW₂ = a₁ᵀ(ŷ-y) / m
```

For a `2 → 2 → 1` network, `a₁` has shape `(m, 2)` and `dW₂` has shape `(2, 1)`.
`output_weight_gradient()` implements the equation without autograd.

## Videos

- Primary — *Backpropagation Details Pt. 1: Optimizing 3 parameters simultaneously*: https://www.youtube.com/watch?v=iyn2zdALii8
- Support — *Ch 7 — Inverse matrices, column space and null space*: https://www.youtube.com/watch?v=uQhTuRlWMxw

## Evidence

Code, shape checks, and numerical verification are present. Manual arithmetic and
lecture completion remain pending user confirmation.
