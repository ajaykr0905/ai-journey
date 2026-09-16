# Day 9 — Full Manual Backpropagation

**Date:** 2026-09-13

`full_backward()` implements gradients for `W₁`, `b₁`, `W₂`, and `b₂` using the
Day 5 derivation. Correctness is checked in two independent ways:

1. Centered finite differences perturb each parameter by `±ε`; this always runs.
2. `torch.autograd` computes a reference when PyTorch is installed; otherwise the
   runner reports a graceful skip.

The test tolerance is strict enough to catch sign, transpose, and averaging errors.

## Videos

- Primary — *Backpropagation Details Pt. 2*: https://www.youtube.com/watch?v=GKZoOHXGcLo
- Support — *Ch 8 — Nonsquare matrices as transformations between dimensions*: https://www.youtube.com/watch?v=v8VSDg_WQlA

## Evidence

The manual and finite-difference checks are executable in CI. The PyTorch comparison
is conditional on local availability. Lecture completion is pending confirmation.
