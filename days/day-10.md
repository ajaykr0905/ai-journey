# Day 10 — Train a NumPy Network on XOR

**Date:** 2026-09-14

XOR is not linearly separable, so the exercise trains a `2 → 4 → 1` sigmoid network.
The implementation uses full-batch gradient descent, a fixed seed, and fixed
hyperparameters. It reports sampled losses and thresholded predictions.

## Acceptance criteria

- Final loss is lower than initial loss.
- Final loss is below the tested threshold.
- All four XOR labels are correct.
- Two runs with the same seed produce equal outputs.

## Videos

- Primary — *Gradient Descent, Step-by-Step*: https://www.youtube.com/watch?v=sDv4f4s2SB8
- Support — *Ch 9 — Dot products and duality*: https://www.youtube.com/watch?v=LyGKycYT2v0

## Evidence

Training and deterministic tests are present. Lecture completion and a short
learning-curve reflection remain pending user confirmation.
