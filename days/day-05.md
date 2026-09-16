# Day 5 — Two-Layer Backward Pass

**Date:** 2026-09-09

For a two-layer sigmoid network:

```text
z₁ = XW₁ + b₁        a₁ = σ(z₁)
z₂ = a₁W₂ + b₂       ŷ  = σ(z₂)
```

Using mean binary cross-entropy over `m` samples:

```text
dz₂ = ŷ-y
dW₂ = a₁ᵀdz₂ / m            db₂ = sum(dz₂) / m
da₁ = dz₂W₂ᵀ
dz₁ = da₁ ⊙ a₁ ⊙ (1-a₁)
dW₁ = Xᵀdz₁ / m             db₁ = sum(dz₁) / m
```

The implementation in `src/ai_journey/neural_net.py` preserves these shapes and
averages gradients once at each parameter boundary.

## Videos

- Primary — *Backpropagation calculus | Chapter 4*: https://www.youtube.com/watch?v=tIeHLnjs5U8
- Support — *Ch 4 — Matrix multiplication as composition*: https://www.youtube.com/watch?v=XkY2DOUCWMU

## Evidence

Derivation and code are present. Handwritten work and lecture completion remain
pending user confirmation.
