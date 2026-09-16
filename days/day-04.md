# Day 4 — Two-Input Sigmoid Neuron Derivation

**Date:** 2026-09-08

For inputs `x₁, x₂`, weights `w₁, w₂`, bias `b`, and binary target `y`:

```text
z = w₁x₁ + w₂x₂ + b
ŷ = σ(z) = 1 / (1 + e⁻ᶻ)
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

The chain rule gives:

```text
∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)
∂ŷ/∂z = ŷ(1-ŷ)
∂L/∂z = ŷ-y
∂L/∂w₁ = (ŷ-y)x₁
∂L/∂w₂ = (ŷ-y)x₂
∂L/∂b  = ŷ-y
```

For a batch, average each sample's gradient. This simplification relies on the
specific pairing of sigmoid output and binary cross-entropy loss.

## Videos

- Primary — *Backpropagation, intuitively | Chapter 3*: https://www.youtube.com/watch?v=Ilg3gGewQ5U
- Support — *Ch 3 — Linear transformations and matrices*: https://www.youtube.com/watch?v=kYB8IZa5AuE

## Evidence

The typed derivation is complete. **A handwritten derivation photo is pending user
completion; no fabricated photo is included.**
