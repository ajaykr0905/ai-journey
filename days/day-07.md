# Day 7 — Manual Output-Bias Gradient

**Date:** 2026-09-11

With sigmoid output and mean binary cross-entropy, each sample contributes
`dz₂ = ŷ-y`. Because the output bias is broadcast across the batch:

```text
db₂ = (1/m) Σᵢ (ŷᵢ-yᵢ)
```

`output_bias_gradient()` calculates this value directly. Day 9 compares it with a
centered finite-difference estimate.

## Videos

- Primary — *Neural Networks Pt. 2: Backpropagation Main Ideas*: https://www.youtube.com/watch?v=IN2XmBhILt4
- Support — *Ch 6 — The determinant*: https://www.youtube.com/watch?v=Ip3X9LOh2dk

## Evidence

Code and automated verification are present. A separate manual arithmetic check
and lecture completion remain pending user confirmation.
