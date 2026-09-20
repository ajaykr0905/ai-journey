# Day 16 — Manual Backpropagation Through Add and Multiply

**Date:** 2026-09-20

## Goal

Make the reverse pass inspectable one edge at a time. The workbook's hands-on
task is to implement backward functions for addition and multiplication, then
verify a tiny graph by hand. This repository adds an independently written
operation-table oracle and deterministic trace. Ajay's own implementation and
hand calculation remain separate learner evidence.

## The two local rules

For `z = x + y`, changing either input changes the output by the same amount:

```text
dz/dx = 1
dz/dy = 1
```

For `z = x * y`, each local derivative is the other input:

```text
dz/dx = y
dz/dy = x
```

If a downstream calculation has already produced `dL/dz`, the chain rule sends
`dL/dz * dz/dx` to `x` and `dL/dz * dz/dy` to `y`. The implementation records
the upstream gradient, local derivative, contribution, and accumulator before
and after every edge update.

## Hand-verifiable graph

The fixed graph uses only the two operations above:

```text
y = ((a * b) + c) * d + a
a = 2, b = -3, c = 10, d = -2
```

The forward pass gives:

```text
product = a * b       = -6
shifted = product + c =  4
scaled  = shifted * d = -8
y       = scaled + a  = -6
```

Starting with `dy/dy = 1`, the reverse pass gives:

```text
dy/dscaled = 1
dy/da      = 1                         direct addition path
dy/dshifted = 1 * d = -2
dy/dd       = 1 * shifted = 4
dy/dproduct = -2 * 1 = -2
dy/dc       = -2 * 1 = -2
dy/da      += -2 * b = 6  -> total 7  multiplication path
dy/db       = -2 * a = -4
```

The final input gradients are `(a=7, b=-4, c=-2, d=4)`. The second update to
`a` is the concrete reason backward functions use accumulation instead of
assignment: `a` reaches the output along two paths.

## Three-way verification

The Day 16 experiment calculates the gradients in three ways:

1. an explicit manual reverse pass over declarative `NodeSpec` records;
2. the separate Day 15 `Value.backward()` engine;
3. derivatives of the expanded formula:
   `dy/da = b*d + 1`, `dy/db = a*d`, `dy/dc = d`, and
   `dy/dd = a*b + c`.

The validator requires all three paths to agree for every input. It also checks
that the shared input `a` receives two edge contributions and that the second
one starts from a non-zero accumulated gradient.

## Reproducible experiment

```bash
python scripts/run_day_16.py \
  --output artifacts/day-16-manual-backprop.md \
  --json-output artifacts/day-16-manual-backprop.json
```

The Markdown report contains the forward table, edge-by-edge reverse trace,
three-way gradient comparison, and a Mermaid graph labeled with values, final
gradients, and local derivatives. The JSON output preserves the same trace for
machine-readable milestone review.

## What the evidence does and does not establish

The repository verifies add/multiply local rules, reverse ordering, explicit
chain-rule multiplication, shared-path accumulation, graph validation, and
agreement with two independent gradient calculations. It does not establish
lecture completion, a learner-written backward pass, tensor broadcasting,
PyTorch proficiency, or completion of the Week 3 MLP gate.

## Learner evidence still needed

1. Confirm or explicitly skip the assigned Day 16 micrograd segment and support
   video.
2. Implement the backward functions for `add` and `mul` in Ajay's own fresh
   `Value` class without copying this reference. Record the commit or notebook.
3. Verify a tiny graph by hand and provide the calculation or image path.
4. Run the command above and record both output paths.
5. Explain in Ajay's own words why multiplication's local derivative is the
   other operand.
6. Explain why `a` receives two contributions in the fixed graph and why they
   must be added.

No lecture viewing, type-along, hand calculation, or personal explanation is
claimed here.

## Sources

- Curriculum lesson: Andrej Karpathy, *The spelled-out intro to neural networks
  and backpropagation: building micrograd*:
  https://www.youtube.com/watch?v=VMj-3S1tku0
- Scheduled support lesson: 3Blue1Brown, *A quick trick for computing
  eigenvalues*:
  https://www.youtube.com/watch?v=e50Bj7jn9IQ

Both sources are canonical workbook assignments. Their inclusion records the
plan; it does not prove that Ajay watched them.
