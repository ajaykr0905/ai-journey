# Day 17 — Finish micrograd and Train a Tiny MLP

**Date:** 2026-09-21

## Goal

Compose the scalar `Value` engine into neurons, dense layers, and a complete
multilayer perceptron. The canonical exercise is to train a tiny MLP on four
examples and show that its loss decreases. This repository supplies an
independently written, deterministic reference with automated checks. Ajay's
own typed implementation and run remain separate learner evidence.

## From scalar operations to an MLP

Each neuron computes an affine combination and applies `tanh`:

```text
neuron(x) = tanh(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

A dense layer evaluates several neurons against the same input vector. The
Day 17 network chains three dense layers:

```text
3 inputs → 4 tanh units → 4 tanh units → 1 tanh output
```

The parameter count follows directly from the affine maps:

```text
first layer:   4 × (3 weights + 1 bias) = 16
second layer:  4 × (4 weights + 1 bias) = 20
output layer:  1 × (4 weights + 1 bias) =  5
total:                                         41
```

Every weight and bias is a scalar `Value`. A forward pass therefore builds one
large computation graph from the same add, multiply, power, and `tanh`
operations verified on Days 15 and 16.

## The training loop

One full-batch step performs five explicit operations:

1. Build predictions and mean squared error for all four examples.
2. Add a small L2 penalty from the squared parameter-vector norm.
3. Clear parameter gradients before the new reverse pass.
4. Backpropagate once from the total loss.
5. Apply a simultaneous SGD update, with global gradient clipping available as
   a finite-value guard.

The scalar engine also clears every reachable node at the start of
`Value.backward()`. The separate `zero_parameter_gradients()` call keeps the
optimizer contract visible and prevents the stale-gradient mistake highlighted
by the lesson if the underlying engine later changes to ordinary accumulating
semantics.

## Parameters and gradients as vectors

The support lesson introduces abstract vector spaces. The training code makes
that idea concrete: the model state is one point in a 41-dimensional parameter
space, and backpropagation returns a 41-dimensional gradient vector at that
point. SGD moves in the negative-gradient direction:

```text
parameters_next = parameters_now - learning_rate × gradient
```

The Euclidean gradient norm is computed across all 41 scalar components. If it
exceeds the configured maximum, one shared scale preserves its direction while
reducing its magnitude.

## Deterministic evidence

With seed `1709`, 120 full-batch steps, learning rate `0.05`, and L2 coefficient
`0.0001`, the checked run produces:

```text
parameter count:                  41
initial total loss:       1.230079020
final total loss:         0.011585422
loss reduction:                 99.06%
initial sign accuracy:           25.0%
final sign accuracy:            100.0%
gradient probes:                     7
maximum probe error:          9.382e-11
```

Seven evenly spaced parameters, including the first and last, are compared with
centered finite differences before training. The validator also requires a
complete final parameter snapshot, increasing history steps, at least 75% loss
reduction, finite metrics, and correct signs for all four training examples.

These are training-set checks. They do not establish generalization.

## Reproducible experiment

```bash
python scripts/run_day_17.py \
  --output artifacts/day-17-scalar-mlp.md \
  --json-output artifacts/day-17-scalar-mlp.json
```

The Markdown report contains the loss trace, final predictions, gradient
probes, and an architecture diagram. The JSON output records the configuration,
dataset, parameter snapshot, and full trace for machine-readable review. CLI
arguments can change the seed, step count, learning rate, L2 coefficient,
gradient limit, and recording interval.

## What the evidence does and does not establish

The repository verifies deterministic initialization, scalar neuron and layer
composition, parameter enumeration, snapshots and restoration, mean squared
error plus L2 regularization, explicit gradient clearing, global gradient
norms, safe SGD updates, finite-difference agreement, decreasing loss, and full
training-set sign accuracy. It does not establish lecture completion, an
independently typed learner solution, performance on unseen data, tensor-level
efficiency, or completion of the Week 3 MLP gate.

## Learner evidence still needed

1. Confirm or explicitly skip the assigned Day 17 micrograd segment and support
   video.
2. Finish Ajay's own fresh micrograd implementation without copying this
   reference. Record the commit or notebook.
3. Train the four-example MLP and record the command, seed, output paths, and
   observed initial and final losses.
4. Explain in Ajay's own words why gradients must be cleared between optimizer
   steps.
5. Draw the 3→4→4→1 architecture from memory and derive its 41-parameter count.
6. Explain how the parameter vector and gradient vector connect the MLP to the
   abstract-vector-space lesson.

No lecture viewing, type-along work, personal explanation, or milestone pass is
claimed here.

## Sources

- Curriculum lesson: Andrej Karpathy, *The spelled-out intro to neural networks
  and backpropagation: building micrograd*:
  https://www.youtube.com/watch?v=VMj-3S1tku0
- Course index and lecture materials: Andrej Karpathy, *Neural Networks: Zero
  to Hero*: https://github.com/karpathy/nn-zero-to-hero
- Reference project description and educational scope: Andrej Karpathy,
  *micrograd*: https://github.com/karpathy/micrograd
- Scheduled support lesson: 3Blue1Brown, *Abstract vector spaces*:
  https://www.youtube.com/watch?v=TgKwz5Ikpc8

The workbook assigns these sources. Listing them records curriculum provenance;
it does not prove that Ajay watched or completed them. The implementation here
was written independently and does not copy source code from the reference
project.
