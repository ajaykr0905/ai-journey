# Day 18 — Blank-File micrograd Rebuild

**Date:** 2026-09-22

## Goal

Rebuild the scalar autodiff engine and tiny MLP from a blank file, without
opening the Day 15–17 implementation. The canonical plan schedules no new
primary lecture. This is a consolidation day whose useful evidence must come
from Ajay's own attempt, mistakes, and comparison notes.

## Why no automated substitute is recorded

The repository already contains tested reference implementations. Copying or
regenerating another implementation would not demonstrate recall. The intended
exercise measures which concepts remain available without prompts:

1. A scalar value stores data, gradient, graph parents, and a local backward
   rule.
2. Arithmetic operations build a directed acyclic computation graph.
3. Reverse topological order applies the chain rule after downstream gradients
   are known.
4. Shared paths accumulate gradients with `+=`.
5. Neurons compose into layers and an MLP.
6. Each optimizer step clears gradients, runs a forward pass, backpropagates,
   and updates parameters.

## Learner evidence still needed

1. Start from a genuinely blank file and record the start and finish times.
2. Commit Ajay's independent rebuild or store the notebook path.
3. Run at least one tiny graph and the four-example MLP.
4. Diff the rebuild against the reference only after the attempt.
5. Record every missing concept or bug in a short gap log.
6. Confirm or explicitly skip the scheduled 3Blue1Brown calculus support
   lesson, *The essence of calculus*.

No rebuild, lecture viewing, or personal reflection is claimed here.

## Source

- Scheduled support lesson: 3Blue1Brown, *The essence of calculus*:
  https://www.youtube.com/watch?v=WUvTyaaNkzM

The workbook assigns this source. Listing it records curriculum provenance; it
does not prove that Ajay watched it.
