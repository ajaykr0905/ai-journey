# Building neural networks from scratch — 32 days in

This is a repository evidence review, not a learner diary. It summarizes what
the public code can prove after the first 32 scheduled days and separates that
from activities—watching lectures, typing code independently, and reflecting on
mistakes—that still require Ajay's own evidence.

## What the repository now proves

The work progresses through increasingly demanding verification boundaries:

- scalar reverse-mode automatic differentiation with independent numerical
  checks;
- character bigram and context-window models with deterministic data handling;
- a compact decoder-only transformer with held-out evaluation and exact restart;
- parameter-wide gradient audits and controlled, predeclared ablations;
- a hierarchical character model built from registered module containers; and
- an independent reconstruction of that hierarchy from raw parameters and tensor
  operations.

The final item is the halfway checkpoint. Day 32 provides the reference model.
Day 33 deliberately uses a second implementation path: manual embedding lookup,
time grouping, matrix multiplication, normalization, activation, dropout, and
output projection. After copying the same values into both models, automated
checks compare their logits, losses, and every parameter gradient. A centered
finite-difference probe adds a numerical check that does not depend on either
model's backward implementation.

Equivalence alone is not enough for a training system. The rebuild therefore
also checks deterministic initialization and fitting, complete-dataset metrics,
bounded sampling, small-slice memorization, and checkpoint recovery. Its restart
test interrupts dropout training, restores the model, optimizer, batch cursor,
and random-number state, and then requires the resumed result to equal the
uninterrupted control exactly.

## What changed in the engineering standard

Early exercises could be validated with a single expected number. Later work
needs evidence about state and failure behavior. Reports are now written with
atomic replacement, checkpoints are validated before mutating live objects, and
command-line experiments fail when a declared numerical or capacity gate fails.
Continuous integration runs a bounded version of each current milestone so that
the repository's claims remain reproducible on CPU.

This direction matters more than model size. A small experiment with explicit
data boundaries, deterministic controls, restart guarantees, and falsifiable
checks is a stronger portfolio artifact than an opaque training run with an
impressive-looking curve.

## Limits and next milestone

All current results use small public demo corpora and CPU-sized models. They are
not claims about production traffic, GPU throughput, distributed training,
statistical significance, or competitive model quality. The report fingerprints
detect content changes but do not provide cryptographic authorship.

The next milestone is to use these reliable primitives for controlled ablations
and fine-tuning work: predeclare the hypothesis, vary one factor, retain negative
results, and connect each conclusion to a reproducible report. Separately, Ajay
still needs to produce the curriculum's learner evidence: an unaided rebuild,
his own run and gap log, confirmation of the support lesson, and a personal
halfway reflection.
