# Analytical scaling-law planning

The Day 60 planner turns a declared token/parameter ratio into an inspectable
training target. It also exposes optimizer-step rounding, public-corpus reuse,
and alternative model/token allocations under the same analytical compute
budget. It does not fit a scaling law from observations.

## Reproduce the pinned plan

Build and independently recompute the report:

```bash
ai-journey-scaling build \
  --protocol config/day-60-scaling-plan.json \
  --output artifacts/day-60-scaling-plan.json
ai-journey-scaling verify \
  --report artifacts/day-60-scaling-plan.json
```

The checked-in protocol declares 100,000,000 parameters, 20 tokens per
parameter, and the common `6 × parameters × tokens` analytical training-compute
convention. The resulting target is 2,000,000,000 tokens and
1,200,000,000,000,000,000 analytical FLOPs.

The batch geometry uses sequence length 2,048, micro-batch size 2, gradient
accumulation 4, and data-parallel size 8. One optimizer step therefore accounts
for 131,072 tokens. The complete-update schedule requires 15,259 steps and
2,000,027,648 scheduled tokens, an explicit 27,648-token overshoot.

The plan carries forward the Day 59 accepted train partition of 69,626 content
tokens only to make data sufficiency visible. Meeting the 2B-token target with
that tiny fixture would require 28,724 complete passes plus 62,776 tokens from a
final pass. That extreme reuse is a warning, not a recommendation to train the
model or a claim that this corpus is suitable for a 100M-parameter run.

The checked-in corpus policy permits at most two epochs and requires at least
half of the target tokens to be distinct. The Day 59 fixture violates both
controls, so the sealed report records `corpus_admitted=false`. The command still
builds the report so the blocked scenario remains inspectable; it does not turn
the fixture into an authorized training input.

The runtime block is also declared, not observed: two accelerators at 100 TFLOP/s
sustained per accelerator and $3.50 per accelerator-hour. Those assumptions imply
6,000 seconds, 3.333 recurring accelerator-hours, and $11.666 recurring cost.
They are an arithmetic scenario for budgeting and must be replaced with quoted
prices and measured sustained throughput before approving a real run.

## IsoFLOP interpretation

The 1.2e18-FLOP budget is allocated to 50M, 100M, and 200M parameter candidates.
Their token/parameter ratios are 80, 20, and 5 respectively. The deterministic
selector identifies 100M parameters as the closest row to the protocol's
declared 20-token ratio. This planning comparison is not an empirical loss curve
and does not establish that the Chinchilla convention transfers to
this model, tokenizer, corpus, or hardware.

## Evidence and recovery contract

The loader rejects unknown, duplicate, oversized, and invalid protocol fields.
Reports contain every input and derived value, record analytical and learner
claim boundaries, and carry a SHA-256 integrity digest. Verification rebuilds the
report from the embedded protocol rather than trusting totals. Publication flushes
a temporary file before atomic replacement, and invalid input leaves an existing
report untouched.

CI builds an isolated wheel, runs the installed command, retains the report, and
verifies it from the installed package. No GPU, distributed process, training
loop, dataset download, loss measurement, or monetary spend occurs.

## Learner evidence still required

Ajay must separately confirm watching Stanford CS336 Lecture 9 and the assigned
LoRA support video. The workbook also requires Ajay to compute and explain the
100M-model target personally. A future real run needs an appropriate public
corpus, tokenizer and model definition, hardware/software manifest, approved
cost, executed commands, logs, utilization, memory, losses, and limitations.
