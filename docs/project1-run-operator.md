# Project 1 run operator

Day 64 is a build day for launching the first project training run. This
operator turns the existing deterministic GPT-2 training command into a
recoverable evidence workflow: admit immutable inputs, execute the bounded
schedule, journal every measured optimizer update, seal the checkpoint and
report, then verify the bundle from stored bytes.

## Run the checked-in CPU fixture

Use a new destination. The operator refuses to overwrite an existing run.

```bash
ai-journey-project1 init \
  --source data/day-19-demo-names.txt \
  --spec config/gpt2-training.json \
  --contract config/day-64-project1-run.json \
  --run-dir /tmp/day-64-project1

ai-journey-project1 run \
  --source data/day-19-demo-names.txt \
  --spec config/gpt2-training.json \
  --run-dir /tmp/day-64-project1

ai-journey-project1 status --run-dir /tmp/day-64-project1
ai-journey-project1 verify --run-dir /tmp/day-64-project1
```

The final directory contains the immutable contract, analytical preflight,
hash-chained journal, training checkpoint, sealed training report, and bundle
index. Verification re-reads every artifact, checks the journal lifecycle,
recomputes the preflight and bundle seals, and verifies the training report.

## Evidence boundary

The checked-in contract is a tiny byte-level CPU fixture. It validates the run
operator and exact evidence path; it is not a useful pretrained model and does
not establish GPU throughput, distributed scaling, serving performance, or
production readiness. The operator does not upload to Weights & Biases, and an
automation run is not evidence that Ajay watched the RAG lesson, launched a
personal training job, or wrote a reflection. A larger public-data run needs a
new immutable spec and contract plus its own compute approval and measured
hardware evidence.

