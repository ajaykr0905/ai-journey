# Decoder roofline analysis

The Day 55 command compares the decoder's matmul and LayerNorm arithmetic
intensity against explicit compute and memory-bandwidth ceilings. It produces a
sealed JSON report that can be recomputed without an accelerator.

```bash
ai-journey-roofline build \
  --config config/day-55-roofline-plan.json \
  --output /tmp/day-55-roofline-report.json

ai-journey-roofline verify \
  --report /tmp/day-55-roofline-report.json
```

The output path must not alias the protected plan, including resolved symlinks
or existing hardlinks. Reports are bounded to 1 MiB, reject duplicate JSON
fields, use atomic replacement, and carry a canonical SHA-256 digest. The digest
checks internal consistency; it is not a signature.

## Pinned analytical comparison

The checked-in plan reuses the 6,738,415,616-parameter dense SwiGLU geometry
from the resource planner with batch size 1, sequence length 2,048, and BF16
tensors. Its hardware ceiling is intentionally generic: 100 TFLOP/s and 1
TB/s, giving a ridge point of 100 FLOPs per byte. These are declared exercise
inputs, not detected device specifications or vendor benchmark claims.

The deterministic report models ten kernel families. Eight matmul families are
compute-bound under the declared ridge. The two LayerNorm families are
memory-bound at 1.3333 FLOPs per compulsory byte. Across the isolated families,
the report counts 29,265,974,263,808 FLOPs and 45,121,273,856 compulsory bytes.
Its 0.295887679-second serial lower bound is a roofline calculation, not an
observed runtime.

## Boundaries

Matmul traffic includes each input tensor and one output tensor. LayerNorm uses
an explicit eight-operations-per-element, two-input-pass model and one output
write. The analysis excludes cache behavior, fusion, residual traffic, scale and
bias reads, accumulator spills, allocator traffic, kernel-launch overhead,
communication, and overlap. It does not prove GPU availability, measured
throughput, production readiness, lecture viewing, a manual exercise, or learner
understanding.
