# Exact GPT-2 checkpoint interoperability

Day 45 implements the GPT-2 124M architecture without using Transformers at
runtime. The checked-in source manifest pins `openai-community/gpt2` at revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, including exact configuration and
checkpoint SHA-256 digests. The 548 MB checkpoint remains outside Git.

## Run the audited import

Place the pinned `config.json` and `pytorch_model.bin` in one artifact directory.
Generate a CPU reference archive with an independent implementation containing
exactly `token_ids` and `logits` tensors, then run:

```bash
ai-journey-gpt2 import \
  --source config/gpt2-124m-source.json \
  --artifacts /path/to/pinned-gpt2 \
  --reference /path/to/reference_logits.pt \
  --output /path/to/gpt2-evidence.json

ai-journey-gpt2 verify \
  --report /path/to/gpt2-evidence.json
```

The import verifies source bytes before deserialization, uses PyTorch's
weights-only CPU loader, accepts only the known legacy base-model layout,
synthesizes the tied language-model head from the token embedding, and rejects
missing, unexpected, wrong-shaped, non-floating, or non-finite learned state.
Evidence is written atomically and binds the source revision, architecture,
149 learned tensors, unique parameter count, runtime versions, token probe, all
logit metrics, and both output fingerprints.

## Measured result

The verified local CPU run used token IDs `15496, 11, 995, 0, 50256` and an
independent Transformers 4.57.1 reference:

| Metric | Result |
|---|---:|
| Unique parameters | 124,439,808 |
| Learned-state fingerprint | `c59cdbf097c635a5c1e453a2a5762dc9a17873f37855237e265de19731c0c730` |
| Logits compared | 251,285 |
| Top-1 matches | 5 / 5 |
| Maximum absolute error | 0.0001068115 |
| Mean absolute error | 0.0000122083 |

This is CPU inference interoperability evidence for one bounded probe. It is not
training, fine-tuning, throughput, GPU, distributed-scale, or production-serving
evidence. It also does not prove that Ajay watched the assigned lecture, typed the
implementation, or personally explained the architecture; those learner artifacts
remain separate.
