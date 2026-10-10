## Run the verified CPU lab

Install with `python -m pip install -e .`. The installed command and
`python scripts/run_day_37.py` accept the same flags:

```bash
ai-journey-day-37 --output artifacts/attention.json
ai-journey-day-37 --output artifacts/post-sdpa.json --norm-placement post --backend sdpa
ai-journey-day-37 --output artifacts/attention.json \
  --protocol config/day-37-head-count-ablation.json \
  --corpus data/day-19-demo-names.txt --ablation-output artifacts/head-count.json
```

Exit 0 confirms all declared numerical tolerances. Exit 2 reports invalid
controls, failed certification, or I/O errors. Both report outputs must differ
from inputs. Each output is replaced atomically; the pair is not one filesystem
transaction. A failed write leaves that destination's previous file intact.
Fingerprints detect changes but do not authenticate an author or prove that a
learner ran the command. Load certification with
`transformer_evidence.load_certification_report(Path(...))` for digest and
semantic checks.

The default audit uses a tiny CPU float64 model and synthetic inputs. The checked-in
three-configuration protocol varies only `model.head_count` across 1, 2 and 4,
with seeds 37 and 38 and four training updates. A supplied corpus is recorded by
its fingerprint and ordered held-out split. CI uses the public demo corpus;
passing this gate establishes reproducibility, not useful text-generation quality.
For a larger study, predeclare a new protocol before seeing its results and
record duration and peak process memory with `/usr/bin/time -l` on macOS or
`/usr/bin/time -v` on Linux. Peak RSS is a whole-process high-water mark, not
per-trial tensor allocation or GPU memory.

## Model controls and padding

`TransformerConfig` defaults preserve the earlier model equations and config
fingerprint. Optional controls are `normalization_placement="pre"|"post"`,
`feed_forward_expansion` (positive integer, default 4),
`feed_forward_activation="gelu"|"relu"`,
`activation_checkpointing` (default false), and
`attention_backend="manual"|"sdpa"`. Nondefault controls affect identity.
Checkpointing is training-only and preserves dropout RNG during recomputation.
Scratch BatchNorm is incompatible with padding, cached decoding and activation
checkpointing because its cross-token statistics or running-state updates break
those contracts.

Pass `lengths=torch.tensor([...], dtype=torch.long, device=token_ids.device)` to
the model for right-padded batches. Every sequence must contain at least one
valid token. Padded attention queries and decoder logits are zero; padded labels
do not contribute to the loss. All token IDs, including padding IDs, must remain
inside the vocabulary. Valid labels must also remain inside it.

## Restart cached inference

```python
from pathlib import Path
from ai_journey.transformer_cache import prefill, decode, save_cache, load_cache

model.eval()  # the same trusted weights must be available after restart
logits, cache = prefill(model, prompt_ids)
Path("artifacts").mkdir(exist_ok=True)
save_cache(Path("artifacts/prefix.json"), cache)
restored = load_cache(Path("artifacts/prefix.json"), model)
next_logits, cache = decode(model, next_ids, restored)
```

Caches own their state and reject changed weights or mismatched architecture,
dtype, device or batch. `reorder_cache` selects or duplicates batch rows;
`migrate_cache` rebuilds state for an explicitly converted copy of the same
model. `generate_cached(..., new_tokens=..., seed=...)` uses an isolated generator
and restores every caller module's train/eval mode even after failure.

Snapshots use bounded JSON and raw tensor bytes, with strict schema, duplicate
field, finite-value and checksum checks. Restore recomputes the retained context
against trusted weights before accepting it. A snapshot does not contain model
weights. Atomic saves flush file and directory metadata; hardware power-loss
recovery has not been tested.
If directory flushing fails after replacement, the new snapshot is valid but
its durability is uncertain; the save reports that error.

After the context fills, decode rebuilds the cropped context because learned
positions restart at zero. FIFO eviction alone would retain incorrect keys.
This fallback is correctness-first and has no speedup guarantee. Model weights
must remain stable during a cache operation; concurrent mutation is unsupported.

The tests and certification cover CPU parity and fault injection. They provide no
GPU, multi-GPU, vLLM, production latency or throughput evidence. CI pins Ruff
0.6.9 for these reliability files. Day 37's full Block exercise and the primary
and support videos still need Ajay's own completion evidence.
