# Explicit low-rank adapters

This lab implements the frozen-weight, low-rank update from
[Hu et al., LoRA](https://arxiv.org/abs/2106.09685):
`linear(x, W, bias) + (alpha / rank) * linear(linear(x, A), B)`.
A uses a local seeded normal initialization and B starts at zero. The change
has rank at most `rank`; the base matrix itself need not be low rank.

These are bounded CPU teaching primitives. Parameter counts describe trainable
capacity, not measured memory savings, GPU performance, or large-model quality.

## Install and train an adapter

```python
import torch
from ai_journey.transformer_lab import DecoderLanguageModel, TransformerConfig
from ai_journey.lora import (
    LoRAConfig, inject_adapters, build_adapter_optimizer,
    parameter_summary, frozen_fingerprint, audit_adapter_gradients,
)

# This example starts from random weights. The comparison experiment separately
# pretrains a shared base before forking its adaptation arms.
base = DecoderLanguageModel(TransformerConfig(
    vocab_size=8, block_size=4, embedding_dim=8, head_count=2, layer_count=1,
))
model = inject_adapters(base, (
    "blocks.0.attention.query_key_value",
    "blocks.0.attention.projection",
), LoRAConfig(rank=2, alpha=4, seed=42))
optimizer = build_adapter_optimizer(model, learning_rate=0.003)
before = frozen_fingerprint(model)
inputs = torch.tensor([[1, 2, 3, 4]])
targets = torch.tensor([[2, 3, 4, 5]])
optimizer.zero_grad(set_to_none=True)
loss = model(inputs, targets)[1]
loss.backward()
audit_adapter_gradients(model)
optimizer.step()
assert frozen_fingerprint(model) == before
print(parameter_summary(model))
```

Installing adapters returns an independent copy, preserves the original model's
trainability and logits at initialization, and freezes every original parameter.
Targets must be distinct, explicitly named, unshared `nn.Linear` modules. Target
order does not change initialization. Already adapted models are rejected.
Bases using BatchNorm are rejected because freezing its parameters would still
allow running statistics to change. The decoder experiment uses LayerNorm.
The adapter optimizer checks the whole trainability boundary before allocating
AdamW state. The gradient audit accepts A's expected zero first-step gradient
while detecting disconnected factors and accidental base gradients.

## Save, restore, and merge

```python
from ai_journey.lora import merge_adapters
from ai_journey.lora_artifact import (
    write_adapter_artifact, restore_adapter_artifact,
)

write_adapter_artifact("artifacts/adapter.json", model)
restored = inject_adapters(base, (
    "blocks.0.attention.query_key_value",
    "blocks.0.attention.projection",
), LoRAConfig(rank=2, alpha=4, seed=42))
restore_adapter_artifact("artifacts/adapter.json", restored)
model.eval()
dense = merge_adapters(model)
torch.testing.assert_close(dense(inputs)[0], model(inputs)[0])
```

The JSON artifact stores factors, configuration, target names, and a frozen-base
fingerprint. It contains no base weights and cannot restore optimizer state or
resume a training trajectory. Keep the exact base weights separately. Loading
rejects another base, mismatched targets, dtype/shape changes, nonfinite values,
overflowing updates, duplicate JSON keys, and malformed metadata before changing
any factor. Read size is capped at 4 MiB, each matrix at 65,536 elements, and
the envelope at 128 targets. The SHA-256 digest detects corruption; it does not
authenticate a publisher. No pickle or serialized Python is loaded.

Publication uses a unique temporary file and atomic replacement. A failed
serialization or replacement preserves a previous artifact. Float16, bfloat16,
float32, and float64 factors have explicit dtype metadata. In-memory exports
also own their tensor storage.

Merging returns another independent model with ordinary dense projections.
It does not change the training model or add the update twice. Evaluation
disables adapter dropout; merging an adapter with active training dropout is
rejected. Dense and factorized multiplication may differ by floating-point
roundoff, so prediction equivalence uses a tolerance rather than byte equality.

## Verification

```sh
python -m unittest discover -s tests -p 'test_lora*.py' -v
```

Tests include independent matrix-calculus gradients and central finite
differences for both factors and inputs, exact zero-update logits, causal-prefix
and padded-sequence checks after nonzero updates, freeze-boundary audits, merge
equivalence, and transactional checkpoint rejection.
