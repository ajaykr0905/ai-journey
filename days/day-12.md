# Day 12 — GPT Dataflow and Tensor Shapes

**Date:** 2026-09-16

## Goal

Trace a decoder-only GPT from token IDs to next-token probabilities while keeping
every tensor shape explicit. The symbols used below are:

- `B`: batch size
- `T`: sequence length
- `V`: vocabulary size
- `C`: model width
- `H`: number of attention heads
- `D = C / H`: width of one head
- `F`: MLP hidden width
- `L`: number of repeated transformer blocks

## Full block diagram

```mermaid
flowchart TD
    IDs["Token IDs<br/>(B, T)"] -->|token lookup| TOK["Token embeddings<br/>(B, T, C)"]
    POS["Position embeddings<br/>(T, C)"] -->|broadcast add| RES0["Residual stream<br/>(B, T, C)"]
    TOK -->|add| RES0
    RES0 --> LN1["Pre-attention norm<br/>(B, T, C)"]
    LN1 --> QKV["QKV projection<br/>(B, T, 3C)"]
    QKV --> Q["Queries<br/>(B, H, T, D)"]
    QKV --> K["Keys<br/>(B, H, T, D)"]
    QKV --> VEC["Values<br/>(B, H, T, D)"]
    Q --> SCORES["Scaled QK scores<br/>(B, H, T, T)"]
    K --> SCORES
    MASK["Causal mask<br/>(1, 1, T, T)"] --> WEIGHTS["Attention weights<br/>(B, H, T, T)"]
    SCORES -->|mask + row softmax| WEIGHTS
    WEIGHTS --> HEADS["Weighted values<br/>(B, H, T, D)"]
    VEC --> HEADS
    HEADS --> CONCAT["Concatenated heads<br/>(B, T, C)"]
    CONCAT --> ATTN["Attention output<br/>(B, T, C)"]
    RES0 --> ADD1["Post-attention residual<br/>(B, T, C)"]
    ATTN --> ADD1
    ADD1 --> LN2["Pre-MLP norm<br/>(B, T, C)"]
    LN2 --> UP["MLP expansion<br/>(B, T, F)"]
    UP --> ACT["GELU activation<br/>(B, T, F)"]
    ACT --> DOWN["MLP projection<br/>(B, T, C)"]
    ADD1 --> ADD2["Post-MLP residual<br/>(B, T, C)"]
    DOWN --> ADD2
    ADD2 -->|repeat block L times| FINAL["Final norm<br/>(B, T, C)"]
    FINAL --> LOGITS["Vocabulary logits<br/>(B, T, V)"]
    LOGITS -->|softmax over V| PROBS["Next-token probabilities<br/>(B, T, V)"]
```

The residual stream always keeps shape `(B, T, C)`. Attention temporarily splits
`C` into `H` heads of width `D`, creates a `(T, T)` score matrix for every batch
item and head, and then concatenates the heads back to `C`. The MLP expands only
the channel dimension from `C` to `F` and projects it back before the residual add.

## Executable shape ledger

`src/ai_journey/transformer_shapes.py` models 24 tensors and 27 operation edges.
It rejects invalid head dimensions, unknown nodes, cycles, and key shape breaks.
It also renders the complete graph, a symbolic-and-concrete shape table, and a
parameter-count breakdown without requiring a machine-learning framework.

```bash
python scripts/run_day_12.py \
  --batch-size 2 \
  --context-length 8 \
  --vocab-size 32000 \
  --d-model 256 \
  --n-heads 8 \
  --d-ff 1024 \
  --n-layers 4 \
  --output artifacts/day-12-gpt-shapes.md
```

## Sources scheduled by the curriculum

- Primary — *Transformers, the tech behind LLMs | Chapter 5*:
  https://www.youtube.com/watch?v=wjZofJX0v4M
- Support — *Cross products in the light of linear transformations*:
  https://www.youtube.com/watch?v=BaM7OCEm3G0

These links and topics come from the canonical workbook. They are planning input,
not evidence that either lecture was watched.

## Evidence status

The deterministic reference implementation, diagram, shape validator, CLI, and
tests are present. Ajay still needs to watch the scheduled lectures, draw the GPT
block diagram from memory without opening this reference, label every shape, and
compare that drawing with the validated ledger. No personal completion is claimed.
