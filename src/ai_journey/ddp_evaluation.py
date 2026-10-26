"""Document-isolated byte-level evaluation for the distributed GPT-2 lab."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
import torch
import math

from .ddp_contracts import integer
from .ddp_metrics import token_nll
from .training_models import base_model_config


@dataclass(frozen=True)
class Document:
    identifier: str
    content: bytes

    def __post_init__(self):
        if (
            not isinstance(self.identifier, str)
            or not self.identifier.strip()
            or len(self.identifier) > 200
        ):
            raise ValueError("document identifier must be bounded nonempty text")
        if not isinstance(self.content, bytes) or len(self.content) < 2:
            raise ValueError("document content must contain at least two bytes")

    @property
    def digest(self) -> str:
        return sha256(self.content).hexdigest()


def documents_from_lines(source: bytes) -> tuple[Document, ...]:
    """Treat every nonempty public line as one document, rejecting duplicates."""
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    documents = tuple(
        Document(f"line-{number}", line + b"\n")
        for number, line in enumerate(source.splitlines(), 1)
        if line
    )
    if not documents or len({document.digest for document in documents}) != len(
        documents
    ):
        raise ValueError("documents must be nonempty and content-distinct")
    return documents


def split_documents(documents, *, validation_count: int, seed: int = 48):
    """Assign whole documents using keyed hashes, independent of input order."""
    documents = tuple(documents)
    integer(validation_count, "validation_count", 1)
    integer(seed, "seed")
    if validation_count >= len(documents):
        raise ValueError("training and validation must both contain documents")
    if len({doc.digest for doc in documents}) != len(documents) or len(
        {doc.identifier for doc in documents}
    ) != len(documents):
        raise ValueError("document identities and contents must be distinct")
    order = sorted(
        documents, key=lambda doc: sha256(f"{seed}:{doc.digest}".encode()).digest()
    )
    return tuple(order[validation_count:]), tuple(order[:validation_count])


def evaluation_batches(documents, *, context_length: int, batch_size: int = 4):
    """Score each within-document target once using masked right-padded windows."""
    integer(context_length, "context_length", 1)
    integer(batch_size, "batch_size", 1)

    def windows():
        for document in documents:
            for start in range(0, len(document.content) - 1, context_length):
                piece = document.content[start : start + context_length + 1]
                yield list(piece[:-1]), list(piece[1:])

    pending = iter(windows())
    while chunk := tuple(islice(pending, batch_size)):
        width = max(len(inputs) for inputs, _ in chunk)
        inputs = torch.zeros((len(chunk), width), dtype=torch.long)
        targets = torch.zeros_like(inputs)
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        for row, (prefix, suffix) in enumerate(chunk):
            inputs[row, : len(prefix)] = torch.tensor(prefix)
            targets[row, : len(suffix)] = torch.tensor(suffix)
            mask[row, : len(prefix)] = True
        yield inputs, targets, mask


def evaluate_documents(model, documents, *, batch_size: int = 4) -> dict:
    """Measure additive NLL without changing gradients, RNG or module modes."""
    modes = [(module, module.training) for module in model.modules()]
    rng = torch.get_rng_state().clone()
    loss_sum = 0.0
    tokens = 0
    try:
        model.eval()
        with torch.inference_mode():
            for inputs, targets, mask in evaluation_batches(
                documents,
                context_length=base_model_config(model).context_length,
                batch_size=batch_size,
            ):
                losses = token_nll(model(inputs), targets)
                loss_sum += losses[mask].double().sum().item()
                tokens += int(mask.sum())
    finally:
        torch.set_rng_state(rng)
        for module, mode in modes:
            module.training = mode
    if tokens == 0:
        raise ValueError("evaluation requires at least one target")
    mean = loss_sum / tokens
    return {
        "loss_sum": loss_sum,
        "tokens": tokens,
        "mean_nll": mean,
        "bits_per_byte": mean / math.log(2),
        "perplexity": math.exp(mean) if mean < 700 else None,
    }


def evaluate_distributed(model, documents, topology, *, batch_size: int = 4) -> dict:
    """Partition whole documents without padding or duplicating validation targets."""
    from .ddp_runtime import reduce_token_totals, require_model_consensus

    require_model_consensus(model)
    local = tuple(documents)[topology.rank :: topology.world_size]
    result = (
        evaluate_documents(model, local, batch_size=batch_size)
        if local
        else {"loss_sum": 0.0, "tokens": 0}
    )
    loss_sum, tokens, mean = reduce_token_totals(result["loss_sum"], result["tokens"])
    return {
        "loss_sum": loss_sum,
        "tokens": tokens,
        "mean_nll": mean,
        "bits_per_byte": mean / math.log(2),
    }


def score_choices(model, prompt: bytes, choices) -> dict:
    """Rank candidates by completion-only mean NLL with sliding context."""
    choices = tuple(choices)
    if not isinstance(prompt, bytes) or not prompt or len(prompt) > 4096:
        raise ValueError("prompt must contain 1 to 4096 bytes")
    if not 2 <= len(choices) <= 8 or any(
        not isinstance(choice, bytes) or not 1 <= len(choice) <= 1024
        for choice in choices
    ):
        raise ValueError("provide two to eight bounded nonempty completions")
    if len(set(choices)) != len(choices):
        raise ValueError("completion candidates must be distinct")
    modes = [(module, module.training) for module in model.modules()]
    scores = []
    try:
        model.eval()
        with torch.inference_mode():
            for choice in choices:
                context = list(prompt)
                losses = []
                for target in choice:
                    inputs = torch.tensor(
                        [context[-base_model_config(model).context_length :]],
                        dtype=torch.long,
                    )
                    logits = model(inputs)[:, -1:, :]
                    losses.append(token_nll(logits, torch.tensor([[target]])).item())
                    context.append(target)
                scores.append(
                    {
                        "tokens": len(choice),
                        "loss_sum": math.fsum(losses),
                        "mean_nll": math.fsum(losses) / len(choice),
                    }
                )
    finally:
        for module, mode in modes:
            module.training = mode
    return {
        "prediction": min(
            range(len(scores)), key=lambda index: scores[index]["mean_nll"]
        ),
        "scores": scores,
        "normalization": "completion_mean_nll",
    }
