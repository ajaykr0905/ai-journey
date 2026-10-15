"""Document-isolated, once-per-target likelihood for byte-normalized comparisons."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import math

import torch
from torch.nn import functional as F

from .bpe import SpecialTokenizer, TokenizerError
from .bpe_corpus import EOS_MARKER
from .tokenizer_codec import BPECodec
from .tokenizer_training import training_segments
from .transformer_lab import DecoderLanguageModel


@dataclass(frozen=True)
class DocumentLikelihood:
    text_sha256: str
    utf8_bytes: int
    codepoints: int
    content_tokens: int
    content_nll: float
    eos_nll: float


def summarize_likelihood(documents: tuple[DocumentLikelihood, ...]) -> dict:
    """Use sums, not means of document rates; EOS is never counted as text bytes."""
    if not documents:
        raise TokenizerError("likelihood requires nonempty documents")
    size = sum(item.utf8_bytes for item in documents)
    tokens = sum(item.content_tokens for item in documents)
    loss = math.fsum(item.content_nll for item in documents)
    return {
        "documents": [asdict(item) for item in documents],
        "document_count": len(documents),
        "utf8_bytes": size,
        "content_tokens": tokens,
        "content_nll": loss,
        "eos_nll": math.fsum(item.eos_nll for item in documents),
        "content_nll_per_token": loss / tokens,
        "content_bits_per_byte": loss / (size * math.log(2)),
        "bytes_per_content_token": size / tokens,
    }


@torch.no_grad()
def score_documents(
    model: DecoderLanguageModel,
    codec: BPECodec,
    documents: tuple[str, ...],
    *,
    batch_size: int = 64,
) -> tuple[DocumentLikelihood, ...]:
    """Score each target once using up to block_size preceding tokens.

    Each document starts from EOS, ends with an independently scored EOS, and
    has no context from another document. Positions reset on context rollover.
    This is canonical-segmentation likelihood, not a sum over BPE segmentations.
    """
    training_segments(documents)
    if (
        not isinstance(codec, BPECodec)
        or not isinstance(codec.tokenizer, SpecialTokenizer)
        or codec.tokenizer.names != (EOS_MARKER,)
        or model.config.vocab_size != codec.vocab_size
    ):
        raise TokenizerError("likelihood requires the matching EOS-only BPE codec")
    if model.config.normalization_mode != "layer_norm" or any(
        parameter.device.type != "cpu" for parameter in model.parameters()
    ):
        raise TokenizerError("likelihood supports CPU layer-normalized models")
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise TokenizerError("likelihood batch_size must be an integer in [1, 128]")
    encoded = [codec.encode(text) for text in documents]
    if len(documents) > 100 or sum(map(len, encoded)) > 8192:
        raise TokenizerError("likelihood exceeds 100 documents or 8192 content tokens")
    eos = codec.vocab_size - 1
    modes = [(module, module.training) for module in model.modules()]
    results = []
    try:
        model.eval()
        for text, ids in zip(documents, encoded):
            sequence = [eos, *ids, eos]
            losses = []
            for start in range(1, len(sequence), batch_size):
                stops = list(range(start, min(start + batch_size, len(sequence))))
                contexts = [
                    sequence[max(0, end - model.config.block_size) : end]
                    for end in stops
                ]
                lengths = torch.tensor([len(context) for context in contexts])
                x = torch.zeros((len(stops), int(lengths.max())), dtype=torch.long)
                for row, context in enumerate(contexts):
                    x[row, : len(context)] = torch.tensor(context)
                logits, _ = model(x, lengths=lengths)
                last = logits[torch.arange(len(stops)), lengths - 1].double()
                targets = torch.tensor([sequence[end] for end in stops])
                nll = F.cross_entropy(last, targets, reduction="none")
                if not bool(torch.isfinite(nll).all()):
                    raise TokenizerError("likelihood logits must yield finite losses")
                losses.extend(nll.tolist())
            results.append(
                DocumentLikelihood(
                    sha256(text.encode()).hexdigest(),
                    len(text.encode()),
                    len(text),
                    len(ids),
                    math.fsum(losses[:-1]),
                    losses[-1],
                )
            )
    finally:
        for module, training in modes:
            module.training = training
    return tuple(results)
