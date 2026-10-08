"""Day 34: deterministic character bigram training on Tiny Shakespeare."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class GPTBigramError(ValueError):
    """Raised when Day 34 data, model, or evidence contracts are invalid."""


@dataclass(frozen=True)
class CorpusSource:
    """Immutable provenance for an exact public corpus payload."""

    name: str
    url: str
    sha256: str
    byte_count: int
    license: str

    def __post_init__(self) -> None:
        for field_name in ("name", "url", "license"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise GPTBigramError(f"{field_name} must be a non-empty string")
        if not self.url.startswith("https://"):
            raise GPTBigramError("url must use HTTPS")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise GPTBigramError("sha256 must be 64 lowercase hexadecimal characters")
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int):
            raise TypeError("byte_count must be an integer")
        if self.byte_count <= 0:
            raise GPTBigramError("byte_count must be positive")


@dataclass(frozen=True)
class CharacterVocabulary:
    """Stable character-to-id mapping derived from one corpus."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) < 2:
            raise GPTBigramError("vocabulary requires at least two characters")
        if any(not isinstance(token, str) or len(token) != 1 for token in self.tokens):
            raise GPTBigramError("vocabulary tokens must be single characters")
        if self.tokens != tuple(sorted(set(self.tokens))):
            raise GPTBigramError("vocabulary tokens must be unique and sorted")

    @classmethod
    def from_text(cls, text: str) -> CharacterVocabulary:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return cls(tuple(sorted(set(text))))

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        indexes = {token: index for index, token in enumerate(self.tokens)}
        try:
            return tuple(indexes[token] for token in text)
        except KeyError as exc:
            raise GPTBigramError(f"unknown character: {exc.args[0]!r}") from exc

    def decode(self, token_ids: tuple[int, ...] | list[int]) -> str:
        decoded: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token ids must be integers")
            if not 0 <= token_id < len(self.tokens):
                raise GPTBigramError(f"token id out of range: {token_id}")
            decoded.append(self.tokens[token_id])
        return "".join(decoded)


@dataclass(frozen=True)
class CorpusSplit:
    """Ordered train and held-out token partitions."""

    train: Tensor
    validation: Tensor
    split_index: int


@dataclass(frozen=True)
class BigramTrainingConfig:
    """Explicit controls for a bounded CPU training run."""

    steps: int = 100
    batch_size: int = 32
    block_size: int = 8
    learning_rate: float = 1e-2
    seed: int = 34
    eval_interval: int = 20
    eval_batch_size: int = 64

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "batch_size",
            "block_size",
            "seed",
            "eval_interval",
            "eval_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        for name in (
            "steps",
            "batch_size",
            "block_size",
            "eval_interval",
            "eval_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise GPTBigramError(f"{name} must be positive")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise GPTBigramError("learning_rate must be positive and finite")


@dataclass(frozen=True)
class TrainingStep:
    """One optimizer update and its measured minibatch loss."""

    step: int
    loss: float


def tokenize_and_split(
    text: str,
    vocabulary: CharacterVocabulary,
    *,
    validation_fraction: float = 0.1,
) -> CorpusSplit:
    """Encode text and reserve its final contiguous portion for validation."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(vocabulary, CharacterVocabulary):
        raise TypeError("vocabulary must be CharacterVocabulary")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not 0 < validation_fraction < 1
    ):
        raise GPTBigramError("validation_fraction must be between zero and one")
    tokens = torch.tensor(vocabulary.encode(text), dtype=torch.long)
    split_index = int(len(tokens) * (1 - validation_fraction))
    if split_index < 2 or len(tokens) - split_index < 2:
        raise GPTBigramError("train and validation partitions need at least two tokens")
    return CorpusSplit(
        train=tokens[:split_index].clone(),
        validation=tokens[split_index:].clone(),
        split_index=split_index,
    )


class WindowBatcher:
    """Seeded random sampler for contiguous next-token windows."""

    def __init__(self, tokens: Tensor, *, block_size: int, seed: int) -> None:
        if not isinstance(tokens, Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.ndim != 1 or tokens.dtype != torch.long:
            raise GPTBigramError("tokens must be a one-dimensional torch.long tensor")
        for name, value in (("block_size", block_size), ("seed", seed)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if block_size <= 0:
            raise GPTBigramError("block_size must be positive")
        if len(tokens) <= block_size:
            raise GPTBigramError("tokens must contain more than block_size entries")
        self._tokens = tokens.detach().clone()
        self.block_size = block_size
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Return aligned input and next-token target windows."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise GPTBigramError("batch_size must be positive")
        starts = torch.randint(
            0,
            len(self._tokens) - self.block_size,
            (batch_size,),
            generator=self._generator,
        )
        inputs = torch.stack(
            [self._tokens[index : index + self.block_size] for index in starts.tolist()]
        )
        targets = torch.stack(
            [
                self._tokens[index + 1 : index + self.block_size + 1]
                for index in starts.tolist()
            ]
        )
        return inputs, targets

    def rng_state(self) -> Tensor:
        """Return an isolated copy of the sampler RNG state."""

        return self._generator.get_state().clone()

    def restore_rng_state(self, state: Tensor) -> None:
        """Restore a state produced by :meth:`rng_state`."""

        if not isinstance(state, Tensor):
            raise TypeError("state must be a torch.Tensor")
        if state.device.type != "cpu" or state.dtype != torch.uint8 or state.ndim != 1:
            raise GPTBigramError("state must be a one-dimensional CPU uint8 tensor")
        try:
            self._generator.set_state(state.detach().clone())
        except RuntimeError as exc:
            raise GPTBigramError("state is not a valid CPU generator state") from exc


class BigramLanguageModel(nn.Module):
    """One learned logit row for every current character id."""

    def __init__(self, vocab_size: int, *, seed: int) -> None:
        super().__init__()
        for name, value in (("vocab_size", vocab_size), ("seed", seed)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if vocab_size < 2:
            raise GPTBigramError("vocab_size must be at least two")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    @property
    def vocab_size(self) -> int:
        return self.token_embedding_table.num_embeddings

    def forward(self, token_ids: Tensor) -> Tensor:
        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.dtype != torch.long:
            raise GPTBigramError("token_ids must use torch.long")
        if token_ids.ndim not in {1, 2}:
            raise GPTBigramError("token_ids must have rank one or two")
        if token_ids.numel() == 0:
            raise GPTBigramError("token_ids must not be empty")
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= self.vocab_size:
            raise GPTBigramError("token_ids contain an out-of-range id")
        return self.token_embedding_table(token_ids)


def next_token_loss(
    model: BigramLanguageModel, inputs: Tensor, targets: Tensor
) -> Tensor:
    """Compute mean categorical NLL for aligned next-token targets."""

    if not isinstance(model, BigramLanguageModel):
        raise TypeError("model must be BigramLanguageModel")
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("inputs and targets must be torch.Tensor values")
    if inputs.shape != targets.shape or inputs.ndim != 2:
        raise GPTBigramError("inputs and targets must have the same rank-two shape")
    if inputs.dtype != torch.long or targets.dtype != torch.long:
        raise GPTBigramError("inputs and targets must use torch.long")
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))


def evaluate_partition(
    model: BigramLanguageModel, tokens: Tensor, *, chunk_size: int = 16_384
) -> float:
    """Compute deterministic NLL over every consecutive pair in a partition."""

    if not isinstance(model, BigramLanguageModel):
        raise TypeError("model must be BigramLanguageModel")
    if not isinstance(tokens, Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    if tokens.ndim != 1 or tokens.dtype != torch.long or len(tokens) < 2:
        raise GPTBigramError(
            "tokens must be one-dimensional torch.long with length at least two"
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise GPTBigramError("chunk_size must be positive")
    was_training = model.training
    model.eval()
    total = 0.0
    pair_count = len(tokens) - 1
    with torch.no_grad():
        for start in range(0, pair_count, chunk_size):
            end = min(start + chunk_size, pair_count)
            logits = model(tokens[start:end])
            total += float(
                F.cross_entropy(logits, tokens[start + 1 : end + 1], reduction="sum")
            )
    model.train(was_training)
    return total / pair_count


def train_steps(
    model: BigramLanguageModel,
    optimizer: torch.optim.Optimizer,
    batcher: WindowBatcher,
    *,
    start_step: int,
    steps: int,
    batch_size: int,
) -> tuple[TrainingStep, ...]:
    """Apply a bounded sequence of deterministic next-token updates."""

    if not isinstance(model, BigramLanguageModel):
        raise TypeError("model must be BigramLanguageModel")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer")
    if not isinstance(batcher, WindowBatcher):
        raise TypeError("batcher must be WindowBatcher")
    for name, value in (
        ("start_step", start_step),
        ("steps", steps),
        ("batch_size", batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if start_step < 0:
        raise GPTBigramError("start_step must be non-negative")
    if steps <= 0 or batch_size <= 0:
        raise GPTBigramError("steps and batch_size must be positive")
    trace: list[TrainingStep] = []
    model.train()
    for offset in range(steps):
        inputs, targets = batcher.sample(batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, inputs, targets)
        if not bool(torch.isfinite(loss)):
            raise GPTBigramError("training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        trace.append(TrainingStep(start_step + offset + 1, float(loss.detach())))
    return tuple(trace)


def generate_tokens(
    model: BigramLanguageModel,
    *,
    start_token_id: int,
    max_new_tokens: int,
    seed: int,
    temperature: float = 1.0,
) -> tuple[int, ...]:
    """Sample a bounded continuation with an isolated CPU generator."""

    if not isinstance(model, BigramLanguageModel):
        raise TypeError("model must be BigramLanguageModel")
    for name, value in (
        ("start_token_id", start_token_id),
        ("max_new_tokens", max_new_tokens),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not 0 <= start_token_id < model.vocab_size:
        raise GPTBigramError("start_token_id is out of range")
    if max_new_tokens <= 0:
        raise GPTBigramError("max_new_tokens must be positive")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not isfinite(temperature)
        or temperature <= 0
    ):
        raise GPTBigramError("temperature must be positive and finite")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    token_ids = [start_token_id]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(torch.tensor([token_ids[-1]], dtype=torch.long))[0]
            probabilities = F.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probabilities, 1, generator=generator))
            token_ids.append(next_id)
    model.train(was_training)
    return tuple(token_ids)


def model_fingerprint(model: BigramLanguageModel) -> str:
    """Hash parameter names, dtypes, shapes, and exact CPU tensor bytes."""

    if not isinstance(model, BigramLanguageModel):
        raise TypeError("model must be BigramLanguageModel")
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


TINY_SHAKESPEARE = CorpusSource(
    name="Tiny Shakespeare",
    url=(
        "https://raw.githubusercontent.com/karpathy/char-rnn/"
        "master/data/tinyshakespeare/input.txt"
    ),
    sha256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
    byte_count=1_115_394,
    license="Public-domain Shakespeare text; compilation distributed by Andrej Karpathy",
)


def validate_corpus_bytes(payload: bytes, source: CorpusSource) -> str:
    """Validate exact source bytes before decoding them as UTF-8 text."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not isinstance(source, CorpusSource):
        raise TypeError("source must be CorpusSource")
    if len(payload) != source.byte_count:
        raise GPTBigramError(
            f"corpus byte count mismatch: expected {source.byte_count}, got {len(payload)}"
        )
    digest = sha256(payload).hexdigest()
    if digest != source.sha256:
        raise GPTBigramError(
            f"corpus sha256 mismatch: expected {source.sha256}, got {digest}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GPTBigramError("corpus must be valid UTF-8") from exc
    if len(text) < 2:
        raise GPTBigramError("corpus must contain at least two characters")
    return text


def load_verified_corpus(path: Path, source: CorpusSource) -> str:
    """Read a local corpus snapshot and enforce its pinned source contract."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GPTBigramError(f"unable to read corpus snapshot: {path}") from exc
    return validate_corpus_bytes(payload, source)
