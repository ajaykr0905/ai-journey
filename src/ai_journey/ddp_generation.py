"""Bounded sampling and inference for checkpoints produced by the CPU DDP lab."""

from dataclasses import dataclass
import math
import time

import torch

from .ddp_contracts import integer
from .training_models import base_model_config


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int = 48

    def __post_init__(self):
        integer(self.max_new_tokens, "max_new_tokens", 1)
        integer(self.seed, "seed")
        if self.seed >= 2**64:
            raise ValueError("seed must fit an unsigned 64-bit integer")
        if self.max_new_tokens > 1024:
            raise ValueError("max_new_tokens exceeds CPU lab budget")
        if self.top_k is not None:
            integer(self.top_k, "top_k", 1)
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature and top_p are outside supported ranges")


def filter_logits(logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    """Keep the minimal nucleus after a deterministic exact top-k cutoff."""
    if (
        logits.ndim != 2
        or not logits.is_floating_point()
        or torch.isnan(logits).any()
        or torch.isposinf(logits).any()
        or not torch.isfinite(logits).any(dim=1).all()
    ):
        raise ValueError("each logit row requires at least one finite value")
    values = logits.double().clone()
    if config.temperature > 0:
        values -= values.amax(dim=-1, keepdim=True)
        values /= config.temperature
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    ranked = values.gather(1, order)
    if config.top_k is not None:
        ranked[:, min(config.top_k, values.shape[-1]) :] = -torch.inf
    probabilities = ranked.softmax(-1)
    previous_mass = probabilities.cumsum(-1) - probabilities
    ranked[previous_mass >= config.top_p] = -torch.inf
    return torch.full_like(values, -torch.inf).scatter(1, order, ranked)


def process_logits(
    logits: torch.Tensor,
    history: torch.Tensor,
    *,
    banned_tokens=(),
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Apply sign-aware repetition penalties and explicit token exclusions."""
    if (
        history.dtype != torch.long
        or history.ndim != 2
        or logits.ndim != 2
        or history.shape[0] != logits.shape[0]
    ):
        raise ValueError("history must contain one integer sequence per logit row")
    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or not math.isfinite(repetition_penalty)
        or repetition_penalty < 1
    ):
        raise ValueError("repetition_penalty must be finite and at least one")
    values = logits.clone()
    if history.numel() and (history.min() < 0 or history.max() >= logits.shape[-1]):
        raise ValueError("history contains an invalid token id")
    for token in banned_tokens:
        integer(token, "banned token")
        if token >= logits.shape[-1]:
            raise ValueError("banned token is outside the vocabulary")
        values[:, token] = -torch.inf
    for row in range(len(values)):
        repeated = history[row].unique()
        scores = values[row, repeated]
        values[row, repeated] = torch.where(
            scores < 0, scores * repetition_penalty, scores / repetition_penalty
        )
    if not torch.isfinite(values).any(dim=1).all():
        raise ValueError("token exclusions removed every possible output")
    return values


class TokenSampler:
    """Request-local sampling RNG that can be checkpointed without global effects."""

    def __init__(self, config: SamplingConfig):
        self.config = config
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        values = filter_logits(logits, self.config).cpu()
        if self.config.temperature == 0:
            return values.argmax(-1, keepdim=True)
        return torch.multinomial(values.softmax(-1), 1, generator=self.generator)

    def state(self) -> torch.Tensor:
        return self.generator.get_state().clone()

    def restore(self, state: torch.Tensor) -> None:
        before = self.state()
        try:
            self.generator.set_state(state.clone())
        except (RuntimeError, TypeError):
            self.generator.set_state(before)
            raise ValueError("invalid request RNG state") from None


def generate_tokens(
    model,
    prompt,
    config: SamplingConfig,
    *,
    eos_token: int | None = None,
    banned_tokens=(),
    repetition_penalty: float = 1.0,
    use_cache: bool = False,
    cancelled=None,
    deadline: float | None = None,
    clock=time.monotonic,
    on_token=None,
) -> dict:
    """Generate under a bounded sliding context while preserving caller modes."""
    prompt = tuple(prompt)
    base = base_model_config(model)
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be a callable")
    if on_token is not None and not callable(on_token):
        raise TypeError("on_token must be callable")
    if deadline is not None and (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("deadline must be a finite monotonic timestamp")
    if not prompt or len(prompt) > 4096:
        raise ValueError("prompt requires 1 to 4096 token ids")
    for token in prompt:
        integer(token, "prompt token")
        if token >= base.vocab_size:
            raise ValueError("prompt token exceeds vocabulary")
    if eos_token is not None:
        integer(eos_token, "eos_token")
        if eos_token >= base.vocab_size:
            raise ValueError("eos_token exceeds vocabulary")
    modes = [(module, module.training) for module in model.modules()]
    sampler = TokenSampler(config)
    history = list(prompt)
    reason = "length"
    cache = None
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(config.max_new_tokens):
                if cancelled is not None and cancelled():
                    reason = "cancelled"
                    break
                if deadline is not None and clock() >= deadline:
                    reason = "deadline"
                    break
                inputs = torch.tensor(
                    [history[-base.context_length :]], dtype=torch.long
                )
                if use_cache:
                    from .ddp_cache import cached_forward

                    if cache is None or cache.length == base.context_length:
                        output, cache = cached_forward(model, inputs)
                    else:
                        output, cache = cached_forward(model, inputs[:, -1:], cache)
                else:
                    output = model(inputs)
                logits = process_logits(
                    output[:, -1, :],
                    torch.tensor([history]),
                    banned_tokens=banned_tokens,
                    repetition_penalty=repetition_penalty,
                )
                token = sampler.sample(logits).item()
                history.append(token)
                if on_token is not None:
                    on_token(len(history) - len(prompt) - 1, token)
                if token == eos_token:
                    reason = "eos"
                    break
    finally:
        for module, mode in modes:
            module.training = mode
    return {
        "prompt_tokens": len(prompt),
        "tokens": history[len(prompt) :],
        "finish_reason": reason,
    }
