"""Bounded CPU generation timings, not production or trained-model evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
from time import perf_counter_ns

import torch
from torch import Tensor

from .transformer_cache import generate_cached
from .transformer_lab import DecoderLanguageModel, TransformerConfig


def _integer(name: str, value: int, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return value


def _tensor_digest(tensor: Tensor) -> str:
    tensor = tensor.detach().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def benchmark_generation(
    model: DecoderLanguageModel,
    prompt: Tensor,
    *,
    new_tokens: int = 4,
    repeats: int = 3,
    warmup: int = 1,
    seed: int = 37,
) -> dict:
    """Time complete greedy generation calls, including cache validation/hashing.

    Caller-supplied CPU models are limited to one million parameters, four
    layers, 128 context positions and 128 embedding channels. Prompts allow
    batch 1..4 and 1..128 positions; generation 1..32, repeats 1..20, warmup
    0..5. Local identically seeded generators leave caller RNG/modes untouched.
    """
    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be DecoderLanguageModel")
    new_tokens = _integer("new_tokens", new_tokens, 1, 32)
    repeats = _integer("repeats", repeats, 1, 20)
    warmup = _integer("warmup", warmup, 0, 5)
    seed = _integer("seed", seed, 0, 2**63 - 1)
    if any(t.device.type != "cpu" for t in model.state_dict().values()):
        raise ValueError("benchmark requires a CPU model")
    if model.config.normalization_mode != "layer_norm":
        raise ValueError("benchmark requires layer_norm")
    for name, maximum in (
        ("block_size", 128),
        ("embedding_dim", 128),
        ("layer_count", 4),
    ):
        _integer(name, getattr(model.config, name), 1, maximum)
    _integer("parameter_count", model.parameter_count, 1, 1_000_000)
    if not isinstance(prompt, Tensor) or prompt.device.type != "cpu":
        raise TypeError("prompt must be a CPU tensor")
    if prompt.ndim != 2:
        raise ValueError("prompt must have shape [batch, time]")
    batch, context = prompt.shape
    _integer("batch_size", batch, 1, 4)
    _integer("prompt_tokens", context, 1, 128)
    model._validate_token_ids(prompt, limit_time=False)
    identity = sha256(model.config.fingerprint().encode())
    for name, tensor in sorted(model.state_dict().items()):
        identity.update(name.encode())
        identity.update(_tensor_digest(tensor).encode())
    samples: dict[str, list[int]] = {"full": [], "cached": []}
    reference = None
    caller_rng = torch.get_rng_state().clone()
    caller_modes = [(module, module.training) for module in model.modules()]
    try:
        for iteration in range(warmup + repeats):
            phase_iteration = iteration if iteration < warmup else iteration - warmup
            methods = (
                tuple(samples) if phase_iteration % 2 == 0 else tuple(reversed(samples))
            )
            for method in methods:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                start = perf_counter_ns()
                if method == "full":
                    output = model.generate(
                        prompt, new_tokens=new_tokens, top_k=1, generator=generator
                    )
                else:
                    output = generate_cached(
                        model, prompt, new_tokens=new_tokens, top_k=1, seed=seed
                    )
                elapsed = perf_counter_ns() - start
                if elapsed <= 0 or not math.isfinite(elapsed):
                    raise ValueError("generation duration must be positive and finite")
                if reference is None:
                    reference = output.clone()
                if output.shape != (batch, context + new_tokens) or not torch.equal(
                    reference, output
                ):
                    raise ValueError("generation outputs do not match exactly")
                if iteration >= warmup:
                    samples[method].append(elapsed)
    finally:
        torch.set_rng_state(caller_rng)
        for module, training in caller_modes:
            module.training = training
    results = {}
    for method, durations in samples.items():
        median = statistics.median(durations)
        p95 = sorted(durations)[math.ceil(0.95 * len(durations)) - 1]
        results[method] = {
            "sample_ns": durations,
            "median_ns": median,
            "p95_ns": p95,
            "generated_tokens_per_second": batch * new_tokens * 1e9 / median,
        }
    return {
        "schema_version": 1,
        "scope": "caller-supplied CPU model; complete generation calls include validation, cache hashing and rollover",
        "percentile_method": "nearest rank: sorted samples[ceil(0.95*n)-1]",
        "method_order": ["full", "cached"],
        "execution_order": (
            "alternating timing pairs; measured pairs start full then cached"
        ),
        "config": asdict(model.config),
        "controls": {
            "new_tokens": new_tokens,
            "repeats": repeats,
            "warmup": warmup,
            "seed": seed,
            "batch_size": batch,
            "prompt_tokens": context,
            "top_k": 1,
        },
        "identity": {
            "model_sha256": identity.hexdigest(),
            "prompt_sha256": _tensor_digest(prompt),
            "output_sha256": _tensor_digest(reference),
        },
        "runtime": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
        "output_shape": list(reference.shape),
        "methods": results,
        "outputs_match": True,
    }


def _publish(path: Path, report: dict) -> None:
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a tiny untrained CPU control, not production throughput."
    )
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (
        ("seed", 37),
        ("repeats", 3),
        ("warmup", 1),
        ("new-tokens", 4),
        ("prompt-tokens", 4),
        ("batch-size", 1),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        for name, lower, upper in (
            ("seed", 0, 2**63 - 1),
            ("repeats", 1, 20),
            ("warmup", 0, 5),
            ("new_tokens", 1, 32),
            ("prompt_tokens", 1, 128),
            ("batch_size", 1, 4),
        ):
            _integer(name, getattr(args, name), lower, upper)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        model = (
            DecoderLanguageModel(
                TransformerConfig(
                    vocab_size=32,
                    block_size=16,
                    embedding_dim=16,
                    head_count=2,
                    layer_count=1,
                    dropout=0,
                )
            )
            .cpu()
            .float()
        )
    prompt = (
        torch.arange(args.batch_size * args.prompt_tokens).reshape(
            args.batch_size, args.prompt_tokens
        )
        % 32
    )
    report = benchmark_generation(
        model,
        prompt,
        new_tokens=args.new_tokens,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
    )
    report["model_source"] = (
        "seeded untrained float32 control; synthetic public-safe token IDs"
    )
    _publish(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
