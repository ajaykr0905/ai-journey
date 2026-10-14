"""Paired cache/full-forward timings bound to verified BPE weights and tokens."""

from hashlib import sha256
import json
import statistics
from time import perf_counter_ns

import torch

from .bpe import TokenizerError
from .bpe_artifact import BPEArtifact
from .bpe_serving import generate_utf8
from .transformer_lab import model_fingerprint


def benchmark_bpe_serving(
    artifact, prompt, *, new_tokens=4, repeats=3, seed=40, temperature=0.8, top_k=1
):
    if not isinstance(artifact, BPEArtifact):
        raise TypeError("serving benchmark requires a verified BPEArtifact")
    if (
        type(repeats) is not int
        or not 1 <= repeats <= 10
        or type(new_tokens) is not int
        or not 1 <= new_tokens <= 32
    ):
        raise TokenizerError("serving benchmark exceeds repeat or token limits")
    expected_model = artifact.report["experiment"]["model_fingerprint"]
    if model_fingerprint(artifact.model) != expected_model:
        raise TokenizerError("benchmark model no longer matches its training evidence")
    controls = dict(
        new_tokens=new_tokens, seed=seed, temperature=temperature, top_k=top_k
    )
    reference = generate_utf8(artifact.model, artifact.corpus.codec, prompt, **controls)
    reference.pop("backend")
    samples = {"full-forward": [], "kv-cache": []}
    for repeat in range(repeats + 1):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for cached in order:
            started = perf_counter_ns()
            sample = generate_utf8(
                artifact.model, artifact.corpus.codec, prompt, cached=cached, **controls
            )
            elapsed = perf_counter_ns() - started
            backend = sample.pop("backend")
            if sample != reference or elapsed <= 0:
                raise TokenizerError(
                    "paired serving benchmark token parity or timing failed"
                )
            if repeat:
                samples[backend].append(elapsed)
    if model_fingerprint(artifact.model) != expected_model:
        raise TokenizerError("benchmark changed its model weights")
    count = len(reference["token_ids"]) - len(artifact.corpus.codec.encode(prompt))
    return {
        "schema_version": 1,
        "model_sha256": expected_model,
        "codec_sha256": artifact.corpus.codec.fingerprint(),
        "corpus_sha256": artifact.corpus.fingerprint(),
        "controls": {"prompt": prompt, "repeats": repeats, **controls},
        "sample_sha256": sha256(
            json.dumps(reference, sort_keys=True).encode()
        ).hexdigest(),
        "generated_tokens": count,
        "elapsed_ns": samples,
        "median_ns": {
            name: statistics.median(times) for name, times in samples.items()
        },
        "runtime": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "threads": torch.get_num_threads(),
        },
        "scope": "Local CPU paired observations after one warmup; includes UTF-8 masking and cache identity checks. No production speed claim.",
    }
