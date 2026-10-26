"""End-to-end public-data training, held-out scoring and serving acceptance."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

import torch

from .ddp_benchmark import benchmark_service
from .ddp_checkpoint import bounded_checkpoint_load
from .ddp_contracts import integer
from .ddp_evaluation import documents_from_lines, split_documents, evaluate_documents
from .ddp_evidence import build_evidence
from .ddp_generation import SamplingConfig
from .ddp_launch import launch_training
from .ddp_limits import preflight
from .ddp_service import GenerationService, GenerationRequest
from .gpt2_training import (
    training_spec_from_mapping,
    build_gpt2_optimizer,
    gpt2_model_fingerprint,
    _validate_optimizer_contract,
)
from .training_models import (
    build_training_model,
    expected_training_parameters,
    model_config_mapping,
)


def load_serving_model(checkpoint: Path, spec, source: bytes):
    if expected_training_parameters(spec.model) > 1_000_000:
        raise ValueError("serving model exceeds CPU lab parameter budget")
    payload = bounded_checkpoint_load(checkpoint, 64 * 1024 * 1024)
    fields = {
        "schema_version",
        "world_size",
        "step",
        "model_config",
        "training_config",
        "source_sha256",
        "model_sha256",
        "model",
        "optimizer",
        "rank_rngs",
    }
    if (
        type(payload) is not dict
        or set(payload) != fields
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("serving checkpoint schema does not match")
    integer(payload["step"], "step")
    integer(payload["world_size"], "world_size", 1)
    if payload["step"] > spec.training.optimizer_steps or payload["world_size"] > 4:
        raise ValueError("serving checkpoint progress or topology exceeds lab limits")
    if (
        type(payload["rank_rngs"]) is not list
        or len(payload["rank_rngs"]) != payload["world_size"]
    ):
        raise ValueError("serving checkpoint rank RNG states do not match")
    for state in payload["rank_rngs"]:
        if (
            not isinstance(state, torch.Tensor)
            or state.device.type != "cpu"
            or state.dtype != torch.uint8
            or state.shape != torch.get_rng_state().shape
        ):
            raise ValueError("serving checkpoint rank RNG state is invalid")
        try:
            torch.Generator(device="cpu").set_state(state)
        except RuntimeError as error:
            raise ValueError("serving checkpoint rank RNG state is invalid") from error
    if (
        payload["model_config"] != model_config_mapping(spec.model)
        or payload["training_config"] != asdict(spec.training)
        or payload["source_sha256"] != sha256(source).hexdigest()
        or payload["source_sha256"] != spec.source.sha256
        or not isinstance(payload["model_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", payload["model_sha256"]) is None
    ):
        raise ValueError("serving checkpoint provenance does not match")
    with torch.random.fork_rng(devices=[]):
        model = (
            build_training_model(spec.model, max_parameters=1_000_000).double().eval()
        )
    expected = model.state_dict()
    state = payload["model"]
    if (
        not isinstance(state, dict)
        or set(state) != set(expected)
        or any(
            not isinstance(state[name], torch.Tensor)
            or state[name].shape != tensor.shape
            or state[name].dtype != tensor.dtype
            for name, tensor in expected.items()
        )
    ):
        raise ValueError("serving checkpoint tensor layout does not match")
    model.load_state_dict(state, strict=True)
    if gpt2_model_fingerprint(model) != payload["model_sha256"]:
        raise ValueError("serving checkpoint model digest does not match")
    optimizer = build_gpt2_optimizer(model, spec.training)
    optimizer.load_state_dict(payload["optimizer"])
    _validate_optimizer_contract(model, optimizer, spec.training, step=payload["step"])
    model.requires_grad_(False)
    return model


def run_pipeline(
    *,
    source: Path,
    spec_path: Path,
    directory: Path,
    validation_count=4,
    world_size=2,
    steps=2,
):
    original_spec = preflight(source, spec_path, world_size=world_size, steps=steps)
    original = source.read_bytes()
    if sha256(original).hexdigest() != original_spec.source.sha256:
        raise ValueError("public fixture does not match source specification")
    train_docs, validation_docs = split_documents(
        documents_from_lines(original),
        validation_count=validation_count,
        seed=original_spec.training.seed,
    )
    training_source = b"".join(document.content for document in train_docs)
    payload = json.loads(spec_path.read_text())
    payload["source"]["sha256"] = sha256(training_source).hexdigest()
    payload["source"]["revision"] += "-document-isolated-train-split"
    spec = training_spec_from_mapping(payload)
    directory.mkdir(parents=True, exist_ok=True)
    training_path = directory / "train.txt"
    derived_spec = directory / "training-spec.json"
    training_path.write_bytes(training_source)
    derived_spec.write_text(json.dumps(payload))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(spec.training.seed)
        initial = build_training_model(spec.model, max_parameters=1_000_000).double()
    before = evaluate_documents(initial, validation_docs)
    checkpoint = directory / "trained.pt"
    ranks = launch_training(
        source=training_path,
        spec=derived_spec,
        checkpoint=checkpoint,
        output_dir=directory / "ranks",
        world_size=world_size,
        steps=steps,
    )
    model = load_serving_model(checkpoint, spec, training_source)
    after = evaluate_documents(model, validation_docs)
    service = GenerationService(model)
    request = GenerationRequest(
        "acceptance", tuple(b"a"), SamplingConfig(max_new_tokens=4, temperature=0)
    )
    generation = service.generate(request)
    benchmark = benchmark_service(service, request, repeats=3, warmup=1)
    return {
        "schema_version": 1,
        "public_source_sha256": sha256(original).hexdigest(),
        "training_documents": [document.digest for document in train_docs],
        "validation_documents": [document.digest for document in validation_docs],
        "validation_before": before,
        "validation_after": after,
        "distributed": build_evidence(training_source, spec, ranks),
        "generation": generation,
        "serving_benchmark": benchmark,
        "limitations": "CPU fixture acceptance only; retain held-out regressions and timing variability.",
    }
