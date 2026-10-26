"""End-to-end public-data training, held-out scoring and serving acceptance."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import torch

from .ddp_benchmark import benchmark_service
from .ddp_checkpoint import bounded_checkpoint_load, validate_optimizer_state
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
    if (
        payload["model_config"] != model_config_mapping(spec.model)
        or payload["training_config"] != asdict(spec.training)
        or payload["source_sha256"] != sha256(source).hexdigest()
    ):
        raise ValueError("serving checkpoint provenance does not match")
    with torch.random.fork_rng(devices=[]):
        model = (
            build_training_model(spec.model, max_parameters=1_000_000).double().eval()
        )
    model.load_state_dict(payload["model"], strict=True)
    if gpt2_model_fingerprint(model) != payload["model_sha256"]:
        raise ValueError("serving checkpoint model digest does not match")
    optimizer = build_gpt2_optimizer(model, spec.training)
    optimizer.load_state_dict(payload["optimizer"])
    validate_optimizer_state(model, optimizer, step=payload["step"])
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
