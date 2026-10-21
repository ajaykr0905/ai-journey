"""Predeclared three-arm experiment isolating global-batch decomposition."""

import json
from pathlib import Path

from .ddp_checkpoint import bounded_checkpoint_load
from .ddp_evidence import build_evidence, verify_evidence, digest
from .ddp_launch import launch_training
from .gpt2_training import load_training_spec


ARMS = (
    ("single_accumulation", 1, 1, 4),
    ("ddp_accumulation", 2, 1, 2),
    ("ddp_microbatch", 2, 2, 1),
)


def compare_decompositions(
    *, source: Path, spec_path: Path, directory: Path, steps=4
) -> dict:
    """Keep model/data/seed/optimizer/token budget fixed while redistributing work."""
    directory.mkdir(parents=True, exist_ok=True)
    base = json.loads(spec_path.read_text())
    records, states = [], []
    for name, world, microbatch, accumulation in ARMS:
        payload = json.loads(json.dumps(base))
        payload["training"].update(
            micro_batch_size=microbatch, accumulation_steps=accumulation
        )
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload))
        checkpoint = directory / f"{name}.pt"
        ranks = launch_training(
            source=source,
            spec=path,
            checkpoint=checkpoint,
            output_dir=directory / name,
            world_size=world,
            steps=steps,
        )
        states.append(bounded_checkpoint_load(checkpoint, 64 * 1024 * 1024)["model"])
        records.append(
            {
                "name": name,
                "world_size": world,
                "micro_batch_size": microbatch,
                "accumulation_steps": accumulation,
                "evidence": build_evidence(
                    source.read_bytes(), load_training_spec(path), ranks
                ),
            }
        )
    errors = [
        max((state[key] - states[0][key]).abs().max().item() for key in state)
        for state in states
    ]
    if max(errors) > 1e-10:
        raise RuntimeError(
            "global-batch decomposition changes the trained state beyond tolerance"
        )
    report = {
        "schema_version": 1,
        "hypothesis": "Equivalent global batches reproduce parameter updates within FP64 rounding tolerance.",
        "fixed_conditions": [
            "model",
            "public_source",
            "seed",
            "optimizer",
            "schedule",
            "global_batch",
            "target_tokens",
            "cpu_dtype_float64",
        ],
        "global_batch_size": 4,
        "absolute_tolerance": 1e-10,
        "max_parameter_errors": errors,
        "arms": records,
        "limitations": "One CPU fixture and seed; numerical equivalence is not a scaling-performance or model-quality result.",
    }
    report["evidence_sha256"] = digest(report)
    return verify_comparison(report)


def verify_comparison(report):
    if report.get("schema_version") != 1 or digest(
        {key: value for key, value in report.items() if key != "evidence_sha256"}
    ) != report.get("evidence_sha256"):
        raise ValueError("comparison schema or digest does not match")
    if len(report["arms"]) != len(ARMS) or len(report["max_parameter_errors"]) != len(
        ARMS
    ):
        raise ValueError("comparison requires all predeclared arms")
    specs = []
    for expected, arm, error in zip(
        ARMS, report["arms"], report["max_parameter_errors"]
    ):
        name, world, microbatch, accumulation = expected
        if (
            (
                arm["name"],
                arm["world_size"],
                arm["micro_batch_size"],
                arm["accumulation_steps"],
            )
            != expected
            or not isinstance(error, (int, float))
            or not 0 <= error <= report["absolute_tolerance"]
        ):
            raise ValueError("comparison controls or observed error are invalid")
        evidence = verify_evidence(arm["evidence"])
        spec = json.loads(json.dumps(evidence["spec"]))
        training = spec["training"]
        if (
            training.pop("micro_batch_size") != microbatch
            or training.pop("accumulation_steps") != accumulation
            or evidence["world_size"] != world
        ):
            raise ValueError("comparison arm does not match measured execution")
        specs.append(spec)
    if any(spec != specs[0] for spec in specs):
        raise ValueError("comparison changed a fixed experimental condition")
    return report
