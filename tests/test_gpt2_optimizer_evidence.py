from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path

import pytest
import torch

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_training import (
    GPT2TrainingError,
    optimizer_group_summary,
    run_training_experiment,
    training_spec_from_mapping,
)
from ai_journey.gpt2_training_evidence import (
    _dense_parameter_shapes,
    build_training_evidence,
    verify_training_evidence,
)
from ai_journey.gqa_config import GQAConfig
from ai_journey.moe_config import MoEConfig
from ai_journey.training_models import (
    build_training_model,
    expected_training_parameters,
    model_config_mapping,
)
from test_gpt2_training_evidence import run_fixture


def reseal(report):
    report["evidence_sha256"] = sha256(
        json.dumps(
            {key: value for key, value in report.items() if key != "evidence_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return report


@pytest.fixture(scope="module", params=["baseline", "modern", "grouped"])
def measured_report(tmp_path_factory, request):
    directory = tmp_path_factory.mktemp("real-optimizer-evidence")
    run = run_fixture(directory)
    if request.param != "baseline":
        config = DecoderVariantConfig(run.spec.model, "rmsnorm", "swiglu", "rope")
        if request.param == "grouped":
            config = GQAConfig(config, kv_heads=1)
        run = run_training_experiment(
            b"public training text\n" * 8,
            replace(run.spec, model=config),
            checkpoint_path=directory / "variant.pt",
        )
    return build_training_evidence(run)


@pytest.mark.parametrize(
    "change",
    ["unknown_name", "moved_group", "shifted_counts", "alias", "mapping", "tuple"],
)
def test_resealed_optimizer_claims_must_match_the_declared_architecture(
    measured_report, change
):
    report = deepcopy(measured_report)
    groups = report["optimizer_groups"]
    if change == "unknown_name":
        groups["decay_names"][0] = "nonexistent.weight"
    elif change == "moved_group":
        groups["decay_names"][0], groups["no_decay_names"][0] = (
            groups["no_decay_names"][0],
            groups["decay_names"][0],
        )
    elif change == "shifted_counts":
        groups["decay_parameters"] += 1
        groups["no_decay_parameters"] -= 1
    elif change == "alias":
        groups["decay_names"][0] = groups["decay_names"][0].replace("wte", "lm_head")
    elif change == "mapping":
        groups["decay_names"] = dict.fromkeys(groups["decay_names"], 0)
    else:
        groups["decay_names"] = tuple(groups["decay_names"])
    reseal(report)
    with pytest.raises(GPT2TrainingError, match="optimizer"):
        verify_training_evidence(report)


def test_published_legacy_training_evidence_remains_valid():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/gpt2-training-2026-10-21/training-evidence.json"
    )
    report = json.loads(path.read_bytes())
    assert verify_training_evidence(report) == report


def architectures():
    base = GPT2Config(
        256,
        8,
        16,
        4,
        2,
        embedding_dropout=0,
        attention_dropout=0,
        residual_dropout=0,
    )
    result = [pytest.param(base, id="baseline")]
    for norm, mlp, position in product(
        ("layernorm", "rmsnorm"), ("gelu", "swiglu"), ("learned", "rope")
    ):
        config = DecoderVariantConfig(
            base, norm, mlp, position, swiglu_hidden_dim=17 if mlp == "swiglu" else None
        )
        result.append(pytest.param(config, id=f"{norm}-{mlp}-{position}"))
        if position == "rope":
            for heads in (1, 2, 4):
                result.append(
                    pytest.param(
                        GQAConfig(config, heads), id=f"gqa-{heads}-{norm}-{mlp}"
                    )
                )
    return result


@pytest.mark.parametrize("config", architectures())
def test_allocation_free_schema_matches_independent_actual_model_names_shapes_and_counts(
    config,
):
    with torch.random.fork_rng(devices=[]):
        model = build_training_model(config)
    actual = {
        name: tuple(parameter.shape) for name, parameter in model.named_parameters()
    }
    schema = _dense_parameter_shapes(config, name_count=len(actual))
    assert list(schema.items()) == list(actual.items())
    summary = optimizer_group_summary(model)
    assert summary.decay_names == tuple(
        name for name, shape in schema.items() if len(shape) >= 2
    )
    assert summary.no_decay_names == tuple(
        name for name, shape in schema.items() if len(shape) < 2
    )
    assert summary.decay_parameters == sum(
        math.prod(schema[name]) for name in summary.decay_names
    )
    assert summary.no_decay_parameters == sum(
        math.prod(schema[name]) for name in summary.no_decay_names
    )
    assert sum(
        math.prod(shape) for shape in schema.values()
    ) == expected_training_parameters(config)
    assert all(not name.endswith("lm_head.weight") for name in schema)


def test_valid_report_verification_constructs_no_model_or_tensors(
    measured_report, monkeypatch
):
    def unexpected(*args, **kwargs):
        raise AssertionError("offline verification must not allocate model tensors")

    monkeypatch.setattr(torch.nn.Module, "__init__", unexpected)
    for name in ("empty", "zeros", "ones", "tensor"):
        monkeypatch.setattr(torch, name, unexpected)
    assert verify_training_evidence(measured_report) == measured_report


def test_impossible_layer_count_rejects_before_name_enumeration(
    measured_report, monkeypatch
):
    report = deepcopy(measured_report)
    original = training_spec_from_mapping(report["spec"])
    base = original.model.gpt2 if hasattr(original.model, "gpt2") else original.model
    report["spec"] = replace(
        original, model=replace(base, layer_count=10**1000)
    ).to_mapping()
    parsed = training_spec_from_mapping(report["spec"])
    report["spec_sha256"] = parsed.fingerprint()
    report["parameter_count"] = expected_training_parameters(parsed.model)
    report["optimizer_groups"]["decay_parameters"] = (
        report["parameter_count"] - report["optimizer_groups"]["no_decay_parameters"]
    )
    reseal(report)

    def unexpected(*args):
        raise AssertionError("unbounded layer enumeration must not start")

    monkeypatch.setattr(
        "ai_journey.gpt2_training_evidence.range", unexpected, raising=False
    )
    with pytest.raises(GPT2TrainingError, match="optimizer names"):
        verify_training_evidence(report)


@pytest.mark.parametrize("grouped", [False, True])
def test_plain_ce_report_cannot_claim_a_moe_architecture(measured_report, grouped):
    report = deepcopy(measured_report)
    parsed = training_spec_from_mapping(report["spec"])
    base = parsed.model.gpt2 if hasattr(parsed.model, "gpt2") else parsed.model
    config = MoEConfig(base, position_kind="rope")
    if grouped:
        config = GQAConfig(config, 1)
    report["spec"]["model"] = model_config_mapping(config)
    report["spec"]["schema_version"] = 3 if grouped else 2
    reseal(report)
    with pytest.raises(GPT2TrainingError, match="auxiliary objective"):
        verify_training_evidence(report)
