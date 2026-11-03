import json

import pytest

from ai_journey.scaling_laws import ScalingLawError
from ai_journey.scaling_protocol import (
    MAX_SCALING_PROTOCOL_BYTES,
    load_scaling_protocol,
)


def payload():
    return {
        "schema_version": 1,
        "name": "day-60-100m-chinchilla-plan",
        "parameter_count": 100_000_000,
        "tokens_per_parameter": "20",
        "training_flops_per_parameter_token": "6",
        "unique_train_tokens": 69_626,
        "batch": {
            "sequence_length": 2_048,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "data_parallel_size": 8,
        },
        "maximum_overshoot_fraction": 0.0001,
        "isoflop_parameter_counts": [50_000_000, 100_000_000, 200_000_000],
    }


def test_protocol_loads_a_complete_preregistered_scenario(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload()))

    protocol = load_scaling_protocol(path)

    assert protocol.parameter_count == 100_000_000
    assert protocol.batch.tokens_per_optimizer_step == 131_072
    assert protocol.isoflop_parameter_counts == (
        50_000_000,
        100_000_000,
        200_000_000,
    )


def test_protocol_rejects_unknown_duplicate_and_invalid_nested_fields(tmp_path):
    data = payload()
    data["extra"] = 1
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ScalingLawError, match="fields"):
        load_scaling_protocol(path)

    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ScalingLawError, match="duplicate"):
        load_scaling_protocol(path)

    data = payload()
    data["batch"]["extra"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ScalingLawError, match="batch fields"):
        load_scaling_protocol(path)


def test_protocol_rejects_excessive_or_unfunded_inputs(tmp_path):
    path = tmp_path / "oversize.json"
    path.write_bytes(b"{" + b" " * MAX_SCALING_PROTOCOL_BYTES)
    with pytest.raises(ScalingLawError, match="exceeds"):
        load_scaling_protocol(path)

    data = payload()
    data["maximum_overshoot_fraction"] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ScalingLawError, match="overshoot"):
        load_scaling_protocol(path)
