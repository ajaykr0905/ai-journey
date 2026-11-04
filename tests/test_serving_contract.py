import pytest

from ai_journey.serving_contract import (
    ClaimBoundary,
    EngineIdentity,
    HardwareManifest,
    ModelIdentity,
    RunLimits,
    ServingContractError,
    SoftwareManifest,
    WorkloadShape,
)


def test_workload_shape_bounds_total_token_demand():
    workload = WorkloadShape(
        request_count=8,
        prompt_tokens_per_request=128,
        output_tokens_per_request=32,
        max_concurrency=4,
    )

    assert workload.total_requested_tokens == 1280


@pytest.mark.parametrize(
    "field,value,exception",
    [
        ("request_count", 0, ServingContractError),
        ("prompt_tokens_per_request", True, TypeError),
        ("output_tokens_per_request", -1, ServingContractError),
        ("max_concurrency", 9, ServingContractError),
    ],
)
def test_workload_shape_rejects_ambiguous_or_unbounded_controls(
    field, value, exception
):
    values = {
        "request_count": 8,
        "prompt_tokens_per_request": 128,
        "output_tokens_per_request": 32,
        "max_concurrency": 4,
    }
    values[field] = value

    with pytest.raises(exception):
        WorkloadShape(**values)


def test_engine_identity_pins_name_version_and_revision():
    identity = EngineIdentity("vllm", "0.12.0", "abcdef1")

    assert (identity.name, identity.version, identity.revision) == (
        "vllm",
        "0.12.0",
        "abcdef1",
    )


@pytest.mark.parametrize("value", ["", " leading", "trailing "])
def test_engine_identity_rejects_noncanonical_text(value):
    with pytest.raises(ServingContractError, match="canonical"):
        EngineIdentity("vllm", "0.12.0", value)


def test_model_identity_pins_model_and_tokenizer_revisions():
    identity = ModelIdentity("org/model", "model-sha", "tokenizer-sha", 7_000_000_000)

    assert identity.parameter_count == 7_000_000_000
    assert identity.revision == "model-sha"
    assert identity.tokenizer_revision == "tokenizer-sha"


def test_model_identity_rejects_unbounded_parameter_counts():
    with pytest.raises(ServingContractError, match="positive"):
        ModelIdentity("org/model", "model-sha", "tokenizer-sha", 0)


def test_hardware_manifest_computes_declared_total_device_memory():
    manifest = HardwareManifest("NVIDIA H100", 2, 80 * 1024**3, "NVLink")

    assert manifest.total_device_memory_bytes == 160 * 1024**3


def test_hardware_manifest_rejects_zero_devices():
    with pytest.raises(ServingContractError, match="device_count"):
        HardwareManifest("CPU fixture", 0, 1024, "host memory")


def test_software_manifest_requires_explicit_stack_versions():
    manifest = SoftwareManifest("Linux 6.8", "3.12.12", "torch 2.9.0", "CUDA 13.0")

    assert manifest.framework_version == "torch 2.9.0"


def test_software_manifest_rejects_blank_driver_identity():
    with pytest.raises(ServingContractError, match="driver_version"):
        SoftwareManifest("Linux", "3.12", "torch 2.9", "")


def test_run_limits_pin_duration_and_deterministic_seed():
    limits = RunLimits(max_duration_seconds=300, random_seed=42)

    assert limits == RunLimits(300, 42)


@pytest.mark.parametrize("duration,seed", [(0, 1), (1, -1), (1, True)])
def test_run_limits_reject_unbounded_or_ambiguous_controls(duration, seed):
    with pytest.raises((ServingContractError, TypeError)):
        RunLimits(duration, seed)


def test_claim_boundary_records_only_sorted_explicit_claims():
    boundary = ClaimBoundary("cpu_fixture", ("functional", "restart"))

    assert boundary.allowed_claims == ("functional", "restart")


@pytest.mark.parametrize(
    "kind,claims",
    [
        ("unknown", ("functional",)),
        ("cpu_fixture", ("restart", "functional")),
        ("measured_hardware", ("production",)),
    ],
)
def test_claim_boundary_rejects_unsupported_or_overstated_claims(kind, claims):
    with pytest.raises(ServingContractError):
        ClaimBoundary(kind, claims)
