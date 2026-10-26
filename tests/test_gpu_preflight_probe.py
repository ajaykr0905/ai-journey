"""Host discovery must report unavailable hardware without a CUDA claim."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from ai_journey.gpu_preflight_probe import probe_gpu_host, validate_gpu_probe


def simulated_nvidia_host(monkeypatch):
    import ai_journey.gpu_preflight_probe as probe

    monkeypatch.setattr(probe.platform, "system", lambda: "Linux")
    monkeypatch.setattr(probe.torch.version, "cuda", "12.4")
    monkeypatch.setattr(probe.torch.distributed, "is_nccl_available", lambda: True)
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(probe.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        probe.torch.cuda,
        "get_device_properties",
        lambda ordinal: SimpleNamespace(
            name=f"simulated GPU {ordinal}", major=8, minor=0
        ),
    )
    monkeypatch.setattr(
        probe.torch.cuda, "mem_get_info", lambda ordinal: (900000, 1000000)
    )
    return probe


def test_darwin_stops_before_cuda_driver_access(monkeypatch):
    import ai_journey.gpu_preflight_probe as probe

    monkeypatch.setattr(probe.platform, "system", lambda: "Darwin")

    def unexpected():
        raise AssertionError("CUDA driver must not be queried")

    monkeypatch.setattr(probe.torch.cuda, "is_available", unexpected)
    observed = probe_gpu_host()
    assert observed["available"] is False
    assert observed["reason"] == "unsupported_operating_system"
    assert observed["devices"] == []


def test_simulated_properties_include_runtime_memory_and_visibility_identity(
    monkeypatch,
):
    simulated_nvidia_host(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    observed = probe_gpu_host()
    assert observed["available"] is True
    assert len(observed["devices"]) == 2
    assert observed["devices"][1]["free_bytes"] == 900000
    assert len(observed["visible_devices_sha256"]) == 64
    assert "2,5" not in str(observed)


def test_runtime_query_failure_is_unavailable_not_partial_success(monkeypatch):
    probe = simulated_nvidia_host(monkeypatch)

    def failing(ordinal):
        raise RuntimeError("simulated driver failure")

    monkeypatch.setattr(probe.torch.cuda, "mem_get_info", failing)
    observed = probe_gpu_host()
    assert observed["reason"] == "device_query_failed"
    assert observed["available"] is False
    assert observed["devices"] == []


def test_claimed_availability_needs_supporting_observations(monkeypatch):
    simulated_nvidia_host(monkeypatch)
    observed = probe_gpu_host()
    for name, value in (
        ("nccl_available", False),
        ("devices", []),
        ("system", "Darwin"),
        ("cuda_runtime", None),
    ):
        forged = deepcopy(observed)
        forged[name] = value
        with pytest.raises(ValueError, match="supporting"):
            validate_gpu_probe(forged)
