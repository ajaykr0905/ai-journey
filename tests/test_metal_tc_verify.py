import copy

import pytest

from ai_journey.memory_access_report import seal_report
from ai_journey.metal_tc_ceiling import load_tc_ceiling
from ai_journey.metal_tc_report import GPU_ONLY, verification_scope
from ai_journey.metal_tc_verify import MetalTcVerificationError, verify_report
from test_metal_fused_ceiling import write_recorded
from test_metal_tc_emulator import TcEmulator
from test_metal_tc_report import small_config, synthetic_report


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return synthetic_report(tmp_path_factory.mktemp("ceiling"))


def resealed(report, path, value):
    changed = copy.deepcopy(report)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value(target[path[-1]]) if callable(value) else value
    changed.pop("report_sha256")
    return seal_report(changed)


def test_a_synthetic_report_verifies_with_its_config_and_ceiling(report, tmp_path):
    ceiling = load_tc_ceiling(write_recorded(tmp_path))
    summary = verify_report(report, config=small_config(), ceiling=ceiling)

    assert summary["measured"] is False
    assert summary["label"] == "synthetic timing, not a measurement"
    assert summary["correctness_cases"] == 5
    assert summary["supported"] == {"m3": True, "torch_bfloat16": True}
    assert summary["max_bound_ratio"]["m3"] <= 1


def test_a_report_without_the_bfloat_path_verifies(tmp_path):
    backend = TcEmulator(bfloat=False, torch_bfloat=False)
    summary = verify_report(synthetic_report(tmp_path, backend=backend))

    assert summary["supported"] == {"m3": False, "torch_bfloat16": False}
    assert summary["hypotheses"]["m3_over_torch_bfloat16"]["not_run"] == 2


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("measured",), True),
        (("evidence_boundary", "tensor_cores_used"), True),
        (("plan", "steps", 0, "predicted"), ["slower"]),
        (("config", "seed"), 83),
        (("correctness", "cases", 2, "inputs", "float16", "a_sha256"), "0" * 64),
        (("correctness", "cases", 2, "bound_sha256", "bfloat16"), "0" * 64),
        (("correctness", "cases", 4, "outputs", "m2", "max_bound_ratio"), 1.5),
        (
            ("correctness", "cases", 4, "outputs", "m3", "max_abs_error"),
            lambda value: value * 4,
        ),
        (("correctness", "cases", 1, "m1_same_bits_as_s1"), False),
        (("benchmark", "points", 0, "sides", "m1", "samples_ns", 0), 1),
        (("benchmark", "points", 1, "checks", "torch_float16", "max_bound_ratio"), 2.0),
        (("benchmark", "points", 1, "supported"), ["s1", "m1"]),
        (("speedup", "points", 0, "steps", 0, "verdict"), "faster"),
        (("speedup", "points", 1, "rates", "m2", "fraction_of_fma"), 0.5),
        (("support", "metal_language_version"), "400"),
        (("support", "sides", "m3", "supported"), False),
        (("shaders", "kernels", 1, "limits", "thread_execution_width"), 64),
        (("summary", "hypotheses", "m1_over_s1", "held"), 9),
        (("run", "timed_ns"), 1),
        (("verification", "gpu_only"), list(GPU_ONLY[1:])),
        (("compute_ceiling", "fma_gflops_per_second"), 1.0),
    ],
)
def test_a_changed_field_that_is_resealed_is_rejected(report, path, value):
    with pytest.raises(MetalTcVerificationError):
        verify_report(resealed(report, path, value))


def test_an_unsealed_edit_and_a_foreign_config_are_rejected(report):
    edited = copy.deepcopy(report)
    edited["summary"]["warnings"] = 5
    with pytest.raises(MetalTcVerificationError, match="digest"):
        verify_report(edited)
    other = small_config(comparison={"min_faster_ratio": 1.2})
    with pytest.raises(MetalTcVerificationError, match="supplied config"):
        verify_report(report, config=other)
    with pytest.raises(MetalTcVerificationError):
        verify_report([report])


def test_the_scope_names_what_only_the_gpu_saw():
    scope = verification_scope()

    assert any("samples_ns" in line for line in scope["gpu_only"])
    assert any("identical_repeats" in line for line in scope["gpu_only"])
    assert any("support" in line for line in scope["gpu_only"])
    assert any("bound digest" in line for line in scope["recomputed"])
