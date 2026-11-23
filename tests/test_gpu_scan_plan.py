import dataclasses
import hashlib

import pytest

from ai_journey import gpu_scan_source
from ai_journey.gpu_scan_plan import (
    COPY,
    LENGTHS,
    PLAN,
    RAGGED_LENGTH,
    SIDES,
    VENDOR,
    ScanHypothesis,
    hypothesis_of,
    plan_mapping,
    plan_sha256,
    require_plan,
)
from ai_journey.gpu_scan_source import (
    BLOCK,
    KERNELS,
    S0_HILLIS_STEELE,
    S1_BLELLOCH,
    S2_SIMD,
    GpuScanError,
    level_lengths,
)
from ai_journey.memory_access_plan import canonical_json_bytes


def test_the_plan_declares_one_hypothesis_for_each_kernel_in_order():
    rows = [(entry.step, entry.kernel, entry.base, entry.predictions) for entry in PLAN]

    assert rows == [
        ("S0", S0_HILLIS_STEELE, VENDOR, tuple((n, ("slower",)) for n in LENGTHS)),
        (
            "S1",
            S1_BLELLOCH,
            S0_HILLIS_STEELE,
            (
                (65536, ("no_clear_difference",)),
                (1048576, ("no_clear_difference", "slower")),
                (16777216, ("no_clear_difference", "slower")),
                (5000003, ("no_clear_difference", "slower")),
            ),
        ),
        (
            "S2",
            S2_SIMD,
            S0_HILLIS_STEELE,
            (
                (65536, ("no_clear_difference",)),
                (1048576, ("faster",)),
                (16777216, ("faster",)),
                (5000003, ("faster",)),
            ),
        ),
    ]
    assert SIDES == (*KERNELS, "torch_cumsum", "stream_copy")
    assert hypothesis_of(S1_BLELLOCH) is PLAN[1]
    assert PLAN[2].predicted(1 << 20) == ("faster",)
    assert PLAN[2].predicted(4096) is None


def test_the_pinned_lengths_include_a_partial_block_at_every_level():
    assert LENGTHS[:3] == (1 << 16, 1 << 20, 1 << 24)
    assert all(
        length % BLOCK for length in level_lengths(RAGGED_LENGTH)
    ), level_lengths(RAGGED_LENGTH)
    assert len(level_lengths(RAGGED_LENGTH)) == 3


def test_the_plan_digest_binds_every_hypothesis_and_kernel_text(monkeypatch):
    mapping = plan_mapping()

    assert plan_sha256() == hashlib.sha256(canonical_json_bytes(mapping)).hexdigest()
    assert mapping["kernels"] == {
        name: kernel.source_sha256 for name, kernel in KERNELS.items()
    }
    assert mapping["library_sha256"] == gpu_scan_source.library_sha256()
    assert (mapping["vendor"], mapping["copy"]) == (VENDOR, COPY)
    assert [entry["kernel"] for entry in mapping["hypotheses"]] == list(KERNELS)
    before = plan_sha256()
    changed = dataclasses.replace(
        PLAN[2], hypothesis=PLAN[2].hypothesis.replace("1.1", "1.2")
    )
    assert plan_sha256((*PLAN[:2], changed)) != before
    kernel = KERNELS[S2_SIMD]
    edited = dataclasses.replace(kernel, source=kernel.source + "// edited\n")
    monkeypatch.setitem(KERNELS, S2_SIMD, edited)
    assert plan_sha256() != before


def test_the_hypotheses_keep_the_evidence_honest():
    texts = " ".join(entry.hypothesis for entry in PLAN)

    assert "PyTorch does not report which kernel torch.cumsum runs." in texts
    assert (
        "SIMD-group functions are the Metal counterpart of CUDA warp shuffles, "
        "not the same instruction."
    ) in texts
    assert "untimed check before this plan" in texts
    for word in ("NVIDIA", "occupancy", "cuBLAS", "production"):
        assert word not in texts


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step", "S10", "S and one digit"),
        ("kernel", "scan_s9_other", "not a registered"),
        ("base", S2_SIMD, "another kernel or the vendor"),
        ("base", COPY, "another kernel or the vendor"),
        ("change", "", "ASCII text"),
        ("hypothesis", "x" * 801, "ASCII text"),
        ("hypothesis", "faster → slower", "ASCII text"),
        ("predictions", ((65536, ()),), "distinct verdicts"),
        ("predictions", ((65536, ("faster", "faster")),), "distinct verdicts"),
        ("predictions", ((65536, ("quicker",)),), "distinct verdicts"),
        ("predictions", ((65536, ("faster",)),), "each pinned length once"),
    ],
)
def test_a_hypothesis_with_a_wrong_field_is_refused(field, value, message):
    with pytest.raises(GpuScanError, match=message):
        dataclasses.replace(PLAN[2], **{field: value})


def test_a_plan_out_of_order_or_without_a_kernel_is_refused():
    with pytest.raises(GpuScanError, match="S0, S1"):
        require_plan((PLAN[1], PLAN[0], PLAN[2]))
    with pytest.raises(GpuScanError, match="one step for each kernel"):
        require_plan(PLAN[:2])
    with pytest.raises(TypeError):
        require_plan((PLAN[0], "S1"))
    with pytest.raises(TypeError):
        ScanHypothesis("S0", S0_HILLIS_STEELE, VENDOR, "a", "b", [])
    with pytest.raises(GpuScanError, match="no declared hypothesis"):
        hypothesis_of("scan_s9_other")
