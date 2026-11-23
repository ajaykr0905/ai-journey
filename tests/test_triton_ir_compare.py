"""Tests of the comparison of the six configs and two targets of the recorded dump.

Every number comes from the committed manifest or the committed texts. No
test imports triton.
"""

import json
from pathlib import Path

import pytest

from ai_journey.triton_ir_compare import (
    NEEDS_FULL_DUMPS,
    compare,
    compile_rows,
    target_rows,
)
from ai_journey.triton_ir_manifest import (
    TritonIrError,
    check_fixtures,
    load_manifest,
    parse_manifest,
)
from ai_journey.triton_ir_ptx import read_ptx

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "day80_triton_ir"


@pytest.fixture(scope="module")
def inputs():
    manifest = load_manifest(FIXTURES)
    texts = check_fixtures(manifest, FIXTURES)
    return manifest, texts["ttgir"].text, read_ptx(texts["ptx"].text)


def test_each_ring_holds_stages_minus_one_tiles_on_both_targets(inputs):
    rows = compile_rows(inputs[0])
    assert len(rows) == 12
    assert all(row["ring_is_stages_minus_one"] for row in rows)
    rings = {(row["config"], row["num_stages"], row["ring_buffers"]) for row in rows}
    assert rings == {
        ("m128_n128_k32_g8_w4_s4", 4, 3),
        ("m128_n64_k32_g8_w4_s4", 4, 3),
        ("m64_n128_k32_g8_w4_s4", 4, 3),
        ("m128_n32_k32_g8_w4_s4", 4, 3),
        ("m64_n32_k32_g8_w2_s5", 5, 4),
        ("m32_n64_k32_g8_w2_s5", 5, 4),
    }


def test_the_line_counts_and_the_derived_fma_of_each_compile(inputs):
    rows = {(row["target"], row["config"]): row for row in compile_rows(inputs[0])}
    big = rows[("sm_80", "m128_n128_k32_g8_w4_s4")]
    assert big["lines"] == {"ttir": 214, "ttgir": 312, "llir": 10151, "ptx": 9303}
    assert big["derived_fma_per_thread_per_iteration"] == 4096
    assert big["threads"] == 128
    small = rows[("sm_80", "m32_n64_k32_g8_w2_s5")]
    assert small["committed_texts"] is True
    assert small["lines"] == {"ttir": 210, "ttgir": 336, "llir": 4133, "ptx": 4539}
    assert sum(row["committed_texts"] for row in rows.values()) == 1
    ptx = {key: row["lines"]["ptx"] for key, row in rows.items()}
    assert max(ptx, key=ptx.get) == ("sm_90", "m128_n128_k32_g8_w4_s4")
    assert ptx[("sm_90", "m128_n128_k32_g8_w4_s4")] == 9692


def test_what_changes_from_sm_80_to_sm_90(inputs):
    rows = target_rows(inputs[0], inputs[1])
    assert len(rows) == 6
    for row in rows:
        assert row["identical"] == {
            "ttir": True,
            "ttgir": False,
            "llir": False,
            "ptx": False,
            "cubin": False,
        }
        assert row["shared_bytes_equal"] is True
        assert row["line_change"]["ttir"] == row["line_change"]["ttgir"] == 0
        assert row["line_change"]["llir"] < 0 < row["line_change"]["ptx"]
    changes = {row["config"]: row["line_change"]["ptx"] for row in rows}
    assert changes == {
        "m128_n128_k32_g8_w4_s4": 389,
        "m128_n64_k32_g8_w4_s4": 197,
        "m64_n128_k32_g8_w4_s4": 197,
        "m128_n32_k32_g8_w4_s4": 101,
        "m64_n32_k32_g8_w2_s5": 101,
        "m32_n64_k32_g8_w2_s5": 101,
    }
    checked = {row["config"]: row["ttgir_is_retargeted_sm_80"] for row in rows}
    assert checked.pop("m32_n64_k32_g8_w2_s5") is True
    assert set(checked.values()) == {None}


def test_a_ttgir_that_does_not_name_its_target_once_is_refused(inputs):
    with pytest.raises(TritonIrError, match="target once"):
        target_rows(inputs[0], inputs[1].replace('"cuda:80"', '"cuda:86"'))


def test_a_changed_sm_90_digest_breaks_the_retarget_check(inputs):
    value = json.loads((FIXTURES / "index.json").read_bytes())
    value["targets"]["sm_90"][5]["stages"]["ttgir"]["sha256"] = "0" * 64
    rows = target_rows(parse_manifest(value), inputs[1])
    assert rows[5]["ttgir_is_retargeted_sm_80"] is False


def test_the_committed_counts_confirm_the_derived_fma(inputs):
    result = compare(*inputs)
    assert result["committed"]["derived_fma_confirmed"] is True
    assert result["committed"]["ptx_families"]["mma"] == 0
    assert result["committed"]["ptx_k_loop_families"]["fma.rn.f32"] == 1024
    assert result["needs_full_dumps"] == list(NEEDS_FULL_DUMPS)
    assert len(NEEDS_FULL_DUMPS) == 4
