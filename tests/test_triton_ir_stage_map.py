"""Tests of the map from kernel lines to TTIR, TTGIR and PTX, and of the pipeline facts.

The map comes from the locations in the committed texts. The pinned counts
show which stage carries the loads, the masks, ``tl.dot`` and the store. No
test imports triton.
"""

import dataclasses
from pathlib import Path

import pytest

from ai_journey.triton_ir_manifest import (
    COMMITTED_CONFIG,
    COMMITTED_TARGET,
    TritonIrError,
    check_fixtures,
    load_manifest,
)
from ai_journey.triton_ir_mlir import read_mlir
from ai_journey.triton_ir_ptx import read_ptx
from ai_journey.triton_ir_stage_map import (
    STATEMENTS,
    pipeline_facts,
    stage_map,
    statement_line,
)
from ai_journey.triton_matmul_kernel import TRITON_SOURCE

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "day80_triton_ir"


@pytest.fixture(scope="module")
def stages():
    manifest = load_manifest(FIXTURES)
    texts = check_fixtures(manifest, FIXTURES)
    return (
        manifest.record(COMMITTED_TARGET, COMMITTED_CONFIG),
        read_mlir(texts["ttir"].text),
        read_mlir(texts["ttgir"].text),
        read_ptx(texts["ptx"].text),
    )


def rows(stages):
    _, ttir, ttgir, ptx = stages
    return {row["key"]: row for row in stage_map(ttir, ttgir, ptx)["statements"]}


def test_each_statement_is_one_line_of_the_kernel_text():
    lines = {key: statement_line(statement) for key, statement in STATEMENTS}
    assert lines == {
        "k_loop": 36,
        "a_mask": 38,
        "b_mask": 39,
        "a_load": 40,
        "b_load": 41,
        "dot": 42,
        "a_advance": 43,
        "b_advance": 44,
        "c_mask": 46,
        "store": 47,
    }
    for key, statement in STATEMENTS:
        assert TRITON_SOURCE.split("\n")[lines[key] - 1].strip() == statement
    with pytest.raises(TritonIrError, match="once"):
        statement_line("acc = tl.zeros", TRITON_SOURCE)


def test_the_dot_is_one_tt_dot_and_1024_fma_in_one_k_iteration(stages):
    dot = rows(stages)["dot"]
    assert dot["ttir"] == {"tt.dot": 1}
    assert dot["ttgir"] == {"tt.dot": 1}
    assert dot["ptx"] == {"fma.rn.f32": 1024}
    assert dot["ptx_k_loop"] == {"fma.rn.f32": 1024}


def test_each_load_becomes_async_copies_and_shared_loads(stages):
    table = rows(stages)
    assert table["a_load"]["ttir"] == table["b_load"]["ttir"] == {"tt.load": 1}
    assert table["a_load"]["ttgir"]["ttg.async_copy_global_to_local"] == 5
    assert table["b_load"]["ttgir"]["ttg.async_copy_global_to_local"] == 5
    assert table["a_load"]["ttgir"]["ttg.local_load"] == 1
    # A: 32 x 32 floats over 64 threads, B: 32 x 64 floats over 64 threads.
    assert table["a_load"]["ptx_k_loop"]["cp.async"] == 16
    assert table["b_load"]["ptx_k_loop"]["cp.async"] == 32
    assert table["a_load"]["ptx"]["cp.async"] == 5 * 16
    assert table["b_load"]["ptx"]["cp.async"] == 5 * 32
    assert table["a_load"]["ptx_k_loop"]["ld.shared"] == 64
    assert table["b_load"]["ptx_k_loop"]["ld.shared"] == 32
    assert "ld.global" not in table["a_load"]["ptx"]
    assert table["a_load"]["ptx_k_loop"]["cp.async.wait_group"] == 1


def test_the_masks_are_compares_and_selects(stages):
    table = rows(stages)
    assert table["a_mask"]["ttir"] == {
        "arith.andi": 1,
        "arith.cmpi": 2,
        "arith.subi": 1,
        "tt.broadcast": 2,
        "tt.splat": 2,
    }
    assert table["a_mask"]["ptx"] == {"other": 3, "setp": 21}
    assert table["b_mask"]["ptx"] == {"setp": 64}
    assert table["c_mask"]["ptx"] == {"other": 32, "setp": 32}
    # The selects that zero the copy size of a masked lane name the load line.
    assert table["a_load"]["ptx"]["selp"] == 150
    assert table["b_load"]["ptx"]["selp"] == 225


def test_the_store_converts_its_layout_through_shared_memory(stages):
    store = rows(stages)["store"]
    assert store["ttir"] == {"tt.store": 1}
    assert store["ttgir"] == {"tt.store": 1, "ttg.convert_layout": 1}
    assert store["ptx"] == {
        "bar.sync": 15,
        "ldmatrix": 8,
        "other": 43,
        "st.global": 32,
        "st.shared": 8,
    }
    assert store["ptx_k_loop"] == {}


def test_the_loop_line_holds_the_scf_for_and_the_pipeline_bookkeeping(stages):
    loop = rows(stages)["k_loop"]
    assert loop["ttir"] == {"arith.constant": 1, "scf.for": 1, "scf.yield": 1}
    assert loop["ttgir"]["scf.for"] == 1
    assert loop["ttgir"]["ttg.local_dealloc"] == 2
    assert loop["ptx_k_loop"] == {"branch": 1, "other": 7, "selp": 2, "setp": 4}


def test_every_line_with_code_and_the_ops_without_a_line(stages):
    _, ttir, ttgir, ptx = stages
    mapped = stage_map(ttir, ttgir, ptx)
    assert [row["line"] for row in mapped["lines"]] == [
        14,
        *range(21, 32),
        33,
        34,
        36,
        *range(38, 48),
    ]
    assert mapped["without_source_line"] == {
        "ttir_ops": 5,
        "ttgir_ops": 16,
        "ptx_instructions": 38,
    }
    totals = {
        stage: sum(row[f"{stage}"] for row in mapped["lines"])
        + mapped["without_source_line"][stage]
        for stage in ("ttir_ops", "ttgir_ops", "ptx_instructions")
    }
    assert totals == {"ttir_ops": 97, "ttgir_ops": 220, "ptx_instructions": 3097}


def test_the_pipeline_facts_of_the_committed_config(stages):
    record, _, ttgir, ptx = stages
    facts = pipeline_facts(record, ttgir, ptx)
    assert facts == {
        "config": "m32_n64_k32_g8_w2_s5",
        "target": "sm_80",
        "num_stages": 5,
        "buffers_per_operand": 4,
        "buffers_are_stages_minus_one": True,
        "tile_bytes": 12288,
        "ring_bytes": 49152,
        "shared_bytes": 49152,
        "ring_fills_shared": True,
        "commit_groups": 10,
        "commit_groups_in_k_loop": 2,
        "prologue_commit_groups": 8,
        "wait_group_in_k_loop": [6],
        "wait_group_after_k_loop": [0],
        "cp_async_per_thread_per_iteration": 48,
        "tile_elements_per_thread": 48,
        "threads": 64,
        "reqntid": "64",
        "input_precision": "ieee",
        "mma_encoding": False,
        "dot_parent_kinds": ["ttg.blocked"],
        "mma_instructions": 0,
        "fma_in_k_loop": 1024,
        "fma_per_thread_per_iteration": 1024,
        "fma_matches_tile": True,
        "ld_global": 0,
    }


def test_the_pipeline_facts_show_a_record_that_does_not_fit(stages):
    record, _, ttgir, ptx = stages
    other = dataclasses.replace(record, shared_bytes=61440)
    facts = pipeline_facts(other, ttgir, ptx)
    assert facts["ring_fills_shared"] is False


def test_a_ptx_text_without_one_loop_is_refused(stages):
    record, ttir, ttgir, ptx = stages
    flat = dataclasses.replace(ptx, loops=())
    with pytest.raises(TritonIrError, match="exactly one loop"):
        stage_map(ttir, ttgir, flat)
    with pytest.raises(TritonIrError, match="exactly one loop"):
        pipeline_facts(record, ttgir, flat)
    first = next(op for op in ttgir.ops if op.name == "ttg.local_alloc")
    single = dataclasses.replace(
        ttgir, ops=tuple(op for op in ttgir.ops if op != first)
    )
    with pytest.raises(TritonIrError, match="one shared ring"):
        pipeline_facts(record, single, ptx)
