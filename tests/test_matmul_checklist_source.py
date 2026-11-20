import re
from dataclasses import replace

import pytest

from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_checklist_source import (
    INTERIOR_TEST,
    LOOP_STATEMENT,
    MAX_THREADGROUP_BYTES,
    STORE_GUARD,
    MatmulChecklistError,
    block_kernel_source,
)
from ai_journey.matmul_checklist_variants import DESIGNS
from ai_journey.matmul_tiled_source import BARRIER, HEADER

# The only conditions that may enclose a barrier: both are the same for every
# thread of a group, so every thread reaches the barrier or none does.
UNIFORM_CONDITIONS = {"interior", "start < k"}
_IF = re.compile(r"^\s*(?:\} else )?if \((.*)\) \{$")


def enclosing_conditions(source):
    """Yield each line with the conditions of the ``if`` blocks around it."""

    stack = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("}"):
            stack.pop()
        match = _IF.match(line)
        if match:
            stack.append(match.group(1))
        elif stripped.endswith("{"):
            stack.append(None)
        yield stripped, [condition for condition in stack if condition is not None]


def phase_steps(design, k):
    """Walk the loop bounds of the kernel in Python: the k index of each fma."""

    steps, start = [], 0
    interior = design.interior_path
    while interior and start + design.block_k <= k:
        steps += range(start, start + design.block_k)
        start += design.block_k
    while start < k:
        steps += range(start, start + min(design.block_k, k - start))
        start += design.block_k
    return steps


def staged_slots(design):
    """Return the slot of every staging chunk for every thread of a group."""

    source = block_kernel_source(design)
    slots = []
    pattern = re.compile(r"if \(index < (\d+)u\) \{|uint e = index \+ (\d+)u;")
    limit = None
    for match in pattern.finditer(source):
        if match.group(1) is not None:
            limit = int(match.group(1))
            continue
        offset = int(match.group(2))
        threads = design.threads_per_group if limit is None else limit
        slots.append([offset + index for index in range(threads)])
        limit = None
    return slots


@pytest.mark.parametrize("design", DESIGNS, ids=lambda design: design.name)
def test_each_kernel_declares_float4_tiles_of_its_declared_bytes(design):
    source = block_kernel_source(design)
    a_vectors = design.block_m * design.a_stride // 4
    b_vectors = design.block_k * design.b_stride // 4

    assert f"kernel void {design.name}(" in source
    assert f"    threadgroup float4 a_store[{a_vectors}];\n" in source
    assert f"    threadgroup float4 b_store[{b_vectors}];\n" in source
    assert source.count("threadgroup float4 ") == 2
    assert design.threadgroup_bytes == 16 * (a_vectors + b_vectors)
    assert "[[thread_index_in_threadgroup]]" in source
    assert source.isascii() and not source.startswith(HEADER)
    assert "return" not in source and "simdgroup" not in source


@pytest.mark.parametrize("design", DESIGNS, ids=lambda design: design.name)
def test_every_barrier_lies_outside_every_thread_dependent_condition(design):
    source = block_kernel_source(design)
    barriers = [
        conditions
        for line, conditions in enclosing_conditions(source)
        if line == BARRIER
    ]

    assert len(barriers) == (6 if design.interior_path else 2)
    for conditions in barriers:
        assert set(conditions) <= UNIFORM_CONDITIONS, conditions


@pytest.mark.parametrize("design", DESIGNS, ids=lambda design: design.name)
def test_only_the_stores_are_guarded_and_each_output_is_stored_once(design):
    source = block_kernel_source(design)
    outputs = design.thread_m * design.thread_n
    lines = list(enclosing_conditions(source))
    stores = [(line, conditions) for line, conditions in lines if line[:2] == "c["]

    guarded = [
        (line, conditions)
        for line, conditions in stores
        if conditions and conditions[-1].startswith("row_")
    ]
    assert len(guarded) == outputs
    for line, conditions in guarded:
        row, column = re.fullmatch(
            r"c\[row_(\d) \* n \+ column_(\d)\] = sum_\1_\2;", line
        ).groups()
        assert f"if ({conditions[-1]}) {{" == STORE_GUARD.format(i=row, j=column)
    unguarded = [store for store in stores if store not in guarded]
    assert len(unguarded) == (outputs if design.interior_path else 0)
    assert all(conditions == ["interior"] for _, conditions in unguarded)
    assert source.count("c[") == len(stores)


@pytest.mark.parametrize("design", DESIGNS, ids=lambda design: design.name)
def test_each_output_takes_one_fma_per_k_in_increasing_order(design):
    source = block_kernel_source(design)
    loops = re.findall(r"for \(uint p = 0; p < (\w+); \+\+p\) \{", source)
    expected = ["steps"] * (2 if design.interior_path else 1)
    if design.interior_path:
        expected.insert(0, f"{design.block_k}u")

    assert loops == expected
    for i in range(design.thread_m):
        for j in range(design.thread_n):
            statement = LOOP_STATEMENT.format(i=i, j=j)
            assert source.count(statement) == len(loops)
    assert source.count("fma(") == len(loops) * design.thread_m * design.thread_n
    assert source.count(f"uint steps = min({design.block_k}u, k - start);") == (
        len(loops) - design.interior_path
    )
    for k in (*range(1, 3 * design.block_k + 2), 1000, 4096):
        assert phase_steps(design, k) == list(range(k)), k


@pytest.mark.parametrize("design", DESIGNS, ids=lambda design: design.name)
def test_the_threads_of_a_group_cover_its_tile_once(design):
    x, y = design.group_size
    outputs = [
        (row + y * i, column + x * j)
        for row in range(y)
        for column in range(x)
        for i in range(design.thread_m)
        for j in range(design.thread_n)
    ]

    assert sorted(outputs) == [
        (row, column)
        for row in range(design.block_m)
        for column in range(design.block_n)
    ]


@pytest.mark.parametrize("design", DESIGNS, ids=lambda design: design.name)
def test_staging_writes_every_slot_of_both_tiles_once_per_path(design):
    a_scalars = design.block_m * design.block_k
    b_scalars = design.block_k * design.block_n
    slots = staged_slots(design)
    groups = []
    expected = [a_scalars, b_scalars]
    if design.vector_loads:
        vectors = [a_scalars // 4, a_scalars, b_scalars // 4, b_scalars]
        expected = vectors * (2 if design.interior_path else 1)
        if design.interior_path:
            expected = [a_scalars // 4, b_scalars // 4, *vectors, *vectors]
    for count in expected:
        group = []
        while len(group) < count:
            group += slots.pop(0)
        groups.append(sorted(group))

    assert slots == []
    assert groups == [list(range(count)) for count in expected]


def test_slots_past_the_matrices_are_zero_filled_and_the_interior_reads_bare():
    guarded = block_kernel_source(DESIGNS[3])
    interior = block_kernel_source(DESIGNS[-1])

    assert "a_row < m && a_column < k ? a[a_row * k + a_column] : 0.0f;" in guarded
    assert "b_row < k && b_column < n ? b[b_row * n + b_column] : 0.0f;" in guarded
    assert guarded.count(": float4(0.0f);") == 2
    assert "    bool vector_a = (k & 3u) == 0u;\n" in guarded
    assert "    bool vector_b = (n & 3u) == 0u;\n" in guarded
    assert "    " + INTERIOR_TEST.format(bm=64, bn=64) + "\n" in interior
    bare = "= *(device const float4*)(a + a_row * k + a_column);"
    assert guarded.count(bare) == 0 and interior.count(bare) == 1
    assert interior.count("for (; start + 16u <= k; start += 16u) {") == 1


def test_a_design_is_refused_before_any_text_exists():
    base = DESIGNS[3]

    for changes, message in (
        ({"name": "tiled_64"}, "matmul_ and a short slug"),
        ({"block_m": 62}, "whole thread blocks"),
        ({"block_m": 128, "block_n": 128, "thread_m": 2}, "exceeds 1024"),
        ({"pad": 1}, "whole float4 values"),
        ({"vector_loads": False}, "interior path needs vector loads"),
        ({"block_k": 64, "block_m": 128, "block_n": 128}, "threadgroup memory"),
    ):
        if "vector_loads" in changes:
            changes = {**changes, "interior_path": True}
        with pytest.raises(MatmulChecklistError, match=message):
            replace(base, **changes)
    with pytest.raises(KernelMetalError, match="block_k must be in"):
        replace(base, block_k=0)
    with pytest.raises(TypeError, match="vector_loads must be a bool"):
        replace(base, vector_loads=1)
    with pytest.raises(TypeError, match="design must be a BlockDesign"):
        block_kernel_source(base.to_mapping())
    largest = max(design.threadgroup_bytes for design in DESIGNS)
    assert largest <= MAX_THREADGROUP_BYTES == 32768
