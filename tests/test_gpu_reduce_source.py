import re

import pytest

from ai_journey.gpu_reduce_source import (
    BARRIER,
    COMBINE,
    GRID_STRIDE,
    HEADER,
    KERNELS,
    R0,
    R3,
    R4,
    R5,
    SIMD_WIDTH,
    STEP_KERNELS,
    GpuReduceError,
    ReduceKernel,
    library_sha256,
    library_source,
    require_count,
    step_kernel,
    text_sha256,
)

# The digests of the kernels whose hypotheses the plan declares; a change to
# any text moves its digest and the plan digest that the config pins.
DIGESTS = {
    "reduce_r0_interleaved": (
        "40303012a3f4b3bcdd1909023d2ee8af860d79dbbde9cbf0714af7ece8f8f6c2"
    ),
    "reduce_r1_sequential": (
        "b230db3f18d0db381928728e77c376adc34532c49d421d4cd36c4adc6e244173"
    ),
    "reduce_r2_first_add": (
        "d52e657352c87e5cd8ae8baf6ce45cc92c08cb5ed2f887d7f7f58f1c48f414a1"
    ),
    "reduce_r3_simd_shuffle": (
        "9ddceaf22450689e0761dbfc5a81ffca455866fc2772e14eff229bebb875f31a"
    ),
    "reduce_r4_simd_sum": (
        "8ee154263d19714e61dd6849d78ff74afdbb5e702194248f5c7c65b69dd53db0"
    ),
    "reduce_r5_grid_stride": (
        "fb97a05461e22dafc3a0c345051a73bcba9088d9e9e14dcefb29114917713460"
    ),
    "reduce_combine": (
        "551bea995a2f88e2d91426e40862f8bf65e58c2de9e76a1a7d537b006233d770"
    ),
}
LIBRARY_SHA256 = "753b102d673402b6ce260b644639d1f09a8c81f16647e831b2e35c8ac54f7edc"
# A loop header whose bound depends on the group size or a constant only.
UNIFORM_LOOP = re.compile(r"for \(uint s = (1u|size / 2u); s [<>=]+ \w+; s [^)]+\) \{")
SIMD_CALL = re.compile(r"\bsimd_(?:sum|shuffle_down)\(")


def blocks(source):
    """Yield each statement line with the headers of the blocks around it.

    The kernel signature spans several lines, so they are joined first.
    """

    stack, pending = [], ""
    for raw in source.splitlines():
        line = raw.strip()
        if line.startswith("//") or not line:
            continue
        if pending or line.startswith("kernel void"):
            pending += line
            if not line.endswith("{"):
                continue
            line, pending = pending, ""
        if line == "}":
            stack.pop()
            continue
        yield line, tuple(stack)
        if line.endswith("{"):
            stack.append(line)
    assert not stack and not pending


def test_every_kernel_is_registered_with_its_pinned_digest():
    assert list(KERNELS) == [*STEP_KERNELS, COMBINE]
    assert {name: kernel.source_sha256 for name, kernel in KERNELS.items()} == DIGESTS
    assert library_sha256() == LIBRARY_SHA256 == text_sha256(library_source())
    library = library_source()
    assert library.startswith(HEADER)
    for name, kernel in KERNELS.items():
        assert library.count(f"kernel void {name}(") == 1
        assert kernel.source in library
        assert kernel.arg_casts == {2: "int32"}
    positions = [library.index(f"kernel void {name}(") for name in KERNELS]
    assert positions == sorted(positions)
    assert [step_kernel(f"R{index}").name for index in range(6)] == list(STEP_KERNELS)
    with pytest.raises(GpuReduceError, match="not a reduction step"):
        step_kernel("R6")


def test_every_barrier_is_reached_by_every_thread_of_the_group():
    for name, kernel in KERNELS.items():
        barriers = 0
        for line, around in blocks(kernel.source):
            if line != BARRIER:
                continue
            barriers += 1
            assert around[0].startswith(f"kernel void {name}("), name
            for header in around[1:]:
                assert UNIFORM_LOOP.fullmatch(header), (name, header)
        assert barriers >= 1, name
    assert "if (" not in "".join(
        header
        for kernel in KERNELS.values()
        for line, around in blocks(kernel.source)
        if line == BARRIER
        for header in around
    )


def test_simd_functions_run_only_with_whole_simd_groups_where_declared():
    for name, kernel in KERNELS.items():
        calls = [
            (line, around)
            for line, around in blocks(kernel.source)
            if SIMD_CALL.search(line)
        ]
        if kernel.simd_lanes is None:
            assert not calls and "simdgroup" not in kernel.source, name
            continue
        assert kernel.simd_lanes == SIMD_WIDTH
        assert calls, name
        for line, around in calls:
            # Either every lane of the group runs it, or SIMD group 0 does.
            assert set(around[1:]) <= {"if (simd_group == 0u) {"}, (name, line)
    shuffles = [
        line for line, _ in blocks(KERNELS[R3].source) if SIMD_CALL.search(line)
    ]
    assert shuffles == [
        f"w += simd_shuffle_down(w, {delta}u);" for delta in (16, 8, 4, 2, 1)
    ]
    for name in (R4, R5):
        assert len(re.findall(r"\bsimd_sum\(", KERNELS[name].source)) == 2


def test_every_load_is_guarded_and_no_kernel_uses_an_atomic():
    guards = {
        R0: ["i < n ? x[i] : 0.0f"],
        "reduce_r1_sequential": ["i < n ? x[i] : 0.0f"],
        R5: ["for (uint e = gid; e < n; e += grid) {", "v += x[e];"],
        COMBINE: ["e + 3072u < count", "for (; e < count; e += 1024u) {"],
    }
    two = "(i < n ? x[i] : 0.0f) + (i + 256u < n ? x[i + 256u] : 0.0f)"
    for name in ("reduce_r2_first_add", R3, R4):
        guards[name] = [two]
    for name, kernel in KERNELS.items():
        assert "atomic" not in kernel.source
        for text in guards[name]:
            assert text in kernel.source, (name, text)
        reads = re.findall(r"\b(x|partials)\[[^\]]+\]", kernel.source)
        assert reads, name
        declared = re.findall(r"threadgroup float shared\[(\d+)\];", kernel.source)
        assert [int(size) for size in declared] == [kernel.threadgroup_floats], name
    assert KERNELS[R0].source.count("t % (2u * s) == 0u") == 1
    assert "size" in KERNELS[R0].source.split("for (", 1)[1].split(")", 1)[0]


def test_the_registry_records_each_launch_shape_and_memory():
    mappings = {name: kernel.to_mapping() for name, kernel in KERNELS.items()}

    assert mappings[R0] == {
        "name": R0,
        "step": "R0",
        "mapping": "one_element_per_thread",
        "tree": "interleaved_tree",
        "group_size": 256,
        "threadgroup_bytes": 1024,
        "simd_lanes": None,
        "fixed_groups": None,
        "source_sha256": DIGESTS[R0],
    }
    assert mappings[R5]["fixed_groups"] == 1024
    assert mappings[R5]["mapping"] == GRID_STRIDE
    assert mappings[R4]["threadgroup_bytes"] == 32
    assert mappings[COMBINE]["step"] is None
    assert mappings[COMBINE]["group_size"] == 1024
    assert mappings[COMBINE]["threadgroup_bytes"] == 4096


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"name": "sum_r0"}, "reduce_ and a short slug"),
        ({"step": "R9"}, "R0 to R5"),
        ({"tree": "spiral"}, "unknown mapping or tree"),
        ({"simd_lanes": 64}, "simd_lanes must be None or 32"),
        ({"fixed_groups": 8}, "exactly a grid-stride kernel"),
        ({"source": "kernel void reduce_other() {}"}, "ASCII source that defines"),
        (
            {"source": KERNELS[R0].source + "\n// atomic_fetch_add"},
            "with no atomic",
        ),
    ],
)
def test_a_kernel_with_a_wrong_declaration_is_refused(change, message):
    fields = {**vars(KERNELS[R0]), **change}

    with pytest.raises(GpuReduceError, match=message):
        ReduceKernel(**fields)


def test_counts_are_bounded_as_this_labs_error():
    assert require_count(5, "n", 1, 9) == 5
    with pytest.raises(GpuReduceError, match=r"n must be in \[1, 9\]"):
        require_count(10, "n", 1, 9)
    with pytest.raises(TypeError, match="n must be an integer"):
        require_count(True, "n", 1, 9)
