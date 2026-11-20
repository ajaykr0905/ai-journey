"""CPU tests of the compile-free reduction checker. No test compiles or runs CUDA."""

import json
import re
from itertools import permutations
from pathlib import Path

import numpy
import pytest

from ai_journey.gpu_checklist_check import parse_body
from ai_journey.gpu_reduce_cuda import (
    COMBINE_LAUNCH,
    PINNED_KERNELS,
    R1_SHARED,
    SHUFFLE_OFFSETS,
    WARP_SHUFFLE,
    GpuReduceCudaError,
    cuda_source,
)
from ai_journey.gpu_reduce_cuda_check import (
    BENCHMARK_SIZES,
    CHECK_SIZES,
    COMBINE_CHECK_NAMES,
    CORRECTNESS_SIZES,
    GridStrideLoops,
    barriers_reached,
    check_names,
    check_reduce_source,
    combine_coverage,
    grid_stride_loops,
    require_reduce_source,
    selected_threads,
    shuffle_coverage,
    tree_trip,
)
from ai_journey.kernel_launch_config import Dim3, LaunchConfig

ROOT = Path(__file__).resolve().parents[1]
R1_TEXT = cuda_source(R1_SHARED)
SHUFFLE_TEXT = cuda_source(WARP_SHUFFLE)
PINNED_LOOPS = GridStrideLoops(
    guard=3072,
    step=4096,
    offsets=((0, 0), (1, 1024), (2, 2048), (3, 3072)),
    tail_step=1024,
    tail_offsets=((0, 0),),
)
# The tree of R1 pass 1 and the start of its store: once in the R1 text.
R1_TREE = (
    "    for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1u) {\n"
    "        if (t < s) {\n"
    "            shared[t] = __fadd_rn(shared[t], shared[t + s]);\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (t == 0u) {\n"
    "        partials"
)


def in_tree(old, new):
    return R1_TREE, R1_TREE.replace(old, new, 1)


def check(text, kernel=R1_SHARED, n=1000003):
    return check_reduce_source(text, kernel=kernel, n=n)


def mutate(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def test_the_sizes_are_the_day_78_correctness_and_timing_sizes():
    config = json.loads((ROOT / "config/day-78-gpu-reduce.json").read_bytes())
    sizes = [n for _, n in config["correctness"]["cases"]]

    assert list(CORRECTNESS_SIZES) == list(dict.fromkeys(sizes))
    assert list(BENCHMARK_SIZES) == config["benchmark"]["sizes"]
    assert CHECK_SIZES == (*CORRECTNESS_SIZES, *BENCHMARK_SIZES)
    assert len(set(CHECK_SIZES)) == 14


@pytest.mark.parametrize("kernel", PINNED_KERNELS, ids=lambda item: item.name)
def test_every_pinned_text_passes_every_check_for_every_size(kernel):
    text = cuda_source(kernel)

    for n in (*CHECK_SIZES, 255, 256, 512, 513, 786432, 786433, 1 << 26):
        result = check(text, kernel, n)

        assert result.passed, (n, result.failures)
        assert tuple(name for name, _ in result.checks) == check_names(kernel)
        assert result.launch == kernel.launch(n)
        assert result.combine_launch == COMBINE_LAUNCH
        assert result.shared_memory_bytes == kernel.shared_memory_bytes == 1024
        assert result.barriers_per_block == kernel.barriers_per_block
    assert require_reduce_source(text, kernel=kernel, n=33).passed


def test_the_names_and_the_mapping_follow_the_kernel():
    mapping = check(SHUFFLE_TEXT, WARP_SHUFFLE, 2000003).to_mapping()

    assert check_names(R1_SHARED) == (
        "cdiv_definition",
        "r1_shared_memory",
        "r1_index",
        "r1_loads",
        "r1_tree",
        "r1_barriers",
        "r1_partial_store",
        *COMBINE_CHECK_NAMES,
        "order",
        "launch",
    )
    assert check_names(WARP_SHUFFLE)[6:8] == (
        "shuffle_warp_stage",
        "shuffle_partial_store",
    )
    assert len(check_names(R1_SHARED)) == 13 and len(check_names(WARP_SHUFFLE)) == 14
    assert mapping == {
        "kernel": "reduce_warp_shuffle",
        "n": 2000003,
        "checks": dict.fromkeys(check_names(WARP_SHUFFLE), True),
        "passed": True,
        "launch": {"grid": [3907, 1, 1], "block": [256, 1, 1]},
        "combine_launch": {"grid": [1, 1, 1], "block": [1024, 1, 1]},
        "shared_memory_bytes": 1024,
        "barriers_per_block": 4,
    }
    # The text of one kernel is not the text of the other kernel.
    assert check(R1_TEXT, WARP_SHUFFLE).failures == check_names(WARP_SHUFFLE)
    assert check(SHUFFLE_TEXT, R1_SHARED).failures == check_names(R1_SHARED)


@pytest.mark.parametrize(
    ("kernel", "text"), [(R1_SHARED, R1_TEXT), (WARP_SHUFFLE, SHUFFLE_TEXT)]
)
def test_formatting_and_comments_do_not_change_the_structure(kernel, text):
    # Spaces go around whole tokens: "> >=" is another text, below.
    spaced = re.sub(r"(<<<|>>>|>>=|\+=|>=|==|[(),;<>+*%?:\[\]{}=])", r" \1 ", text)
    commented = text.replace("{\n", "{ /* block */\n").replace(";\n", "; // end\n")

    assert check(spaced, kernel).passed
    assert check(commented, kernel).passed


@pytest.mark.parametrize(
    ("old", "new"),
    [
        in_tree("s >>= 1u", "s > >= 1u"),
        ("e += 4096u", "e + = 4096u"),
        ("e += 4096u", "e +/* split */= 4096u"),
        ("<<<grid", "< <<grid"),
    ],
)
def test_a_space_inside_an_operator_fails_every_check(old, new):
    # A compiler reads "e + = 4096u" as an error and "> >=" as two operators.
    result = check(mutate(R1_TEXT, old, new))

    assert result.failures == check_names(R1_SHARED)
    assert result.launch is None and result.barriers_per_block is None


@pytest.mark.parametrize(
    "extra",
    [
        "#define __fadd_rn(a, b) ((b) + (a))",
        "#define __syncthreads()",
        "#define __shfl_down_sync(m, v, d) (v)",
        "__device__ float __fadd_rn(float a, float b) { return a; }",
    ],
)
@pytest.mark.parametrize(
    ("kernel", "text"), [(R1_SHARED, R1_TEXT), (WARP_SHUFFLE, SHUFFLE_TEXT)]
)
def test_code_outside_the_four_functions_fails_every_check(kernel, text, extra):
    # A macro or an overload changes what the pinned statements mean.
    marker = f"\n__global__ void {kernel.kernel_name}("
    result = check(mutate(text, marker, f"\n{extra}\n{marker}"), kernel)

    assert result.failures == check_names(kernel)


@pytest.mark.parametrize(
    ("name", "old", "new", "failures"),
    [
        (
            "the barrier after the load is gone",
            "0.0f;\n    __syncthreads();\n",
            "0.0f;\n",
            ("r1_barriers",),
        ),
        (
            "a barrier inside the guard of a level",
            *in_tree(
                "        }\n        __syncthreads();\n",
                "            __syncthreads();\n        }\n",
            ),
            ("r1_barriers",),
        ),
        (
            "the tree stops at 2 values",
            *in_tree("s > 0u", "s > 1u"),
            ("r1_tree", "r1_barriers"),
        ),
        (
            "a stride that falls by 4",
            *in_tree("s >>= 1u", "s >>= 2u"),
            ("r1_tree", "r1_barriers"),
        ),
        (
            "an unguarded load",
            "shared[t] = i < n ? x[i] : 0.0f;",
            "shared[t] = x[i];",
            ("r1_loads",),
        ),
        (
            "a guard that reads x[n]",
            "shared[t] = i < n ? x[i] : 0.0f;",
            "shared[t] = i <= n ? x[i] : 0.0f;",
            ("r1_loads",),
        ),
        (
            "blocks 128 elements apart",
            "blockIdx.x * 256u + t;",
            "blockIdx.x * 128u + t;",
            ("r1_loads",),
        ),
        (
            "the partial of thread 1",
            "    if (t == 0u) {\n        partials",
            "    if (t == 1u) {\n        partials",
            ("r1_partial_store",),
        ),
        (
            "a rounding toward zero",
            *in_tree("__fadd_rn", "__fadd_rz"),
            ("r1_tree",),
        ),
        (
            "half the shared memory",
            "__shared__ float shared[256];",
            "__shared__ float shared[128];",
            ("r1_shared_memory",),
        ),
        (
            "an early exit before the barriers",
            "    unsigned int i = blockIdx.x * 256u + t;\n",
            "    unsigned int i = blockIdx.x * 256u + t;\n"
            "    if (i >= n + 256u) {\n        return;\n    }\n",
            ("r1_index", "r1_barriers", "order"),
        ),
        (
            "a grid of 512 elements per block",
            "dim3 grid(cdiv(n, 256));",
            "dim3 grid(cdiv(n, 512));",
            ("launch",),
        ),
        (
            "a cdiv that rounds down",
            "return (a + b - 1) / b;",
            "return a / b;",
            ("cdiv_definition",),
        ),
    ],
)
def test_each_change_to_the_r1_text_fails_the_check_of_its_part(
    name, old, new, failures
):
    assert check(mutate(R1_TEXT, old, new)).failures == failures, name


@pytest.mark.parametrize(
    ("name", "old", "new", "failures"),
    [
        (
            "a mask of 16 lanes",
            "__shfl_down_sync(0xffffffffu, w, 16u)",
            "__shfl_down_sync(0x0000ffffu, w, 16u)",
            ("shuffle_warp_stage",),
        ),
        (
            "offset 8 before offset 16",
            "w, 16u));\n        w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, 8u));",
            "w, 8u));\n        w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, 16u));",
            ("shuffle_warp_stage",),
        ),
        (
            "no shuffle at offset 1",
            "        w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, 1u));\n",
            "",
            ("shuffle_warp_stage",),
        ),
        (
            "shuffles in 16 threads only",
            "if (warp == 0u)",
            "if (t < 16u)",
            ("shuffle_warp_stage",),
        ),
        (
            "shuffles in warp 1",
            "if (warp == 0u)",
            "if (warp == 1u)",
            ("shuffle_warp_stage", "shuffle_partial_store"),
        ),
        (
            "a barrier in the last warp",
            "w, 1u));\n",
            "w, 1u));\n        __syncthreads();\n",
            ("shuffle_barriers", "shuffle_warp_stage"),
        ),
        (
            "a warp of 16 lanes",
            "unsigned int warp = t / 32u;",
            "unsigned int warp = t / 16u;",
            ("shuffle_index", "shuffle_warp_stage"),
        ),
        (
            "the tree stops at 16 values",
            "s >= 32u;",
            "s >= 16u;",
            ("shuffle_tree", "shuffle_barriers"),
        ),
        (
            "an unguarded second load",
            "i + 256u < n ? x[i + 256u] : 0.0f",
            "x[i + 256u]",
            ("shuffle_loads",),
        ),
        (
            "loads 128 apart",
            "i + 256u < n ? x[i + 256u] : 0.0f",
            "i + 128u < n ? x[i + 128u] : 0.0f",
            ("shuffle_loads",),
        ),
        (
            "a store before the shuffles",
            "        float w = shared[lane];\n",
            "        float w = shared[lane];\n"
            "        if (lane == 0u) {\n            partials[blockIdx.x] = w;\n"
            "        }\n",
            ("shuffle_partial_store", "order"),
        ),
        (
            "a grid of 256 elements per block",
            "dim3 grid(cdiv(n, 512));",
            "dim3 grid(cdiv(n, 256));",
            ("launch",),
        ),
        (
            "a grid-stride step of 2048",
            "e += 4096u",
            "e += 2048u",
            ("combine_grid_stride",),
        ),
        (
            "two reads of the same partial",
            "partials[e + 2048u]",
            "partials[e + 1024u]",
            ("combine_grid_stride",),
        ),
        (
            "a guard that lets e + 3072 pass count",
            "e + 3072u < count",
            "e + 2048u < count",
            ("combine_grid_stride",),
        ),
        (
            "the accumulators joined in another order",
            "__fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3))",
            "__fadd_rn(__fadd_rn(a0, a2), __fadd_rn(a1, a3))",
            ("combine_tree",),
        ),
        (
            "no barrier after the join",
            "__fadd_rn(a2, a3));\n    __syncthreads();\n",
            "__fadd_rn(a2, a3));\n",
            ("combine_barriers",),
        ),
        (
            "the sum of thread 1",
            "    if (t == 0u) {\n        out[0]",
            "    if (t == 1u) {\n        out[0]",
            ("combine_store",),
        ),
        (
            "a combine of two blocks",
            "<<<1, 1024>>>",
            "<<<2, 1024>>>",
            ("launch",),
        ),
        (
            "a combine count of n",
            "(out, partials, grid.x)",
            "(out, partials, n)",
            ("launch",),
        ),
    ],
)
def test_each_change_to_the_shuffle_text_fails_the_check_of_its_part(
    name, old, new, failures
):
    result = check(mutate(SHUFFLE_TEXT, old, new), WARP_SHUFFLE)

    assert result.failures == failures, name


def test_a_combine_without_its_tail_loop_skips_partials():
    tail = "    for (; e < count; e += 1024u) {\n        a0 = __fadd_rn(a0, partials[e]);\n"
    text = mutate(SHUFFLE_TEXT, tail + "    }\n", "")

    assert check(text, WARP_SHUFFLE).failures == ("combine_grid_stride", "order")
    # Below 3073 partials, only the tail loop reads them.
    assert check(text, WARP_SHUFFLE, 3).failures == ("combine_grid_stride", "order")


def test_the_tree_trip_follows_the_header_for_a_block():
    header = "for(unsigned int s=blockDim.x/2u;s{}u;s>>=1u)"

    assert tree_trip(header.format(">0"), 256) == (128, 64, 32, 16, 8, 4, 2, 1)
    assert tree_trip(header.format(">=32"), 256) == (128, 64, 32)
    assert tree_trip(header.format(">0"), 1024) == (
        512,
        256,
        128,
        64,
        32,
        16,
        8,
        4,
        2,
        1,
    )
    # An unsigned stride stays at or above 0, so that loop never ends.
    assert tree_trip(header.format(">=0"), 256) is None
    assert tree_trip("for(unsigned int s=128u;s>0u;s>>=1u)", 256) is None


def test_guards_select_the_threads_of_a_block():
    assert selected_threads(("if(t==0u)",), 256) == (0,)
    assert selected_threads(("if(warp==0u)", "if(lane==0u)"), 256) == (0,)
    assert selected_threads(("if(warp==0u)",), 256) == tuple(range(32))
    assert selected_threads(("if(warp==1u)", "if(lane==0u)"), 256) == (32,)
    assert selected_threads(("if(t<16u)",), 256) == tuple(range(16))
    assert selected_threads(("if(t<s)",), 256) is None


def test_the_offsets_give_lane_0_each_value_once():
    assert shuffle_coverage(SHUFFLE_OFFSETS) == (1,) * 32
    # Every order of the offsets covers the warp; the order sets the bits.
    for order in permutations(SHUFFLE_OFFSETS):
        assert shuffle_coverage(order) == (1,) * 32
    assert shuffle_coverage((16, 8, 4, 2)).count(0) == 16
    assert max(shuffle_coverage((16, 8, 4, 2, 1, 1))) > 1
    # Lane 0 + 32 is outside the warp, so lane 0 adds its own value again.
    assert shuffle_coverage((32,))[0] == 2


def test_the_grid_stride_loops_read_each_partial_once():
    nodes = parse_body(
        "for(;e+3072u<count;e+=4096u){a0=__fadd_rn(a0,partials[e]);"
        "a1=__fadd_rn(a1,partials[e+1024u]);a2=__fadd_rn(a2,partials[e+2048u]);"
        "a3=__fadd_rn(a3,partials[e+3072u]);}for(;e<count;e+=1024u){"
        "a0=__fadd_rn(a0,partials[e]);}"
    )

    assert grid_stride_loops(nodes) == PINNED_LOOPS
    for count in (1, 1023, 1024, 1025, 3072, 3073, 4096, 4097, 7169, 9000, 1 << 18):
        reads = combine_coverage(PINNED_LOOPS, count)
        assert reads is not None and reads.shape == (count,), count
        assert bool((reads == 1).all()), count
    stepped = GridStrideLoops(3072, 2048, PINNED_LOOPS.offsets, 1024, ((0, 0),))
    assert int(combine_coverage(stepped, 9000).max()) == 2
    loose = GridStrideLoops(2048, 4096, PINNED_LOOPS.offsets, 1024, ((0, 0),))
    assert combine_coverage(loose, 3000) is None
    stuck = GridStrideLoops(3072, 0, PINNED_LOOPS.offsets, 1024, ((0, 0),))
    assert combine_coverage(stuck, 9000) is None
    # A stride of 1024 with 512 threads skips the partials 512 to 1023.
    fewer = combine_coverage(PINNED_LOOPS, 5000, threads=512)
    assert fewer is not None and int(fewer[512:1024].max()) == 0


def test_barriers_count_only_where_every_thread_reaches_them():
    tree = (
        "for(unsigned int s=blockDim.x/2u;s>0u;s>>=1u){if(t<s){a=b;}__syncthreads();}"
    )

    assert barriers_reached(parse_body("__syncthreads();" + tree), 256) == 9
    assert barriers_reached(parse_body(tree), 1024) == 10
    guarded = "if(t<s){__syncthreads();}"
    assert barriers_reached(parse_body(guarded), 256) is None
    # A loop bound that a thread can change is not uniform.
    custom = tree.replace("s>0u", "s>n")
    assert barriers_reached(parse_body(custom), 256) is None
    other = "for(unsigned int s=0u;s<8u;++s){__syncthreads();}"
    assert barriers_reached(parse_body(other), 256) is None


def test_a_text_must_be_a_string_and_an_n_inside_the_bound():
    with pytest.raises(TypeError, match="must be a string"):
        check(R1_TEXT.encode("ascii"))
    with pytest.raises(GpuReduceCudaError, match="between 1 and"):
        check(R1_TEXT, n=0)
    edited = mutate(R1_TEXT, "dim3 block(256);", "dim3 block(512);")
    with pytest.raises(GpuReduceCudaError, match="fails structural checks: launch"):
        require_reduce_source(edited, kernel=R1_SHARED, n=33)


def test_a_launch_that_breaks_a_limit_fails_the_launch_check_only():
    text = mutate(R1_TEXT, "dim3 block(256);", "dim3 block(2048);")
    result = check(text)

    assert result.failures == ("launch",)
    assert result.launch is None and result.combine_launch is None
    assert check(R1_TEXT).launch == LaunchConfig(Dim3(3907), Dim3(256))


def test_the_combine_coverage_matches_the_partials_of_every_size():
    for kernel in PINNED_KERNELS:
        for n in CHECK_SIZES:
            reads = combine_coverage(PINNED_LOOPS, kernel.partial_count(n))
            assert numpy.array_equal(reads, numpy.ones(kernel.partial_count(n)))
