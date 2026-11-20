"""CPU tests of the pinned Day 78 CUDA texts. No test compiles or runs CUDA."""

import re
from hashlib import sha256

import pytest

from ai_journey.gpu_reduce_bound import partial_count
from ai_journey.gpu_reduce_cuda import (
    COMBINE_BARRIERS,
    COMBINE_LAUNCH,
    COMBINE_STRIDES,
    PINNED_KERNELS,
    R1_SHARED,
    SHUFFLE_OFFSETS,
    WARP_SHUFFLE,
    CudaReduceKernel,
    GpuReduceCudaError,
    combine_mapping,
    combine_source,
    cuda_source,
    require_kernel,
    require_n,
    source_files,
    source_sha256,
    tree_strides,
)
from ai_journey.gpu_reduce_source import (
    COMBINE,
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    GpuReduceError,
)
from ai_journey.kernel_launch_config import Dim3, LaunchConfig

SOURCE_SHA256 = {
    "reduce_r1_shared.cu": (
        "86e4ed8683896e63c03038972bfd627837ea24841176ab6110b00bca0cdb5948"
    ),
    "reduce_warp_shuffle.cu": (
        "db3ddc27468f7f8eea0a0c29601116a2ab64681e6f1709802c69504b01a4bf81"
    ),
}
COMBINE_SHA256 = "f6190e3a7352134e2abb08f3e70beab86c69206055913432938d5e4887b51cf4"
# The correctness and timing sizes of config/day-78-gpu-reduce.json, and the
# edges of each launch.
SIZES = (1, 31, 33, 255, 256, 257, 511, 512, 513, 300007, 1000003, 2000003)
SIZES += (65536, 1048576, 16777216, 67108864, 10000019)


def lines_of(kernel):
    return cuda_source(kernel).splitlines()


def test_pinned_texts_are_stable_and_state_that_nothing_ran():
    files = source_files()

    assert [name for name, _ in files] == list(SOURCE_SHA256)
    for (name, text), kernel in zip(files, PINNED_KERNELS):
        assert text == cuda_source(kernel) and name == kernel.file_name
        assert source_sha256(text) == SOURCE_SHA256[name]
        assert sha256(text.encode("ascii")).hexdigest() == SOURCE_SHA256[name]
        assert text.isascii() and text.endswith("}\n")
        assert max(len(line) for line in text.splitlines()) <= 80
        assert "The lab that generated this text did not compile or run it." in text
        assert "It is source for a later run on an NVIDIA GPU." in text
        assert "#" not in text and "\\" not in text and "??" not in text
        assert "atomic" not in text
    with pytest.raises(GpuReduceCudaError, match="ASCII") as raised:
        source_sha256("café")
    assert isinstance(raised.value, GpuReduceError)


def test_each_kernel_takes_its_load_and_tree_from_its_metal_twin():
    assert R1_SHARED.to_mapping() == {
        "name": "reduce_r1_shared",
        "metal_twin": R1,
        "same_order_as": [R1],
        "block_threads": 256,
        "elements_per_thread": 1,
        "shared_memory_bytes": 1024,
        "tree_strides": [128, 64, 32, 16, 8, 4, 2, 1],
        "shuffle_offsets": [],
        "full_mask": None,
        "barriers_per_block": 9,
    }
    assert WARP_SHUFFLE.to_mapping() == {
        "name": "reduce_warp_shuffle",
        "metal_twin": R3,
        "same_order_as": [R2, R3],
        "block_threads": 256,
        "elements_per_thread": 2,
        "shared_memory_bytes": 1024,
        "tree_strides": [128, 64, 32],
        "shuffle_offsets": [16, 8, 4, 2, 1],
        "full_mask": "0xffffffff",
        "barriers_per_block": 4,
    }
    assert [kernel.prefix for kernel in PINNED_KERNELS] == ["r1", "shuffle"]
    assert R1_SHARED.kernel_name == "reduce_r1_shared_kernel"
    assert WARP_SHUFFLE.launcher_name == "launch_reduce_warp_shuffle"
    # The shuffles add 16 + 8 + 4 + 2 + 1 = 31 lanes to lane 0: one warp.
    assert sum(SHUFFLE_OFFSETS) == 31 and WARP_SHUFFLE.tree_strides[-1] == 32


def test_the_r1_text_loads_one_element_and_halves_the_stride_to_1():
    lines = lines_of(R1_SHARED)

    for line in (
        "__global__ void reduce_r1_shared_kernel(",
        "    __shared__ float shared[256];",
        "    unsigned int i = blockIdx.x * 256u + t;",
        "    shared[t] = i < n ? x[i] : 0.0f;",
        "    for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1u) {",
        "            shared[t] = __fadd_rn(shared[t], shared[t + s]);",
        "    if (t == 0u) {",
        "        partials[blockIdx.x] = shared[0];",
        "    dim3 block(256);",
        "    dim3 grid(cdiv(n, 256));",
        "    reduce_r1_shared_kernel<<<grid, block>>>(partials, x, n);",
        "    reduce_combine_kernel<<<1, 1024>>>(out, partials, grid.x);",
    ):
        assert line in lines, line
    assert "__shfl" not in cuda_source(R1_SHARED)


def test_the_shuffle_text_ends_in_warp_0_with_five_full_mask_shuffles():
    text = cuda_source(WARP_SHUFFLE)
    lines = text.splitlines()

    for line in (
        "    unsigned int lane = t % 32u;",
        "    unsigned int warp = t / 32u;",
        "    unsigned int i = blockIdx.x * 512u + t;",
        "    float v = __fadd_rn(i < n ? x[i] : 0.0f, "
        "i + 256u < n ? x[i + 256u] : 0.0f);",
        "    for (unsigned int s = blockDim.x / 2u; s >= 32u; s >>= 1u) {",
        "    dim3 grid(cdiv(n, 512));",
    ):
        assert line in lines, line
    stage = text.split("    if (warp == 0u) {\n", 1)[1].split("\n    }\n}\n", 1)[0]
    assert stage.splitlines() == [
        "        float w = shared[lane];",
        *(
            f"        w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, {offset}u));"
            for offset in (16, 8, 4, 2, 1)
        ),
        "        if (lane == 0u) {",
        "            partials[blockIdx.x] = w;",
        "        }",
    ]
    # The last warp has no barrier: the 2 barriers of pass 1 come before it.
    kernel = text.split("__global__ void reduce_warp_shuffle_kernel(", 1)[1]
    kernel = kernel.split("\n}\n", 1)[0]
    assert kernel.count("__syncthreads();") == 2
    assert kernel.rindex("__syncthreads();") < kernel.index("if (warp == 0u)")


def test_both_files_hold_one_combine_kernel_with_its_own_digest():
    combine = combine_source()

    for _, text in source_files():
        assert text.count(combine) == 1
        assert text.count("__global__ void") == 2
    assert source_sha256(combine) == COMBINE_SHA256
    assert combine_mapping() == {
        "name": "reduce_combine_kernel",
        "metal_twin": COMBINE,
        "launch": {"grid": [1, 1, 1], "block": [1024, 1, 1]},
        "accumulators": 4,
        "stride": 1024,
        "span": 4096,
        "shared_memory_bytes": 4096,
        "tree_strides": [512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        "barriers_per_block": 11,
        "source_sha256": COMBINE_SHA256,
    }
    assert COMBINE_LAUNCH == LaunchConfig(Dim3(1), Dim3(1024))
    assert COMBINE_BARRIERS == 1 + len(COMBINE_STRIDES) == 11
    for line in (
        "    for (; e + 3072u < count; e += 4096u) {",
        "        a3 = __fadd_rn(a3, partials[e + 3072u]);",
        "    for (; e < count; e += 1024u) {",
        "    shared[t] = __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));",
        "        out[0] = shared[0];",
    ):
        assert line in combine.splitlines(), line


def test_every_float_addition_is_one_fadd_rn():
    # R1: the tree, then the combine's 4 + 1 loads, 3 joins and its tree.
    counts = {R1_SHARED: 1 + 9, WARP_SHUFFLE: 1 + 1 + 5 + 9}

    for kernel, count in counts.items():
        text = cuda_source(kernel)
        assert text.count("__fadd_rn(") == count
        # Only the index e of the grid-stride loops uses +=.
        assert re.findall(r"(\w+) \+=", text) == ["e", "e"]


@pytest.mark.parametrize("kernel", PINNED_KERNELS, ids=lambda item: item.name)
def test_the_launch_covers_n_with_the_metal_twin_partials(kernel):
    for n in SIZES:
        launch = kernel.launch(n)
        blocks = launch.grid.x

        assert launch.block == Dim3(256) and launch.grid.to_list()[1:] == [1, 1]
        assert (blocks - 1) * kernel.elements_per_block < n
        assert n <= blocks * kernel.elements_per_block
        assert kernel.partial_count(n) == partial_count(kernel.metal_twin, n)
    assert kernel.partial_count(1 << 26) == (1 << 26) // kernel.elements_per_block


@pytest.mark.parametrize("n", [0, -1, (1 << 26) + 1])
def test_lengths_outside_the_texts_are_refused(n):
    with pytest.raises(GpuReduceCudaError, match="between 1 and 67108864"):
        require_n(n)
    with pytest.raises(GpuReduceCudaError):
        R1_SHARED.launch(n)


@pytest.mark.parametrize("n", [True, 2.0, "3"])
def test_a_length_must_be_an_integer(n):
    with pytest.raises(TypeError, match="n must be an integer"):
        require_n(n)


@pytest.mark.parametrize("twin", [R0, R2, R4, R5, COMBINE, "reduce_r6"])
def test_only_twins_whose_order_a_text_states_are_kernels(twin):
    with pytest.raises(GpuReduceCudaError, match="Metal twin must be"):
        CudaReduceKernel(name="reduce_other", metal_twin=twin)


def test_only_the_pinned_kernels_have_a_text():
    other = CudaReduceKernel(name="reduce_other", metal_twin=R1)

    for value, error, message in (
        (other, GpuReduceCudaError, "not a pinned CUDA kernel"),
        ("reduce_r1_shared", TypeError, "CudaReduceKernel"),
    ):
        with pytest.raises(error, match=message):
            require_kernel(value)
        with pytest.raises(error, match=message):
            cuda_source(value)
    for name in ("Reduce_r1", "r1_shared", "reduce_", 7):
        with pytest.raises(GpuReduceCudaError, match="reduce_ and a short slug"):
            CudaReduceKernel(name=name, metal_twin=R1)
    assert require_kernel(WARP_SHUFFLE) is WARP_SHUFFLE


def test_tree_strides_halve_from_half_the_block_to_the_stop():
    assert tree_strides(256, 1) == (128, 64, 32, 16, 8, 4, 2, 1)
    assert tree_strides(256, 32) == (128, 64, 32)
    assert tree_strides(1024, 1) == COMBINE_STRIDES
    assert tree_strides(32, 32) == ()
