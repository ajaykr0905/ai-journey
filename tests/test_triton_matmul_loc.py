"""Tests of the counting rule and of the pinned counts of the real texts."""

from pathlib import Path

import pytest

from ai_journey.triton_matmul_kernel import (
    GROUPED_ORDER,
    TRITON_SOURCE,
    TritonMatmulError,
)
from ai_journey.triton_matmul_loc import (
    COUNTING_RULE,
    CountedText,
    LineCount,
    count_c_like,
    count_python,
    counted_texts,
    loc_mapping,
)

ROOT = Path(__file__).resolve().parents[1]
# lab, language prefix, name: kernel, host, other, comment, blank lines.
PINNED_COUNTS = {
    ("day-79", "Triton", "matmul_kernel"): (33, 9, 2, 10, 4),
    ("day-76", "CUDA", "tiled_matmul_t8_kernel"): (24, 9, 0, 10, 3),
    ("day-76", "CUDA", "tiled_matmul_t16_kernel"): (24, 9, 0, 10, 3),
    ("day-76", "CUDA", "tiled_matmul_t32_kernel"): (24, 9, 0, 10, 3),
    ("day-77", "CUDA", "coarsened_matmul_t16_c2_kernel"): (33, 9, 0, 11, 3),
    ("day-77", "CUDA", "coarsened_matmul_t16_c4_kernel"): (33, 9, 0, 11, 3),
    ("day-77", "CUDA", "coarsened_matmul_t32_c2_kernel"): (33, 9, 0, 11, 3),
    ("day-77", "CUDA", "blocked_matmul_m64_n64_k16_t4x4_kernel"): (84, 9, 0, 15, 3),
    ("day-77", "CUDA", "blocked_matmul_m128_n128_k8_t8x8_kernel"): (84, 9, 0, 15, 3),
    ("day-76", "Metal", "matmul_tiled_8"): (31, None, 0, 4, 0),
    ("day-76", "Metal", "matmul_tiled_16"): (31, None, 0, 4, 0),
    ("day-76", "Metal", "matmul_tiled_32"): (31, None, 0, 4, 0),
    ("day-77", "Metal", "matmul_s2_coarse_4x1"): (126, None, 0, 3, 0),
    ("day-77", "Metal", "matmul_s3_block_4x4"): (192, None, 0, 3, 0),
    ("day-77", "Metal", "matmul_s4_vec4"): (222, None, 0, 3, 0),
    ("day-77", "Metal", "matmul_s5_pad4"): (222, None, 0, 3, 0),
    ("day-77", "Metal", "matmul_s6_group_8x8"): (316, None, 0, 3, 0),
    ("day-77", "Metal", "matmul_s6_group_32x32"): (186, None, 0, 3, 0),
    ("day-77", "Metal", "matmul_s7_interior"): (420, None, 0, 3, 0),
}
# The guide of each lab pins the digest of each of its texts.
GUIDES = {
    ("day-76", "CUDA"): "docs/gpu-tiled-cuda-contract.md",
    ("day-77", "CUDA"): "docs/gpu-checklist-cuda-contract.md",
    ("day-76", "Metal"): "docs/matmul-tiled-lab.md",
    ("day-77", "Metal"): "docs/matmul-checklist-lab.md",
}


def key(text):
    return (text.lab, text.language.split()[0], text.name)


def test_each_real_text_has_its_pinned_count():
    texts = counted_texts()

    assert [key(text) for text in texts] == list(PINNED_COUNTS)
    for text in texts:
        count = text.count()
        fields = (count.kernel, count.host, count.other, count.comment, count.blank)
        assert fields == PINNED_COUNTS[key(text)], key(text)
        assert count.total == text.text.count("\n")


def test_each_counted_text_is_the_text_that_its_guide_pins():
    for text in counted_texts()[1:]:
        guide = (ROOT / GUIDES[key(text)[:2]]).read_text(encoding="utf-8")
        assert f"`{text.sha256}`" in guide, key(text)


def test_without_the_grouped_order_the_triton_kernel_has_the_cuda_tiled_count():
    triton = counted_texts()[0].count()
    without = "".join(
        line
        for line in TRITON_SOURCE.splitlines(keepends=True)
        if line.strip() not in GROUPED_ORDER
    )

    assert len(GROUPED_ORDER) == 9
    assert count_python(without).kernel == triton.kernel - 9 == 24
    assert counted_texts()[2].count().kernel == 24


def test_the_mapping_names_the_rule_and_every_text():
    mapping = loc_mapping()

    assert mapping["rule"] == COUNTING_RULE
    assert len(mapping["texts"]) == len(PINNED_COUNTS)
    assert mapping["texts"][0] == {
        "lab": "day-79",
        "language": "Triton (Python)",
        "name": "matmul_kernel",
        "origin": "ai_journey.triton_matmul_kernel.TRITON_SOURCE",
        "sha256": "9eb35e273b0fd229e83e223861ef7ce13c4f15ea46d2694baf13920851a31d56",
        "lines": {
            "kernel": 33,
            "host": 9,
            "other": 2,
            "comment": 10,
            "blank": 4,
            "total": 58,
        },
    }
    assert mapping["texts"][-1]["lines"]["host"] is None


def test_the_c_rule_on_comments_helpers_includes_and_host_code():
    text = (
        "#include <cuda_runtime.h>\n"
        "\n"
        "// a comment line\n"
        "__device__ float twice(float x) { return 2.0f * x; } // trailing\n"
        "__host__ __device__ inline int cdiv(int a, int b) {\n"
        "    /* one-line block */ return (a + b - 1) / b;\n"
        "}\n"
        "__global__ void k(float* c) {\n"
        "    /* only a comment */\n"
        "    c[0] = twice(1.0f);\n"
        "}\n"
        "void launch(float* c) {\n"
        "    k<<<cdiv(1, 1), 1>>>(c);\n"
        "}\n"
    )

    assert count_c_like(text, host_code=True) == LineCount(
        kernel=4, host=6, other=1, comment=2, blank=1
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("/* two\nlines */\n__global__ void k() {}\n", "block comment across"),
        ("#define A \\\n  1\n", "line splice"),
        ("__global__ void k() {\n", "inside a definition"),
        ("}\n", "never opened"),
        ("__global__ void k() {}", "newline"),
    ],
)
def test_the_c_rule_refuses_a_text_it_cannot_count(text, message):
    with pytest.raises(TritonMatmulError, match=message):
        count_c_like(text, host_code=True)


def test_a_metal_text_must_hold_kernels_only():
    kernel = "kernel void k(device float* c [[buffer(0)]]) {\n    c[0] = 1.0f;\n}\n"

    assert count_c_like(kernel, host_code=False) == LineCount(3, None, 0, 0, 0)
    with pytest.raises(TritonMatmulError, match="kernel functions only"):
        count_c_like(kernel + "void host() {\n}\n", host_code=False)


def test_the_python_rule_counts_decorators_and_skips_docstrings():
    text = (
        '"""Module docstring."""\n'
        "import triton\n"
        "\n"
        "@triton.jit\n"
        "def kernel(x):\n"
        '    """Kernel docstring\n'
        '    on two lines."""\n'
        "    # a comment\n"
        "    y = x  # trailing\n"
        "\n"
        "def launch(x):\n"
        "    kernel[(1,)](x)\n"
        "SIZE = 4\n"
    )

    assert count_python(text) == LineCount(
        kernel=3, host=2, other=2, comment=4, blank=2
    )


def test_a_counted_text_needs_a_known_lab_and_language():
    with pytest.raises(TritonMatmulError, match="known lab and language"):
        CountedText("day-78", "CUDA C++", "k", "x", "__global__ void k() {}\n")
    with pytest.raises(TritonMatmulError, match="known lab and language"):
        CountedText("day-76", "OpenCL", "k", "x", "kernel void k() {}\n")
