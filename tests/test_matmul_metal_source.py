import re
from hashlib import sha256

import pytest

from ai_journey.kernel_metal_runtime import probe_metal
from ai_journey.matmul_metal_mapping import MatmulMetalError
from ai_journey.matmul_metal_source import (
    GUARDED_LOOP,
    KERNEL_FOR_MAPPING,
    KERNELS,
    LOOP_STATEMENT,
    MATMUL_X_COLUMNS,
    MATMUL_X_ROWS,
    SHADER_SOURCE,
    MatmulKernelSpec,
    source_sha256,
)

AVAILABILITY = probe_metal()
requires_metal = pytest.mark.skipif(
    not AVAILABILITY.available,
    reason="needs an Apple GPU through torch.mps.compile_shader; "
    f"this host: {AVAILABILITY.message}",
)


def kernel_texts():
    pieces = SHADER_SOURCE.split("kernel void ")[1:]
    return {piece.split("(", 1)[0]: piece for piece in pieces}


def test_both_kernels_share_one_guarded_loop_and_differ_in_coordinates():
    texts = kernel_texts()

    assert set(texts) == {MATMUL_X_COLUMNS, MATMUL_X_ROWS}
    for text in texts.values():
        assert text.count(GUARDED_LOOP) == 1
        assert "if (row < m && column < n) {" in text
    assert (
        "    uint column = position.x;\n    uint row = position.y;\n"
        in texts[MATMUL_X_COLUMNS]
    )
    assert (
        "    uint row = position.x;\n    uint column = position.y;\n"
        in texts[MATMUL_X_ROWS]
    )
    for text in texts.values():
        body = text.split(") {\n", 1)[1]
        # Two coordinate lines, then the shared template closes the kernel.
        assert body.split("\n", 2)[2].startswith(f"{GUARDED_LOOP}\n}}\n")


def test_the_loop_rounds_once_per_step_through_an_explicit_fma():
    assert LOOP_STATEMENT == "sum = fma(a[row * k + i], b[i * n + column], sum);"
    assert SHADER_SOURCE.count("fma(") == 2
    assert "+=" not in SHADER_SOURCE
    assert "for (uint i = 0; i < k; ++i) {" in GUARDED_LOOP
    assert "float sum = 0.0f;" in GUARDED_LOOP


def test_specs_match_the_buffer_bindings_and_cast_scalars_to_32_bits():
    for name, text in kernel_texts().items():
        bound = re.findall(r"(\w+) \[\[buffer\((\d)\)\]\]", text)
        spec = KERNELS[name]
        assert [binding for binding, _ in bound] == list(spec.buffers)
        assert [int(index) for _, index in bound] == list(range(6))
        for index in spec.scalar_buffers:
            assert f"constant uint& {spec.buffers[index]}" in text
        assert spec.arg_casts == {3: "int32", 4: "int32", 5: "int32"}
    assert KERNEL_FOR_MAPPING == {
        "x_columns": MATMUL_X_COLUMNS,
        "x_rows": MATMUL_X_ROWS,
    }
    assert KERNELS[MATMUL_X_ROWS].to_mapping() == {
        "name": "matmul_x_rows",
        "mapping": "x_rows",
        "axes": 2,
        "buffers": ["c", "a", "b", "m", "n", "k"],
        "scalar_buffers": [3, 4, 5],
    }
    with pytest.raises(MatmulMetalError, match="mapping must be one of"):
        MatmulKernelSpec("matmul_x_columns", "columns")


def test_the_digest_covers_the_exact_ascii_source():
    assert SHADER_SOURCE.isascii()
    assert source_sha256() == sha256(SHADER_SOURCE.encode("ascii")).hexdigest()


@requires_metal
def test_the_metal_compiler_accepts_both_kernels():
    import torch

    library = torch.mps.compile_shader(SHADER_SOURCE)

    for name in KERNELS:
        kernel = getattr(library, name)
        assert kernel.max_threads_per_threadgroup >= 256
        assert kernel.thread_execution_width >= 1
