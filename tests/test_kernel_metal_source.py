import re
from fractions import Fraction

import numpy

from ai_journey.kernel_metal_reference import GRAY_WEIGHTS
from ai_journey.kernel_metal_source import (
    CONTRACTION_PROBE,
    GRAY_EXPRESSION,
    KERNELS,
    PROBE_EXPRESSION,
    PROBE_FUSED_RESULT,
    PROBE_INPUTS,
    PROBE_UNFUSED_RESULT,
    RGB_TO_GRAY,
    SHADER_SOURCE,
    VECTOR_ADD,
    source_sha256,
)

# Recorded with the source below; reports bind to this exact text.
RECORDED_SOURCE_SHA256 = (
    "f12a5e8af97c2baf1f835af8ec166d3c4cdf02a07bcf53d32c658a7fb0938802"
)


def kernel_text(name):
    match = re.search(
        rf"kernel void {name}\((.*?)\) \{{(.*?)\n\}}\n", SHADER_SOURCE, re.DOTALL
    )
    assert match is not None, name
    return match.group(1), match.group(2)


def test_each_kernel_binds_its_buffers_in_the_declared_order():
    for spec in KERNELS.values():
        parameters, _ = kernel_text(spec.name)
        bindings = re.findall(r"(\w+) \[\[buffer\((\d)\)\]\]", parameters)

        assert bindings == [(name, str(i)) for i, name in enumerate(spec.buffers)]
        for index in spec.scalar_buffers:
            assert f"constant uint& {spec.buffers[index]}" in parameters
        assert spec.arg_casts == {index: "int32" for index in spec.scalar_buffers}
    assert KERNELS[VECTOR_ADD].to_mapping() == {
        "name": "vector_add",
        "axes": 1,
        "buffers": ["out", "a", "b", "n"],
        "scalar_buffers": [3],
        "bytes_per_item": 12,
    }
    assert KERNELS[RGB_TO_GRAY].bytes_per_item == 3 + 1


def test_kernels_index_from_the_grid_position_and_guard_the_padding():
    parameters, body = kernel_text(VECTOR_ADD)
    assert "uint i [[thread_position_in_grid]]" in parameters
    assert "if (i < n) {\n        out[i] = a[i] + b[i];" in body

    parameters, body = kernel_text(RGB_TO_GRAY)
    assert "uint2 position [[thread_position_in_grid]]" in parameters
    assert "uint column = position.x;\n    uint row = position.y;" in body
    assert "if (column < width && row < height) {" in body
    assert "uint pixel = row * width + column;" in body
    assert "out[pixel] = uchar(clamp(floor(y + 0.5f), 0.0f, 255.0f));" in body


def test_the_kernel_weights_are_the_reference_weights_in_the_same_order():
    assert GRAY_EXPRESSION == "0.2989f * r + 0.5870f * g + 0.1140f * b"
    assert f"float y = {GRAY_EXPRESSION};" in kernel_text(RGB_TO_GRAY)[1]
    literals = re.findall(r"(\d\.\d{4})f \* [rgb]", GRAY_EXPRESSION)
    assert tuple(float(value) for value in literals) == GRAY_WEIGHTS


def test_the_probe_separates_a_fused_multiply_add_from_two_roundings():
    first, second, third = (numpy.float32(value) for value in PROBE_INPUTS)
    exact = Fraction(PROBE_INPUTS[0]) * Fraction(PROBE_INPUTS[1])

    assert all(float(numpy.float32(value)) == value for value in PROBE_INPUTS)
    assert float(first * second + third) == PROBE_UNFUSED_RESULT
    assert exact + Fraction(PROBE_INPUTS[2]) == Fraction(PROBE_FUSED_RESULT)
    assert f"out[0] = {PROBE_EXPRESSION};" in kernel_text(CONTRACTION_PROBE)[1]


def test_the_source_digest_pins_the_compiled_text():
    assert source_sha256() == RECORDED_SOURCE_SHA256
    assert SHADER_SOURCE.isascii()
    assert SHADER_SOURCE.startswith("#include <metal_stdlib>\nusing namespace metal;")
