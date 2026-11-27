"""Tests of the pinned Day 82 Tensor Core text, its four variants and its config.

None of these tests imports triton: they read the text as Python source and
check the declared variants against it.
"""

import ast
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.triton_tc_kernel import (
    CONSTEXPRS,
    KERNEL_NAME,
    PINNED_SHA256,
    TC_CONFIG,
    TRITON_SOURCE,
    VARIANTS,
    VARIANTS_BY_NAME,
    TcConfig,
    TcVariant,
    TritonTcError,
    kernel_mapping,
    require_pinned,
    require_variant,
    source_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
# Pinned here apart from the module: the SHA-256 of the plan kernel file.
PLAN_SHA256 = "6833430d2fea2ac4557766b4fef78b335d34d97b513058c6ec1ad7c2ee84fd9a"
IMPORT_CHECK = """
import sys
sys.modules["triton"] = None
import ai_journey.triton_tc_kernel as kernel
assert "triton" not in [name for name in sys.modules if sys.modules[name]]
print(kernel.source_sha256())
"""


def kernel_function():
    tree = ast.parse(TRITON_SOURCE)
    (function,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == KERNEL_NAME
    ]
    return function


def test_the_text_is_the_plan_kernel_byte_for_byte():
    assert PINNED_SHA256 == PLAN_SHA256
    assert sha256(TRITON_SOURCE.encode("ascii")).hexdigest() == PLAN_SHA256
    assert source_sha256() == PLAN_SHA256
    assert require_pinned(TRITON_SOURCE) is TRITON_SOURCE
    assert TRITON_SOURCE.startswith("import triton\nimport triton.language as tl\n")
    assert TRITON_SOURCE.endswith("& (offs_n[None, :] < N))\n")


def test_a_changed_text_is_not_the_pinned_text():
    with pytest.raises(TritonTcError, match="not the pinned"):
        require_pinned(TRITON_SOURCE.replace("BLOCK_K: tl.constexpr", "BLOCK_K"))
    with pytest.raises(TritonTcError, match="ASCII"):
        source_sha256("tl.dot ×")


def test_every_signature_names_the_parameters_of_the_kernel_in_order():
    function = kernel_function()
    names = [argument.arg for argument in function.args.args]
    annotated = [
        argument.arg
        for argument in function.args.args
        if argument.annotation is not None
        and ast.unparse(argument.annotation) == "tl.constexpr"
    ]
    assert [ast.unparse(item) for item in function.decorator_list] == ["triton.jit"]
    assert annotated == list(CONSTEXPRS)
    for variant in VARIANTS:
        signature = variant.signature()
        assert list(signature) == names
        assert [key for key, kind in signature.items() if kind == "constexpr"] == (
            annotated
        )


def test_the_dot_takes_the_precision_constexpr_and_accumulates_in_float32():
    calls = [
        node
        for node in ast.walk(kernel_function())
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "tl.dot"
    ]
    (dot,) = calls
    assert [ast.unparse(argument) for argument in dot.args] == ["a", "b", "acc"]
    assert [(item.arg, ast.unparse(item.value)) for item in dot.keywords] == [
        ("input_precision", "INPUT_PRECISION")
    ]
    assert "acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)" in TRITON_SOURCE
    assert "tl.store(c_ptrs, acc, mask=" in TRITON_SOURCE


def test_the_four_variants_change_only_the_input_type_and_the_precision():
    assert [variant.name for variant in VARIANTS] == [
        "fp32-ieee",
        "fp32-tf32",
        "fp16",
        "bf16",
    ]
    rows = [
        (
            variant.signature()["a_ptr"],
            variant.signature()["b_ptr"],
            variant.signature()["c_ptr"],
            variant.constexprs()["INPUT_PRECISION"],
            variant.tensor_cores,
            variant.math,
            variant.mma_operand_type,
            variant.element_bytes,
        )
        for variant in VARIANTS
    ]
    assert rows == [
        ("*fp32", "*fp32", "*fp32", "ieee", False, "fp32", None, 4),
        ("*fp32", "*fp32", "*fp32", "tf32", True, "tf32", "tf32", 4),
        ("*fp16", "*fp16", "*fp32", "ieee", True, "fp16", "f16", 2),
        ("*bf16", "*bf16", "*fp32", "ieee", True, "bf16", "bf16", 2),
    ]
    others = {
        key: value
        for variant in VARIANTS
        for key, value in variant.signature().items()
        if key not in ("a_ptr", "b_ptr")
    }
    assert set(others.values()) == {"*fp32", "i32", "constexpr"}
    assert VARIANTS_BY_NAME["bf16"] is require_variant("bf16")


def test_the_one_declared_config_is_128_by_128_by_32_with_4_warps_3_stages():
    assert TC_CONFIG.name == "m128_n128_k32_w4_s3"
    assert TC_CONFIG.options() == {"num_warps": 4, "num_stages": 3}
    assert VARIANTS_BY_NAME["fp32-tf32"].constexprs() == {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
        "INPUT_PRECISION": "tf32",
    }
    assert TC_CONFIG.tile_bytes(4) == 32768
    assert TC_CONFIG.tile_bytes(2) == 16384
    assert kernel_mapping()["config"] == TC_CONFIG.to_mapping()
    assert kernel_mapping()["sha256"] == PLAN_SHA256


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ((96, 128, 32, 4, 3), TritonTcError, "power of two"),
        ((128, 512, 32, 4, 3), TritonTcError, "block_n"),
        ((128, 128, 32, 3, 3), TritonTcError, "num_warps"),
        ((128, 128, 32, 4, 0), TritonTcError, "num_stages"),
        ((128, 128, 32, True, 3), TypeError, "num_warps"),
        ((128.0, 128, 32, 4, 3), TypeError, "block_m"),
    ],
)
def test_a_config_refuses_values_that_triton_would_not_compile(
    arguments, error, message
):
    with pytest.raises(error, match=message):
        TcConfig(*arguments)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("bf16-tf32", "bf16", "tf32"), "only to fp32 inputs"),
        (("fp8", "fp8e4nv", "ieee"), "input_type"),
        (("fp32-x3", "fp32", "tf32x3"), "input_precision"),
        (("", "fp32", "ieee"), "name"),
    ],
)
def test_a_variant_refuses_undeclared_types_and_precisions(arguments, message):
    with pytest.raises(TritonTcError, match=message):
        TcVariant(*arguments)
    with pytest.raises(TritonTcError, match="variant must be one of"):
        require_variant("fp64")
    with pytest.raises(TypeError, match="TcConfig"):
        VARIANTS[0].constexprs({"BLOCK_M": 128})


def test_the_kernel_module_imports_without_triton():
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_CHECK],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == PLAN_SHA256
