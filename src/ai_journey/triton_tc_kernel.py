"""Pinned Triton text of the Day 82 Tensor Core matmul, its variants and config.

``TRITON_SOURCE`` is the exact text of the Day 82 plan kernel
``tc_matmul_kernel``, and ``PINNED_SHA256`` is the SHA-256 of its ASCII bytes.
One program computes one ``BLOCK_M x BLOCK_N`` tile of ``C = A @ B`` on a 2D
grid. Masks on the loads fill each slot outside A or B with 0, and a mask on
the store keeps each write inside C. The K loop adds each tile product to a
float32 accumulator with ``tl.dot(..., input_precision=INPUT_PRECISION)``.

The lab compiles one text in four declared ``VARIANTS``, so that only the
input type and the ``input_precision`` argument change between them:

- ``fp32-ieee``: float32 inputs, ``"ieee"``. ``tl.dot`` must keep IEEE
  float32 arithmetic, so it cannot use a Tensor Core;
- ``fp32-tf32``: float32 inputs, ``"tf32"``. ``tl.dot`` may round each input
  to TF32 and use a Tensor Core;
- ``fp16`` and ``bf16``: 16-bit inputs, ``"ieee"``. ``input_precision``
  applies only to float32 inputs, so Triton ignores it here and uses a Tensor
  Core for the 16-bit product. C stays float32 in every variant.

``TC_CONFIG`` is the one declared block config: 128 x 128 x 32 with 4 warps
and 3 stages. This module never imports triton, so the package imports on a
host without it. Triton has no macOS wheel.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kernel_metal_geometry import KernelMetalError, require_integer
from .triton_matmul_kernel import MAX_BLOCK, MAX_STAGES, MIN_BLOCK, WARP_COUNTS
from .triton_matmul_kernel import TritonMatmulError
from .triton_matmul_kernel import source_sha256 as _text_sha256

KERNEL_NAME = "tc_matmul_kernel"
LANGUAGE = "Triton (Python)"
ACCUMULATOR = "float32"
OUTPUT_TYPE = "fp32"
PLAN_FILE = "plans/d82/tc_kernel.py"
PINNED_SHA256 = "6833430d2fea2ac4557766b4fef78b335d34d97b513058c6ec1ad7c2ee84fd9a"
POINTERS = ("a_ptr", "b_ptr", "c_ptr")
INTEGERS = (
    "M",
    "N",
    "K",
    "stride_am",
    "stride_ak",
    "stride_bk",
    "stride_bn",
    "stride_cm",
    "stride_cn",
)
CONSTEXPRS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "INPUT_PRECISION")
INPUT_TYPES = {"fp32": 4, "fp16": 2, "bf16": 2}
PRECISIONS = ("ieee", "tf32")

TRITON_SOURCE = """\
import triton
import triton.language as tl


@triton.jit
def tc_matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
"""


class TritonTcError(ValueError):
    """Raised when a Day 82 Tensor Core lab input, compile or report breaks."""


def _integer(value: object, name: str, low: int, high: int) -> int:
    try:
        return require_integer(value, name, low, high)
    except KernelMetalError as error:
        raise TritonTcError(str(error)) from error


def source_sha256(text: str = TRITON_SOURCE) -> str:
    """Return the SHA-256 of an ASCII text, by default of ``TRITON_SOURCE``."""

    try:
        return _text_sha256(text)
    except TritonMatmulError as error:
        raise TritonTcError(str(error)) from error


def require_pinned(text: str) -> str:
    """Return the text only if it is the pinned plan kernel, byte for byte."""

    if source_sha256(text) != PINNED_SHA256:
        raise TritonTcError(f"the kernel text is not the pinned {PLAN_FILE}")
    return text


@dataclass(frozen=True)
class TcConfig:
    """The block sizes and compile options of one compile of the text."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        for name in ("block_m", "block_n", "block_k"):
            value = _integer(getattr(self, name), name, MIN_BLOCK, MAX_BLOCK)
            if value & (value - 1):
                raise TritonTcError(f"{name} must be a power of two")
        if type(self.num_warps) is not int:
            raise TypeError("num_warps must be an integer")
        if self.num_warps not in WARP_COUNTS:
            raise TritonTcError(f"num_warps must be one of {WARP_COUNTS}")
        _integer(self.num_stages, "num_stages", 1, MAX_STAGES)

    @property
    def name(self) -> str:
        return (
            f"m{self.block_m}_n{self.block_n}_k{self.block_k}"
            f"_w{self.num_warps}_s{self.num_stages}"
        )

    def options(self) -> dict[str, int]:
        """Return the ``options`` mapping of ``triton.compile``."""

        return {"num_warps": self.num_warps, "num_stages": self.num_stages}

    def tile_bytes(self, element_bytes: int) -> int:
        """Return the bytes of one A tile and one B tile at an element size.

        The compiler decides how many copies of the two tiles it keeps in
        shared memory for ``num_stages``, so this is not the shared memory of
        a compile. The compile metadata states that.
        """

        size = _integer(element_bytes, "element_bytes", 1, 8)
        return (self.block_m + self.block_n) * self.block_k * size

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_k": self.block_k,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


TC_CONFIG = TcConfig(128, 128, 32, 4, 3)


@dataclass(frozen=True)
class TcVariant:
    """One declared input type and ``input_precision`` of the pinned text."""

    name: str
    input_type: str
    input_precision: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TritonTcError("variant name must be a non-empty string")
        if self.input_type not in INPUT_TYPES:
            raise TritonTcError(f"input_type must be one of {tuple(INPUT_TYPES)}")
        if self.input_precision not in PRECISIONS:
            raise TritonTcError(f"input_precision must be one of {PRECISIONS}")
        if self.input_type != "fp32" and self.input_precision != "ieee":
            raise TritonTcError(
                "input_precision applies only to fp32 inputs, so a 16-bit "
                "variant declares ieee"
            )

    @property
    def element_bytes(self) -> int:
        return INPUT_TYPES[self.input_type]

    @property
    def tensor_cores(self) -> bool:
        """Whether ``tl.dot`` may use a Tensor Core: not for IEEE float32."""

        return not (self.input_type == "fp32" and self.input_precision == "ieee")

    @property
    def math(self) -> str:
        """Return the math path of the speedup model: fp32, tf32, fp16 or bf16."""

        if self.input_type == "fp32":
            return "fp32" if self.input_precision == "ieee" else "tf32"
        return self.input_type

    @property
    def mma_operand_type(self) -> str | None:
        """Return the PTX type of the A and B operands of an MMA, if any."""

        return {"tf32": "tf32", "fp16": "f16", "bf16": "bf16"}.get(self.math)

    def signature(self) -> dict[str, str]:
        """Return the ``ASTSource`` signature, in the parameter order of the text."""

        return {
            "a_ptr": f"*{self.input_type}",
            "b_ptr": f"*{self.input_type}",
            "c_ptr": f"*{OUTPUT_TYPE}",
            **dict.fromkeys(INTEGERS, "i32"),
            **dict.fromkeys(CONSTEXPRS, "constexpr"),
        }

    def constexprs(self, config: TcConfig = TC_CONFIG) -> dict[str, object]:
        """Return the ``tl.constexpr`` values of one compile."""

        if type(config) is not TcConfig:
            raise TypeError("config must be TcConfig")
        return {
            "BLOCK_M": config.block_m,
            "BLOCK_N": config.block_n,
            "BLOCK_K": config.block_k,
            "INPUT_PRECISION": self.input_precision,
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "input_type": self.input_type,
            "output_type": OUTPUT_TYPE,
            "accumulator": ACCUMULATOR,
            "input_precision": self.input_precision,
            "element_bytes": self.element_bytes,
            "tensor_cores": self.tensor_cores,
            "math": self.math,
            "mma_operand_type": self.mma_operand_type,
        }


VARIANTS = (
    TcVariant("fp32-ieee", "fp32", "ieee"),
    TcVariant("fp32-tf32", "fp32", "tf32"),
    TcVariant("fp16", "fp16", "ieee"),
    TcVariant("bf16", "bf16", "ieee"),
)
VARIANTS_BY_NAME = {variant.name: variant for variant in VARIANTS}


def require_variant(name: object) -> TcVariant:
    if type(name) is not str or name not in VARIANTS_BY_NAME:
        raise TritonTcError(f"variant must be one of {tuple(VARIANTS_BY_NAME)}")
    return VARIANTS_BY_NAME[name]


def kernel_mapping() -> dict[str, object]:
    """Return the pinned text, its digest, the variants and the config."""

    return {
        "language": LANGUAGE,
        "kernel": KERNEL_NAME,
        "plan_file": PLAN_FILE,
        "sha256": source_sha256(),
        "text": TRITON_SOURCE,
        "config": TC_CONFIG.to_mapping(),
        "variants": [variant.to_mapping() for variant in VARIANTS],
    }
