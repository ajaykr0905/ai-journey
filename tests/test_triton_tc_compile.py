"""Tests of the offline compile backend and of the dump folder, with no triton.

Stand-ins for triton record what the backend passes to ``triton.compile``, so
these tests run on macOS, where triton has no wheel. The texts that the
stand-ins return are short synthetic strings, not compiler output. The
compile contract test compiles for real on a Linux host with triton 3.9.0.
"""

import inspect
import json
import linecache
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from ai_journey.triton_tc_compile import (
    COMPILE_ORDER,
    ENVIRONMENT_FIELDS,
    INDEX_NAME,
    SUMMARY_NAME,
    CompiledVariant,
    CompilerBackend,
    TritonTcUnavailableError,
    compile_all,
    compile_text,
    load_dump,
    source_filename,
    stage_path,
    write_dump,
)
from ai_journey.triton_tc_kernel import (
    PINNED_SHA256,
    VARIANTS,
    VARIANTS_BY_NAME,
    TritonTcError,
)

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = dict.fromkeys(ENVIRONMENT_FIELDS, "stand-in")
NO_TRITON = """
import sys
sys.modules["triton"] = None
from ai_journey.triton_tc_compile import CompilerBackend, TritonTcUnavailableError
try:
    CompilerBackend()
except TritonTcUnavailableError as error:
    print(error.reason)
"""


def synthetic(variant, target, shared=32768, warps=4, stages=3):
    """A compile record with short synthetic texts, not compiler output."""

    return CompiledVariant(
        variant=variant,
        target=target,
        ttgir=f"// synthetic ttgir {variant} {target}\n",
        ptx=f"// synthetic ptx {variant} {target}\n.target {target}\n",
        shared_bytes=shared,
        num_warps=warps,
        num_stages=stages,
    )


class SyntheticBackend:
    """Return a synthetic compile for each request, and record the requests."""

    name = "synthetic"
    environment = ENVIRONMENT

    def __init__(self):
        self.calls = []

    def compile(self, variant, target):
        self.calls.append((variant.name, target))
        return synthetic(variant.name, target)


def stand_in_triton(monkeypatch, kind, error=None):
    """Install stand-in triton modules whose jit returns a ``kind`` object."""

    calls = []

    class Function:
        pass

    Function.__name__ = kind

    def jit(fn):
        wrapped = Function()
        wrapped.fn = fn
        return wrapped

    def compile_(source, target, options):
        calls.append((source, target, options))
        if error is not None:
            raise error
        metadata = types.SimpleNamespace(shared=65536, num_warps=4, num_stages=3)
        asm = {"ttgir": "synthetic ttgir\n", "ptx": "synthetic ptx\n"}
        return types.SimpleNamespace(asm=asm, metadata=metadata)

    triton = types.ModuleType("triton")
    triton.__version__ = "0.0.0"
    triton.jit = jit
    triton.compile = compile_
    triton.cdiv = lambda a, b: -(-a // b)
    language = types.ModuleType("triton.language")
    language.constexpr = object
    compiler = types.ModuleType("triton.compiler")
    compiler.ASTSource = lambda **kwargs: kwargs
    backends = types.ModuleType("triton.backends")
    backends_compiler = types.ModuleType("triton.backends.compiler")
    backends_compiler.GPUTarget = lambda *args: args
    triton.language = language
    for name, module in {
        "triton": triton,
        "triton.language": language,
        "triton.compiler": compiler,
        "triton.backends": backends,
        "triton.backends.compiler": backends_compiler,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_the_backend_compiles_one_variant_for_a_declared_target(monkeypatch):
    calls = stand_in_triton(monkeypatch, "JITFunction")
    backend = CompilerBackend()
    compiled = backend.compile(VARIANTS_BY_NAME["bf16"], "sm_90")

    (source, target, options) = calls[0]
    assert target == ("cuda", 90, 32)
    assert options == {"num_warps": 4, "num_stages": 3}
    assert source["signature"] == VARIANTS_BY_NAME["bf16"].signature()
    assert source["constexprs"] == {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
        "INPUT_PRECISION": "ieee",
    }
    assert compiled.key == ("bf16", "sm_90")
    assert (compiled.ttgir, compiled.ptx) == ("synthetic ttgir\n", "synthetic ptx\n")
    assert (compiled.shared_bytes, compiled.num_warps, compiled.num_stages) == (
        65536,
        4,
        3,
    )
    assert backend.environment["triton"] == "0.0.0"
    assert tuple(backend.environment) == ENVIRONMENT_FIELDS
    assert backend.source_sha256 == PINNED_SHA256


def test_the_jit_reads_the_pinned_text_through_linecache(monkeypatch):
    stand_in_triton(monkeypatch, "JITFunction")
    text, namespace = compile_text()
    kernel = namespace["tc_matmul_kernel"]

    assert source_filename(text) in linecache.cache
    assert source_filename(text).endswith(f"sha256={PINNED_SHA256}>")
    source = inspect.getsource(kernel.fn)
    assert source.startswith("@triton.jit\ndef tc_matmul_kernel(\n")
    assert source in text


def test_the_backend_refuses_the_interpreter(monkeypatch):
    stand_in_triton(monkeypatch, "InterpretedFunction")
    with pytest.raises(TritonTcUnavailableError) as raised:
        CompilerBackend()
    assert raised.value.reason == "interpreter_enabled"


def test_the_backend_names_a_missing_triton_or_compiler(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    with pytest.raises(TritonTcUnavailableError) as raised:
        CompilerBackend()
    assert raised.value.reason == "triton_not_importable"
    stand_in_triton(monkeypatch, "JITFunction")
    monkeypatch.setitem(sys.modules, "triton.compiler", None)
    with pytest.raises(TritonTcUnavailableError) as raised:
        CompilerBackend()
    assert raised.value.reason == "compiler_not_importable"
    with pytest.raises(ValueError, match="reason must be one of"):
        TritonTcUnavailableError("other", "x")


def test_a_compile_error_names_the_variant_and_the_target(monkeypatch):
    stand_in_triton(monkeypatch, "JITFunction", error=RuntimeError("no mma"))
    backend = CompilerBackend()
    with pytest.raises(TritonTcError, match="could not compile fp16 for sm_80: no"):
        backend.compile(VARIANTS_BY_NAME["fp16"], "sm_80")
    with pytest.raises(TritonTcError, match="target must be one of"):
        backend.compile(VARIANTS_BY_NAME["fp16"], "sm_86")
    with pytest.raises(TypeError, match="TcVariant"):
        backend.compile("fp16", "sm_80")


def test_compile_all_compiles_every_variant_for_both_targets_in_order():
    backend = SyntheticBackend()
    compiled = compile_all(backend)

    assert backend.calls == list(COMPILE_ORDER)
    assert len(COMPILE_ORDER) == 2 * len(VARIANTS) == 8
    assert COMPILE_ORDER[:2] == (("fp32-ieee", "sm_80"), ("fp32-ieee", "sm_90"))
    assert [item.key for item in compiled] == list(COMPILE_ORDER)


def test_compile_all_refuses_a_backend_that_breaks_its_contract():
    class Silent:
        environment = {"python": "3.11"}

    class Wrong(SyntheticBackend):
        def compile(self, variant, target):
            return synthetic("bf16", target)

    class Loose(SyntheticBackend):
        def compile(self, variant, target):
            return {"variant": variant.name}

    with pytest.raises(TritonTcError, match="must name its environment"):
        compile_all(Silent())
    with pytest.raises(TritonTcError, match="returned"):
        compile_all(Wrong())
    with pytest.raises(TritonTcError, match="must return CompiledVariant"):
        compile_all(Loose())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"warps": 8}, "num_warps"),
        ({"stages": 4}, "num_stages"),
        ({"shared": 227 * 1024 + 1}, "shared_bytes"),
        ({"shared": -1}, "shared_bytes"),
    ],
)
def test_a_compile_refuses_metadata_off_the_declared_config(change, message):
    with pytest.raises(TritonTcError, match=message):
        synthetic("bf16", "sm_80", **change)


def test_a_compile_refuses_an_unknown_variant_target_or_empty_text():
    with pytest.raises(TritonTcError, match="variant must be one of"):
        synthetic("fp8", "sm_80")
    with pytest.raises(TritonTcError, match="target must be one of"):
        synthetic("bf16", "sm_100")
    with pytest.raises(TritonTcError, match="ptx must be a text"):
        CompiledVariant("bf16", "sm_80", "ttgir", "", 0, 4, 3)


def test_a_dump_round_trips_and_names_each_text_by_digest(tmp_path):
    compiled = compile_all(SyntheticBackend())
    summary = {"synthetic": True}
    index = write_dump(tmp_path, compiled, ENVIRONMENT, summary)

    assert (tmp_path / stage_path("bf16", "sm_90", "ptx")).is_file()
    assert (tmp_path / "fp32-tf32" / "sm_80" / "ttgir.mlir").is_file()
    assert json.loads((tmp_path / SUMMARY_NAME).read_text("ascii")) == summary
    assert json.loads((tmp_path / INDEX_NAME).read_text("ascii")) == index
    assert index["kernel_sha256"] == PINNED_SHA256
    assert index["compiles"][7]["stages"]["ptx"]["lines"] == 2
    read_index, loaded = load_dump(tmp_path)
    assert loaded == compiled
    assert read_index == index


def test_a_dump_refuses_a_changed_text_or_index(tmp_path):
    compiled = compile_all(SyntheticBackend())
    write_dump(tmp_path, compiled, ENVIRONMENT)
    text = tmp_path / stage_path("fp16", "sm_80", "ptx")
    text.write_text(text.read_text("ascii") + "mma.sync\n", encoding="ascii")
    with pytest.raises(TritonTcError, match="does not match its index digest"):
        load_dump(tmp_path)

    write_dump(tmp_path, compiled, ENVIRONMENT)
    index = json.loads((tmp_path / INDEX_NAME).read_text("ascii"))
    index["compiles"][0], index["compiles"][1] = (
        index["compiles"][1],
        index["compiles"][0],
    )
    (tmp_path / INDEX_NAME).write_text(json.dumps(index), encoding="ascii")
    with pytest.raises(TritonTcError, match="COMPILE_ORDER"):
        load_dump(tmp_path)
    index["compiles"] = index["compiles"][:2]
    index["kernel_sha256"] = "0" * 64
    (tmp_path / INDEX_NAME).write_text(json.dumps(index), encoding="ascii")
    with pytest.raises(TritonTcError, match="another text"):
        load_dump(tmp_path)
    with pytest.raises(TritonTcError, match="COMPILE_ORDER"):
        write_dump(tmp_path, compiled[:7], ENVIRONMENT)


def test_a_compile_without_triton_stops_before_it_compiles():
    completed = subprocess.run(
        [sys.executable, "-c", NO_TRITON],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "triton_not_importable"
