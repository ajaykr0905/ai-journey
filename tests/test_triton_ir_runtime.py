"""Tests of the triton compiler backend and of the digest comparison with the dump.

Most tests use stand-ins for triton, so they run on macOS, where triton has no
wheel. One test compiles the committed config with the real triton and skips
cleanly when triton cannot be imported.
"""

import importlib.util
import json
import sys
import types
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.triton_ir_manifest import (
    COMMITTED_CONFIG,
    COMMITTED_STAGES,
    COMMITTED_TARGET,
    STAGES,
    TritonIrError,
    check_fixtures,
    load_manifest,
    parse_manifest,
)
from ai_journey.triton_ir_runtime import (
    ENVIRONMENT_FIELDS,
    CompiledStages,
    CompilerBackend,
    TritonIrUnavailableError,
    regenerate,
    signature,
)
from ai_journey.triton_matmul_kernel import CONFIGS_BY_NAME

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "day80_triton_ir"
ENVIRONMENT = dict.fromkeys(ENVIRONMENT_FIELDS, "stand-in")


class FixtureBackend:
    """Return the committed texts for the committed config, other text elsewhere."""

    name = "fixture"
    environment = ENVIRONMENT

    def __init__(self, cubin=b"cubin"):
        manifest = load_manifest(FIXTURES)
        self.texts = {
            stage: fixture.text
            for stage, fixture in check_fixtures(manifest, FIXTURES).items()
        }
        self.manifest = manifest
        self.cubin = cubin
        self.calls = []

    def compile(self, target, config):
        self.calls.append((target, config.name))
        record = self.manifest.record(target, config.name)
        committed = (target, config.name) == (COMMITTED_TARGET, COMMITTED_CONFIG)
        texts = {
            stage: self.texts[stage]
            if committed and stage in COMMITTED_STAGES
            else f"{target} {config.name} {stage}\n"
            for stage in STAGES
        }
        return CompiledStages(
            texts=texts,
            shared_bytes=record.shared_bytes,
            num_warps=record.num_warps,
            num_stages=record.num_stages,
            cubin=self.cubin,
        )


def test_the_signature_is_the_signature_of_the_dump():
    assert signature() == {
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "c_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
        "GROUP_M": "constexpr",
    }


def test_regenerate_compiles_every_record_and_matches_only_equal_texts():
    backend = FixtureBackend()
    result = regenerate(load_manifest(FIXTURES), backend)
    assert len(backend.calls) == 12
    assert result["backend"] == "injected"
    assert result["environment"] == ENVIRONMENT
    summary = result["summary"]
    assert summary == {
        "texts_matched": 3,
        "texts": 48,
        "metadata_matched": 12,
        "compiles": 12,
        "cubins_matched": 0,
        "cubins_by_target": {"sm_80": 0, "sm_90": 0},
        "all_texts_and_metadata_match": False,
    }
    row = result["compiles"][5]
    assert (row["target"], row["config"]) == (COMMITTED_TARGET, COMMITTED_CONFIG)
    assert [row["stages"][stage]["match"] for stage in STAGES] == [
        True,
        True,
        False,
        True,
    ]
    assert row["stages"]["ptx"]["regenerated_lines"] == 4539
    assert row["cubin"]["regenerated_sha256"] == sha256(b"cubin").hexdigest()


def test_regenerate_passes_when_every_text_and_metadata_match():
    backend = FixtureBackend()
    value = json.loads((FIXTURES / "index.json").read_bytes())
    for target, rows in value["targets"].items():
        for row in rows:
            for stage, entry in row["stages"].items():
                text = backend.compile(target, CONFIGS_BY_NAME[row["config"]["name"]])
                body = text.texts[stage]
                entry["sha256"] = sha256(body.encode()).hexdigest()
                entry["lines"] = len(body.splitlines())
    result = regenerate(parse_manifest(value), FixtureBackend())
    assert result["summary"]["all_texts_and_metadata_match"] is True
    assert result["summary"]["texts_matched"] == 48


def test_a_changed_metadata_breaks_the_metadata_match():
    class Wider(FixtureBackend):
        def compile(self, target, config):
            compiled = super().compile(target, config)
            return CompiledStages(compiled.texts, compiled.shared_bytes + 4, 2, 5, b"")

    result = regenerate(load_manifest(FIXTURES), Wider())
    assert result["summary"]["metadata_matched"] == 0


def test_a_backend_must_name_its_environment_and_return_compiled_stages():
    class Bare(FixtureBackend):
        environment = {"python": "3.11"}

    with pytest.raises(TritonIrError, match="must name its environment"):
        regenerate(load_manifest(FIXTURES), Bare())

    class Loose(FixtureBackend):
        def compile(self, target, config):
            return {"ttir": ""}

    with pytest.raises(TritonIrError, match="must return CompiledStages"):
        regenerate(load_manifest(FIXTURES), Loose())
    with pytest.raises(TypeError, match="TritonIrManifest"):
        regenerate({}, FixtureBackend())


@pytest.mark.parametrize(
    ("texts", "message"),
    [
        ({"ttir": "", "ttgir": "", "llir": ""}, "stages"),
        ({"ttgir": "", "ttir": "", "llir": "", "ptx": ""}, "stages"),
        ({"ttir": b"", "ttgir": "", "llir": "", "ptx": ""}, "string"),
    ],
)
def test_compiled_stages_must_hold_four_texts_in_order(texts, message):
    with pytest.raises(TritonIrError, match=message):
        CompiledStages(texts, 0, 2, 5, b"")
    with pytest.raises(TritonIrError, match="cubin must be bytes"):
        CompiledStages(dict.fromkeys(STAGES, ""), 0, 2, 5, "x")
    with pytest.raises(TritonIrError, match="shared_bytes"):
        CompiledStages(dict.fromkeys(STAGES, ""), -1, 2, 5, b"")


def stand_in_triton(monkeypatch, kind):
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
        metadata = types.SimpleNamespace(shared=49152, num_warps=2, num_stages=5)
        asm = {stage: f"{stage}\n" for stage in STAGES}
        asm["cubin"] = b"\x7fELF"
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


def test_the_backend_compiles_with_the_declared_target_and_options(monkeypatch):
    monkeypatch.delenv("TRITON_DEFAULT_FP_FUSION", raising=False)
    calls = stand_in_triton(monkeypatch, "JITFunction")
    backend = CompilerBackend()
    assert backend.environment["triton"] == "0.0.0"
    assert backend.environment["fp_fusion"] == "1"
    compiled = backend.compile("sm_90", CONFIGS_BY_NAME[COMMITTED_CONFIG])
    (source, target, options) = calls[0]
    assert target == ("cuda", 90, 32)
    assert options == {"num_warps": 2, "num_stages": 5}
    assert source["constexprs"] == {
        "BLOCK_M": 32,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "GROUP_M": 8,
    }
    assert source["signature"] == signature()
    assert compiled.texts == {stage: f"{stage}\n" for stage in STAGES}
    assert (compiled.shared_bytes, compiled.cubin) == (49152, b"\x7fELF")


def test_the_backend_refuses_the_interpreter(monkeypatch):
    monkeypatch.delenv("TRITON_DEFAULT_FP_FUSION", raising=False)
    stand_in_triton(monkeypatch, "InterpretedFunction")
    with pytest.raises(TritonIrUnavailableError) as raised:
        CompilerBackend()
    assert raised.value.reason == "interpreter_enabled"


def test_the_backend_names_a_missing_triton(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    with pytest.raises(TritonIrUnavailableError) as raised:
        CompilerBackend()
    assert raised.value.reason == "triton_not_importable"
    with pytest.raises(ValueError, match="reason must be one of"):
        TritonIrUnavailableError("other", "x")


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="triton is not importable on this host; it has no macOS wheel",
)
def test_the_real_triton_regenerates_the_committed_texts(monkeypatch):
    monkeypatch.setenv("TRITON_INTERPRET", "0")
    monkeypatch.setenv("TRITON_DEFAULT_FP_FUSION", "1")
    backend = CompilerBackend()
    manifest = load_manifest(FIXTURES)
    record = manifest.record(COMMITTED_TARGET, COMMITTED_CONFIG)
    compiled = backend.compile(COMMITTED_TARGET, record.config)
    for stage in COMMITTED_STAGES:
        digest = sha256(compiled.texts[stage].encode()).hexdigest()
        assert digest == record.stage(stage).sha256, stage
    assert compiled.shared_bytes == record.shared_bytes
