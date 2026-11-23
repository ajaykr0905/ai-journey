"""Tests of the strict model of the recorded Triton stage dump and its fixtures.

None of these tests imports triton. They read the committed ``index.json`` and
the three committed stage texts, and they break copies of them in ``tmp_path``.
"""

import copy
import json
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.triton_ir_manifest import (
    COMMITTED_CONFIG,
    COMMITTED_STAGES,
    COMMITTED_TARGET,
    MANIFEST_NAME,
    RECORDED_MANIFEST_SHA256,
    STAGES,
    TARGETS,
    TritonIrError,
    check_fixtures,
    check_source,
    load_manifest,
    parse_manifest,
    read_manifest,
    require_recorded,
    stage_path,
)
from ai_journey.triton_matmul_kernel import AUTOTUNE_CONFIGS, source_sha256

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "day80_triton_ir"


def raw():
    return json.loads((FIXTURES / MANIFEST_NAME).read_bytes())


def copy_fixtures(tmp_path):
    target = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, target)
    return target


def test_the_recorded_manifest_round_trips_through_the_model():
    payload, manifest = read_manifest(FIXTURES)
    require_recorded(payload)
    assert sha256(payload).hexdigest() == RECORDED_MANIFEST_SHA256
    assert manifest.to_mapping() == json.loads(payload)
    assert (manifest.triton, manifest.python, manifest.machine) == (
        "3.9.0",
        "3.11.18",
        "aarch64",
    )
    assert manifest.fp_fusion == "1"
    assert len(manifest.records) == len(TARGETS) * len(AUTOTUNE_CONFIGS) == 12


def test_the_recorded_source_digest_is_the_digest_of_the_day_79_text():
    manifest = load_manifest(FIXTURES)
    assert check_source(manifest) == source_sha256()
    assert manifest.source_sha256 == (
        "9eb35e273b0fd229e83e223861ef7ce13c4f15ea46d2694baf13920851a31d56"
    )


def test_the_recorded_metadata_of_each_compile():
    manifest = load_manifest(FIXTURES)
    shared = {
        record.name: record.shared_bytes
        for record in manifest.records
        if record.target == "sm_80"
    }
    assert shared == {
        "m128_n128_k32_g8_w4_s4": 98304,
        "m128_n64_k32_g8_w4_s4": 73728,
        "m64_n128_k32_g8_w4_s4": 73728,
        "m128_n32_k32_g8_w4_s4": 61440,
        "m64_n32_k32_g8_w2_s5": 49152,
        "m32_n64_k32_g8_w2_s5": 49152,
    }
    for record in manifest.records:
        twin = manifest.record("sm_90", record.name)
        assert twin.shared_bytes == record.shared_bytes
        assert twin.stage("ttir").sha256 == record.stage("ttir").sha256
    small = manifest.record(COMMITTED_TARGET, COMMITTED_CONFIG)
    assert small.to_mapping()["stages"]["ptx"]["lines"] == 4539
    assert small.cubin_bytes == 200224


def test_the_committed_fixtures_match_their_digests():
    manifest = load_manifest(FIXTURES)
    texts = check_fixtures(manifest, FIXTURES)
    assert tuple(texts) == COMMITTED_STAGES == ("ttir", "ttgir", "ptx")
    lines = {stage: text.record.lines for stage, text in texts.items()}
    assert lines == {"ttir": 210, "ttgir": 336, "ptx": 4539}
    sizes = {stage: text.to_mapping()["bytes"] for stage, text in texts.items()}
    assert sizes == {"ttir": 17027, "ttgir": 32142, "ptx": 197766}
    committed = sorted(
        str(path.relative_to(FIXTURES))
        for path in FIXTURES.rglob("*")
        if path.is_file()
    )
    assert committed == sorted(
        [MANIFEST_NAME]
        + [stage_path(COMMITTED_TARGET, COMMITTED_CONFIG, s) for s in COMMITTED_STAGES]
    )


def test_a_changed_byte_in_a_fixture_is_refused(tmp_path):
    folder = copy_fixtures(tmp_path)
    path = folder / stage_path(COMMITTED_TARGET, COMMITTED_CONFIG, "ptx")
    text = path.read_text()
    path.write_text(text.replace("fma.rn.f32", "fma.rz.f32", 1))
    with pytest.raises(TritonIrError, match="does not match its manifest digest"):
        check_fixtures(load_manifest(folder), folder)


def test_a_missing_or_linked_fixture_is_refused(tmp_path):
    folder = copy_fixtures(tmp_path)
    path = folder / stage_path(COMMITTED_TARGET, COMMITTED_CONFIG, "ttir")
    path.unlink()
    with pytest.raises(TritonIrError, match="readable regular non-symlink"):
        check_fixtures(load_manifest(folder), folder)
    path.symlink_to(FIXTURES / stage_path(COMMITTED_TARGET, COMMITTED_CONFIG, "ttir"))
    with pytest.raises(TritonIrError, match="readable regular non-symlink"):
        check_fixtures(load_manifest(folder), folder)


def test_another_manifest_is_not_the_recorded_one(tmp_path):
    with pytest.raises(TritonIrError, match="not the recorded"):
        require_recorded((FIXTURES / MANIFEST_NAME).read_bytes() + b"\n")


def test_another_kernel_text_is_refused():
    value = raw()
    value["source_sha256"] = "0" * 64
    with pytest.raises(TritonIrError, match="another kernel text"):
        check_source(parse_manifest(value))


def mutate(edit):
    value = copy.deepcopy(raw())
    edit(value)
    return value


def first(value, target="sm_80", index=0):
    return value["targets"][target][index]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda v: v.pop("machine"), "manifest fields"),
        (lambda v: v.update(extra=1), "manifest fields"),
        (lambda v: v.update(triton="3.9"), "triton must be a version"),
        (lambda v: v.update(python=3.11), "python must be a version"),
        (lambda v: v.update(machine="arm 64"), "machine"),
        (lambda v: v.update(TRITON_DEFAULT_FP_FUSION=None), "FP_FUSION"),
        (lambda v: v.update(source_sha256="9EB3" + "0" * 60), "source_sha256"),
        (lambda v: v["targets"].pop("sm_90"), "targets"),
        (lambda v: v["targets"]["sm_80"].pop(), "one record for each"),
        (lambda v: v["targets"]["sm_80"].reverse(), "config must be"),
        (lambda v: first(v).update(shared_bytes=-1), "shared_bytes"),
        (lambda v: first(v).update(shared_bytes=True), "shared_bytes"),
        (lambda v: first(v).update(shared_bytes=300000), "shared_bytes"),
        (lambda v: first(v).update(num_warps=8), "num_warps"),
        (lambda v: first(v).update(num_stages=3), "num_stages"),
        (lambda v: first(v).update(cubin_bytes=0), "cubin_bytes"),
        (lambda v: first(v).update(cubin_sha256="z" * 64), "cubin_sha256"),
        (lambda v: first(v).pop("cubin_sha256"), "record fields"),
        (lambda v: first(v)["config"].update(block_k=64), "config must be"),
        (lambda v: first(v)["stages"].pop("llir"), "must name the stages"),
        (lambda v: first(v)["stages"]["ptx"].update(lines=0), "ptx lines"),
        (lambda v: first(v)["stages"]["ptx"].update(lines=1.0), "ptx lines"),
        (lambda v: first(v)["stages"]["ptx"].update(extra=1), "ptx fields"),
        (
            lambda v: first(v)["stages"]["ptx"].update(file="../../etc/passwd"),
            "dump path",
        ),
        (
            lambda v: first(v)["stages"]["ttir"].update(
                file=first(v)["stages"]["ttgir"]["file"]
            ),
            "dump path",
        ),
    ],
)
def test_a_manifest_the_dump_could_not_write_is_refused(edit, message):
    with pytest.raises(TritonIrError, match=message):
        parse_manifest(mutate(edit))


def test_a_manifest_with_a_duplicate_key_or_too_many_bytes_is_refused(tmp_path):
    folder = tmp_path / "dup"
    folder.mkdir()
    (folder / MANIFEST_NAME).write_text('{"triton": "3.9.0", "triton": "3.9.0"}')
    with pytest.raises(TritonIrError, match="duplicate JSON key"):
        load_manifest(folder)
    (folder / MANIFEST_NAME).write_text(" " * (64 * 1024 + 1))
    with pytest.raises(TritonIrError, match="exceeds"):
        load_manifest(folder)


def test_every_stage_of_every_record_names_its_dump_path():
    manifest = load_manifest(FIXTURES)
    for record in manifest.records:
        assert [stage.stage for stage in record.stages] == list(STAGES)
        for stage in record.stages:
            assert stage.file == stage_path(record.target, record.name, stage.stage)
    with pytest.raises(TritonIrError, match="no sm_90 m1"):
        manifest.record("sm_90", "m1")
