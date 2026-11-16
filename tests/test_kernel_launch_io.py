import os
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.kernel_launch_config import KernelLaunchError
from ai_journey.kernel_launch_io import (
    canonical_json_bytes,
    canonical_sha256,
    loads_strict,
    paths_alias,
    read_bounded,
    write_atomic,
)


def test_canonical_json_is_sorted_compact_ascii_and_finite():
    value = {"b": [1, 2.5], "a": "é", "c": {"z": None, "y": True}}

    assert canonical_json_bytes(value) == (
        b'{"a":"\\u00e9","b":[1,2.5],"c":{"y":true,"z":null}}'
    )
    assert canonical_sha256(value) == sha256(canonical_json_bytes(value)).hexdigest()
    for bad in (float("nan"), float("inf"), {1, 2}, b"bytes"):
        with pytest.raises(KernelLaunchError, match="canonical JSON"):
            canonical_json_bytes({"value": bad})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"a":1,"a":2}', "duplicate JSON key: a"),
        (b'{"a":{"b":1,"b":1}}', "duplicate JSON key: b"),
        (b'{"a":NaN}', "non-finite JSON constant: NaN"),
        (b'{"a":-Infinity}', "non-finite JSON constant: -Infinity"),
        (b'{"a":1e999}', "non-finite JSON number: 1e999"),
        (b'{"a":' + b"1" * 5_000 + b"}", "integer literal too long"),
        (b'{"a":"\xff"}', "not strict UTF-8 JSON"),
        (b'{"a":', "not strict UTF-8 JSON"),
    ],
)
def test_strict_loading_rejects_ambiguous_json(payload, message):
    with pytest.raises(KernelLaunchError, match=message):
        loads_strict(payload, label="plan")


def test_strict_loading_keeps_valid_values_and_needs_bytes():
    assert loads_strict(b'{"n":1000,"x":[0.5,true,null]}', label="plan") == {
        "n": 1000,
        "x": [0.5, True, None],
    }
    with pytest.raises(TypeError, match="plan must be bytes"):
        loads_strict('{"n":1}', label="plan")


def test_bounded_reads_accept_only_small_regular_files(tmp_path: Path):
    plain = tmp_path / "plan.json"
    plain.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(plain)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)

    assert read_bounded(plain, limit=2, label="plan") == b"{}"
    with pytest.raises(KernelLaunchError, match="exceeds 1 bytes"):
        read_bounded(plain, limit=1, label="plan")
    for path in (link, tmp_path / "missing.json"):
        with pytest.raises(KernelLaunchError, match="readable regular non-symlink"):
            read_bounded(path, limit=64, label="plan")
    for path in (fifo, tmp_path):
        with pytest.raises(KernelLaunchError, match="regular"):
            read_bounded(path, limit=64, label="plan")
    with pytest.raises(TypeError, match="pathlib.Path"):
        read_bounded(str(plain), limit=64, label="plan")
    with pytest.raises(KernelLaunchError, match="limit must be an integer"):
        read_bounded(plain, limit=0, label="plan")


def test_atomic_writes_replace_the_target_and_leave_no_temporary(tmp_path: Path):
    target = tmp_path / "out" / "nested" / "report.json"

    write_atomic(target, b"first\n")
    write_atomic(target, b"second\n")

    assert target.read_bytes() == b"second\n"
    assert sorted(path.name for path in target.parent.iterdir()) == ["report.json"]


def test_atomic_writes_refuse_protected_inputs_and_odd_targets(tmp_path: Path):
    plan = tmp_path / "plan.json"
    plan.write_bytes(b"{}")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(plan)
    hard = tmp_path / "hard.json"
    os.link(plan, hard)
    loop = tmp_path / "loop"
    loop.symlink_to("loop")

    for target in (plan, symlink, hard):
        with pytest.raises(KernelLaunchError, match="alias a protected input"):
            write_atomic(target, b"x", protected=[plan])
    assert paths_alias(hard, plan) and not paths_alias(loop, plan)
    with pytest.raises(KernelLaunchError, match="regular non-symlink"):
        write_atomic(symlink, b"x")
    with pytest.raises(KernelLaunchError, match="regular non-symlink"):
        write_atomic(tmp_path, b"x")
    with pytest.raises(KernelLaunchError, match="regular non-symlink"):
        write_atomic(loop, b"x")
    with pytest.raises(OSError):
        write_atomic(loop / "report.json", b"x")
    for target in (Path("."), tmp_path / ".."):
        with pytest.raises(KernelLaunchError, match="must name a file"):
            write_atomic(target, b"x")
    with pytest.raises(TypeError, match="payload must be bytes"):
        write_atomic(tmp_path / "report.json", "text")
    with pytest.raises(TypeError, match="protected paths"):
        write_atomic(tmp_path / "report.json", b"x", protected=["plan.json"])
    assert plan.read_bytes() == b"{}"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "hard.json",
        "loop",
        "plan.json",
        "symlink.json",
    ]
