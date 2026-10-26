from argparse import Namespace
import os

import pytest

from ai_journey import ddp_operations


def arguments(tmp_path, command):
    for name in ("source", "spec", "checkpoint"):
        (tmp_path / name).write_bytes(b"preserve original " + name.encode())
    return Namespace(
        command=command,
        source=tmp_path / "source",
        spec=tmp_path / "spec",
        checkpoint=tmp_path / "checkpoint",
        output=tmp_path / "report.json",
        workdir=tmp_path / "work",
        steps=1,
        repeats=1,
    )


@pytest.mark.parametrize("command", ["benchmark", "matrix", "pipeline"])
@pytest.mark.parametrize("alias", ["source", "spec", "hardlink", "symlink"])
def test_operation_aliases_reject_before_runtime_work(
    tmp_path, monkeypatch, command, alias
):
    args = arguments(tmp_path, command)
    if alias in ("source", "spec"):
        args.output = getattr(args, alias)
    elif alias == "hardlink":
        os.link(args.source, args.output)
    else:
        args.output.symlink_to(args.source)

    def unexpected(*args, **kwargs):
        raise AssertionError("unsafe paths must reject before changing runtime")

    monkeypatch.setattr(ddp_operations.torch, "set_num_threads", unexpected)
    before = args.source.read_bytes()
    with pytest.raises(ValueError, match="output"):
        ddp_operations.run_operation(args)
    assert args.source.read_bytes() == before


def test_benchmark_cannot_replace_its_checkpoint(tmp_path):
    args = arguments(tmp_path, "benchmark")
    args.output = args.checkpoint
    with pytest.raises(ValueError, match="aliases"):
        ddp_operations.run_operation(args)
    assert args.checkpoint.read_bytes() == b"preserve original checkpoint"


@pytest.mark.parametrize("command", ["matrix", "pipeline"])
@pytest.mark.parametrize(
    "layout", ["nonempty", "report_inside", "input_inside", "symlink"]
)
def test_workdir_admission_preserves_unrelated_files_and_inputs(
    tmp_path, command, layout
):
    args = arguments(tmp_path, command)
    args.workdir.mkdir()
    if layout == "nonempty":
        (args.workdir / "user-work").write_bytes(b"keep")
    elif layout == "report_inside":
        args.output = args.workdir / "trained.pt"
    elif layout == "input_inside":
        args.workdir = tmp_path
    else:
        alias = tmp_path / "work-alias"
        alias.symlink_to(args.workdir, target_is_directory=True)
        args.workdir = alias
    with pytest.raises(ValueError, match="workdir"):
        ddp_operations.run_operation(args)
    assert args.source.read_bytes() == b"preserve original source"
    if layout == "nonempty":
        assert (args.workdir / "user-work").read_bytes() == b"keep"
