from dataclasses import asdict
import json
from pathlib import Path

import pytest

from ai_journey.resource_accounting import SEVEN_B_DECODER, ResourceAccountingError
from ai_journey.resource_cli import load_resource_plan, main, verify_main
from ai_journey.resource_report import build_resource_report
from ai_journey.resource_report_io import load_resource_report
from test_resource_report import make_plan

ROOT = Path(__file__).resolve().parents[1]


def test_resource_cli_builds_verified_report(tmp_path: Path, capsys):
    config = tmp_path / "plan.json"
    output = tmp_path / "report.json"
    config.write_text(json.dumps(asdict(make_plan())), encoding="utf-8")

    assert main(["--config", str(config), "--output", str(output)]) == 0

    report = load_resource_report(output)
    assert report["parameters"]["count"] == 7_888
    stdout = capsys.readouterr().out
    assert "fits=true" in stdout
    assert "measured=false" in stdout

    assert verify_main(["--report", str(output)]) == 0
    verified = capsys.readouterr().out
    assert report["integrity"]["digest"] in verified
    assert "measured=false" in verified


def test_pinned_day_51_plan_matches_the_manual_exercise():
    plan = load_resource_plan(ROOT / "config" / "day-51-resource-plan.json")
    assert plan.model == SEVEN_B_DECODER
    assert plan.workload.tokens_per_optimizer_step == 8 * 2_048
    assert plan.workload.total_training_tokens == 16_384_000
    report = build_resource_report(plan)
    assert report["measured"] is False
    assert report["memory"]["capacity"]["fits"] is True
    assert report["cost"]["estimated_cost"] > 0


def test_resource_cli_cannot_replace_its_protected_config(tmp_path: Path):
    config = tmp_path / "plan.json"
    original = (ROOT / "config/day-51-resource-plan.json").read_bytes()
    config.write_bytes(original)

    with pytest.raises(ResourceAccountingError, match="protected config"):
        main(["--config", str(config), "--output", str(config)])

    assert config.read_bytes() == original
    assert load_resource_plan(config).model == SEVEN_B_DECODER
    assert sorted(path.name for path in tmp_path.iterdir()) == ["plan.json"]


@pytest.mark.parametrize(
    "layout",
    [
        "relative",
        "input_symlink",
        "output_symlink",
        "input_parent_symlink",
        "output_parent_symlink",
        "hardlink",
    ],
)
def test_resource_cli_rejects_config_alias_layouts_without_changing_files(
    tmp_path: Path, monkeypatch, layout
):
    config = tmp_path / "plan.json"
    original = (ROOT / "config/day-51-resource-plan.json").read_bytes()
    config.write_bytes(original)
    source, output = config, config
    alias = tmp_path / "alias.json"
    if layout == "relative":
        monkeypatch.chdir(tmp_path)
        source = Path("plan.json")
    elif layout == "input_symlink":
        alias.symlink_to(config)
        source = alias
    elif layout == "output_symlink":
        alias.symlink_to(config)
        output = alias
    elif layout.endswith("parent_symlink"):
        parent = tmp_path / "linked-directory"
        parent.symlink_to(tmp_path, target_is_directory=True)
        if layout == "input_parent_symlink":
            source = parent / "plan.json"
        else:
            output = parent / "plan.json"
    else:
        alias.hardlink_to(config)
        output = alias
    names = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(ResourceAccountingError, match="protected config"):
        main(["--config", str(source), "--output", str(output)])
    assert config.read_bytes() == original
    assert source.read_bytes() == output.read_bytes() == original
    assert source.samefile(output)
    assert sorted(path.name for path in tmp_path.iterdir()) == names


@pytest.mark.parametrize("layout", ["regular", "symlink", "hardlink"])
def test_resource_cli_keeps_disjoint_atomic_output_replacement(tmp_path: Path, layout):
    config, output = tmp_path / "plan.json", tmp_path / "report.json"
    original = (ROOT / "config/day-51-resource-plan.json").read_bytes()
    config.write_bytes(original)
    prior = tmp_path / "prior-output.json"
    prior.write_bytes(b"preserve this independent prior output")
    if layout == "symlink":
        output.symlink_to(prior)
    elif layout == "hardlink":
        output.hardlink_to(prior)
    else:
        output.write_bytes(b"old report may be replaced")
    assert main(["--config", str(config), "--output", str(output)]) == 0
    assert verify_main(["--report", str(output)]) == 0
    assert config.read_bytes() == original
    assert prior.read_bytes() == b"preserve this independent prior output"
    assert not output.is_symlink()
    assert not output.samefile(prior)
