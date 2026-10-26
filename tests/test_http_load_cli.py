import json
from pathlib import Path
import tomllib

import pytest

from ai_journey.http_load_cli import main
from ai_journey.http_load_profile import HTTPWorkloadProfile
from test_http_measurements import running_service


def test_cli_entry_point_generates_real_evidence_and_verifies_without_mutation(
    tmp_path,
):
    package = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    assert (
        package["project"]["scripts"]["ai-journey-http-load"]
        == "ai_journey.http_load_cli:main"
    )
    profile = tmp_path / "profile.json"
    report = tmp_path / "report.json"
    assert main(["profile", "--output", str(profile)]) == 0
    profile.write_text(
        json.dumps(
            HTTPWorkloadProfile(
                measured_requests=2, warmup_requests=0, concurrency=2, max_new_tokens=2
            ).to_mapping()
        )
    )
    with running_service() as (endpoint, service):
        assert (
            main(
                [
                    "measure",
                    "--endpoint",
                    endpoint,
                    "--profile",
                    str(profile),
                    "--output",
                    str(report),
                ]
            )
            == 0
        )
    original = report.read_bytes()
    assert main(["verify", "--report", str(report)]) == 0
    assert report.read_bytes() == original
    with pytest.raises(SystemExit):
        main(["measure", "--endpoint", endpoint, "--output", str(report)])
    assert report.read_bytes() == original


def test_cli_rejects_symlinked_oversized_and_tampered_reports(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("{}")
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)
    with pytest.raises(SystemExit):
        main(["verify", "--report", str(linked)])
    target.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(SystemExit):
        main(["verify", "--report", str(target)])
