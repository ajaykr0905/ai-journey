import base64
from hashlib import sha256
import json

import pytest

from ai_journey.web_extraction_cli import main


def _write_inputs(tmp_path):
    snapshot = tmp_path / "pages.jsonl"
    rows = []
    for index in range(2):
        payload = (
            f"<article>Useful public page {index} with enough text to accept.</article>"
        ).encode()
        rows.append(
            {
                "url": f"https://example.com/page-{index}",
                "content_type": "text/html; charset=utf-8",
                "payload_base64": base64.b64encode(payload).decode(),
                "payload_sha256": sha256(payload).hexdigest(),
            }
        )
    snapshot.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": {
                    "minimum_pages": 2,
                    "minimum_text_characters": 20,
                    "maximum_replacement_fraction": 0.01,
                },
            }
        )
    )
    return snapshot, config


def test_cli_builds_then_independently_verifies_extraction_evidence(tmp_path, capsys):
    snapshot, config = _write_inputs(tmp_path)
    output = tmp_path / "evidence"

    assert (
        main(
            [
                "build",
                "--snapshot",
                str(snapshot),
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    built = json.loads(capsys.readouterr().out)
    assert built["accepted_pages"] == 2
    assert built["rejected_pages"] == 0

    assert main(["verify", "--evidence", str(output)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified == built


def test_cli_reports_invalid_snapshot_without_traceback(tmp_path, capsys):
    snapshot, config = _write_inputs(tmp_path)
    snapshot.write_text("not-json\n")

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "build",
                "--snapshot",
                str(snapshot),
                "--config",
                str(config),
                "--output",
                str(tmp_path / "evidence"),
            ]
        )

    assert "invalid web snapshot JSON" in capsys.readouterr().err
