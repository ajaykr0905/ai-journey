import base64
from hashlib import sha256
import json

import pytest

from ai_journey.bpe import TokenizerError
from ai_journey.web_snapshot import load_web_snapshot


def _row(url="https://example.com/page", payload=b"<p>public page</p>"):
    return {
        "url": url,
        "content_type": "text/html; charset=utf-8",
        "payload_base64": base64.b64encode(payload).decode(),
        "payload_sha256": sha256(payload).hexdigest(),
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_snapshot_loads_verified_pages_in_declared_order(tmp_path):
    path = tmp_path / "pages.jsonl"
    _write(path, [_row(), _row("https://example.com/next", b"<p>next</p>")])

    pages = load_web_snapshot(path)

    assert [page.url for page in pages] == [
        "https://example.com/page",
        "https://example.com/next",
    ]
    assert pages[1].payload == b"<p>next</p>"


def test_snapshot_rejects_digest_drift(tmp_path):
    path = tmp_path / "pages.jsonl"
    row = _row()
    row["payload_sha256"] = "0" * 64
    _write(path, [row])

    with pytest.raises(TokenizerError, match="digest mismatch"):
        load_web_snapshot(path)


def test_snapshot_rejects_duplicate_urls_and_duplicate_fields(tmp_path):
    path = tmp_path / "pages.jsonl"
    _write(path, [_row(), _row()])
    with pytest.raises(TokenizerError, match="duplicate page URLs"):
        load_web_snapshot(path)

    path.write_text(
        '{"url":"https://example.com/","url":"https://example.com/",'
        '"content_type":"text/html","payload_base64":"eA==",'
        f'"payload_sha256":"{sha256(b"x").hexdigest()}"}}\n'
    )
    with pytest.raises(TokenizerError, match="duplicate web snapshot field"):
        load_web_snapshot(path)


@pytest.mark.parametrize("payload", [b"", b"{}", b"{}\n\n"])
def test_snapshot_requires_nonempty_newline_terminated_jsonl(tmp_path, payload):
    path = tmp_path / "pages.jsonl"
    path.write_bytes(payload)

    with pytest.raises(TokenizerError):
        load_web_snapshot(path)


def test_snapshot_rejects_symlinks(tmp_path):
    target = tmp_path / "target.jsonl"
    _write(target, [_row()])
    link = tmp_path / "pages.jsonl"
    link.symlink_to(target)

    with pytest.raises(TokenizerError, match="nonsymlink"):
        load_web_snapshot(link)
