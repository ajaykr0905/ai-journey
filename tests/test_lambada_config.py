import json
from pathlib import Path

from ai_journey.lambada import LambadaProtocol, protocol_mapping


def test_pinned_protocol_identifies_public_source_and_exact_slice():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "config/lambada-test.json").read_text())
    protocol = LambadaProtocol.from_mapping(payload)
    assert protocol_mapping(protocol) == payload
    assert protocol.source.revision == "900124bf3b8235c6daf21033af9948b3f07346c4"
    assert protocol.source.sample_sha256 == (
        "b304b6afffcc2d23ba8b6a67d16df47c60f7511f8ea2ef60016d331b3a485e15"
    )
    assert protocol.start == 0
    assert protocol.count == 100
    assert len(protocol.fingerprint) == 64
