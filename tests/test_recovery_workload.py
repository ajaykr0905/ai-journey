import pytest
from test_recovery_config import valid_config

from ai_journey.recovery_workload import build_recovery_workload


def test_workload_is_ordered_bounded_and_content_private():
    source = b"public recovery fixture\n"

    workload = build_recovery_workload(source, valid_config())

    assert workload.source_bytes == len(source)
    assert len(workload.source_sha256) == 64
    assert len(workload.items) == valid_config().work_items
    assert [item.index for item in workload.items] == list(range(20))
    assert len({item.payload_sha256 for item in workload.items}) == 20
    assert source.decode().strip() not in str(workload)
    assert len(workload.fingerprint()) == 64
    assert (
        workload.fingerprint()
        == build_recovery_workload(source, valid_config()).fingerprint()
    )


def test_workload_identity_changes_with_source_or_item_count():
    first = build_recovery_workload(b"alpha", valid_config())
    second = build_recovery_workload(b"beta", valid_config())
    changed = valid_config().to_mapping()
    changed["work_items"] = 21

    assert first.source_sha256 != second.source_sha256
    assert first.items != second.items
    assert first.fingerprint() != second.fingerprint()


@pytest.mark.parametrize("source", [b"", b"12345"])
def test_workload_rejects_empty_or_oversized_source(source):
    with pytest.raises(ValueError, match="non-empty and bounded"):
        build_recovery_workload(source, valid_config(), max_source_bytes=4)


def test_workload_requires_bytes_and_integer_bound():
    with pytest.raises(TypeError, match="source"):
        build_recovery_workload("text", valid_config())
    with pytest.raises(TypeError, match="integer"):
        build_recovery_workload(b"text", valid_config(), max_source_bytes=True)
