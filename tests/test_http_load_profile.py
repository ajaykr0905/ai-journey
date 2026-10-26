import pytest

from ai_journey.http_load_profile import (
    HTTPWorkloadProfile,
    http_profile_from_mapping,
    http_request_schedule,
)


def test_profile_roundtrip_and_deterministic_schedule_preserve_fresh_identity():
    profile = HTTPWorkloadProfile(prompts=((1, 2), (3,)), cancel_every=2)
    assert http_profile_from_mapping(profile.to_mapping()) == profile
    assert (
        http_profile_from_mapping(profile.to_mapping()).fingerprint()
        == profile.fingerprint()
    )
    first = http_request_schedule(profile, "a" * 32)
    assert first == http_request_schedule(profile, "a" * 32)
    assert [row["request"]["prompt"] for row in first[:3]] == [[1, 2], [3], [1, 2]]
    assert [row["cancel_after_tokens"] for row in first[:3]] == [None, 1, None]
    warmup = http_request_schedule(profile, "a" * 32, phase="warmup")
    ids = {row["request"]["request_id"] for row in first + warmup}
    assert len(ids) == profile.measured_requests + profile.warmup_requests
    assert ids.isdisjoint(
        row["request"]["request_id"] for row in http_request_schedule(profile, "b" * 32)
    )


@pytest.mark.parametrize(
    "controls",
    [
        {"measured_requests": True},
        {"concurrency": 9},
        {"timeout_seconds": float("nan")},
        {"prompts": ((True,),)},
        {"streaming": False, "cancel_every": 2},
        {"max_new_tokens": 1, "cancel_every": 1},
        {"prompts": (tuple(range(64)),), "measured_requests": 128},
        {"warmup_requests": 16, "timeout_seconds": 10.0},
    ],
)
def test_profile_rejects_invalid_and_excessive_work(controls):
    with pytest.raises(ValueError):
        HTTPWorkloadProfile(**controls)
