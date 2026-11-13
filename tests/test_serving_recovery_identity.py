import re

import pytest
from test_serving_recovery_config import valid_config

from ai_journey.serving_recovery_identity import (
    idempotency_key,
    key_request_index,
    lifetime_number,
    require_lifetime_id,
    require_run_key,
    server_lifetime_id,
    serving_run_id,
)

MODEL = "a" * 64


def test_run_identity_is_readable_and_bound_to_config_and_model():
    run = serving_run_id(valid_config(), model_sha256=MODEL)

    assert run.startswith("serving-recovery-test-")
    assert len(run.rsplit("-", 1)[-1]) == 16
    assert run == serving_run_id(valid_config(), model_sha256=MODEL)
    assert run != serving_run_id(valid_config(seed=71), model_sha256=MODEL)
    assert run != serving_run_id(valid_config(), model_sha256="b" * 64)


def test_lifetime_identities_are_numbered_and_bound_to_the_run():
    run = serving_run_id(valid_config(), model_sha256=MODEL)
    other = serving_run_id(valid_config(seed=71), model_sha256=MODEL)
    first = server_lifetime_id(run, 1)
    second = server_lifetime_id(run, 2)

    assert first.startswith("lifetime-1-")
    assert second.startswith("lifetime-2-")
    assert first != second != server_lifetime_id(other, 2)
    assert lifetime_number(second) == 2
    assert require_lifetime_id(first) == first


def test_idempotency_keys_are_stable_safe_and_bound_to_run_and_index():
    run = serving_run_id(valid_config(), model_sha256=MODEL)
    other = serving_run_id(valid_config(seed=71), model_sha256=MODEL)
    keys = [idempotency_key(run, index) for index in range(24)]

    assert keys == [idempotency_key(run, index) for index in range(24)]
    assert len(set(keys)) == 24
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) for key in keys)
    assert [key_request_index(key) for key in keys] == list(range(24))
    assert require_run_key(run, keys[7]) == 7
    with pytest.raises(ValueError, match="does not belong"):
        require_run_key(other, keys[7])


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: serving_run_id(valid_config(), model_sha256="A" * 64), "SHA-256"),
        (
            lambda: serving_run_id(valid_config().to_mapping(), model_sha256=MODEL),
            "ServingRecoveryConfig",
        ),
        (lambda: server_lifetime_id("bad", 1), "run_id"),
        (lambda: server_lifetime_id("lab-0123456789abcdef", True), "integer"),
        (lambda: server_lifetime_id("lab-0123456789abcdef", 0), "lifetime_number"),
        (lambda: server_lifetime_id("lab-0123456789abcdef", 65), "lifetime_number"),
        (lambda: idempotency_key("lab-0123456789abcdef", -1), "request_index"),
        (lambda: idempotency_key("lab-0123456789abcdef", 10_000), "request_index"),
        (lambda: lifetime_number("lifetime-0-0123456789abcdef"), "lifetime_id"),
        (lambda: lifetime_number("lifetime-70-0123456789abcdef"), "range"),
        (lambda: key_request_index("req-1-" + "0" * 24), "idempotency key"),
        (lambda: key_request_index(b"req-0001-" + b"0" * 24), "idempotency key"),
    ],
)
def test_identities_reject_ambiguous_values(call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call()
