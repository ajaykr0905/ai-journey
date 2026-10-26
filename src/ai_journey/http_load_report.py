"""Sealed HTTP evidence with independently recomputed timing and work totals."""

import copy
from datetime import datetime
from hashlib import sha256
import json
import math
import re

from .http_load import HTTP_LOAD_SCOPE, OUTCOMES, http_load_totals
from .http_load_profile import http_profile_from_mapping, http_request_schedule
from .http_measurements import loopback_address

REPORT_SCHEMA = "ai-journey-http-load-report-v1"
PERCENTILE_METHOD = (
    "nearest rank: sorted samples[ceil(p*n)-1]; empty distributions use null"
)
VERIFICATION_SCOPE = "internal consistency of recorded client observations; a digest cannot authenticate a timing measurement"
_OBSERVATION_FIELDS = {
    "request_id",
    "route",
    "started_ns",
    "headers_ns",
    "finished_ns",
    "status",
    "outcome",
    "tokens",
    "token_received_ns",
    "model_sha256",
    "finish_reason",
    "error_kind",
}


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid measured {name}")
    return value


def _distribution(values):
    values = sorted(values)
    return {
        "count": len(values),
        "minimum": values[0] if values else None,
        "maximum": values[-1] if values else None,
        "mean": sum(values) / len(values) if values else None,
        **{
            f"p{percent}": values[math.ceil(percent / 100 * len(values)) - 1]
            if values
            else None
            for percent in (50, 95, 99)
        },
    }


def _validate_outcome(observation, spec, profile):
    """Accept only states produced by the collector, including finally overrides."""

    outcome, status = observation["outcome"], observation["status"]
    tokens = observation["tokens"]
    error = observation["error_kind"]
    if error is not None and (
        type(error) is not str
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", error) is None
    ):
        raise ValueError("invalid measured error classification")
    model, reason = observation["model_sha256"], observation["finish_reason"]
    terminal = model is not None or reason is not None
    if terminal:
        if (
            status != 200
            or type(model) is not str
            or re.fullmatch(r"[0-9a-f]{64}", model) is None
            or type(reason) is not str
            or reason not in {"length", "eos", "cancelled", "deadline"}
        ):
            raise ValueError(
                "HTTP terminal metadata requires a valid HTTP 200 generation output"
            )
        if reason == "length" and len(tokens) != profile.max_new_tokens:
            raise ValueError(
                "completed output does not match the requested token budget"
            )
    if tokens and (status != 200 or (not profile.streaming and not terminal)):
        raise ValueError(
            "output tokens require a received successful generation response"
        )
    cancellation = spec["cancel_after_tokens"]
    if cancellation is not None:
        if len(tokens) > cancellation or (terminal and len(tokens) >= cancellation):
            raise ValueError(
                "observation continued past its planned client cancellation"
            )
        if len(tokens) == cancellation and not (
            (outcome == "cancelled" and error == "client_closed_stream")
            or (outcome == "deadline" and error == "client_request_deadline")
        ):
            raise ValueError(
                "reaching the client cancellation boundary must stop receipt"
            )
    elapsed = observation["finished_ns"] - observation["started_ns"]
    if elapsed >= profile.timeout_seconds * 1e9 and (
        outcome != "deadline" or error != "client_request_deadline"
    ):
        raise ValueError("elapsed client deadline must override the recorded outcome")

    if outcome == "completed":
        if not terminal or reason not in {"length", "eos"} or error is not None:
            raise ValueError("completed HTTP observation lacks valid terminal evidence")
    elif outcome == "cancelled":
        if error == "client_closed_stream":
            if (
                status != 200
                or not profile.streaming
                or cancellation is None
                or len(tokens) != cancellation
                or terminal
            ):
                raise ValueError(
                    "client cancellation does not match its scheduled stream boundary"
                )
        elif not terminal or reason != "cancelled" or error is not None:
            raise ValueError(
                "server cancellation requires a validated cancelled terminal"
            )
    elif outcome == "deadline":
        if error is None:
            if not terminal or reason != "deadline":
                raise ValueError(
                    "server deadline requires a validated deadline terminal"
                )
        elif error == "TimeoutError":
            if terminal:
                raise ValueError("socket timeout cannot include a validated terminal")
        elif error != "client_request_deadline":
            raise ValueError("deadline observation has an invalid error classification")
    elif outcome in {"rejected", "http_error"}:
        if terminal or tokens or error is not None:
            raise ValueError(
                "HTTP rejection/error responses cannot carry generation tokens or terminal metadata"
            )
        if outcome == "rejected" and status not in {429, 503}:
            raise ValueError("rejected requests require HTTP 429 or 503")
        if outcome == "http_error" and (status is None or status in {200, 429, 503}):
            raise ValueError(
                "HTTP error classification contradicts the received status"
            )
    elif outcome == "protocol_error":
        if terminal or error not in {
            "StreamProtocolError",
            "ValueError",
            "TypeError",
            "server_error_event",
        }:
            raise ValueError(
                "protocol failure cannot carry validated terminal metadata"
            )
        if error == "server_error_event" and (status != 200 or not profile.streaming):
            raise ValueError(
                "server error events require a successful streaming response"
            )
    elif outcome == "transport_error":
        if (
            terminal
            or error is None
            or error
            in {
                "TimeoutError",
                "client_request_deadline",
                "client_closed_stream",
                "StreamProtocolError",
                "ValueError",
                "TypeError",
                "server_error_event",
            }
        ):
            raise ValueError(
                "transport failure requires a transport exception without terminal metadata"
            )


def _validate_observation(observation, spec, profile):
    if type(observation) is not dict or set(observation) != _OBSERVATION_FIELDS:
        raise ValueError("HTTP observation fields differ from the schema")
    if observation["request_id"] != spec["request"]["request_id"] or observation[
        "route"
    ] != ("/stream" if profile.streaming else "/generate"):
        raise ValueError("HTTP observation does not match its scheduled request")
    started = _integer(observation["started_ns"], "started_ns")
    finished = _integer(observation["finished_ns"], "finished_ns")
    if finished <= started or observation["outcome"] not in OUTCOMES:
        raise ValueError("HTTP observation duration or outcome is invalid")
    status = observation["status"]
    headers = observation["headers_ns"]
    if status is not None:
        if (
            type(status) is not int
            or not 100 <= status <= 599
            or type(headers) is not int
            or not started <= headers <= finished
        ):
            raise ValueError("HTTP response headers or status are invalid")
    elif headers is not None:
        raise ValueError("HTTP headers require a received status")
    tokens, times = observation["tokens"], observation["token_received_ns"]
    if (
        type(tokens) is not list
        or len(tokens) > profile.max_new_tokens
        or any(type(token) is not int or not 0 <= token < 2**31 for token in tokens)
    ):
        raise ValueError("invalid measured output token ids")
    if (
        type(times) is not list
        or any(
            type(value) is not int
            or headers is None
            or not headers <= value <= finished
            for value in times
        )
        or times != sorted(times)
    ):
        raise ValueError("token receipt timestamps are outside the request window")
    if (profile.streaming and len(times) != len(tokens)) or (
        not profile.streaming and times
    ):
        raise ValueError("token receipt timestamps differ from the transport semantics")
    _validate_outcome(observation, spec, profile)


def validate_http_measurement(raw):
    fields = {
        "schema",
        "scope",
        "created_at_utc",
        "endpoint",
        "run_id",
        "profile",
        "profile_sha256",
        "warmup",
        "measured",
        "work_started_ns",
        "work_finished_ns",
        "totals",
    }
    if (
        type(raw) is not dict
        or set(raw) != fields
        or raw["schema"] != "ai-journey-http-load-raw-v1"
        or raw["scope"] != HTTP_LOAD_SCOPE
    ):
        raise ValueError("HTTP measurement fields or scope differ from the schema")
    profile = http_profile_from_mapping(raw["profile"])
    if raw["profile_sha256"] != profile.fingerprint():
        raise ValueError("HTTP workload profile digest differs")
    loopback_address(raw["endpoint"])
    created = datetime.fromisoformat(raw["created_at_utc"])
    if created.tzinfo is None or created.utcoffset().total_seconds() != 0:
        raise ValueError("HTTP measurement creation time must use UTC")
    started = _integer(raw["work_started_ns"], "work_started_ns")
    finished = _integer(raw["work_finished_ns"], "work_finished_ns")
    if finished <= started:
        raise ValueError("HTTP work window must be positive")
    models = set()
    intervals = []
    for phase in ("warmup", "measured"):
        rows = raw[phase]
        schedule = http_request_schedule(profile, raw["run_id"], phase=phase)
        if type(rows) is not list or len(rows) != len(schedule):
            raise ValueError("HTTP report omitted scheduled attempts")
        for row, spec in zip(rows, schedule, strict=True):
            if (
                type(row) is not dict
                or set(row) != {"index", "observation"}
                or type(row["index"]) is not int
                or row["index"] != spec["index"]
            ):
                raise ValueError("HTTP observation index differs from its schedule")
            observation = row["observation"]
            _validate_observation(observation, spec, profile)
            if phase == "measured":
                if (
                    not started
                    <= observation["started_ns"]
                    < observation["finished_ns"]
                    <= finished
                ):
                    raise ValueError("measured request lies outside its work window")
                intervals.extend(
                    ((observation["started_ns"], 1), (observation["finished_ns"], -1))
                )
            elif observation["finished_ns"] > started:
                raise ValueError("warmup overlaps the measured work window")
            if observation["model_sha256"] is not None:
                models.add(observation["model_sha256"])
    if len(models) > 1:
        raise ValueError("HTTP load observations mix different model identities")
    active = 0
    for _, change in sorted(intervals):
        active += change
        if active > profile.concurrency:
            raise ValueError("observed HTTP concurrency exceeds the declared budget")
    expected = http_load_totals(
        raw["measured"], work_started_ns=started, work_finished_ns=finished
    )
    if _canonical(raw["totals"]) != _canonical(expected):
        raise ValueError(
            "HTTP request counts, output tokens or work denominator differ"
        )
    return profile


def http_latency_summary(raw):
    validate_http_measurement(raw)
    observations = [row["observation"] for row in raw["measured"]]
    completed = [row for row in observations if row["outcome"] == "completed"]
    return {
        "percentile_method": PERCENTILE_METHOD,
        "all_request_duration_ns": _distribution(
            [row["finished_ns"] - row["started_ns"] for row in observations]
        ),
        "completed_request_duration_ns": _distribution(
            [row["finished_ns"] - row["started_ns"] for row in completed]
        ),
        "completed_ttft_ns": _distribution(
            [
                row["token_received_ns"][0] - row["started_ns"]
                for row in completed
                if row["token_received_ns"]
            ]
        ),
        "completed_inter_token_gap_ns": _distribution(
            [
                later - earlier
                for row in completed
                for earlier, later in zip(
                    row["token_received_ns"], row["token_received_ns"][1:]
                )
            ]
        ),
    }


def http_report_digest(report):
    return sha256(
        _canonical(
            {name: value for name, value in report.items() if name != "report_sha256"}
        ).encode()
    ).hexdigest()


def build_http_load_report(raw):
    report = {
        "schema": REPORT_SCHEMA,
        "measurement": copy.deepcopy(raw),
        "latencies": http_latency_summary(raw),
        "verification_scope": VERIFICATION_SCOPE,
    }
    report["report_sha256"] = http_report_digest(report)
    return report


def verify_http_load_report(report):
    if (
        type(report) is not dict
        or set(report)
        != {"schema", "measurement", "latencies", "verification_scope", "report_sha256"}
        or report["schema"] != REPORT_SCHEMA
        or report["verification_scope"] != VERIFICATION_SCOPE
    ):
        raise ValueError("HTTP report fields or verification scope differ")
    if report["report_sha256"] != http_report_digest(report):
        raise ValueError("HTTP report digest differs")
    if _canonical(report["latencies"]) != _canonical(
        http_latency_summary(report["measurement"])
    ):
        raise ValueError(
            "HTTP report latency percentiles do not match recorded observations"
        )
    return report
