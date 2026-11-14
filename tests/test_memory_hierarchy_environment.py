import getpass
import json
import os
import platform
from pathlib import Path

import numpy

from ai_journey import memory_hierarchy_environment as environment_module
from ai_journey.memory_hierarchy_environment import (
    ENVIRONMENT_FIELDS,
    TOKEN,
    host_environment,
)


def test_record_holds_only_version_and_platform_tokens():
    record = host_environment()

    assert tuple(record) == ENVIRONMENT_FIELDS
    assert record["numpy_version"] == numpy.__version__
    assert record["python_version"] == platform.python_version()
    for name in ENVIRONMENT_FIELDS[:-1]:
        assert TOKEN.fullmatch(record[name]), name
    assert record["cpu_count"] is None or record["cpu_count"] >= 1


def test_record_never_carries_host_user_or_path_identifiers():
    record = host_environment()
    values = [value for value in record.values() if isinstance(value, str)]
    identifiers = {platform.node(), getpass.getuser(), str(Path.home())}

    for identifier in identifiers - {"", "unknown"}:
        assert identifier not in values
    assert not any(os.sep in value for value in values)
    assert json.loads(json.dumps(record)) == record


def test_missing_or_odd_build_metadata_becomes_unknown(monkeypatch):
    monkeypatch.delattr(numpy.__config__, "CONFIG", raising=False)
    monkeypatch.setattr(platform, "machine", lambda: "")
    monkeypatch.setattr(platform, "system", lambda: "Some OS 1.0")
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    record = host_environment()

    assert record["numpy_blas"] == record["numpy_blas_version"] == "unknown"
    assert record["platform_machine"] == "unknown"
    assert record["platform_system"] == "unknown"
    assert record["cpu_count"] is None


def test_blas_metadata_is_read_from_the_numpy_build_record(monkeypatch):
    build = {"Build Dependencies": {"blas": {"name": "openblas64", "version": "0.3"}}}
    monkeypatch.setattr(numpy.__config__, "CONFIG", build, raising=False)
    assert host_environment()["numpy_blas"] == "openblas64"

    monkeypatch.setattr(
        numpy.__config__,
        "CONFIG",
        {"Build Dependencies": {"blas": {"name": "/opt/lib/blas"}}},
        raising=False,
    )
    assert environment_module.host_environment()["numpy_blas"] == "unknown"
    monkeypatch.setattr(numpy.__config__, "CONFIG", {"Build Dependencies": []})
    assert host_environment()["numpy_blas_version"] == "unknown"
