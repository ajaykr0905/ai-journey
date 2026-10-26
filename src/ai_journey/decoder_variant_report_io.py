"""Atomic, durable publication of new decoder reports without overwriting."""

import json
import os
from pathlib import Path
import tempfile


def write_new_report(path: Path, report: dict) -> None:
    """Link a complete same-filesystem temporary file into an unused path."""

    rendered = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike replace(), link() fails if another writer won the destination.
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
