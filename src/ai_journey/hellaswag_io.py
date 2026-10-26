"""Bounded input reads and collision checks for benchmark operations."""

import json
import os
from pathlib import Path
import stat

from .hellaswag import HellaSwagError


def read_evaluation_bytes(path: Path, *, maximum: int, name: str) -> bytes:
    """Read one bounded descriptor without following a symlink or blocking on FIFO."""
    if type(maximum) is not int or maximum <= 0:
        raise HellaSwagError("evaluation byte budget must be a positive integer")
    try:
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
                raise HellaSwagError(f"{name} exceeds regular-file constraints")
            raw = stream.read(maximum + 1)
    except OSError as error:
        raise HellaSwagError(f"{name} exceeds regular-file constraints") from error
    if len(raw) > maximum:
        raise HellaSwagError(f"{name} exceeds its byte budget")
    return raw


def parse_evaluation_json(raw: bytes, *, name: str) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise HellaSwagError(f"{name} contains duplicate JSON fields")
            result[key] = value
        return result

    def invalid_constant(value):
        raise HellaSwagError(f"{name} must contain finite JSON numbers")

    try:
        payload = json.loads(
            raw, object_pairs_hook=unique_object, parse_constant=invalid_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise HellaSwagError(f"{name} must be valid bounded UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise HellaSwagError(f"{name} must contain a JSON object")
    return payload


def validate_evaluation_output(output: Path, inputs: tuple[Path, ...]):
    """Reject destructive aliases before loading models or evaluating examples."""
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise HellaSwagError("evaluation output must be a regular non-symlink path")
    for source in inputs:
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise HellaSwagError("evaluation output aliases an input artifact")
