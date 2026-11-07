"""Content identities for Project 1 runs and attempts."""

from __future__ import annotations

from .project1_contract import Project1RunContract


def run_id(contract: Project1RunContract) -> str:
    """Return a readable identity bound to every run control."""

    if not isinstance(contract, Project1RunContract):
        raise TypeError("contract must be Project1RunContract")
    return f"{contract.run_name}-{contract.fingerprint()[:16]}"
