"""Transitive duplicate clusters from exact candidate assessments."""

from .bpe import TokenizerError
from .minhash_assessment import CandidateAssessment


def duplicate_clusters(
    document_ids: tuple[str, ...],
    assessments: tuple[CandidateAssessment, ...],
) -> tuple[tuple[str, ...], ...]:
    """Union accepted duplicate edges into stable connected components."""
    if (
        not isinstance(document_ids, tuple)
        or not document_ids
        or len(set(document_ids)) != len(document_ids)
        or any(not isinstance(value, str) or not value for value in document_ids)
    ):
        raise TokenizerError("duplicate clustering requires unique document ids")
    if not isinstance(assessments, tuple) or any(
        not isinstance(value, CandidateAssessment) for value in assessments
    ):
        raise TokenizerError("duplicate clustering requires candidate assessments")
    identities = set(document_ids)
    parent = {identity: identity for identity in identities}

    def find(identity: str) -> str:
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            canonical, duplicate = sorted((left_root, right_root))
            parent[duplicate] = canonical

    for assessment in assessments:
        if (
            assessment.left_id not in identities
            or assessment.right_id not in identities
        ):
            raise TokenizerError("duplicate assessment references an unknown document")
        if assessment.duplicate:
            union(assessment.left_id, assessment.right_id)
    groups: dict[str, list[str]] = {}
    for identity in sorted(identities):
        groups.setdefault(find(identity), []).append(identity)
    return tuple(
        sorted(tuple(members) for members in groups.values() if len(members) > 1)
    )
