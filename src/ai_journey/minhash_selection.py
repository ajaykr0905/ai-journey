"""Deterministic representative selection for duplicate clusters."""

from dataclasses import dataclass

from .bpe import TokenizerError


@dataclass(frozen=True)
class ClusterDecision:
    canonical_id: str
    duplicate_ids: tuple[str, ...]


def select_representatives(
    document_ids: tuple[str, ...], clusters: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, ...], tuple[ClusterDecision, ...]]:
    """Keep the smallest document id in each disjoint duplicate cluster."""
    if (
        not isinstance(document_ids, tuple)
        or not document_ids
        or len(set(document_ids)) != len(document_ids)
    ):
        raise TokenizerError("representative selection requires unique document ids")
    identities = set(document_ids)
    clustered = set()
    decisions = []
    if not isinstance(clusters, tuple):
        raise TokenizerError("representative selection requires cluster tuples")
    for cluster in clusters:
        if (
            not isinstance(cluster, tuple)
            or len(cluster) < 2
            or len(set(cluster)) != len(cluster)
            or any(identity not in identities for identity in cluster)
            or clustered.intersection(cluster)
        ):
            raise TokenizerError("duplicate clusters must be disjoint known documents")
        ordered = tuple(sorted(cluster))
        clustered.update(ordered)
        decisions.append(ClusterDecision(ordered[0], ordered[1:]))
    removed = {
        duplicate for decision in decisions for duplicate in decision.duplicate_ids
    }
    kept = tuple(sorted(identities - removed))
    return kept, tuple(sorted(decisions, key=lambda value: value.canonical_id))
