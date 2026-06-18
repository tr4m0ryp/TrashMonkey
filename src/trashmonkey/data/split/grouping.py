"""Instance grouping: union-find over the dedup near-dup graph + stratum pick.

A connected component of near-duplicate edges is "the same physical object" and
must never straddle a split boundary, so every split decision is made per group,
never per image. ``group_instances`` builds those components; ``primary_stratum``
picks a group's majority (source, class) for stratified allocation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from trashmonkey.data.dedup import Item, NearEdge


def group_instances(keys: Iterable[str], edges: Iterable[NearEdge]) -> dict[str, str]:
    """Union-find over the near-dup graph; group id = smallest member key."""
    parent: dict[str, str] = {key: key for key in keys}

    def find(key: str) -> str:
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    for edge in edges:
        if edge.key_a in parent and edge.key_b in parent:
            root_a, root_b = find(edge.key_a), find(edge.key_b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    return {key: find(key) for key in parent}


def primary_stratum(members: Sequence[Item]) -> tuple[str, str]:
    """Majority (source, class) of a group; ties break lexicographically."""
    tally: dict[tuple[str, str], int] = {}
    for item in members:
        stratum = (item.source, item.class_name)
        tally[stratum] = tally.get(stratum, 0) + 1
    return min(tally, key=lambda s: (-tally[s], s))
