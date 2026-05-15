"""
engine/registry/union_find.py — Path-compressed Union-Find for service identity.

Solves cascading rename chains A→B→C→D by collapsing all aliases to a single
root canonical ID via path compression. Replaces the single-hop dict lookup
in AliasTracker that breaks on multi-hop renames.
"""
from __future__ import annotations


class UnionFind:
    """
    Path-compressed Union-Find for stable service canonical IDs.

    After union(A→B), union(B→C), union(C→D):
        canonical(D) == canonical(C) == canonical(B) == canonical(A)

    All resolve to the original root (A's initial ID), so historical
    incident fingerprints remain aligned even after multiple renames.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, name: str) -> None:
        if name not in self.parent:
            self.parent[name] = name

    def union(self, old_name: str, new_name: str) -> None:
        """new_name inherits the canonical root of old_name."""
        self.add(old_name)
        self.add(new_name)
        root = self.find(old_name)
        self.parent[new_name] = root  # path compress immediately

    def find(self, name: str) -> str:
        """Return canonical root with full path compression."""
        self.add(name)
        if self.parent[name] != name:
            self.parent[name] = self.find(self.parent[name])
        return self.parent[name]

    def canonical(self, name: str) -> str:
        """Public alias for find()."""
        return self.find(name)
