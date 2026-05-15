"""
Lightweight rename tracker for the benchmark adapter (Path A).

No SQLite, no async, no external dependencies — plain dicts only.
Boot time: ~0ms (just dict allocation).

CRITICAL: rename events use "from_" (with underscore), NOT "from".
Python reserves "from" as a keyword. The benchmark generator outputs
events with key "from_". Reading event.get("from") returns None and
silently breaks all renames.

Fix 1: UnionFind is used alongside _name_to_cid so cascading rename
chains (A→B→C→D) always collapse to a single path-compressed root.
"""
from __future__ import annotations

import uuid

from engine.registry.union_find import UnionFind


class AliasTracker:
    """
    Maps service names (current and historical) to stable canonical UUIDs.

    A rename from "payments-svc" to "billing-svc" makes both names resolve
    to the same canonical_id. Old names are NEVER deleted — historical
    lookups across the full benchmark window must work.

    Thread safety: NOT required. Used only in synchronous benchmark path.
    """

    def __init__(self) -> None:
        self._name_to_cid: dict[str, str] = {}
        self._cid_to_names: dict[str, list[str]] = {}
        self._edges: set[tuple[str, str]] = set()
        # Fix 1: path-compressed union-find for multi-hop rename chains
        self.uf = UnionFind()

    def process_event(self, event: dict) -> None:
        """
        Call on every event in ingest(). Two jobs:
        1. Register unseen services.
        2. Handle rename and dep_add topology events (reads from_, NOT from).

        Never raises. Silently skips malformed events.
        """
        # Job 1: register unseen services
        svc = event.get("service")
        if svc:
            if svc not in self._name_to_cid:
                self._register(svc)

        # Job 2: handle topology events
        # CRITICAL: "from_" with underscore — NOT "from"
        if event.get("kind") == "topology":
            change = event.get("change")
            old = event.get("from_")
            new = event.get("to")
            if old and new:
                if change == "rename":
                    self._rename(old, new)
                elif change == "dep_add":
                    from_cid = self.resolve(old)
                    to_cid = self.resolve(new)
                    self._edges.add((from_cid, to_cid))
                elif change == "dep_remove":
                    from_cid = self.resolve(old)
                    to_cid = self.resolve(new)
                    if (from_cid, to_cid) in self._edges:
                        self._edges.remove((from_cid, to_cid))

    def get_role(self, cid: str) -> str:
        in_degree = sum(1 for (f, t) in self._edges if t == cid)
        out_degree = sum(1 for (f, t) in self._edges if f == cid)
        
        if in_degree == 0:
            return "role_ingress"
        if out_degree == 0:
            return "role_leaf"
        return "role_transit"

    def resolve(self, name: str) -> str:
        """
        Returns canonical_id for name. Auto-registers if not seen.
        NEVER raises. Always returns a string.
        """
        if name not in self._name_to_cid:
            return self._register(name)
        return self._name_to_cid[name]

    def get_family_id(self, name: str) -> str:
        """Alias for resolve(). Used by callers who prefer this name."""
        return self.resolve(name)

    def get_current_name(self, canonical_id: str) -> str:
        """Return the most recent name for a canonical_id."""
        return self._cid_to_names.get(canonical_id, ["unknown"])[-1]

    def get_all_names(self, canonical_id: str) -> list[str]:
        """Return all names (historical + current) for a canonical_id."""
        return self._cid_to_names.get(canonical_id, []).copy()

    def known_services(self) -> list[str]:
        """Return all known service names (current and historical)."""
        return list(self._name_to_cid.keys())

    def _register(self, name: str) -> str:
        """Register a new service name with a fresh canonical UUID."""
        cid = str(uuid.uuid4())
        self._name_to_cid[name] = cid
        self._cid_to_names[cid] = [name]
        return cid

    def _rename(self, old: str, new: str) -> None:
        """
        Map new name to same canonical_id as old name.
        DO NOT delete old name — historical lookups must still work.

        Fix 1: also union in the UnionFind so that multi-hop chains
        (A→B, B→C, C→D) all compress to A's root transparently.
        """
        if old not in self._name_to_cid:
            self._register(old)
        cid = self._name_to_cid[old]
        self._name_to_cid[new] = cid   # new name → same cid
        # Track name history (ordered: first=original, last=current)
        if new not in self._cid_to_names.get(cid, []):
            self._cid_to_names[cid].append(new)
        # Union-Find: collapses multi-hop chains via path compression
        self.uf.union(old, new)

    def canonical(self, name: str) -> str:
        """Return path-compressed canonical name via UnionFind."""
        return self.uf.canonical(name)
