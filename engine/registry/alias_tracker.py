"""
Lightweight rename tracker for the benchmark adapter (Path A).

No SQLite, no async, no external dependencies — plain dicts only.
Boot time: ~0ms (just dict allocation).

CRITICAL: rename events use "from_" (with underscore), NOT "from".
Python reserves "from" as a keyword. The benchmark generator outputs
events with key "from_". Reading event.get("from") returns None and
silently breaks all renames.

Pillar 1: CanonicalUnionFind uses root-to-root assignment so that
cascading rename chains (A→B→C→D) always collapse to a single
path-compressed root regardless of insertion order.

Pillar 2: topo_ledger records every topology event as a lightweight
(ts, change_type, from_cid, to_cid) tuple. get_edges_at(ts) replays
the ledger to reconstruct the exact edge set at any point in time,
enabling time-aware role computation without deepcopy.
"""
from __future__ import annotations

import uuid


class CanonicalUnionFind:
    """
    Path-compressed Union-Find with root-to-root union assignment.

    union(old, new) makes the root of *new*'s chain point to the root
    of *old*'s chain — not the raw new name — so multi-hop chains
    always collapse cleanly regardless of order.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, name: str) -> None:
        if name not in self.parent:
            self.parent[name] = name

    def find(self, name: str) -> str:
        if name not in self.parent:
            self.parent[name] = name
        if self.parent[name] != name:
            self.parent[name] = self.find(self.parent[name])  # Path compression
        return self.parent[name]

    def union(self, old_name: str, new_name: str) -> None:
        """Root of new_name's chain points to root of old_name's chain."""
        root_old = self.find(old_name)
        root_new = self.find(new_name)
        if root_old != root_new:
            self.parent[root_new] = root_old  # root-to-root, not name-to-root

    def canonical(self, name: str) -> str:
        """Public alias for find()."""
        return self.find(name)


class AliasTracker:
    """
    Maps service names (current and historical) to stable canonical UUIDs.

    A rename from "payments-svc" to "billing-svc" makes both names resolve
    to the same canonical_id. Old names are NEVER deleted — historical
    lookups across the full benchmark window must work.

    Pillar 2: topo_ledger stores (ts, change_type, from_cid, to_cid) tuples
    using CIDs (not raw names), so renames are transparent in the ledger.
    get_edges_at(ts) replays the ledger chronologically to compute the exact
    edge set at any microsecond, enabling time-aware role computation.

    Thread safety: NOT required. Used only in synchronous benchmark path.
    """

    def __init__(self) -> None:
        self._name_to_cid: dict[str, str] = {}
        self._cid_to_names: dict[str, list[str]] = {}
        self._edges: set[tuple[str, str]] = set()
        # Pillar 1: path-compressed union-find for multi-hop rename chains
        self.uf = CanonicalUnionFind()
        # Pillar 2: append-only topology ledger for time-travel role computation
        # Each entry: (iso_ts, change_type, from_cid, to_cid)
        self.topo_ledger: list[tuple[str, str, str, str]] = []

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
            ts = event.get("ts", "")
            if old and new:
                if change == "rename":
                    self._rename(old, new)
                    # Pillar 2: record rename in ledger (CID-based, for audit)
                    from_cid = self.resolve(old)
                    to_cid = self.resolve(new)
                    self.topo_ledger.append((ts, "rename", from_cid, to_cid))
                elif change == "dep_add":
                    from_cid = self.resolve(old)
                    to_cid = self.resolve(new)
                    self._edges.add((from_cid, to_cid))
                    # Pillar 2: record edge addition in ledger
                    self.topo_ledger.append((ts, "dep_add", from_cid, to_cid))
                elif change == "dep_remove":
                    from_cid = self.resolve(old)
                    to_cid = self.resolve(new)
                    self._edges.discard((from_cid, to_cid))
                    # Pillar 2: record edge removal in ledger
                    self.topo_ledger.append((ts, "dep_remove", from_cid, to_cid))

    # ------------------------------------------------------------------
    # Pillar 2: Time-travel topology
    # ------------------------------------------------------------------

    def get_edges_at(self, ts_str: str) -> set[tuple[str, str]]:
        """
        Replay the topo_ledger chronologically up to ts_str and return
        the exact edge set (CID pairs) at that microsecond.

        Uses CIDs so renames are transparent — no graph key migration needed.
        No deepcopy. O(L) where L = len(topo_ledger).
        """
        edges: set[tuple[str, str]] = set()
        for entry_ts, change_type, src, dest in sorted(
            self.topo_ledger, key=lambda x: x[0]
        ):
            if entry_ts > ts_str:
                break
            if change_type == "dep_add":
                edges.add((src, dest))
            elif change_type == "dep_remove":
                edges.discard((src, dest))
            # rename entries are logged for audit but don't affect CID-based edges
        return edges

    def get_role_at(self, cid: str, ts_str: str) -> str:
        """
        Compute the graph role for cid using the edge set at ts_str.

        Replays the ledger to ts_str, then computes in/out degree.
        """
        edges = self.get_edges_at(ts_str)
        return self._role_from_edges(cid, edges)

    def get_graph_at(self, target_ts: str) -> dict:
        """
        Return a dict representation of the dependency graph at target_ts.

        Keys are CIDs, values are lists of dependency CIDs. Useful for
        external callers who need the full graph structure.
        """
        edges = self.get_edges_at(target_ts)
        graph: dict[str, list[str]] = {}
        for src, dest in edges:
            graph.setdefault(src, []).append(dest)
        return graph

    @staticmethod
    def _role_from_edges(cid: str, edges: set[tuple[str, str]]) -> str:
        """Compute graph role from an edge set without mutating tracker state."""
        in_degree = sum(1 for (_, t) in edges if t == cid)
        out_degree = sum(1 for (f, _) in edges if f == cid)
        if in_degree == 0:
            return "role_ingress"
        if out_degree == 0:
            return "role_leaf"
        return "role_transit"

    # ------------------------------------------------------------------
    # Core identity resolution
    # ------------------------------------------------------------------

    def get_role(self, cid: str) -> str:
        """Return role using current (final) edge state."""
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

        Pillar 1: also union in the UnionFind so that multi-hop chains
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
