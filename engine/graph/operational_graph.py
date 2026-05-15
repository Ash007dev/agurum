"""
Operational dependency graph with role-based node classification.

Uses NetworkX DiGraph. Called ONLY from run_in_executor workers —
all methods are synchronous. The async wrapping happens in ContextReconstructor.

CRITICAL INVARIANT for WL hash: uses ROLE LABELS as node attributes,
NEVER canonical_ids or service names. Two graphs with identical topology
but different service names → SAME hash. This is the rename-robustness
guarantee for structural matching.

Thread safety: threading.RLock — CORRECT here because this class is
called from executor worker threads, NOT from async coroutines.
"""
from __future__ import annotations

import asyncio
import hashlib
import threading
from typing import Optional, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from engine.registry.entity_registry import EntityRegistry


class OperationalGraph:
    """
    Directed dependency graph where nodes are canonical entity UUIDs
    and edges represent relationships (calls, depends_on, consumes, writes_to).

    Each node carries computed role labels (topology + functional + temporal)
    that are used for rename-invariant WL hashing and embedding strings.
    """

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self._dirty_nodes: set[str] = set()
        self._debounce_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()  # CORRECT — executor threads, not coroutines
        self._registry: Optional[EntityRegistry] = None

    def set_registry(self, registry: EntityRegistry) -> None:
        """Store reference for role writeback to SQLite."""
        self._registry = registry

    def upsert_edge(self, caller_cid: str, callee_cid: str,
                    relationship: str = "calls",
                    weight: float = 1.0) -> None:
        """Add or update a directed edge between two canonical entities."""
        with self._lock:
            # Add nodes if not present
            if not self.G.has_node(caller_cid):
                self.G.add_node(caller_cid, roles=[])
            if not self.G.has_node(callee_cid):
                self.G.add_node(callee_cid, roles=[])

            # Add or update edge
            self.G.add_edge(
                caller_cid, callee_cid,
                relationship=relationship,
                weight=weight,
                inv_weight=1.0 / max(weight, 1e-9),
            )

            self._dirty_nodes.add(caller_cid)
            self._dirty_nodes.add(callee_cid)

        self._schedule_role_recompute()

    def _schedule_role_recompute(self) -> None:
        """Debounced role recomputation: async if loop running, sync otherwise."""
        try:
            loop = asyncio.get_running_loop()
            if self._debounce_task is None or self._debounce_task.done():
                self._debounce_task = loop.create_task(
                    self._bulk_recompute_after(delay=0.1)
                )
        except RuntimeError:
            # No event loop — sync/test context
            self._force_recompute()

    async def _bulk_recompute_after(self, delay: float = 0.1) -> None:
        """Debounced async role recomputation."""
        await asyncio.sleep(delay)
        nodes = self._dirty_nodes.copy()
        self._dirty_nodes.clear()
        new_attrs = {n: self._compute_roles(n) for n in nodes}
        with self._lock:
            nx.set_node_attributes(
                self.G, {n: {"roles": r} for n, r in new_attrs.items()}
            )
        if self._registry:
            for cid, roles in new_attrs.items():
                self._registry.update_roles(cid, roles)

    def _force_recompute(self) -> None:
        """Synchronous fallback for tests and executor-called contexts."""
        nodes = self._dirty_nodes.copy()
        self._dirty_nodes.clear()
        new_attrs = {n: self._compute_roles(n) for n in nodes}
        with self._lock:
            nx.set_node_attributes(
                self.G, {n: {"roles": r} for n, r in new_attrs.items()}
            )

    def _compute_roles(self, cid: str) -> list[str]:
        """
        Compute role labels for a node using degree analysis.

        Returns sorted list of ALL applicable roles:
          Topology:   HUB_SVC, CORE_SVC, BRIDGE_SVC, LEAF_SVC, EDGE_SVC, SOLO_SVC
          Functional: REQUEST_HANDLER, QUEUE_CONSUMER, STORE, COMPUTE
          Temporal:   TRIGGER_SVC, UPSTREAM_SVC, DOWNSTREAM_SVC
        """
        if cid not in self.G:
            return ["SOLO_SVC"]

        in_deg = self.G.in_degree(cid)
        out_deg = self.G.out_degree(cid)
        total_deg = in_deg + out_deg

        roles: list[str] = []

        # ── Topology roles (one primary) ──
        if total_deg == 0:
            roles.append("SOLO_SVC")
        elif total_deg >= 4:
            roles.append("HUB_SVC")
        elif in_deg >= 2 and out_deg >= 2:
            roles.append("CORE_SVC")
        elif in_deg >= 1 and out_deg >= 1:
            roles.append("BRIDGE_SVC")
        elif in_deg == 0 and out_deg >= 1:
            roles.append("LEAF_SVC")
        elif out_deg == 0 and in_deg >= 1:
            roles.append("EDGE_SVC")

        # ── Functional roles (check edge attributes) ──
        for _, _, data in self.G.in_edges(cid, data=True):
            rel = data.get("relationship", "")
            if rel == "calls":
                if "REQUEST_HANDLER" not in roles:
                    roles.append("REQUEST_HANDLER")
            elif rel == "consumes":
                if "QUEUE_CONSUMER" not in roles:
                    roles.append("QUEUE_CONSUMER")
            elif rel == "writes_to":
                if "STORE" not in roles:
                    roles.append("STORE")

        if out_deg >= 2:
            has_outgoing_calls = any(
                data.get("relationship") == "calls"
                for _, _, data in self.G.out_edges(cid, data=True)
            )
            if has_outgoing_calls and "COMPUTE" not in roles:
                roles.append("COMPUTE")

        # ── Temporal roles ──
        if in_deg == 0 and out_deg >= 1:
            if "TRIGGER_SVC" not in roles:
                roles.append("TRIGGER_SVC")
        if out_deg > 0:
            if "UPSTREAM_SVC" not in roles:
                roles.append("UPSTREAM_SVC")
        if in_deg > 0:
            if "DOWNSTREAM_SVC" not in roles:
                roles.append("DOWNSTREAM_SVC")

        return sorted(roles)

    def get_neighborhood(self, canonical_id: str, depth: int = 2) -> list[str]:
        """
        BFS neighborhood within depth hops (including self).
        Max allowed depth: 4.
        """
        depth = min(depth, 4)
        try:
            lengths = nx.single_source_shortest_path_length(
                self.G, canonical_id, cutoff=depth
            )
            return list(lengths.keys())
        except nx.NodeNotFound:
            return [canonical_id]

    def get_causal_path(self, from_cid: str, to_cid: str) -> list[str]:
        """
        Highest-confidence path (inverted weights for shortest-path).
        Returns list of canonical_ids, or [] if no path.
        """
        try:
            return nx.shortest_path(
                self.G, from_cid, to_cid, weight="inv_weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_wl_hash(self, canonical_id: str, depth: int = 2) -> str:
        """
        Weisfeiler-Lehman structural hash.

        CRITICAL INVARIANT: uses ROLE LABELS as node attributes — NEVER
        canonical_ids or service names. Two graphs with identical topology
        but different service names → SAME hash.
        """
        if canonical_id not in self.G:
            return hashlib.md5(canonical_id.encode()).hexdigest()[:16]

        # Get subgraph: all nodes within depth hops
        try:
            neighbors = nx.single_source_shortest_path_length(
                self.G, canonical_id, cutoff=depth
            )
        except nx.NodeNotFound:
            return hashlib.md5(canonical_id.encode()).hexdigest()[:16]

        # Build fresh subgraph with role labels as node attributes
        # Use role strings — NOT canonical_ids or service names
        subgraph = nx.DiGraph()
        node_list = list(neighbors.keys())
        for node in node_list:
            node_roles = self.G.nodes.get(node, {}).get("roles", [])
            subgraph.add_node(node, roles="_".join(sorted(node_roles)))

        for u, v, data in self.G.edges(data=True):
            if u in neighbors and v in neighbors:
                subgraph.add_edge(u, v, **data)

        return nx.weisfeiler_lehman_graph_hash(
            subgraph, node_attr="roles", iterations=3, digest_size=16
        )

    def get_all_roles(self, canonical_id: str) -> list[str]:
        """Return computed roles for a node."""
        return self.G.nodes.get(canonical_id, {}).get("roles", [])

    def node_count(self) -> int:
        """Return number of nodes in the graph."""
        return self.G.number_of_nodes()

    def edge_count(self) -> int:
        """Return number of edges in the graph."""
        return self.G.number_of_edges()
