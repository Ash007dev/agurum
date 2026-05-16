"""
engine/graph/versioned_dep_graph.py — Versioned dependency graph via event-log ledger.

Fix 4: Replaces the deepcopy snapshot list with an append-only event log.
graph_at(ts) reconstructs topology iteratively by replaying events up to the
target timestamp — O(E) time, O(1) space per event vs O(N*E) for deepcopy list.

This eliminates both linear scan latency on the snapshot list AND the memory
pressure from storing N full graph deepcopies.
"""
from __future__ import annotations


class VersionedDepGraph:
    """
    Append-only event log ledger for dependency graph time travel.

    Every topology event is stored as a (timestamp, change_type, data) tuple.
    graph_at(ts) iterates through the sorted log and reconstructs the graph
    state up to the requested timestamp without storing full snapshots.
    """

    def __init__(self) -> None:
        # List of (iso_ts_str, change_type, data) tuples — no deepcopies
        self.changes: list[tuple[str, str, tuple]] = []

    def on_topology(self, event: dict) -> None:
        """Append a lightweight change record for every topology event.

        Generator uses 'from_' (with underscore) to avoid Python keyword clash.
        Also accepts 'from' as fallback for robustness.
        """
        ts = event.get("ts", "")
        change = event.get("change", "")

        if change == "rename":
            old = event.get("from_") or event.get("from", "")
            new = event.get("to", "")
            if old and new:
                self.changes.append((ts, "rename", (old, new)))

        elif change == "dep_add":
            old = event.get("from_") or event.get("from", "")
            new = event.get("to", "")
            if old and new:
                self.changes.append((ts, "shift", (old, new)))

        elif change == "dep_remove":
            old = event.get("from_") or event.get("from", "")
            new = event.get("to", "")
            if old and new:
                self.changes.append((ts, "dep_remove", (old, new)))

    def graph_at(self, ts_str: str) -> dict[str, list[str]]:
        """
        Compute the topology state iteratively up to ts_str.

        Replays the event log in sorted timestamp order, applying each
        change whose timestamp <= ts_str. This is the timeline index approach:
        no snapshots stored, no deepcopy overhead.
        """
        graph: dict[str, list[str]] = {}
        for ts, change_type, data in sorted(self.changes, key=lambda x: x[0]):
            if ts > ts_str:
                break
            if change_type == "rename":
                old, new = data
                if old in graph:
                    graph[new] = graph.pop(old)
                for svc in list(graph):
                    graph[svc] = [new if d == old else d for d in graph[svc]]
            elif change_type == "shift":
                src, dest = data
                graph.setdefault(src, [])
                if dest not in graph[src]:
                    graph[src].append(dest)
            elif change_type == "dep_remove":
                src, dest = data
                if src in graph and dest in graph[src]:
                    graph[src].remove(dest)
        return graph

    def upstream_callers_at(self, service_name: str, ts_str: str) -> list[str]:
        """Who was calling service_name at time ts_str?"""
        g = self.graph_at(ts_str)
        return [svc for svc, deps in g.items() if service_name in deps]
