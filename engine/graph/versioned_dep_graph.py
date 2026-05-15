"""
engine/graph/versioned_dep_graph.py — Versioned dependency graph with time travel.

Snapshots the service dependency graph after every topology event so that
mid-incident renames/shifts don't corrupt historical context reconstruction.
Use graph_at(ts) to get the graph state as it existed at incident time.
"""
from __future__ import annotations

import copy


class VersionedDepGraph:
    """
    Maintains an append-only log of dependency graph snapshots.

    Every topology event (rename, dep_add, dep_remove) triggers a new
    snapshot.  graph_at(ts) returns the snapshot that was current at ts,
    so mid-eval renames don't corrupt historical context reconstruction.
    """

    def __init__(self) -> None:
        self._versions: list[tuple[str, dict]] = []  # (iso_ts, graph_snapshot)
        self._current: dict[str, list[str]] = {}     # service → [dependencies]

    def on_topology(self, event: dict) -> None:
        """Call on every topology event during ingest."""
        ts = event.get("ts", "")
        change = event.get("change", "")
        # Generator uses "from_" (with underscore) to avoid Python keyword clash
        old = event.get("from_") or event.get("from", "")
        new = event.get("to", "")

        if change == "rename" and old and new:
            # Migrate all entries that mention old name
            if old in self._current:
                self._current[new] = self._current.pop(old)
            for svc in list(self._current):
                self._current[svc] = [
                    new if d == old else d for d in self._current[svc]
                ]

        elif change == "dep_add" and old and new:
            self._current.setdefault(old, [])
            if new not in self._current[old]:
                self._current[old].append(new)

        elif change == "dep_remove" and old and new:
            if old in self._current and new in self._current[old]:
                self._current[old].remove(new)

        # Snapshot after every change
        self._versions.append((ts, copy.deepcopy(self._current)))

    def graph_at(self, ts_str: str) -> dict:
        """Return the dependency graph as it existed at ts_str (ISO format)."""
        result: dict = {}
        for version_ts, snapshot in self._versions:
            if version_ts <= ts_str:
                result = snapshot
            else:
                break
        return result

    def upstream_callers_at(self, service_name: str, ts_str: str) -> list[str]:
        """Who was calling service_name at time ts_str?"""
        g = self.graph_at(ts_str)
        return [svc for svc, deps in g.items() if service_name in deps]
