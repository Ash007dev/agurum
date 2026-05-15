"""
engine/store/in_memory_store.py — fast in-memory event store for the benchmark adapter.

No DuckDB, no disk I/O. Just a Python list + timestamp parsing.
Used in Path A (benchmark adapter). DuckDB EventStore is for Path B.
"""
from __future__ import annotations

from datetime import datetime, timezone


def _parse_ts(ts_str: str) -> float:
    """Parse ISO timestamp string → unix float. Returns 0.0 on failure."""
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


class InMemoryStore:
    """
    Append-only list-based event store.
    Supports canonical-ID-based windowed queries via an AliasTracker.
    """

    def __init__(self) -> None:
        self._events: list[dict] = []

    def append(self, event: dict) -> None:
        self._events.append(event)

    def append_batch(self, events: list[dict]) -> None:
        self._events.extend(events)

    def get_by_canonical_id(
        self,
        canonical_id: str,
        tracker,               # AliasTracker
        window_start_ts: float,
        window_end_ts: float,
    ) -> list[dict]:
        """Return events for a canonical service within a time window."""
        result = []
        for e in self._events:
            svc = e.get("service", "")
            if not svc:
                continue
            if tracker.resolve(svc) != canonical_id:
                continue
            e_ts = _parse_ts(e.get("ts", ""))
            if window_start_ts <= e_ts <= window_end_ts:
                result.append(e)
        return result

    def get_all(self) -> list[dict]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)
