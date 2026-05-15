"""
Sliding window hot cache for recent events per canonical entity.

⚠️ v2 FIX: uses asyncio.Lock — NOT threading.RLock.

v1 used threading.RLock which causes deadlock when a coroutine yields
while holding the lock and another coroutine tries to acquire it on the
same thread. asyncio.Lock cooperatively yields, never deadlocks.

This cache is called DIRECTLY from async coroutines (not via executor),
so asyncio.Lock is the only correct choice here.

grep -rn "threading.RLock" engine/cache/ must return EMPTY.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone


class SlidingWindowCache:
    """
    Per-entity deque of recent events with asyncio.Lock for async safety.

    All mutation methods are async because they acquire asyncio.Lock.
    Called directly from FastAPI coroutines — NOT from executor threads.
    """

    WINDOW_SECONDS: int = 300
    MAX_DEQUE_LEN: int = 500

    def __init__(self) -> None:
        self._windows: defaultdict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.MAX_DEQUE_LEN)
        )
        self._lock = asyncio.Lock()  # ← NOT threading.RLock

    async def push(self, event: dict) -> None:
        """
        Push an event into the appropriate entity window.
        Performance target: <0.01ms (lock + deque append).
        """
        async with self._lock:
            cid = event.get("canonical_id") or event.get("service", "unknown")
            self._windows[cid].append(event)

    async def get_window(self, canonical_id: str,
                         since_ts: str, until_ts: str) -> list[dict]:
        """
        Return events for canonical_id within [since_ts, until_ts].
        Sorted by ts ascending. Handles missing/malformed ts gracefully.
        """
        async with self._lock:
            dq = list(self._windows.get(canonical_id, deque()))

        # Filter and sort outside lock
        filtered = []
        for event in dq:
            ts = event.get("ts", "")
            if not ts:
                continue
            try:
                if since_ts <= ts <= until_ts:
                    filtered.append(event)
            except (TypeError, ValueError):
                continue

        filtered.sort(key=lambda e: e.get("ts", ""))
        return filtered

    async def get_neighborhood_window(self, canonical_ids: list[str],
                                       since_ts: str,
                                       until_ts: str) -> list[dict]:
        """
        Gather events from multiple entities, merge, sort, deduplicate.
        """
        tasks = [
            self.get_window(cid, since_ts, until_ts)
            for cid in canonical_ids
        ]
        results = await asyncio.gather(*tasks)

        # Merge all results
        merged = []
        seen_ids: set = set()
        for window in results:
            for event in window:
                # Deduplicate by event_id if present
                eid = event.get("event_id")
                if eid:
                    if eid in seen_ids:
                        continue
                    seen_ids.add(eid)
                merged.append(event)

        merged.sort(key=lambda e: e.get("ts", ""))
        return merged

    async def evict_stale(self, until_ts: str) -> int:
        """
        Remove events older than (until_ts - WINDOW_SECONDS).
        Returns count of evicted events.
        """
        try:
            cutoff_dt = datetime.fromisoformat(
                until_ts.replace("Z", "+00:00")
            )
            cutoff_epoch = cutoff_dt.timestamp() - self.WINDOW_SECONDS
        except (ValueError, AttributeError):
            return 0

        evicted = 0
        async with self._lock:
            for cid in list(self._windows.keys()):
                dq = self._windows[cid]
                before = len(dq)
                # Remove from left (oldest) while stale
                while dq:
                    ts = dq[0].get("ts", "")
                    if not ts:
                        dq.popleft()
                        evicted += 1
                        continue
                    try:
                        event_epoch = datetime.fromisoformat(
                            ts.replace("Z", "+00:00")
                        ).timestamp()
                        if event_epoch < cutoff_epoch:
                            dq.popleft()
                            evicted += 1
                        else:
                            break
                    except (ValueError, AttributeError):
                        dq.popleft()
                        evicted += 1

        return evicted

    def entity_count(self) -> int:
        """Number of entities with cached events. Sync ok — just dict len."""
        return len(self._windows)

    def event_count(self) -> int:
        """Total events across all entity windows."""
        return sum(len(d) for d in self._windows.values())

    async def clear(self, canonical_id: str) -> None:
        """Remove all cached events for a canonical_id."""
        async with self._lock:
            self._windows.pop(canonical_id, None)
