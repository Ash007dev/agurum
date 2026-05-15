"""
engine/store/event_store.py — DuckDB-backed event store for production Path B.

NOT used by the benchmark adapter. Path A uses InMemoryStore.

All public methods are synchronous; the FastAPI production path calls them
via asyncio.get_event_loop().run_in_executor(None, ...) to avoid blocking
the event loop.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Lazy import so benchmark adapter doesn't need duckdb installed
try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False


_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_DEFAULT_DB = os.environ.get("PCE_DB_PATH", ":memory:")


class EventStore:
    """
    DuckDB-backed persistent event store for the production pipeline (Path B).

    Thread safety: DuckDB connections are NOT thread-safe. Each call to a
    public method should be dispatched via run_in_executor — the executor
    serializes calls into the thread pool.

    Alternatively, construct one EventStore per thread-pool worker thread
    using a threading.local() wrapper (production pattern).
    """

    def __init__(self, db_path: str = _DEFAULT_DB) -> None:
        if not _DUCKDB_AVAILABLE:
            raise ImportError(
                "duckdb is required for EventStore (production path). "
                "Benchmark adapter uses InMemoryStore instead."
            )
        self._conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        if _SCHEMA_PATH.exists():
            self._conn.execute(_SCHEMA_PATH.read_text())

    # ── Write ──────────────────────────────────────────────────────────────────

    def append_event(
        self,
        event: dict,
        canonical_id: str,
        ts: datetime | None = None,
    ) -> None:
        """Persist a single raw event with its resolved canonical_id."""
        if ts is None:
            ts_str = event.get("ts", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)

        self._conn.execute(
            """
            INSERT INTO events (ts, kind, canonical_id, raw_service, incident_id, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ts,
                event.get("kind", "unknown"),
                canonical_id,
                event.get("service"),
                event.get("incident_id"),
                json.dumps(event),
            ],
        )

    def upsert_episode(
        self,
        incident_id: str,
        canonical_id: str,
        action: str = "rollback",
        outcome: str = "resolved",
        event_count: int = 0,
        family: int = -1,
    ) -> None:
        """Insert or replace episode metadata."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO episodes
                (incident_id, canonical_id, action, outcome, event_count, family)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [incident_id, canonical_id, action, outcome, event_count, family],
        )

    def record_alias(self, canonical_id: str, name: str) -> None:
        """Record a new name for a canonical_id in the alias log."""
        # Mark all previous entries as not current
        self._conn.execute(
            "UPDATE alias_log SET is_current = FALSE WHERE canonical_id = ?",
            [canonical_id],
        )
        self._conn.execute(
            "INSERT INTO alias_log (canonical_id, name) VALUES (?, ?)",
            [canonical_id, name],
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_window_events(
        self,
        canonical_id: str,
        window_start: datetime,
        window_end: datetime,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return raw event dicts for a canonical service within a time window."""
        rows = self._conn.execute(
            """
            SELECT payload FROM events
            WHERE canonical_id = ?
              AND ts BETWEEN ? AND ?
            ORDER BY ts ASC
            LIMIT ?
            """,
            [canonical_id, window_start, window_end, limit],
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_episode(self, incident_id: str) -> dict[str, Any] | None:
        """Fetch episode metadata by incident_id."""
        row = self._conn.execute(
            "SELECT incident_id, canonical_id, action, outcome, event_count, family "
            "FROM episodes WHERE incident_id = ?",
            [incident_id],
        ).fetchone()
        if row is None:
            return None
        return {
            "incident_id": row[0],
            "canonical_id": row[1],
            "action": row[2],
            "outcome": row[3],
            "event_count": row[4],
            "family": row[5],
        }

    def get_all_episodes(self) -> list[dict[str, Any]]:
        """Return all episode metadata rows."""
        rows = self._conn.execute(
            "SELECT incident_id, canonical_id, action, outcome, event_count, family "
            "FROM episodes ORDER BY created_at ASC"
        ).fetchall()
        return [
            {
                "incident_id": r[0],
                "canonical_id": r[1],
                "action": r[2],
                "outcome": r[3],
                "event_count": r[4],
                "family": r[5],
            }
            for r in rows
        ]

    def rebuild_aliases(self) -> dict[str, str]:
        """
        Rebuild name → canonical_id mapping from alias_log.
        Called on startup to warm up AliasTracker from persistent state.
        Returns: {name: canonical_id}
        """
        rows = self._conn.execute(
            "SELECT name, canonical_id FROM alias_log ORDER BY first_seen ASC"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    # ── Continuous Learning ────────────────────────────────────────────────────

    def update_pair_stats_ewma(self, cid_a: str, cid_b: str, alpha: float, observation: float) -> float:
        """
        Apply EWMA update to a pair of canonical IDs. Returns the new weight.
        """
        # Ensure consistent ordering
        cid_1, cid_2 = sorted([cid_a, cid_b])
        
        row = self._conn.execute(
            "SELECT ewma_weight FROM entity_pair_stats WHERE canonical_id_a = ? AND canonical_id_b = ?",
            [cid_1, cid_2]
        ).fetchone()
        
        old_val = row[0] if row else 0.0
        new_val = alpha * observation + (1.0 - alpha) * old_val
        
        self._conn.execute(
            """
            INSERT OR REPLACE INTO entity_pair_stats (canonical_id_a, canonical_id_b, ewma_weight)
            VALUES (?, ?, ?)
            """,
            [cid_1, cid_2, new_val]
        )
        return new_val

    def get_all_pair_stats(self) -> dict[tuple[str, str], float]:
        """Load all pair stats into memory."""
        rows = self._conn.execute(
            "SELECT canonical_id_a, canonical_id_b, ewma_weight FROM entity_pair_stats"
        ).fetchall()
        return {(r[0], r[1]): r[2] for r in rows}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
