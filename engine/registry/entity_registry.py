"""
Production-path entity registry with SQLite temporal alias resolution.

Called ONLY via run_in_executor in the async FastAPI path — never called
directly from a coroutine. Uses threading.RLock (CORRECT — executor
worker threads, NOT async coroutines).

Performance targets:
  resolve()  : <0.2ms cached, <0.5ms uncached
  rename()   : <5ms (two writes in one transaction)
  register() : <1ms
"""
from __future__ import annotations

import collections
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Path to schema.sql relative to this file
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# LRU cache size limit
_MAX_CACHE_SIZE = 1000


class EntityRegistry:
    """
    SQLite-backed entity registry with temporal validity for time-range lookups.

    Every service name is an alias of a canonical UUID. When a service is
    renamed, the old alias is retired (ts_valid_to set) and a new alias is
    created pointing to the same canonical_id. The canonical_id NEVER changes.

    Thread safety: threading.RLock — correct here because this class is
    called from ThreadPoolExecutor workers, NOT from async coroutines.
    """

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Create tables from schema.sql
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        self._conn.executescript(schema_sql)

        # LRU cache: (name, ts_bucket) → canonical_id
        self._cache: dict[tuple[str, int], str] = {}
        self._cache_order: collections.OrderedDict = collections.OrderedDict()

        # threading.RLock is CORRECT — executor threads, not coroutines
        self._lock = threading.RLock()

        # Pre-warm cache from most recent aliases
        self._prewarm()

    def _prewarm(self) -> None:
        """Load the 100 most recent active aliases into cache."""
        cursor = self._conn.execute(
            "SELECT name, canonical_id, ts_valid_from FROM aliases "
            "ORDER BY alias_id DESC LIMIT 100"
        )
        for name, cid, ts_from in cursor.fetchall():
            try:
                ts_bucket = int(
                    datetime.fromisoformat(
                        ts_from.replace("Z", "+00:00")
                    ).timestamp() / 60
                )
                key = (name, ts_bucket)
                self._cache[key] = cid
                self._cache_order[key] = None
            except (ValueError, AttributeError):
                continue

    def _cache_put(self, key: tuple[str, int], value: str) -> None:
        """Add to LRU cache, evicting oldest if over limit."""
        if key in self._cache:
            self._cache_order.move_to_end(key)
            return
        self._cache[key] = value
        self._cache_order[key] = None
        while len(self._cache) > _MAX_CACHE_SIZE:
            oldest = next(iter(self._cache_order))
            del self._cache_order[oldest]
            del self._cache[oldest]

    def _cache_invalidate_name(self, name: str) -> None:
        """Remove all cache entries for a given service name."""
        to_remove = [k for k in self._cache if k[0] == name]
        for k in to_remove:
            del self._cache[k]
            self._cache_order.pop(k, None)

    def resolve(self, name: str, at_ts: str) -> str:
        """
        Returns canonical_id for name at the given ISO8601 timestamp.
        Raises KeyError if not found.

        Cache key: (name, int(timestamp / 60))
        Performance: <0.2ms cached, <0.5ms uncached.
        """
        with self._lock:
            ts_bucket = int(
                datetime.fromisoformat(
                    at_ts.replace("Z", "+00:00")
                ).timestamp() / 60
            )
            key = (name, ts_bucket)

            # Cache hit
            if key in self._cache:
                self._cache_order.move_to_end(key)
                return self._cache[key]

            # Cache miss — query SQLite
            cursor = self._conn.execute(
                "SELECT canonical_id FROM aliases "
                "WHERE name = ? AND ts_valid_from <= ? "
                "AND (ts_valid_to IS NULL OR ts_valid_to > ?) "
                "ORDER BY ts_valid_from DESC LIMIT 1",
                (name, at_ts, at_ts),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(
                    f"No entity found for name={name!r} at ts={at_ts!r}"
                )

            cid = row[0]
            self._cache_put(key, cid)
            return cid

    def register(self, name: str, at_ts: str,
                 metadata: Optional[dict] = None) -> str:
        """
        Register a new service name. If a current alias already exists
        (ts_valid_to IS NULL), return its canonical_id. Otherwise create
        a new entity + alias.
        """
        with self._lock:
            # Check if already registered with an active alias
            cursor = self._conn.execute(
                "SELECT canonical_id FROM aliases "
                "WHERE name = ? AND ts_valid_to IS NULL "
                "LIMIT 1",
                (name,),
            )
            row = cursor.fetchone()
            if row is not None:
                return row[0]

            # Create new entity
            cid = str(uuid.uuid4())
            meta_json = json.dumps(metadata) if metadata else None
            self._conn.execute(
                "INSERT INTO entities (canonical_id, created_at, metadata) "
                "VALUES (?, ?, ?)",
                (cid, at_ts, meta_json),
            )
            self._conn.execute(
                "INSERT INTO aliases (canonical_id, name, ts_valid_from) "
                "VALUES (?, ?, ?)",
                (cid, name, at_ts),
            )
            self._conn.commit()

            # Warm cache
            ts_bucket = int(
                datetime.fromisoformat(
                    at_ts.replace("Z", "+00:00")
                ).timestamp() / 60
            )
            self._cache_put((name, ts_bucket), cid)

            return cid

    def rename(self, old_name: str, new_name: str, ts: str) -> None:
        """
        THE CRITICAL METHOD. Rename old_name → new_name at timestamp ts.

        canonical_id MUST be IDENTICAL before and after rename.
        Steps 2 and 3 run in a single SQLite transaction.

        NOTE: caller passes event["from_"] as old_name — the underscore
        is the caller's responsibility; rename() just takes old/new strings.
        """
        with self._lock:
            # Step 1: get canonical_id for old_name
            cursor = self._conn.execute(
                "SELECT canonical_id FROM aliases "
                "WHERE name = ? AND ts_valid_to IS NULL "
                "LIMIT 1",
                (old_name,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(
                    f"Cannot rename: no active alias for {old_name!r}"
                )
            cid = row[0]

            # Steps 2+3: retire old alias, create new — single transaction
            self._conn.execute(
                "UPDATE aliases SET ts_valid_to = ? "
                "WHERE name = ? AND ts_valid_to IS NULL",
                (ts, old_name),
            )
            self._conn.execute(
                "INSERT INTO aliases (canonical_id, name, ts_valid_from) "
                "VALUES (?, ?, ?)",
                (cid, new_name, ts),
            )
            self._conn.commit()

            # Step 4: invalidate cache entries for both names
            self._cache_invalidate_name(old_name)
            self._cache_invalidate_name(new_name)

    def get_current_name(self, canonical_id: str) -> str:
        """Return the currently active name for a canonical_id."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT name FROM aliases "
                "WHERE canonical_id = ? AND ts_valid_to IS NULL "
                "LIMIT 1",
                (canonical_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else canonical_id

    def get_history(self, canonical_id: str) -> list[dict]:
        """Return full alias history for a canonical_id, ordered by time."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT name, ts_valid_from, ts_valid_to FROM aliases "
                "WHERE canonical_id = ? ORDER BY ts_valid_from ASC",
                (canonical_id,),
            )
            return [
                {"name": name, "from": ts_from, "to": ts_to}
                for name, ts_from, ts_to in cursor.fetchall()
            ]

    def update_roles(self, canonical_id: str, roles: list[str]) -> None:
        """Merge roles into entity metadata JSON."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT metadata FROM entities WHERE canonical_id = ?",
                (canonical_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return
            meta = json.loads(row[0]) if row[0] else {}
            meta["roles"] = roles
            self._conn.execute(
                "UPDATE entities SET metadata = ? WHERE canonical_id = ?",
                (json.dumps(meta), canonical_id),
            )
            self._conn.commit()

    def get_role(self, canonical_id: str) -> Optional[list[str]]:
        """Read roles from entity metadata JSON."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT metadata FROM entities WHERE canonical_id = ?",
                (canonical_id,),
            )
            row = cursor.fetchone()
            if row is None or row[0] is None:
                return None
            meta = json.loads(row[0])
            return meta.get("roles")

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
