-- agurum v2 — Entity Registry SQLite Schema
-- Production path (Path B): temporal alias resolution with WAL mode.

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Canonical entities: one row per unique service identity (UUID4).
CREATE TABLE IF NOT EXISTS entities (
    canonical_id  TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    metadata      TEXT
);

-- Alias history: tracks every name a service has ever had,
-- with temporal validity windows. Old aliases are NEVER deleted —
-- ts_valid_to is set when retired, NULL means currently active.
CREATE TABLE IF NOT EXISTS aliases (
    alias_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id  TEXT NOT NULL REFERENCES entities(canonical_id),
    name          TEXT NOT NULL,
    ts_valid_from TEXT NOT NULL,
    ts_valid_to   TEXT
);

-- Hot-path index for resolve(): lookup by name + check if currently active.
-- Without this, every resolve() is a full table scan.
CREATE INDEX IF NOT EXISTS idx_alias_lookup ON aliases(name, ts_valid_to);

-- Entity relationships: directed graph edges between canonical entities.
CREATE TABLE IF NOT EXISTS entity_relationships (
    rel_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    caller_cid    TEXT NOT NULL REFERENCES entities(canonical_id),
    callee_cid    TEXT NOT NULL REFERENCES entities(canonical_id),
    relationship  TEXT NOT NULL,
    weight        REAL DEFAULT 1.0,
    last_seen     TEXT,
    UNIQUE(caller_cid, callee_cid, relationship)
);

CREATE INDEX IF NOT EXISTS idx_rel_caller ON entity_relationships(caller_cid);
