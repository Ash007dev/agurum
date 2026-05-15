-- engine/store/schema.sql
-- DuckDB schema for Path B (production FastAPI pipeline).
-- NOT used by the benchmark adapter (Path A uses InMemoryStore + NumpyBehavioralIndex).
-- Run via: duckdb engine/store/pce.db < engine/store/schema.sql

-- ── Events table ─────────────────────────────────────────────────────────────
-- Append-only. Indexed on (canonical_id, ts) for window queries.

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY,          -- auto-assigned row id
    ts           TIMESTAMPTZ   NOT NULL,       -- event timestamp (UTC)
    kind         VARCHAR(32)   NOT NULL,       -- log|metric|trace|deploy|topology|incident_signal|remediation
    canonical_id VARCHAR(64)   NOT NULL,       -- resolved canonical service ID from AliasTracker
    raw_service  VARCHAR(255),                 -- original service name as received
    incident_id  VARCHAR(128),                 -- NULL for non-incident events
    payload      JSON          NOT NULL        -- full raw event dict
);

CREATE INDEX IF NOT EXISTS idx_events_canonical_ts
    ON events (canonical_id, ts);

CREATE INDEX IF NOT EXISTS idx_events_incident
    ON events (incident_id)
    WHERE incident_id IS NOT NULL;

-- ── Episodes table ────────────────────────────────────────────────────────────
-- One row per resolved incident. seq_vector stored as BLOB (float32 little-endian).
-- Qdrant holds the searchable vectors; this table is the authoritative metadata store.

CREATE TABLE IF NOT EXISTS episodes (
    incident_id  VARCHAR(128)  PRIMARY KEY,
    canonical_id VARCHAR(64)   NOT NULL,
    action       VARCHAR(64)   NOT NULL DEFAULT 'rollback',
    outcome      VARCHAR(32)   NOT NULL DEFAULT 'resolved',
    event_count  INTEGER       NOT NULL DEFAULT 0,
    family       INTEGER       NOT NULL DEFAULT -1,  -- suffix of INC-...-{family}
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT now(),
    -- seq_vector stored in Qdrant under the same incident_id key
    qdrant_id    VARCHAR(64)   -- optional reference if IDs differ
);

CREATE INDEX IF NOT EXISTS idx_episodes_canonical
    ON episodes (canonical_id);

CREATE INDEX IF NOT EXISTS idx_episodes_family
    ON episodes (family)
    WHERE family >= 0;

-- ── Alias / rename log ────────────────────────────────────────────────────────
-- Audit trail of all service renames. AliasTracker in-memory state is rebuilt
-- from this table on restart.

CREATE TABLE IF NOT EXISTS alias_log (
    id           INTEGER PRIMARY KEY,
    canonical_id VARCHAR(64)  NOT NULL,
    name         VARCHAR(255) NOT NULL,
    first_seen   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    is_current   BOOLEAN      NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_alias_name
    ON alias_log (name);

CREATE INDEX IF NOT EXISTS idx_alias_canonical
    ON alias_log (canonical_id);

-- ── Continuous Learning ──────────────────────────────────────────────────────
-- Tracks EWMA (Exponentially Weighted Moving Average) co-occurrence counts
-- between canonical entities for causal confidence boosting.

CREATE TABLE IF NOT EXISTS entity_pair_stats (
    canonical_id_a VARCHAR(64) NOT NULL,
    canonical_id_b VARCHAR(64) NOT NULL,
    ewma_weight    DOUBLE      NOT NULL DEFAULT 0.0,
    PRIMARY KEY (canonical_id_a, canonical_id_b)
);
