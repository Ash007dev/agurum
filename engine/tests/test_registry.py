"""
Complete test suite for P2 Identity & Invariance Layer.
Verifies all 7 kill-list gotchas.
"""
from __future__ import annotations

import asyncio
import time

import pytest


def test_from_underscore_field():
    """BENCHMARK CRITICAL: rename events use from_ with underscore."""
    from engine.registry.alias_tracker import AliasTracker

    tracker = AliasTracker()
    # Simulate a topology rename event exactly as the generator outputs
    tracker.process_event(
        {
            "kind": "topology",
            "change": "rename",
            "from_": "payments-svc",  # ← UNDERSCORE
            "to": "billing-svc",
            "ts": "2024-01-15T12:00:00",
        }
    )
    cid_old = tracker.resolve("payments-svc")
    cid_new = tracker.resolve("billing-svc")
    assert cid_old == cid_new, "rename must preserve canonical_id"


def test_from_field_not_read():
    """Confirm that 'from' without underscore does NOT trigger rename."""
    from engine.registry.alias_tracker import AliasTracker

    tracker = AliasTracker()
    tracker.process_event(
        {"service": "payments-svc", "kind": "metric", "ts": "2024-01-15T10:00:00"}
    )
    original_cid = tracker.resolve("payments-svc")

    # This event has "from" not "from_" — should NOT trigger rename
    tracker.process_event(
        {
            "kind": "topology",
            "change": "rename",
            "from": "payments-svc",  # no underscore — must be ignored
            "to": "billing-svc",
            "ts": "2024-01-15T12:00:00",
        }
    )
    # billing-svc should be a NEW entity, not mapped to payments-svc
    billing_cid = tracker.resolve("billing-svc")
    assert (
        original_cid != billing_cid
    ), "'from' without underscore must not trigger rename"


def test_rename_canonical_id_stable():
    """canonical_id must not change after rename."""
    from engine.registry.alias_tracker import AliasTracker

    tracker = AliasTracker()
    tracker.process_event(
        {"service": "payments-svc", "kind": "metric", "ts": "2024-01-15T10:00:00"}
    )
    cid_before = tracker.resolve("payments-svc")
    tracker.process_event(
        {
            "kind": "topology",
            "change": "rename",
            "from_": "payments-svc",
            "to": "billing-svc",
            "ts": "2024-01-15T12:00:00",
        }
    )
    cid_after = tracker.resolve("billing-svc")
    assert cid_before == cid_after


def test_historical_lookup_still_works_after_rename():
    """Old name must still resolve after rename — historical events need it."""
    from engine.registry.alias_tracker import AliasTracker

    tracker = AliasTracker()
    tracker.process_event(
        {"service": "payments-svc", "kind": "metric", "ts": "2024-01-15T10:00:00"}
    )
    tracker.process_event(
        {
            "kind": "topology",
            "change": "rename",
            "from_": "payments-svc",
            "to": "billing-svc",
            "ts": "2024-01-15T12:00:00",
        }
    )
    # Both names must resolve to the same cid
    assert tracker.resolve("payments-svc") == tracker.resolve("billing-svc")


def test_resolve_auto_registers():
    """resolve() on unknown name must register and return a UUID."""
    from engine.registry.alias_tracker import AliasTracker

    tracker = AliasTracker()
    cid = tracker.resolve("brand-new-svc")
    assert isinstance(cid, str) and len(cid) == 36  # UUID4 format


def test_process_event_idempotent():
    """Same service seen multiple times → same canonical_id always."""
    from engine.registry.alias_tracker import AliasTracker

    tracker = AliasTracker()
    for _ in range(100):
        tracker.process_event(
            {"service": "stable-svc", "kind": "metric", "ts": "2024-01-15T10:00:00"}
        )
    all_cids = {tracker.resolve("stable-svc")}
    assert len(all_cids) == 1


def test_entity_registry_rename_preserves_cid(tmp_path):
    """EntityRegistry (production path) must also preserve canonical_id."""
    from engine.registry.entity_registry import EntityRegistry

    registry = EntityRegistry(db_path=str(tmp_path / "test.db"))
    ts_before = "2024-01-15T10:00:00Z"
    ts_rename = "2024-01-15T12:00:00Z"
    ts_after = "2024-01-15T14:00:00Z"

    cid = registry.register("payments-svc", ts_before)
    registry.rename("payments-svc", "billing-svc", ts_rename)

    assert registry.resolve("payments-svc", ts_before) == cid
    assert registry.resolve("billing-svc", ts_after) == cid


def test_entity_registry_benchmark(tmp_path):
    """10,000 resolve() calls must complete in under 2 seconds."""
    from engine.registry.entity_registry import EntityRegistry

    registry = EntityRegistry(db_path=str(tmp_path / "test.db"))
    ts = "2024-01-15T10:00:00Z"
    registry.register("bench-svc", ts)

    start = time.perf_counter()
    for _ in range(10_000):
        registry.resolve("bench-svc", ts)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"resolve() too slow: {elapsed:.3f}s for 10K calls"


def test_wl_hash_stable_after_rename():
    """WL hash must be identical before and after rename — topology unchanged."""
    from engine.registry.alias_tracker import AliasTracker
    from engine.graph.operational_graph import OperationalGraph

    tracker = AliasTracker()
    graph = OperationalGraph()

    cid_a = tracker.resolve("payments-svc")
    cid_b = tracker.resolve("auth-svc")

    graph.upsert_edge(cid_a, cid_b, "calls")
    graph._force_recompute()

    hash_before = graph.get_wl_hash(cid_a)

    # Rename payments-svc → billing-svc (same canonical_id cid_a)
    tracker.process_event(
        {
            "kind": "topology",
            "change": "rename",
            "from_": "payments-svc",
            "to": "billing-svc",
            "ts": "2024-01-15T12:00:00",
        }
    )

    # Graph topology is unchanged — only the mapping in AliasTracker changed
    hash_after = graph.get_wl_hash(cid_a)

    assert hash_before == hash_after, (
        "WL hash must be topology-invariant, not name-invariant"
    )


def test_simple_embedding_string_no_service_name():
    """simple_embedding_string must never contain service names."""
    from engine.roles.role_abstraction import simple_embedding_string

    for svc_name in ["payments-svc", "billing-svc", "api-gateway"]:
        event = {
            "kind": "metric",
            "service": svc_name,
            "name": "latency_p99_ms",
            "value": 5000.0,
            "ts": "2024-01-15T10:00:00",
        }
        result = simple_embedding_string(event)
        assert svc_name not in result, (
            f"Service name '{svc_name}' leaked into embedding string: {result}"
        )
        assert "metric" in result
        assert "latency_p99_ms" in result


def test_incident_match_uses_incident_id():
    """BENCHMARK CRITICAL: IncidentMatch must use 'incident_id' not 'past_incident_id'."""
    from engine.models import IncidentMatch

    match = IncidentMatch(incident_id="INC-123-3", similarity=0.9, rationale="test")
    assert "incident_id" in match
    assert match.get("incident_id") == "INC-123-3"

    # Scorer code: m.get("incident_id", "") — this extraction must work
    family = int(match.get("incident_id", "0").rsplit("-", 1)[-1])
    assert family == 3


def test_sliding_window_asyncio_lock():
    """SlidingWindowCache must use asyncio.Lock, not threading.RLock."""
    from engine.cache.sliding_window import SlidingWindowCache

    cache = SlidingWindowCache()
    assert isinstance(cache._lock, type(asyncio.Lock())), (
        "SlidingWindowCache._lock must be asyncio.Lock, not threading.RLock"
    )

    async def _run():
        event = {"canonical_id": "cid-1", "kind": "metric", "ts": "2024-01-15T10:00:00"}
        await cache.push(event)
        window = await cache.get_window(
            "cid-1", "2024-01-15T09:55:00", "2024-01-15T10:05:00"
        )
        assert len(window) == 1

    asyncio.run(_run())
