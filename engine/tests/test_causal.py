"""
engine/tests/test_causal.py — unit tests for CausalEdgeExtractor.
"""
from __future__ import annotations
import pytest
from engine.causal.causal_extractor import CausalEdgeExtractor
from engine.tests.conftest import make_deploy, make_metric, make_log, make_incident_signal


def test_deploy_precedes_metric_spike():
    extractor = CausalEdgeExtractor()
    events = [
        make_deploy("svc-a", "v1.0", "2026-05-10T14:00:00Z"),
        make_metric("svc-a", "latency_p99_ms", 5000, "2026-05-10T14:01:00Z"),
    ]
    edges = extractor.extract(events)
    assert len(edges) >= 1
    assert any("DEPLOY" in e.get("evidence", "").upper() or "deploy" in e.get("evidence", "").lower() for e in edges)
    assert all(0 < e["confidence"] <= 1.0 for e in edges)


def test_metric_spike_precedes_error():
    extractor = CausalEdgeExtractor()
    events = [
        make_metric("svc-a", "latency_p99_ms", 5000, "2026-05-10T14:00:00Z"),
        make_log("svc-b", "error", "timeout calling svc-a", "2026-05-10T14:00:30Z"),
    ]
    edges = extractor.extract(events)
    assert len(edges) >= 1


def test_error_precedes_incident():
    extractor = CausalEdgeExtractor()
    events = [
        make_log("svc-a", "error", "connection refused", "2026-05-10T14:00:00Z"),
        make_incident_signal("svc-a", "INC-1", "alert:svc-a/error>5%", "2026-05-10T14:01:00Z"),
    ]
    edges = extractor.extract(events)
    assert len(edges) >= 1
    assert edges[0]["confidence"] >= 0.8


def test_no_edges_when_gap_too_large():
    extractor = CausalEdgeExtractor()
    events = [
        make_deploy("svc-a", "v1.0", "2026-05-10T14:00:00Z"),
        make_metric("svc-a", "latency_p99_ms", 5000, "2026-05-10T15:00:00Z"),  # 1 hour gap
    ]
    edges = extractor.extract(events)
    assert len(edges) == 0


def test_confidence_capped():
    extractor = CausalEdgeExtractor(pair_stats={("a", "b"): 1000})

    from engine.registry.alias_tracker import AliasTracker
    tracker = AliasTracker()
    events = [
        make_deploy("svc-a", "v1.0", "2026-05-10T14:00:00Z"),
        make_metric("svc-a", "latency_p99_ms", 5000, "2026-05-10T14:01:00Z"),
    ]
    edges = extractor.extract(events, tracker=tracker)
    assert all(e["confidence"] <= 0.98 for e in edges)


def test_empty_events():
    extractor = CausalEdgeExtractor()
    assert extractor.extract([]) == []


def test_sorted_by_confidence():
    extractor = CausalEdgeExtractor()
    events = [
        make_deploy("svc-a", "v1.0", "2026-05-10T14:00:00Z"),
        make_metric("svc-a", "latency_p99_ms", 5000, "2026-05-10T14:01:00Z"),
        make_log("svc-a", "error", "crash", "2026-05-10T14:01:30Z"),
        make_incident_signal("svc-a", "INC-1", "alert:svc-a/error>5%", "2026-05-10T14:02:00Z"),
    ]
    edges = extractor.extract(events)
    confidences = [e["confidence"] for e in edges]
    assert confidences == sorted(confidences, reverse=True)
