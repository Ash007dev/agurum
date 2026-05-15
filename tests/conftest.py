import pytest
import os

@pytest.fixture
def toy_topology():
    """5-service directed graph: gateway → api → payments → db, payments → cache"""
    return {
        "gateway": ["api"],
        "api": ["payments"],
        "payments": ["db", "cache"],
        "db": [],
        "cache": []
    }

@pytest.fixture
def deploy_then_spike_events(toy_topology):
    """Deploy on payments-svc followed by latency spike — fires DEPLOY_PRECEDES_METRIC_SPIKE"""
    return [
        {"ts": "2026-05-15T12:00:00Z", "kind": "deploy", "service": "payments"},
        {"ts": "2026-05-15T12:05:00Z", "kind": "metric", "service": "payments", "metric_name": "latency", "value": "high"}
    ]

@pytest.fixture
def rename_then_incident_events(toy_topology):
    """Rename payments-svc → billing-svc, then same incident pattern on billing-svc"""
    return [
        {"ts": "2026-05-15T10:00:00Z", "kind": "rename", "from_": "payments", "to": "billing-svc"},
        {"ts": "2026-05-15T12:00:00Z", "kind": "deploy", "service": "billing-svc"},
        {"ts": "2026-05-15T12:05:00Z", "kind": "metric", "service": "billing-svc", "metric_name": "latency", "value": "high"}
    ]