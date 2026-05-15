"""
tests/conftest.py — P1 writes these; everyone else uses them.

Provides shared fixtures for the gateway test suite.
toy_topology follows the plan spec: {services: [...], edges: [...]}.
"""
import pytest
from engine.models import Context, CausalEdge


@pytest.fixture
def toy_topology():
    """5-service directed graph: gateway-svc → api-svc → payments-svc → db-svc,
    payments-svc → cache-svc.
    Format: {services: list[str], edges: list[tuple[str, str]]}
    """
    return {
        "services": [
            "gateway-svc",
            "api-svc",
            "payments-svc",
            "db-svc",
            "cache-svc",
        ],
        "edges": [
            ("gateway-svc", "api-svc"),
            ("api-svc", "payments-svc"),
            ("payments-svc", "db-svc"),
            ("payments-svc", "cache-svc"),
        ],
    }


@pytest.fixture
def deploy_then_spike_events(toy_topology):
    """Deploy on payments-svc followed by latency spike — fires DEPLOY_PRECEDES_METRIC_SPIKE."""
    return [
        {
            "ts": "2026-05-15T12:00:00Z",
            "kind": "deploy",
            "service": "payments-svc",
            "version": "v2.14.0",
            "actor": "ci",
        },
        {
            "ts": "2026-05-15T12:05:00Z",
            "kind": "metric",
            "service": "payments-svc",
            "name": "latency_p99_ms",
            "value": 4820.0,
        },
        {
            "ts": "2026-05-15T12:05:30Z",
            "kind": "incident_signal",
            "service": "payments-svc",
            "incident_id": "INC-1715774730-3",
            "trigger": "alert:payments-svc/latency_p99_ms>4000",
        },
    ]


@pytest.fixture
def rename_then_incident_events(toy_topology):
    """Rename payments-svc → billing-svc, then the same incident pattern fires on billing-svc."""
    return [
        # Rename event — uses from_ (underscore), matching bench schema B3
        {
            "ts": "2026-05-15T10:00:00Z",
            "kind": "topology",
            "change": "rename",
            "from_": "payments-svc",
            "to": "billing-svc",
        },
        {
            "ts": "2026-05-15T12:00:00Z",
            "kind": "deploy",
            "service": "billing-svc",
            "version": "v2.15.0",
            "actor": "ci",
        },
        {
            "ts": "2026-05-15T12:05:00Z",
            "kind": "metric",
            "service": "billing-svc",
            "name": "latency_p99_ms",
            "value": 5100.0,
        },
        {
            "ts": "2026-05-15T12:05:30Z",
            "kind": "incident_signal",
            "service": "billing-svc",
            "incident_id": "INC-1715774730-3",
            "trigger": "alert:billing-svc/latency_p99_ms>4000",
        },
    ]