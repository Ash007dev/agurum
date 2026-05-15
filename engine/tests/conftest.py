"""
engine/tests/conftest.py — shared fixtures for agurum tests.
"""
from __future__ import annotations
import pytest


def make_deploy(service: str, version: str, ts: str) -> dict:
    return {"ts": ts, "kind": "deploy", "service": service, "version": version, "actor": "ci"}


def make_metric(service: str, name: str, value: float, ts: str) -> dict:
    return {"ts": ts, "kind": "metric", "service": service, "name": name, "value": value}


def make_log(service: str, level: str, msg: str, ts: str, incident_id: str = None) -> dict:
    e = {"ts": ts, "kind": "log", "service": service, "level": level, "msg": msg}
    if incident_id:
        e["incident_id"] = incident_id
    return e


def make_incident_signal(service: str, incident_id: str, trigger: str, ts: str) -> dict:
    return {"ts": ts, "kind": "incident_signal", "service": service, "incident_id": incident_id, "trigger": trigger}


def make_remediation(service: str, incident_id: str, action: str, outcome: str, ts: str) -> dict:
    return {"ts": ts, "kind": "remediation", "service": service, "incident_id": incident_id, "action": action, "outcome": outcome, "target": service}


def make_rename(old: str, new: str, ts: str) -> dict:
    return {"ts": ts, "kind": "topology", "change": "rename", "from_": old, "to": new, "service": new}


@pytest.fixture
def scenario_events() -> list[dict]:
    """Pre-canned 5-service incident scenario."""
    return [
        make_deploy("payments-svc", "v2.14.0", "2026-05-10T14:21:30Z"),
        make_metric("payments-svc", "latency_p99_ms", 4820, "2026-05-10T14:22:01Z"),
        make_log("checkout-api", "error", "timeout calling payments-svc", "2026-05-10T14:22:01Z"),
        make_log("payments-svc", "error", "connection timeout", "2026-05-10T14:22:05Z"),
        make_metric("payments-svc", "latency_p99_ms", 5200, "2026-05-10T14:22:30Z"),
        make_incident_signal("payments-svc", "INC-714", "alert:payments-svc/error-rate>5%", "2026-05-10T14:32:11Z"),
        make_remediation("payments-svc", "INC-714", "rollback", "resolved", "2026-05-10T15:10:00Z"),
    ]
