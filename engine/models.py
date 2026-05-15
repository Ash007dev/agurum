"""
engine/models.py — shared TypedDicts and dataclasses for the Agurum engine.

Written once (Hour 0), locked. All modules import from here.
Matches bench-p02-context/schema.py exactly for benchmark-facing types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


# ── Benchmark interface types (must match bench schema.py exactly) ────────────

class Event(TypedDict, total=False):
    ts: str
    kind: str               # deploy|log|metric|trace|topology|incident_signal|remediation
    service: str
    incident_id: str
    from_: str              # UNDERSCORE — rename events use from_ not from
    to: str
    action: str             # always "rollback" in generated data
    outcome: str            # resolved | failed
    # kind-specific fields (all optional, matching bench schema.py exactly)
    level: str              # log events: "error", "warn", "info"
    msg: str                # log events: e.g. "timeout calling svc-00"
    name: str               # metric events: e.g. "latency_p99_ms" (NOT "metric_name")
    value: float            # metric events: e.g. 4820.0
    version: str            # deploy events: e.g. "v2.14.0"
    actor: str              # deploy events: "ci"
    change: str             # topology events: "rename" | "dep_add" | "dep_remove"
    trigger: str            # incident_signal: "alert:svc-name/metric>threshold"
    target: str             # remediation: service targeted
    trace_id: str
    spans: list[dict[str, Any]]
    attrs: dict[str, Any]


class IncidentSignal(TypedDict, total=False):
    incident_id: str        # format: INC-{timestamp_mod}-{family_index}
    ts: str
    trigger: str            # "alert:service_name/metric>threshold"
    service: str            # trigger service name — SAFEST identity field


class CausalEdge(TypedDict, total=False):
    cause_event_id: str
    effect_event_id: str
    evidence: str
    confidence: float


class IncidentMatch(TypedDict, total=False):
    incident_id: str        # NOT "past_incident_id" — scorer reads m.get("incident_id")
    similarity: float
    rationale: str


class Remediation(TypedDict, total=False):
    action: str             # always "rollback" in generated data
    target: str
    historical_outcome: str
    confidence: float
    # NOTE: no supporting_incidents — not in bench schema.py


class Context(TypedDict, total=False):
    related_events: list[Event]
    causal_chain: list[CausalEdge]
    similar_past_incidents: list[IncidentMatch]
    suggested_remediations: list[Remediation]
    confidence: float
    explain: str


# ── Internal types ────────────────────────────────────────────────────────────

@dataclass
class RoleNormalizedEvent:
    roles: list[str]
    event_type: str           # DEPLOY | METRIC_SPIKE | LOG_ERROR | TRACE | INCIDENT_SIGNAL
    severity_category: str    # CRITICAL | HIGH | MEDIUM | LOW
    metric_category: str      # LATENCY | ERROR_RATE | THROUGHPUT | NONE
    metric_direction: str     # SPIKE | DROP | STABLE | NONE
    error_class: str          # TIMEOUT | CONNECTION | RATE_LIMIT | CRASH | NONE
    temporal_slot: int
    canonical_id: str         # for linkage only — NOT in embeddings


@dataclass
class Episode:
    incident_id: str
    canonical_service_id: str
    action: str
    outcome: str
    event_count: int
    seq_vector: "Any"   # np.ndarray (384,) — sequence embedding
    event_vectors: "Any"  # np.ndarray (n, 384) — per-event embeddings
    family: int = -1    # extracted from incident_id suffix
