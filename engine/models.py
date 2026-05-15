"""
Single source of truth for all shared types.

v2: field names match benchmark harness schema exactly. Do not edit without
team sync — wrong field names silently zero out benchmark scores.

CRITICAL FIELD NAME RULES:
  - IncidentMatch uses "incident_id" — NOT "past_incident_id"
    Scorer calls: m.get("incident_id", "")
  - Remediation uses "action" with value "rollback"
    Scorer calls: any(s.get("action") == expected ...)
  - Event rename events use "from_" (underscore) — NOT "from"
    Python keyword conflict; generator outputs "from_" intentionally.
"""
from __future__ import annotations

from typing import TypedDict, Optional, Literal, Any
from dataclasses import dataclass, field


# ── Benchmark interface types (must match bench schema exactly) ──────────────


class Event(TypedDict, total=False):
    """Represents every event coming from the benchmark generator."""
    ts: str                     # ISO8601 timestamp — present on all events
    kind: str                   # deploy|log|metric|trace|topology|
                                # incident_signal|remediation
    service: str                # service name — present on most events
    incident_id: str            # present on incident_signal and remediation events
    from_: str                  # UNDERSCORE — rename topology events use from_, not from
    to: str                     # rename topology events: new service name
    action: str                 # remediation events: always "rollback" in generated data
    outcome: str                # remediation events: "resolved" | "failed"
    level: str                  # log events: "error" | "warn" | "info"
    msg: str                    # log events: e.g. "timeout calling svc-00"
    name: str                   # metric events: e.g. "latency_p99_ms" — NOT "metric_name"
    value: float                # metric events: e.g. 4820.0
    version: str                # deploy events: e.g. "v2.14.0"
    actor: str                  # deploy events: "ci"
    change: str                 # topology events: "rename" | "dep_add" | "dep_remove"
    trigger: str                # incident_signal: "alert:svc-name/metric>threshold"
    target: str                 # remediation events: service targeted
    trace_id: str               # trace events
    spans: list                 # trace events: list[dict]
    attrs: dict                 # extensible attributes


class IncidentSignal(TypedDict, total=False):
    """Passed to reconstruct_context() by the benchmark harness."""
    incident_id: str            # format: INC-{timestamp_mod}-{family_index}
    ts: str                     # ISO8601
    trigger: str                # "alert:service_name/metric>threshold"
    service: str                # SAFEST way to get trigger service — use this first


class IncidentMatch(TypedDict, total=False):
    """
    CRITICAL: field is "incident_id" — scorer calls m.get("incident_id")
    Using "past_incident_id" here produces recall@5 = 0.0 on every query.
    """
    incident_id: str            # ← MUST be this exact name
    similarity: float           # combined score 0.0–1.0
    rationale: str              # human-readable explanation string


class Remediation(TypedDict, total=False):
    """
    CRITICAL: "action" field must equal "rollback" for remediation_acc credit.
    Scorer: any(s.get("action") == expected for s in suggested_remediations)
    """
    action: str                 # MUST be "rollback" to score
    target: str                 # current name of the target service
    historical_outcome: str     # e.g. "resolved 4/4"
    confidence: float           # 0.0–1.0
    # NOTE: no supporting_incidents field — bench schema does not have it


class Context(TypedDict):
    """Return type of reconstruct_context(). All fields required."""
    related_events: list[dict]
    causal_chain: list[CausalEdge]
    similar_past_incidents: list[IncidentMatch]
    suggested_remediations: list[Remediation]
    confidence: float
    explain: str


# ── Internal types ────────────────────────────────────────────────────────────


@dataclass
class CausalEdge:
    """Represents a causal link between two entities in an incident."""
    cause_id: str               # canonical UUID of cause entity
    effect_id: str              # canonical UUID of effect entity
    evidence: str               # description of why this edge exists
    confidence: float           # 0.0–1.0
    rule: str                   # name of the CAUSAL_RULE that fired


@dataclass
class RoleNormalizedEvent:
    """
    Event projected into role-space. Service names and canonical_ids
    are NEVER used in embedding strings — only roles and event characteristics.
    """
    roles: list[str]            # topology + functional + temporal roles
    event_type: str             # DEPLOY|METRIC_SPIKE|LOG_ERROR|TRACE|INCIDENT_SIGNAL
    severity_category: str      # CRITICAL|HIGH|MEDIUM|LOW|NONE
    metric_category: str        # LATENCY|ERROR_RATE|THROUGHPUT|NONE
    metric_direction: str       # SPIKE|DROP|STABLE|NONE
    error_class: str            # TIMEOUT|CONNECTION|RATE_LIMIT|CRASH|GENERIC|NONE
    temporal_slot: int           # int(unix_timestamp / 60)
    canonical_id: str           # for linkage only — NEVER in embedding strings


@dataclass
class Episode:
    incident_id: str
    canonical_service_id: str
    action: str
    outcome: str
    event_count: int
    seq_vector: Any   # np.ndarray (384,) — sequence embedding
    event_vectors: Any  # np.ndarray (n, 384) — per-event embeddings
    family: int = -1    # extracted from incident_id suffix
