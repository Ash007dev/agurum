"""
Role abstraction layer: converts raw events into role-normalized representations.

CARDINAL RULE: service names and canonical_ids MUST NEVER appear in any string
fed to the embedding model. This is the invariance guarantee that makes
incident matching work across renames.

Provides two levels:
  1. simple_embedding_string() — standalone function, no dependencies, usable
     immediately by P3's benchmark adapter. Uses actual generator field names.
  2. RoleAbstractionLayer class — full production normalization using registry
     and graph lookups.

FIELD NAME NOTE (from generator.py):
  metric events: "name" (e.g. "latency_p99_ms"), "value" (e.g. 4820.0)
  log events:    "level" (e.g. "error"), "msg" (e.g. "timeout calling svc-00")
  deploy events: "version", "actor"
  remediation:   "action" ("rollback"), "outcome" ("resolved")
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from engine.models import RoleNormalizedEvent

if TYPE_CHECKING:
    from engine.registry.entity_registry import EntityRegistry
    from engine.graph.operational_graph import OperationalGraph

logger = logging.getLogger(__name__)


# ── Taxonomy constants ────────────────────────────────────────────────────────

TOPOLOGY_ROLES = [
    "HUB_SVC", "BRIDGE_SVC", "LEAF_SVC", "EDGE_SVC", "CORE_SVC", "SOLO_SVC",
]
FUNCTIONAL_ROLES = [
    "REQUEST_HANDLER", "QUEUE_CONSUMER", "STORE", "COMPUTE",
]
TEMPORAL_ROLES = [
    "TRIGGER_SVC", "UPSTREAM_SVC", "DOWNSTREAM_SVC",
]
EVENT_TYPES = [
    "DEPLOY", "METRIC_SPIKE", "LOG_ERROR", "TRACE", "INCIDENT_SIGNAL", "UNKNOWN",
]
SEVERITY_CATEGORIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
METRIC_CATEGORIES = ["LATENCY", "ERROR_RATE", "THROUGHPUT", "NONE"]
METRIC_DIRECTIONS = ["SPIKE", "DROP", "STABLE", "NONE"]
ERROR_CLASSES = [
    "TIMEOUT", "CONNECTION", "RATE_LIMIT", "CRASH", "GENERIC", "NONE",
]


# ── Standalone function (no class needed — P3 uses this directly) ─────────────


def simple_embedding_string(event: dict) -> str:
    """
    Rename-robust event string using ACTUAL generator field names.

    Uses: name (metric), level (log), msg (log), action/outcome (remediation).
    Does NOT use: metric_name, error_class, severity — those don't exist in data.
    Service name is ABSENT from output — rename-robust by construction.

    Examples:
      "metric latency_p99_ms high"
      "log error timeout"
      "deploy deployment"
      "remediation rollback resolved"
    """
    kind = event.get("kind", "unknown")
    parts = [kind]

    if kind == "metric":
        parts.append(event.get("name", "unknown_metric"))   # "latency_p99_ms"
        val = event.get("value", 0)
        parts.append("high" if val > 3000 else "normal")
    elif kind == "log":
        parts.append(event.get("level", "info"))             # "error"|"warn"|"info"
        msg = event.get("msg", "")
        if "timeout" in msg.lower():
            parts.append("timeout")
        elif "error" in msg.lower():
            parts.append("error")
    elif kind == "deploy":
        parts.append("deployment")
    elif kind == "incident_signal":
        parts.append("alert")
    elif kind == "remediation":
        parts.append(event.get("action", "unknown"))         # "rollback"
        parts.append(event.get("outcome", "unknown"))        # "resolved"

    return " ".join(parts)


# ── Full production class ─────────────────────────────────────────────────────

# Event kind → event_type mapping
_KIND_MAP = {
    "deploy": "DEPLOY",
    "metric": "METRIC_SPIKE",
    "gauge": "METRIC_SPIKE",
    "log": "LOG_ERROR",
    "error": "LOG_ERROR",
    "trace": "TRACE",
    "span": "TRACE",
    "incident_signal": "INCIDENT_SIGNAL",
    "alert": "INCIDENT_SIGNAL",
}


class RoleAbstractionLayer:
    """
    Converts raw events into RoleNormalizedEvent instances using
    registry (identity) and graph (topology) lookups.
    """

    def __init__(self, registry: EntityRegistry, graph: OperationalGraph) -> None:
        self.registry = registry
        self.graph = graph

    def normalize_event(self, raw_event: dict) -> RoleNormalizedEvent:
        """
        Convert a raw event dict into a RoleNormalizedEvent.

        Steps:
          1. Extract service name
          2. Extract timestamp
          3. Resolve canonical_id via registry
          4. Get structural roles from graph
          5-9. Classify event type, severity, metric, error, temporal slot
        """
        # Step 1: extract service name
        service_name = (
            raw_event.get("service")
            or raw_event.get("source")
            or raw_event.get("entity")
            or raw_event.get("name")
        )
        if not service_name:
            raise ValueError(f"Cannot extract service name from event: {raw_event}")

        # Step 2: extract timestamp
        ts = raw_event.get("ts") or raw_event.get("timestamp", "")

        # Step 3: resolve canonical_id
        try:
            canonical_id = self.registry.resolve(service_name, ts)
        except KeyError:
            canonical_id = self.registry.register(service_name, ts)

        # Step 4: get structural roles
        roles = self.graph.get_all_roles(canonical_id)
        if not roles:
            roles = ["UNKNOWN_SVC"]

        # Step 5: map event kind → event_type
        kind = raw_event.get("kind", "")
        event_type = _KIND_MAP.get(kind, "UNKNOWN")

        # Step 6: determine severity_category
        severity_category = self._classify_severity(raw_event)

        # Step 7: determine metric_category and metric_direction
        metric_category = self._classify_metric_category(raw_event)
        metric_direction = self._classify_metric_direction(raw_event, event_type)

        # Step 8: determine error_class
        error_class = self._classify_error(raw_event)

        # Step 9: temporal_slot
        temporal_slot = 0
        if ts:
            try:
                temporal_slot = int(
                    datetime.fromisoformat(
                        ts.replace("Z", "+00:00")
                    ).timestamp() / 60
                )
            except (ValueError, AttributeError):
                pass

        return RoleNormalizedEvent(
            roles=roles,
            event_type=event_type,
            severity_category=severity_category,
            metric_category=metric_category,
            metric_direction=metric_direction,
            error_class=error_class,
            temporal_slot=temporal_slot,
            canonical_id=canonical_id,
        )

    def to_embedding_string(self, rne: RoleNormalizedEvent) -> str:
        """
        THE MOST IMPORTANT METHOD for production path.

        RULE: canonical_id and service names are FORBIDDEN in output.
        Roles are SORTED for deterministic output.

        Example: "BRIDGE_SVC_REQUEST_HANDLER_UPSTREAM_SVC DEPLOY NONE NONE NONE HIGH"
        """
        role_str = "_".join(sorted(rne.roles))
        return (
            f"{role_str} {rne.event_type} {rne.metric_category} "
            f"{rne.metric_direction} {rne.error_class} {rne.severity_category}"
        )

    def batch_normalize(self, raw_events: list[dict]) -> list[RoleNormalizedEvent]:
        """
        Normalize a batch of events. Skips failures with a warning.
        Never crashes the entire batch.
        """
        results = []
        for event in raw_events:
            try:
                results.append(self.normalize_event(event))
            except Exception as e:
                logger.warning("Skipping event in batch_normalize: %s", e)
        return results

    def batch_to_strings(self, rnes: list[RoleNormalizedEvent]) -> list[str]:
        """Convert a batch of RoleNormalizedEvents to embedding strings."""
        return [self.to_embedding_string(rne) for rne in rnes]

    # ── Private classification helpers ────────────────────────────────────

    @staticmethod
    def _classify_severity(event: dict) -> str:
        """Determine severity category from event fields."""
        severity_raw = str(
            event.get("severity", event.get("level", ""))
        ).lower()
        value = event.get("value")

        if "critical" in severity_raw or (isinstance(value, (int, float)) and value > 0.9 and value <= 1.0):
            return "CRITICAL"
        if "high" in severity_raw or "error" in severity_raw or (isinstance(value, (int, float)) and value > 0.7 and value <= 0.9):
            return "HIGH"
        if "medium" in severity_raw or "warn" in severity_raw or (isinstance(value, (int, float)) and value > 0.4 and value <= 0.7):
            return "MEDIUM"
        if severity_raw or (isinstance(value, (int, float)) and value > 0):
            return "LOW"
        return "NONE"

    @staticmethod
    def _classify_metric_category(event: dict) -> str:
        """Determine metric category from field names."""
        field_name = str(
            event.get("name", event.get("metric_name", ""))
        ).lower()

        if "latency" in field_name or "p99" in field_name or "p95" in field_name:
            return "LATENCY"
        if "error_rate" in field_name or "errors" in field_name:
            return "ERROR_RATE"
        if "throughput" in field_name or "rps" in field_name or "qps" in field_name:
            return "THROUGHPUT"
        return "NONE"

    @staticmethod
    def _classify_metric_direction(event: dict, event_type: str) -> str:
        """Determine metric direction for METRIC_SPIKE events."""
        if event_type != "METRIC_SPIKE":
            return "NONE"

        value = event.get("value", 0)
        desc = str(event.get("msg", event.get("description", ""))).lower()

        if (isinstance(value, (int, float)) and value > 3000) or "spike" in desc:
            return "SPIKE"
        if "drop" in desc:
            return "DROP"
        return "STABLE"

    @staticmethod
    def _classify_error(event: dict) -> str:
        """Determine error class from message/error fields."""
        error_str = str(
            event.get("msg", event.get("error", event.get("message", "")))
        ).lower()

        if not error_str:
            return "NONE"
        if "timeout" in error_str:
            return "TIMEOUT"
        if "connection" in error_str:
            return "CONNECTION"
        if "rate_limit" in error_str or "429" in error_str:
            return "RATE_LIMIT"
        if "crash" in error_str or "oom" in error_str or "segfault" in error_str:
            return "CRASH"

        # Check if this is actually an error event
        level = str(event.get("level", "")).lower()
        if level in ("error", "critical", "fatal"):
            return "GENERIC"

        return "NONE"
