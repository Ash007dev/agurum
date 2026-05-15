"""
engine/causal/causal_extractor.py — rule-based causal edge extraction.

5 causal rules with historical frequency boosting via entity_pair_stats.
O(n²) pairwise loop (n≤60) — all in-memory, ≤30ms.

Used by ContextReconstructor Phase 5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from engine.models import CausalEdge


@dataclass(frozen=True)
class CausalRule:
    """A single causal inference rule with time-gap constraints."""
    name: str
    cause_kind: str
    effect_kind: str
    base_confidence: float
    max_gap_seconds: float
    evidence_template: str

    def matches(self, cause: dict, effect: dict, gap_seconds: float) -> bool:
        if gap_seconds < 0 or gap_seconds > self.max_gap_seconds:
            return False
        if cause.get("kind") != self.cause_kind:
            return False
        if effect.get("kind") != self.effect_kind:
            return False
        return True


# ── The 5 causal rules from the architecture spec ────────────────────────────

CAUSAL_RULES: list[CausalRule] = [
    CausalRule(
        name="DEPLOY_PRECEDES_METRIC_SPIKE",
        cause_kind="deploy",
        effect_kind="metric",
        base_confidence=0.78,
        max_gap_seconds=120,
        evidence_template="Deploy {cause_svc} v{version} preceded metric spike {metric}={value}",
    ),
    CausalRule(
        name="DEPLOY_PRECEDES_LOG_ERROR",
        cause_kind="deploy",
        effect_kind="log",
        base_confidence=0.72,
        max_gap_seconds=180,
        evidence_template="Deploy {cause_svc} preceded error log: {msg}",
    ),
    CausalRule(
        name="METRIC_SPIKE_PRECEDES_DOWNSTREAM_ERROR",
        cause_kind="metric",
        effect_kind="log",
        base_confidence=0.74,
        max_gap_seconds=60,
        evidence_template="Metric spike {metric}={value} preceded downstream error: {msg}",
    ),
    CausalRule(
        name="ERROR_PRECEDES_INCIDENT_SIGNAL",
        cause_kind="log",
        effect_kind="incident_signal",
        base_confidence=0.85,
        max_gap_seconds=120,
        evidence_template="Error log '{msg}' preceded incident signal {trigger}",
    ),
    CausalRule(
        name="DEPLOY_PRECEDES_INCIDENT_SIGNAL",
        cause_kind="deploy",
        effect_kind="incident_signal",
        base_confidence=0.88,
        max_gap_seconds=300,
        evidence_template="Deploy {cause_svc} v{version} preceded incident signal",
    ),
]


def _parse_ts(ts_str: str) -> float:
    """Parse ISO timestamp → unix float. Returns 0.0 on failure."""
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _event_id(event: dict, idx: int) -> str:
    """Generate a stable event identifier."""
    return event.get("trace_id") or f"evt-{idx}-{event.get('kind', 'unk')}"


def _format_evidence(rule: CausalRule, cause: dict, effect: dict) -> str:
    """Build a human-readable evidence string from the rule template."""
    try:
        return rule.evidence_template.format(
            cause_svc=cause.get("service", "?"),
            version=cause.get("version", "?"),
            metric=cause.get("name", "?"),
            value=cause.get("value", "?"),
            msg=effect.get("msg", cause.get("msg", "?")),
            trigger=effect.get("trigger", "?"),
        )
    except (KeyError, IndexError):
        return f"{rule.name}: {cause.get('kind')} → {effect.get('kind')}"


class CausalEdgeExtractor:
    """
    Extracts causal edges from a set of events using rule-based heuristics.

    The frequency boost rewards entity pairs that have co-occurred causally
    in the past, strengthening confidence for recurring patterns.
    """

    FREQ_BOOST_SCALE: float = 0.15
    FREQ_BOOST_CAP: float = 0.20
    CONFIDENCE_CAP: float = 0.98

    def __init__(self, pair_stats: dict[tuple[str, str], int] | None = None) -> None:
        """
        Args:
            pair_stats: Optional dict mapping (canonical_id_a, canonical_id_b) → co-occurrence count.
                        Used for historical frequency boost.
        """
        self._pair_stats = pair_stats or {}

    def update_pair_stats(self, stats: dict[tuple[str, str], int]) -> None:
        """Replace the pair stats dictionary."""
        self._pair_stats = stats

    def extract(
        self,
        events: list[dict],
        tracker=None,
    ) -> list[CausalEdge]:
        """
        Run all 5 causal rules across O(n²) event pairs.

        Args:
            events: list of raw event dicts, sorted by timestamp ascending.
            tracker: AliasTracker for resolving canonical IDs (for frequency boost).

        Returns:
            List of CausalEdge TypedDicts, deduplicated, sorted by confidence desc.
        """
        if not events:
            return []

        # Pre-parse timestamps
        parsed = []
        for i, e in enumerate(events):
            parsed.append((e, _parse_ts(e.get("ts", "")), i))

        edges: list[CausalEdge] = []
        seen: set[tuple[str, str, str]] = set()

        for ci, (cause, cause_ts, cause_idx) in enumerate(parsed):
            for ei, (effect, effect_ts, effect_idx) in enumerate(parsed):
                if ci >= ei:
                    continue  # cause must precede effect in list order
                gap = effect_ts - cause_ts
                if gap < 0:
                    continue

                for rule in CAUSAL_RULES:
                    if not rule.matches(cause, effect, gap):
                        continue

                    cause_id = _event_id(cause, cause_idx)
                    effect_id = _event_id(effect, effect_idx)
                    dedup_key = (cause_id, effect_id, rule.name)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    # Historical frequency boost
                    freq_boost = 0.0
                    if tracker and self._pair_stats:
                        c_cid = tracker.resolve(cause.get("service", ""))
                        e_cid = tracker.resolve(effect.get("service", ""))
                        hist_count = self._pair_stats.get((c_cid, e_cid), 0)
                        if hist_count > 0:
                            freq_boost = min(
                                self.FREQ_BOOST_SCALE * math.log1p(hist_count),
                                self.FREQ_BOOST_CAP,
                            )

                    confidence = min(
                        rule.base_confidence + freq_boost,
                        self.CONFIDENCE_CAP,
                    )

                    edges.append({
                        "cause_event_id": cause_id,
                        "effect_event_id": effect_id,
                        "evidence": _format_evidence(rule, cause, effect),
                        "confidence": round(confidence, 4),
                    })

        # Sort by confidence descending
        edges.sort(key=lambda e: e.get("confidence", 0), reverse=True)
        return edges
