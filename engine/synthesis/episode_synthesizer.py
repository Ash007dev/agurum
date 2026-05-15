"""
engine/synthesis/episode_synthesizer.py — builds episodes from resolved incidents.

An Episode = sequence of events from an incident that was resolved.
It is stored in the behavioral index for future similarity matching.

Sync version (used by benchmark adapter).
Production path wraps this in run_in_executor.
"""
from __future__ import annotations

import numpy as np

from engine.ml.embedder import Embedder
from engine.ml.numpy_index import NumpyBehavioralIndex
from engine.registry.alias_tracker import AliasTracker


def _event_to_string(event: dict) -> str:
    """
    Convert an event to a rename-robust embedding string.
    Service name is NEVER included — rename-robust by construction.
    Uses actual field names from generator.py (name, level, msg, value).
    NOT imaginary names: metric_name, error_class, severity.
    """
    kind = event.get("kind", "unknown")
    parts = [kind]
    if kind == "metric":
        parts.append(event.get("name", "unknown_metric"))    # "latency_p99_ms"
        val = event.get("value", 0)
        parts.append("high" if val > 3000 else "normal")
    elif kind == "log":
        parts.append(event.get("level", "info"))              # "error"
        msg = event.get("msg", "")
        if "timeout" in msg:
            parts.append("timeout")
        elif "error" in msg:
            parts.append("error")
    elif kind == "deploy":
        parts.append("deployment")
    elif kind == "incident_signal":
        parts.append("alert")
    elif kind == "remediation":
        parts.append(event.get("action", "unknown"))          # "rollback"
        parts.append(event.get("outcome", "unknown"))         # "resolved"
    elif kind == "topology":
        parts.append(event.get("change", "change"))
    return " ".join(parts)


class EpisodeSynthesizer:
    """
    Builds and indexes episodes from resolved incidents.

    One episode = all events associated with one incident_id that was resolved.
    """

    def __init__(
        self,
        embedder: Embedder,
        index: NumpyBehavioralIndex,
        tracker: AliasTracker,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._tracker = tracker
        # incident_id → per-event vectors (n, 384) for MMD reranking
        self.episode_vectors: dict[str, np.ndarray] = {}
        # track which episodes we've already synthesized (idempotent across double-ingest)
        self._synthesized: set[str] = set()

    def synthesize_all(
        self,
        incidents: dict[str, list[dict]],       # incident_id → event list
        remediations: dict[str, dict],           # incident_id → remediation event
    ) -> None:
        """
        Build episodes for all resolved incidents not yet synthesized.
        Safe to call multiple times — idempotent via self._synthesized set.
        """
        for inc_id, remediation in remediations.items():
            if inc_id in self._synthesized:
                continue  # already indexed — skip (handles double ingest)

            events = incidents.get(inc_id, [])
            if not events:
                continue

            strings = [_event_to_string(e) for e in events]
            if not strings:
                continue

            # Sequence embedding: joint text of all event strings
            seq_str = " ".join(strings)
            seq_vec = self._embedder.encode_single(seq_str)          # (384,)

            # Per-event embeddings for MMD reranking
            event_vecs = self._embedder.encode_batch(strings)        # (n, 384)

            self.episode_vectors[inc_id] = event_vecs

            # Extract family from incident_id: INC-{ts}-{family}
            family = _family_from_id(inc_id)

            self._index.upsert(inc_id, seq_vec, {
                "incident_id": inc_id,
                "action": remediation.get("action", "rollback"),
                "outcome": remediation.get("outcome", "resolved"),
                "target": remediation.get("target", ""),
                "event_count": len(events),
                "family": family,
            })
            self._synthesized.add(inc_id)

    def synthesize_one(
        self,
        inc_id: str,
        events: list[dict],
        remediation: dict,
    ) -> None:
        """Synthesize a single episode. Convenience wrapper."""
        self.synthesize_all(
            incidents={inc_id: events},
            remediations={inc_id: remediation},
        )


def _family_from_id(incident_id: str) -> int:
    """Extract numeric family from INC-{ts}-{family}. Returns -1 on failure."""
    try:
        return int(incident_id.rsplit("-", 1)[-1])
    except (ValueError, IndexError):
        return -1
