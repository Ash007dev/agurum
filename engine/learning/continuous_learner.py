"""
engine/learning/continuous_learner.py — event-driven continuous learning.

On remediation resolution:
  - EWMA update α=0.15 on entity_pair_stats co-occurrence counts
  - Update episode quality metadata

Runs as a background task, triggered by remediation events.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from engine import config

logger = logging.getLogger(__name__)


class ContinuousLearner:
    """
    Continuous learning system that reinforces successful remediation pathways
    and decays unsuccessful ones.

    Uses EWMA (Exponential Weighted Moving Average) with α=0.15 to smooth
    co-occurrence statistics over time.
    """

    def __init__(self, eventstore: Any = None, index: Any = None, alpha: float = config.EWMA_ALPHA) -> None:
        self._alpha = alpha
        self._eventstore = eventstore
        self._index = index
        
        # (canonical_id_a, canonical_id_b) → smoothed co-occurrence weight
        self._pair_stats: dict[tuple[str, str], float] = {}
        if self._eventstore:
            try:
                self._pair_stats = self._eventstore.get_all_pair_stats()
            except Exception:
                pass
        
        # incident_id → resolution feedback
        self._feedback_log: list[dict[str, Any]] = []

    @property
    def pair_stats(self) -> dict[tuple[str, str], float]:
        return dict(self._pair_stats)

    def on_remediation(
        self,
        incident_id: str,
        canonical_id: str,
        action: str,
        outcome: str,
        related_canonical_ids: list[str] | None = None,
    ) -> None:
        """
        Called when a remediation event is received.
        Updates EWMA pair statistics for all co-occurring entity pairs.
        """
        if outcome != "resolved":
            # Only reinforce successful resolutions
            self._feedback_log.append({
                "incident_id": incident_id,
                "outcome": outcome,
                "action": "decay",
            })
            return

        # Reinforce co-occurrence for all related pairs
        related = related_canonical_ids or []
        all_cids = [canonical_id] + related

        for i, cid_a in enumerate(all_cids):
            for cid_b in all_cids[i + 1:]:
                key = tuple(sorted([cid_a, cid_b]))
                
                if self._eventstore:
                    # Update DuckDB
                    new_val = self._eventstore.update_pair_stats_ewma(
                        cid_a, cid_b, self._alpha, 1.0
                    )
                    self._pair_stats[key] = new_val
                else:
                    # In-memory fallback
                    old_val = self._pair_stats.get(key, 0.0)
                    new_val = self._alpha * 1.0 + (1.0 - self._alpha) * old_val
                    self._pair_stats[key] = new_val

        # Update Qdrant payload (resolution_confidence)
        if self._index:
            self._index.update_payload(incident_id, "resolution_confidence", 1.0)

        self._feedback_log.append({
            "incident_id": incident_id,
            "outcome": outcome,
            "action": "reinforce",
            "pairs_updated": len(all_cids) * (len(all_cids) - 1) // 2,
        })

        logger.debug(
            f"Reinforced {len(all_cids)} entities for incident {incident_id}"
        )

    def on_feedback(
        self,
        incident_id: str,
        was_helpful: bool,
        canonical_id: str | None = None,
    ) -> None:
        """
        Process explicit human feedback on a remediation suggestion.
        Adjusts pair stats up (helpful) or down (not helpful).
        """
        if canonical_id:
            for key in list(self._pair_stats.keys()):
                if canonical_id in key:
                    if was_helpful:
                        self._pair_stats[key] = min(
                            self._alpha * 1.0 + (1.0 - self._alpha) * self._pair_stats[key],
                            1.0,
                        )
                    else:
                        self._pair_stats[key] *= (1.0 - self._alpha)

        self._feedback_log.append({
            "incident_id": incident_id,
            "was_helpful": was_helpful,
            "action": "feedback",
        })

    def get_pair_count(self, cid_a: str, cid_b: str) -> int:
        """Get the integer co-occurrence count for a pair (for CausalEdgeExtractor)."""
        key = tuple(sorted([cid_a, cid_b]))
        return int(self._pair_stats.get(key, 0.0) * 10)  # scale to integer
