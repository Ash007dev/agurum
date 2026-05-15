"""
engine/remediation/remediation_advisor.py — suggests remediations from historical episodes.

Aggregates remediation outcomes from matched past incidents and applies
Laplace-smoothed confidence scoring. Resolves canonical IDs back to
current service names for human-readable suggestions.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


class RemediationAdvisor:
    """
    Builds ranked remediation suggestions from MMD-reranked episode matches.

    Laplace smoothing: confidence = (successes + 1) / (total + 2)
    This avoids 0% or 100% confidence from small samples.
    """

    LAPLACE_ALPHA: float = 1.0
    LAPLACE_BETA: float = 2.0
    TOP_K: int = 5

    def __init__(self, tracker=None) -> None:
        """
        Args:
            tracker: AliasTracker for canonical_id → current name resolution.
        """
        self._tracker = tracker

    def suggest(
        self,
        ranked_matches: list[dict],
        trigger_service: str,
    ) -> list[dict[str, Any]]:
        """
        Build top-K remediation suggestions from ranked episode matches.

        Args:
            ranked_matches: Output from MMDReRanker.rerank() — each dict has
                            payload with action, outcome, target fields.
            trigger_service: The current service name from the incident signal.

        Returns:
            List of Remediation TypedDicts sorted by confidence descending.
        """
        if not ranked_matches:
            return [{
                "action": "rollback",
                "target": trigger_service,
                "historical_outcome": "no_prior_matches",
                "confidence": 0.1,
            }]

        # Aggregate outcomes by (action, target)
        action_stats: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"successes": 0, "total": 0, "combined_scores": []}
        )

        for match in ranked_matches:
            payload = match.get("payload", {})
            action = payload.get("action", "rollback")
            target = payload.get("target", trigger_service) or trigger_service

            # Resolve canonical target to current name if tracker available
            if self._tracker and target:
                try:
                    cid = self._tracker.resolve(target)
                    target = self._tracker.get_current_name(cid)
                except Exception:
                    pass

            key = (action, target)
            stats = action_stats[key]
            stats["total"] += 1
            if payload.get("outcome") == "resolved":
                stats["successes"] += 1
            stats["combined_scores"].append(match.get("combined_score", 0.0))

        # Build ranked suggestions with Laplace-smoothed confidence
        suggestions = []
        for (action, target), stats in action_stats.items():
            laplace_conf = (stats["successes"] + self.LAPLACE_ALPHA) / (
                stats["total"] + self.LAPLACE_BETA
            )
            avg_match_score = (
                sum(stats["combined_scores"]) / len(stats["combined_scores"])
                if stats["combined_scores"]
                else 0.0
            )
            # Blend Laplace confidence with match quality
            blended = 0.6 * laplace_conf + 0.4 * avg_match_score

            suggestions.append({
                "action": action,
                "target": target,
                "historical_outcome": f"resolved {stats['successes']}/{stats['total']}",
                "confidence": round(blended, 3),
            })

        suggestions.sort(key=lambda s: s["confidence"], reverse=True)
        return suggestions[: self.TOP_K]
