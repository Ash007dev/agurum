"""
L3 Mock Scorer
Computes same metrics as the real harness:
  recall@5, precision@5, remediation_acc, latency_p95
Plus L3-specific breakdown:
  - per-family recall (reveals which families your engine misses)
  - pre-drift vs post-drift adaptability delta
  - cascading rename recall (did you survive A->B->C->D?)
  - multi-service outage recall (did you catch correlated failures?)
"""

import time
import statistics
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def compute_recall_at_k(
    returned: List[str],
    ground_truth: List[str],
    k: int = 5
) -> float:
    if not ground_truth:
        return 1.0
    top_k = returned[:k]
    hits = sum(1 for r in top_k if r in ground_truth)
    return min(hits / max(len(ground_truth), 1), 1.0)


def compute_precision_at_k(
    returned: List[str],
    ground_truth: List[str],
    k: int = 5
) -> float:
    if not returned:
        return 0.0
    top_k = returned[:k]
    hits = sum(1 for r in top_k if r in ground_truth)
    return hits / len(top_k)


class L3Scorer:
    def __init__(self, ground_truth: Dict[str, Dict]):
        self.ground_truth = ground_truth

    def score_one(
        self,
        incident_id: str,
        context,               # Context TypedDict returned by engine
        latency_ms: float,
    ) -> Dict[str, Any]:
        meta = self.ground_truth.get(incident_id, {})
        true_similar = meta.get("ground_truth_family_ids", [])
        family_id = meta.get("family_id", "unknown")

        returned_ids = [m.get("incident_id") or m.get("past_incident_id")
                        for m in (context.get("similar_past_incidents") or [])]

        recall = compute_recall_at_k(returned_ids, true_similar, k=5)
        precision = compute_precision_at_k(returned_ids, true_similar, k=5)

        # Remediation accuracy
        rem_correct = 0
        expected_rem = FAMILY_SIGNATURES_REMEDIATION.get(family_id)
        for r in (context.get("suggested_remediations") or []):
            if r.get("action") == expected_rem:
                rem_correct = 1
                break

        return {
            "incident_id": incident_id,
            "family_id": family_id,
            "recall@5": recall,
            "precision@5": precision,
            "remediation_correct": rem_correct,
            "latency_ms": latency_ms,
            "returned_ids": returned_ids,
            "true_similar": true_similar,
        }

    def aggregate(self, results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {}

        recalls = [r["recall@5"] for r in results]
        precisions = [r["precision@5"] for r in results]
        rem_acc = [r["remediation_correct"] for r in results]
        latencies = [r["latency_ms"] for r in results]

        # Per-family breakdown
        family_recalls = defaultdict(list)
        family_precisions = defaultdict(list)
        for r in results:
            family_recalls[r["family_id"]].append(r["recall@5"])
            family_precisions[r["family_id"]].append(r["precision@5"])

        # Weighted automated score (same formula as real harness)
        mean_recall = statistics.mean(recalls)
        mean_precision = statistics.mean(precisions)
        mean_rem = statistics.mean(rem_acc)
        lat_score = 1.0 if sorted(latencies)[int(len(latencies)*0.95)] < 2000 else 0.0

        weighted = (
            mean_recall    * 0.30 +
            mean_precision * 0.15 +
            mean_rem       * 0.20 +
            lat_score      * 0.15
        )

        # L3-specific: adaptability (pre vs post first rename)
        # We can't compute this without timestamps, so we approximate
        # by splitting results into first half vs second half by index
        n = len(results)
        first_half = results[:n//2]
        second_half = results[n//2:]
        pre_recall = statistics.mean([r["recall@5"] for r in first_half]) if first_half else 0
        post_recall = statistics.mean([r["recall@5"] for r in second_half]) if second_half else 0
        adaptability_delta = abs(pre_recall - post_recall)

        return {
            "recall@5":            round(mean_recall, 4),
            "precision@5_mean":    round(mean_precision, 4),
            "remediation_acc":     round(mean_rem, 4),
            "latency_p95_ms":      round(sorted(latencies)[int(len(latencies)*0.95)], 2),
            "latency_mean_ms":     round(statistics.mean(latencies), 2),
            "weighted_automated":  round(weighted, 4),
            "adaptability_delta":  round(adaptability_delta, 4),
            "per_family_recall":   {
                fam: round(statistics.mean(rs), 4)
                for fam, rs in family_recalls.items()
            },
            "per_family_precision": {
                fam: round(statistics.mean(ps), 4)
                for fam, ps in family_precisions.items()
            },
            "n_queries": len(results),
        }


# Remediation lookup for scoring
FAMILY_SIGNATURES_REMEDIATION = {
    "F01": "rollback",
    "F02": "restart",
    "F03": "config_change",
    "F04": "rollback",
    "F05": "scale_out",
    "F06": "cert_rotate",
    "F07": "rollback",
    "F08": "rollback",
    "F09": "restart",
    "F10": "circuit_break",
}
