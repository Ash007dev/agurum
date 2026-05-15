"""
engine/ml/mmd_reranker.py — reranks ANN top-20 candidates using MMD distribution match.

Pipeline: NumpyBehavioralIndex.recall(top_k=20) → MMDReRanker.rerank() → top-5

Combined score = mmd_weight * mmd_similarity + (1 - mmd_weight) * cosine_score

mmd_weight = 0.6 when both query and candidate have ≥5 events (trust MMD more)
mmd_weight = 0.2 when insufficient samples (fall back to cosine)
"""
from __future__ import annotations

import numpy as np

from engine.ml.mmd_detector import MMDDriftDetector


class MMDReRanker:
    """
    Promotes candidates whose event distribution matches the query distribution.
    Cosine on the mean vector misses distributional shape; MMD catches it.
    """

    MMD_WEIGHT_HIGH: float = 0.6   # when both sides have ≥ MIN_VECS events
    MMD_WEIGHT_LOW: float = 0.2    # fallback when insufficient samples
    MIN_VECS: int = 5              # minimum events to trust MMD
    MMD_SCALE: float = 0.1         # similarity conversion scale

    def __init__(self, mmd: MMDDriftDetector | None = None) -> None:
        self.mmd = mmd or MMDDriftDetector()

    def rerank(
        self,
        query_vecs: np.ndarray,              # (n, 384) query per-event embeddings
        candidates: list[dict],              # from NumpyBehavioralIndex.recall()
        episode_vectors: dict[str, np.ndarray],  # incident_id → (m, 384)
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank candidates by combined MMD+cosine score.

        Returns list of dicts with added fields:
            cosine_score, mmd_similarity, combined_score
        Sorted descending by combined_score, truncated to top_k.
        """
        if not candidates:
            return []

        scored = []
        for c in candidates:
            inc_id = c["incident_id"]
            cosine = float(c["score"])
            cand_vecs = episode_vectors.get(inc_id)

            if (
                cand_vecs is None
                or len(cand_vecs) < self.MIN_VECS
                or len(query_vecs) < self.MIN_VECS
            ):
                # Insufficient data — trust cosine only
                mmd_sim = cosine
                mmd_weight = 0.0
            else:
                mmd2 = self.mmd.compute_mmd_squared(query_vecs, cand_vecs)
                mmd_sim = self.mmd.mmd_to_similarity(mmd2, scale=self.MMD_SCALE)
                n_q = len(query_vecs)
                n_c = len(cand_vecs)
                mmd_weight = (
                    self.MMD_WEIGHT_HIGH
                    if (n_q >= self.MIN_VECS and n_c >= self.MIN_VECS)
                    else self.MMD_WEIGHT_LOW
                )

            combined = mmd_weight * mmd_sim + (1.0 - mmd_weight) * cosine
            scored.append({
                **c,
                "cosine_score": round(cosine, 4),
                "mmd_similarity": round(mmd_sim, 4),
                "combined_score": round(combined, 4),
            })

        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored[:top_k]
