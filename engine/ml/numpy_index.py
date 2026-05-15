"""
engine/ml/numpy_index.py — in-memory numpy behavioral index.

Used by the benchmark adapter (Path A). Zero dependencies beyond numpy.
Sub-millisecond for <500 episodes. Vectors must be L2-normalized (dot = cosine).
"""
from __future__ import annotations

import numpy as np
from typing import Any


class NumpyBehavioralIndex:
    """
    Pure numpy cosine similarity index.
    Stores episode sequence embeddings and returns top-k candidates by cosine score.
    """

    def __init__(self) -> None:
        self._embeddings: list[np.ndarray] = []   # each is (384,) normalized
        self._metadata: list[dict] = []           # parallel list of metadata dicts

    def upsert(self, incident_id: str, embedding: np.ndarray, metadata: dict) -> None:
        """Add or overwrite an episode embedding."""
        for i, m in enumerate(self._metadata):
            if m["incident_id"] == incident_id:
                self._embeddings[i] = embedding
                self._metadata[i] = {**metadata, "incident_id": incident_id}
                return
        self._embeddings.append(embedding)
        self._metadata.append({**metadata, "incident_id": incident_id})

    def recall(self, query_vector: np.ndarray, top_k: int = 20) -> list[dict]:
        """Return top_k candidates sorted by descending cosine similarity."""
        if not self._embeddings:
            return []
        matrix = np.array(self._embeddings)        # (n_episodes, 384)
        scores = matrix @ query_vector             # cosine sim (normalized vecs)
        n = min(top_k, len(scores))
        top_idx = np.argsort(scores)[::-1][:n]
        return [
            {
                "incident_id": self._metadata[i]["incident_id"],
                "score": float(scores[i]),
                "payload": self._metadata[i],
            }
            for i in top_idx
        ]

    def update_payload(self, incident_id: str, key: str, value: Any) -> None:
        """Update a specific key in the payload metadata for an episode."""
        for i, m in enumerate(self._metadata):
            if m["incident_id"] == incident_id:
                self._metadata[i][key] = value
                return

    def __len__(self) -> int:
        return len(self._embeddings)
