"""
engine/ml/behavioral_index.py — public-facing behavioral index interface.

This module exports the index used by production Path B (DuckDB + Qdrant).
For the benchmark adapter (Path A) use engine.ml.numpy_index.NumpyBehavioralIndex.

Protocol defines the shared interface so both implementations are swappable.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BehavioralIndex(Protocol):
    """
    Interface for episode behavioral indices.
    Both NumpyBehavioralIndex (benchmark) and QdrantBehavioralIndex (production)
    implement this protocol.
    """

    def upsert(self, incident_id: str, embedding: np.ndarray, metadata: dict) -> None:
        """Add or overwrite an episode embedding."""
        ...

    def recall(self, query_vector: np.ndarray, top_k: int = 20) -> list[dict]:
        """
        Return top_k candidates sorted by descending cosine similarity.
        Each result: {"incident_id": str, "score": float, "payload": dict}
        """
        ...

    def __len__(self) -> int:
        ...


# Re-export NumpyBehavioralIndex as the default for convenience
from engine.ml.numpy_index import NumpyBehavioralIndex  # noqa: E402

__all__ = ["BehavioralIndex", "NumpyBehavioralIndex"]
