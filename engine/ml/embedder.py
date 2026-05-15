"""
engine/ml/embedder.py — all-MiniLM-L6-v2 singleton embedder.

Loaded once at startup. Includes warmup to eliminate JIT cold-start penalty.
Used by both benchmark adapter (sync) and production path.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_instance = None


def get_embedder() -> "Embedder":
    """Return the global singleton Embedder. Creates + warms up on first call."""
    global _instance
    if _instance is None:
        _instance = Embedder()
    return _instance


class Embedder:
    """
    Thin wrapper around all-MiniLM-L6-v2.
    - All outputs are L2-normalized (unit vectors) → dot product = cosine sim.
    - show_progress_bar=False for clean benchmark output.
    - convert_to_numpy=True so numpy ops work directly.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.MODEL_NAME)
        # Warmup: first inference is 3× slower due to JIT compilation
        self._model.encode(
            ["warmup ping"],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode one string → (384,) normalized float32 vector."""
        result = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return result[0]  # (384,)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of strings → (n, 384) normalized float32 array."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
