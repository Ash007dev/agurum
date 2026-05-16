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
        self._fallback_mode = False
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
            # Warmup: first inference is 3× slower due to JIT compilation
            self._model.encode(
                ["warmup ping"],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Unable to load SentenceTransformer. Falling back to offline deterministic embedder: %s", e)
            self._fallback_mode = True

    def _fallback_encode(self, text: str) -> np.ndarray:
        """Deterministic hash-based fallback (384-dimensional)."""
        import hashlib
        vec = np.zeros(384, dtype=np.float32)
        words = text.lower().split()
        for i, w in enumerate(words):
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % 384
            vec[idx] += 1.0
            # Add some pseudo-randomness based on word position
            vec[(idx + i) % 384] += 0.5
            
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_single(self, text: str) -> np.ndarray:
        """Encode one string → (384,) normalized float32 vector."""
        if self._fallback_mode:
            return self._fallback_encode(text)

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
            
        if self._fallback_mode:
            return np.array([self._fallback_encode(t) for t in texts], dtype=np.float32)

        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
