"""
engine/ml/mmd_detector.py — unbiased MMD² estimator with multi-scale RBF kernel.

Used to detect distribution shift between pre-deploy and post-deploy event
distributions, and as a similarity metric in MMDReRanker.

Mathematics:
    MMD²(P,Q) = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]
    k(x,y) = Σᵢ exp(-||x-y||² / 2σᵢ²)   σ ∈ {0.5, 1.0, 2.0, 5.0}
    Unbiased: zero diagonals on Kxx, Kyy before summing.
"""
from __future__ import annotations

import numpy as np


class MMDDriftDetector:
    """
    Unbiased MMD² estimator with multi-scale RBF kernel.
    Multi-scale σ covers the full range of pairwise distances in 384-dim space.
    """

    MIN_SAMPLES: int = 5
    # Multi-scale σ tuned for 384-dim space.
    # Typical ||x-y||² between samples from different normal distributions: ~2000-2600
    # Rule: σ ≈ sqrt(median_dist²/2) ≈ 33; use 4 scales spanning that range.
    SIGMAS: np.ndarray = np.array([10.0, 50.0, 100.0, 200.0])
    DRIFT_THRESHOLD: float = 1e-4

    def compute_mmd_squared(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute unbiased MMD² between sample sets X and Y.

        Args:
            X: (n, d) array of embeddings from distribution P
            Y: (m, d) array of embeddings from distribution Q

        Returns:
            float ≥ 0.0; returns 0.0 if either set has fewer than MIN_SAMPLES.
        """
        n, m = len(X), len(Y)
        if n < self.MIN_SAMPLES or m < self.MIN_SAMPLES:
            return 0.0  # insufficient data — never crash

        Kxx = self._kernel(X, X)
        Kyy = self._kernel(Y, Y)
        Kxy = self._kernel(X, Y)

        # Unbiased estimator: zero out diagonal terms
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)

        mmd2 = (
            Kxx.sum() / (n * (n - 1))
            + Kyy.sum() / (m * (m - 1))
            - 2.0 * Kxy.mean()
        )
        return float(max(mmd2, 0.0))  # numerical safety — never negative

    def is_drift(self, X: np.ndarray, Y: np.ndarray) -> bool:
        """Return True if MMD² exceeds the drift threshold."""
        return self.compute_mmd_squared(X, Y) > self.DRIFT_THRESHOLD

    def mmd_to_similarity(self, mmd2: float, scale: float = 0.1) -> float:
        """
        Convert MMD² to [0,1] similarity score.
        mmd2=0 → 1.0 (identical distributions)
        mmd2=0.1 → 0.5 (moderate difference)
        mmd2=1.0 → 0.09 (very different)
        """
        return 1.0 / (1.0 + mmd2 / scale)

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multi-scale RBF kernel: k(A,B) = Σᵢ exp(-||a-b||²/2σᵢ²)"""
        A_sq = np.sum(A ** 2, axis=1, keepdims=True)   # (n, 1)
        B_sq = np.sum(B ** 2, axis=1, keepdims=True)   # (m, 1)
        # ||a-b||² = ||a||² + ||b||² - 2<a,b>
        dist2 = np.maximum(A_sq + B_sq.T - 2.0 * (A @ B.T), 0.0)  # (n, m)
        K = np.zeros_like(dist2)
        for sigma in self.SIGMAS:
            K += np.exp(-dist2 / (2.0 * sigma ** 2))
        return K
