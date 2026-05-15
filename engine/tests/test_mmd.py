"""
engine/tests/test_mmd.py — unit tests for MMDDriftDetector and MMDReRanker.

Run with: pytest engine/tests/test_mmd.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from engine.ml.mmd_detector import MMDDriftDetector
from engine.ml.mmd_reranker import MMDReRanker
from engine.ml.numpy_index import NumpyBehavioralIndex


@pytest.fixture
def detector() -> MMDDriftDetector:
    return MMDDriftDetector()


@pytest.fixture
def reranker(detector: MMDDriftDetector) -> MMDReRanker:
    return MMDReRanker(mmd=detector)


# ── MMDDriftDetector tests ────────────────────────────────────────────────────

class TestMMDDriftDetector:

    def test_mmd_zero_identical(self, detector: MMDDriftDetector):
        """MMD² must be ~0 when X == Y (same distribution)."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 384)).astype(np.float32)
        result = detector.compute_mmd_squared(X, X)
        assert result < 1e-6, f"Expected ~0 for identical inputs, got {result}"

    def test_mmd_detects_shift(self, detector: MMDDriftDetector):
        """MMD² must exceed threshold when distributions are clearly separated."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 384)).astype(np.float32)
        Y = (rng.standard_normal((20, 384)) + 2.0).astype(np.float32)
        result = detector.compute_mmd_squared(X, Y)
        # In 384-dim space, DRIFT_THRESHOLD=1e-4; shifted dists should be well above it
        assert result > 1e-4, f"Expected MMD² > 1e-4 for shifted distributions, got {result}"

    def test_mmd_insufficient_samples(self, detector: MMDDriftDetector):
        """MMD² must return 0.0 when either set has < MIN_SAMPLES=5 items."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((3, 384)).astype(np.float32)   # below MIN_SAMPLES
        Y = rng.standard_normal((20, 384)).astype(np.float32)
        result = detector.compute_mmd_squared(X, Y)
        assert result == 0.0, f"Expected 0.0 for insufficient samples, got {result}"

    def test_mmd_insufficient_samples_both(self, detector: MMDDriftDetector):
        """Returns 0.0 when both sets are too small."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((2, 384)).astype(np.float32)
        Y = rng.standard_normal((2, 384)).astype(np.float32)
        assert detector.compute_mmd_squared(X, Y) == 0.0

    def test_mmd_non_negative(self, detector: MMDDriftDetector):
        """Unbiased MMD² can be slightly negative due to floating point — must be clipped."""
        rng = np.random.default_rng(4)
        X = rng.standard_normal((10, 384)).astype(np.float32)
        Y = rng.standard_normal((10, 384)).astype(np.float32)
        result = detector.compute_mmd_squared(X, Y)
        assert result >= 0.0, f"MMD² must be non-negative, got {result}"

    def test_mmd_similarity_conversion_identical(self, detector: MMDDriftDetector):
        """mmd2=0 → similarity=1.0 (identical distributions)."""
        sim = detector.mmd_to_similarity(0.0)
        assert abs(sim - 1.0) < 1e-6, f"Expected sim=1.0 for mmd2=0, got {sim}"

    def test_mmd_similarity_conversion_scale(self, detector: MMDDriftDetector):
        """mmd2=scale → similarity=0.5 (half-way point)."""
        scale = 0.1
        sim = detector.mmd_to_similarity(scale, scale=scale)
        assert abs(sim - 0.5) < 1e-6, f"Expected sim=0.5 for mmd2=scale, got {sim}"

    def test_mmd_is_drift(self, detector: MMDDriftDetector):
        """is_drift() returns True for clearly shifted distributions."""
        rng = np.random.default_rng(5)
        X = rng.standard_normal((20, 384)).astype(np.float32)
        Y = (rng.standard_normal((20, 384)) + 3.0).astype(np.float32)
        assert detector.is_drift(X, Y) is True

    def test_mmd_not_drift_same_dist(self, detector: MMDDriftDetector):
        """is_drift() returns False for same distribution."""
        rng = np.random.default_rng(6)
        X = rng.standard_normal((20, 384)).astype(np.float32)
        assert detector.is_drift(X, X) is False


# ── MMDReRanker tests ─────────────────────────────────────────────────────────

class TestMMDReRanker:

    def _make_vecs(self, n: int, dim: int = 384, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, dim)).astype(np.float32)
        # Normalize so cosine sim = dot product
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-8)

    def test_rerank_empty_candidates(self, reranker: MMDReRanker):
        """Returns empty list for empty candidates."""
        q_vecs = self._make_vecs(10)
        result = reranker.rerank(q_vecs, [], {}, top_k=5)
        assert result == []

    def test_rerank_returns_top_k(self, reranker: MMDReRanker):
        """Returns at most top_k results."""
        q_vecs = self._make_vecs(10, seed=0)
        candidates = [
            {"incident_id": f"INC-{i}", "score": 0.9 - i * 0.05, "payload": {"incident_id": f"INC-{i}"}}
            for i in range(15)
        ]
        episode_vecs = {f"INC-{i}": self._make_vecs(10, seed=i+1) for i in range(15)}
        result = reranker.rerank(q_vecs, candidates, episode_vecs, top_k=5)
        assert len(result) <= 5

    def test_rerank_fields_present(self, reranker: MMDReRanker):
        """Each result must have cosine_score, mmd_similarity, combined_score."""
        q_vecs = self._make_vecs(10, seed=0)
        candidates = [
            {"incident_id": "INC-1", "score": 0.85, "payload": {"incident_id": "INC-1"}}
        ]
        episode_vecs = {"INC-1": self._make_vecs(10, seed=1)}
        result = reranker.rerank(q_vecs, candidates, episode_vecs, top_k=5)
        assert len(result) == 1
        r = result[0]
        assert "cosine_score" in r
        assert "mmd_similarity" in r
        assert "combined_score" in r
        assert "incident_id" in r

    def test_rerank_sorted_descending(self, reranker: MMDReRanker):
        """Results are sorted by combined_score descending."""
        q_vecs = self._make_vecs(10, seed=0)
        candidates = [
            {"incident_id": f"INC-{i}", "score": float(i) / 10, "payload": {"incident_id": f"INC-{i}"}}
            for i in range(5)
        ]
        episode_vecs = {f"INC-{i}": self._make_vecs(10, seed=i+1) for i in range(5)}
        result = reranker.rerank(q_vecs, candidates, episode_vecs, top_k=5)
        scores = [r["combined_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_fallback_no_episode_vecs(self, reranker: MMDReRanker):
        """Falls back to cosine when no episode vectors available."""
        q_vecs = self._make_vecs(10, seed=0)
        candidates = [
            {"incident_id": "INC-1", "score": 0.7, "payload": {"incident_id": "INC-1"}},
            {"incident_id": "INC-2", "score": 0.5, "payload": {"incident_id": "INC-2"}},
        ]
        result = reranker.rerank(q_vecs, candidates, {}, top_k=5)
        assert len(result) == 2
        # With no episode vecs, mmd_weight=0 so combined_score == cosine_score
        assert result[0]["incident_id"] == "INC-1"  # higher cosine comes first


# ── NumpyBehavioralIndex tests ────────────────────────────────────────────────

class TestNumpyBehavioralIndex:

    def _make_vec(self, dim: int = 384, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_recall_empty(self):
        idx = NumpyBehavioralIndex()
        result = idx.recall(self._make_vec(), top_k=5)
        assert result == []

    def test_upsert_and_recall(self):
        idx = NumpyBehavioralIndex()
        v = self._make_vec(seed=0)
        idx.upsert("INC-1", v, {"incident_id": "INC-1"})
        result = idx.recall(v, top_k=5)
        assert len(result) == 1
        assert result[0]["incident_id"] == "INC-1"
        assert abs(result[0]["score"] - 1.0) < 1e-5  # same vector → cosine=1.0

    def test_upsert_overwrite(self):
        idx = NumpyBehavioralIndex()
        v1 = self._make_vec(seed=0)
        v2 = self._make_vec(seed=1)
        idx.upsert("INC-1", v1, {"incident_id": "INC-1", "version": 1})
        idx.upsert("INC-1", v2, {"incident_id": "INC-1", "version": 2})
        assert len(idx) == 1  # should overwrite, not append
        result = idx.recall(v2, top_k=1)
        assert result[0]["payload"]["version"] == 2

    def test_recall_top_k_respected(self):
        idx = NumpyBehavioralIndex()
        for i in range(10):
            v = self._make_vec(seed=i)
            idx.upsert(f"INC-{i}", v, {"incident_id": f"INC-{i}"})
        result = idx.recall(self._make_vec(seed=99), top_k=3)
        assert len(result) <= 3
