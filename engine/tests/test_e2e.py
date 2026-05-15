"""
engine/tests/test_e2e.py — end-to-end smoke test for the full pipeline.

Tests the complete flow: ingest → synthesize → reconstruct, including
rename robustness (the core differentiator).
"""
from __future__ import annotations
import pytest
from engine.registry.alias_tracker import AliasTracker
from engine.store.in_memory_store import InMemoryStore
from engine.ml.embedder import get_embedder
from engine.ml.numpy_index import NumpyBehavioralIndex
from engine.ml.mmd_detector import MMDDriftDetector
from engine.ml.mmd_reranker import MMDReRanker
from engine.synthesis.episode_synthesizer import EpisodeSynthesizer, _event_to_string
from engine.causal.causal_extractor import CausalEdgeExtractor
from engine.tests.conftest import (
    make_deploy, make_metric, make_log,
    make_incident_signal, make_remediation, make_rename,
)


@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


@pytest.fixture
def pipeline(embedder):
    tracker = AliasTracker()
    store = InMemoryStore()
    index = NumpyBehavioralIndex()
    mmd = MMDDriftDetector()
    reranker = MMDReRanker(mmd=mmd)
    synthesizer = EpisodeSynthesizer(embedder=embedder, index=index, tracker=tracker)
    causal = CausalEdgeExtractor()
    return {
        "tracker": tracker, "store": store, "index": index,
        "mmd": mmd, "reranker": reranker, "synthesizer": synthesizer,
        "causal": causal, "embedder": embedder,
    }


def _ingest(pipeline, events):
    incidents = {}
    remediations = {}
    for e in events:
        e = dict(e)
        pipeline["tracker"].process_event(e)
        pipeline["store"].append(e)
        inc_id = e.get("incident_id")
        if inc_id:
            incidents.setdefault(inc_id, []).append(e)
        if e.get("kind") == "remediation" and e.get("outcome") == "resolved" and inc_id:
            remediations[inc_id] = e
    pipeline["synthesizer"].synthesize_all(incidents, remediations)
    return incidents, remediations


def test_full_pipeline_smoke(pipeline):
    """Basic: ingest events, synthesize episode, query similar."""
    events = [
        make_deploy("svc-a", "v1.0", "2026-05-10T14:00:00Z"),
        make_metric("svc-a", "latency_p99_ms", 5000, "2026-05-10T14:01:00Z"),
        make_log("svc-a", "error", "timeout", "2026-05-10T14:01:30Z"),
        make_incident_signal("svc-a", "INC-1", "alert:svc-a/latency>3000", "2026-05-10T14:02:00Z"),
        make_remediation("svc-a", "INC-1", "rollback", "resolved", "2026-05-10T14:10:00Z"),
    ]
    _ingest(pipeline, events)
    assert len(pipeline["index"]) == 1
    assert "INC-1" in pipeline["synthesizer"].episode_vectors


def test_rename_robustness(pipeline):
    """Core test: incidents match across service renames."""
    # Phase 1: ingest incident under old name
    events1 = [
        make_deploy("payments-svc", "v2.14.0", "2026-05-10T14:00:00Z"),
        make_metric("payments-svc", "latency_p99_ms", 4820, "2026-05-10T14:01:00Z"),
        make_log("payments-svc", "error", "timeout", "2026-05-10T14:01:30Z"),
        make_incident_signal("payments-svc", "INC-OLD", "alert:payments-svc/latency>3000", "2026-05-10T14:02:00Z"),
        make_remediation("payments-svc", "INC-OLD", "rollback", "resolved", "2026-05-10T14:10:00Z"),
    ]
    _ingest(pipeline, events1)

    # Phase 2: rename
    rename_events = [make_rename("payments-svc", "billing-svc", "2026-05-10T15:00:00Z")]
    _ingest(pipeline, rename_events)

    # Phase 3: similar incident under NEW name
    events2 = [
        make_deploy("billing-svc", "v2.15.0", "2026-05-10T16:00:00Z"),
        make_metric("billing-svc", "latency_p99_ms", 5100, "2026-05-10T16:01:00Z"),
        make_log("billing-svc", "error", "timeout", "2026-05-10T16:01:30Z"),
    ]
    _ingest(pipeline, events2)

    # Reconstruct: query with billing-svc events
    strings = [_event_to_string(e) for e in events2]
    seq_vec = pipeline["embedder"].encode_single(" ".join(strings))
    candidates = pipeline["index"].recall(seq_vec, top_k=5)

    # The old incident should be found despite the rename
    found_ids = [c["incident_id"] for c in candidates]
    assert "INC-OLD" in found_ids, f"Rename robustness failed — INC-OLD not in {found_ids}"


def test_causal_edges_from_scenario(pipeline):
    """Causal extractor finds edges in a realistic scenario."""
    events = [
        make_deploy("svc-x", "v3.0", "2026-05-10T10:00:00Z"),
        make_metric("svc-x", "latency_p99_ms", 6000, "2026-05-10T10:01:00Z"),
        make_log("svc-y", "error", "timeout calling svc-x", "2026-05-10T10:01:30Z"),
        make_incident_signal("svc-x", "INC-X", "alert:svc-x/latency>5000", "2026-05-10T10:02:00Z"),
    ]
    edges = pipeline["causal"].extract(events)
    assert len(edges) >= 2, f"Expected ≥2 causal edges, got {len(edges)}"
