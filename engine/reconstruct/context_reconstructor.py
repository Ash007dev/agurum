"""
engine/reconstruct/context_reconstructor.py — 8-phase async reconstruction pipeline.

Translates the synchronous benchmark adapter logic (adapters/agurum.py) into
a non-blocking async pipeline safe for FastAPI.

Golden rule: NEVER call any ML/DuckDB/embedding code directly in the async
context. Always wrap in run_in_executor.

Phase 1: Identity resolution          ≤0.5ms  (sync — dict lookup)
Phase 2: Neighborhood collection      ≤5ms    (sync — in-memory)
Phase 3+4: Event collection + drift   ≤10ms   (parallel via asyncio.gather)
Phase 5: Causal chain extraction      ≤30ms   (run_in_executor)
Phase 6: Behavioral matching          ≤80ms   (run_in_executor — embed+ANN+MMD)
Phase 7: Remediation synthesis        ≤5ms    (sync)
Phase 8: Explain narrative            ≤2ms fast / ≤1200ms deep
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from engine import config
from engine.synthesis.episode_synthesizer import _event_to_string

logger = logging.getLogger(__name__)


def _parse_ts(ts_str: str) -> float:
    """Parse ISO timestamp → unix float. Returns 0.0 on failure."""
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


class ContextReconstructor:
    """
    8-phase async pipeline for incident context reconstruction.

    Takes the synchronous ML components from P3 and wraps them in
    run_in_executor to avoid blocking the asyncio event loop.
    """

    def __init__(self, state: Any) -> None:
        """
        Args:
            state: Application state object with all components attached:
                   - state.tracker (AliasTracker)
                   - state.store (InMemoryStore)
                   - state.embedder (Embedder)
                   - state.index (NumpyBehavioralIndex)
                   - state.mmd (MMDDriftDetector)
                   - state.reranker (MMDReRanker)
                   - state.synthesizer (EpisodeSynthesizer)
                   - state.causal_extractor (CausalEdgeExtractor)
                   - state.remediation_advisor (RemediationAdvisor)
                   - state.llm_synthesizer (LLMSynthesizer)
                   - state.executor (ThreadPoolExecutor)
        """
        self._state = state

    async def reconstruct(
        self,
        signal: dict,
        mode: str = "fast",
    ) -> dict:
        """
        Reconstruct operational context for an incident signal.

        Args:
            signal: IncidentSignal dict with ts, trigger, service, incident_id.
            mode: "fast" (template narrative) or "deep" (LLM narrative).

        Returns:
            Context TypedDict.
        """
        t_start = time.perf_counter()
        loop = asyncio.get_event_loop()
        executor = self._state.executor

        # ── Phase 1: Identity resolution (sync — ≤0.5ms) ────────────────────
        trigger_svc = signal.get("service", "")
        if not trigger_svc:
            raw = signal.get("trigger", "")
            if ":" in raw:
                trigger_svc = raw.split(":")[1].split("/")[0]
            elif "/" in raw:
                trigger_svc = raw.split("/")[0]

        trigger_cid = self._state.tracker.resolve(trigger_svc)
        signal_ts = _parse_ts(signal.get("ts", ""))

        # ── Phase 2: Get neighborhood (sync — ≤5ms) ─────────────────────────
        # For now, we work with the trigger service only.
        # P2's OperationalGraph would provide BFS neighborhood here.
        neighborhood_cids = [trigger_cid]

        # ── Phase 3+4: Event collection + drift check (PARALLEL) ────────────
        window_start = signal_ts - config.WINDOW_SECONDS
        window_end = signal_ts

        # Phase 3: Collect events from InMemoryStore (CPU-bound scan)
        async def _get_window_events():
            return await loop.run_in_executor(
                executor,
                self._state.store.get_by_canonical_id,
                trigger_cid,
                self._state.tracker,
                window_start,
                window_end,
            )

        # Phase 4: Check MMD drift for each service in neighborhood
        async def _check_drift(cid: str):
            try:
                # Get pre-window and post-window events for drift comparison
                pre_events = await loop.run_in_executor(
                    executor,
                    self._state.store.get_by_canonical_id,
                    cid,
                    self._state.tracker,
                    window_start - config.WINDOW_SECONDS,  # 300s before window
                    window_start,
                )
                post_events = await loop.run_in_executor(
                    executor,
                    self._state.store.get_by_canonical_id,
                    cid,
                    self._state.tracker,
                    window_start,
                    window_end,
                )

                if len(pre_events) < 5 or len(post_events) < 5:
                    return None

                pre_strings = [_event_to_string(e) for e in pre_events[-20:]]
                post_strings = [_event_to_string(e) for e in post_events[-20:]]

                pre_vecs = await loop.run_in_executor(
                    executor, self._state.embedder.encode_batch, pre_strings
                )
                post_vecs = await loop.run_in_executor(
                    executor, self._state.embedder.encode_batch, post_strings
                )

                is_drift = await loop.run_in_executor(
                    executor, self._state.mmd.is_drift, pre_vecs, post_vecs
                )
                if is_drift:
                    mmd2 = await loop.run_in_executor(
                        executor, self._state.mmd.compute_mmd_squared, pre_vecs, post_vecs
                    )
                    return {"canonical_id": cid, "mmd_squared": mmd2}
                return None
            except Exception as e:
                logger.warning(f"Drift check failed for {cid}: {e}")
                return None

        # Run Phase 3 and Phase 4 in parallel
        results = await asyncio.gather(
            _get_window_events(),
            *[_check_drift(cid) for cid in neighborhood_cids],
            return_exceptions=True,
        )

        # Unpack results
        window_events = results[0] if not isinstance(results[0], Exception) else []
        drift_signals = [r for r in results[1:] if r and not isinstance(r, Exception)]

        if not window_events:
            return self._empty_context(signal, trigger_svc)

        # ── Phase 5: Causal chain extraction (run_in_executor — ≤30ms) ──────
        try:
            if hasattr(self._state, "learner"):
                self._state.causal_extractor.update_pair_stats(self._state.learner.pair_stats)
            causal_edges = await loop.run_in_executor(
                executor,
                self._state.causal_extractor.extract,
                window_events,
                self._state.tracker,
            )
        except Exception as e:
            logger.warning(f"Causal extraction failed: {e}")
            causal_edges = []

        # ── Phase 6: Behavioral matching (run_in_executor — ≤80ms) ──────────
        try:
            recent_events = window_events[-config.MAX_CONTEXT_EVENTS:]
            strings = [_event_to_string(e) for e in recent_events]
            seq_str = " ".join(strings)

            # 6a: Sequence embedding
            seq_vec = await loop.run_in_executor(
                executor, self._state.embedder.encode_single, seq_str
            )

            # 6b: Per-event embeddings
            event_vecs = await loop.run_in_executor(
                executor, self._state.embedder.encode_batch, strings
            )

            # 6c: ANN recall top-20
            candidates = await loop.run_in_executor(
                executor, self._state.index.recall, seq_vec, config.ANN_TOP_K
            )

            # 6d: MMD rerank → top-5
            ranked = await loop.run_in_executor(
                executor,
                self._state.reranker.rerank,
                event_vecs,
                candidates,
                self._state.synthesizer.episode_vectors,
                config.RERANK_TOP_K,
            )
        except Exception as e:
            logger.warning(f"Behavioral matching failed: {e}")
            ranked = []
            event_vecs = None

        # ── Phase 7: Remediation synthesis (sync — ≤5ms) ────────────────────
        try:
            remediations = self._state.remediation_advisor.suggest(
                ranked, trigger_svc
            )
        except Exception as e:
            logger.warning(f"Remediation synthesis failed: {e}")
            remediations = [{
                "action": "rollback",
                "target": trigger_svc,
                "historical_outcome": "error",
                "confidence": 0.1,
            }]

        # Build similar_past_incidents from ranked matches
        similar_past = [
            {
                "incident_id": r["incident_id"],
                "similarity": r["combined_score"],
                "rationale": (
                    f"cosine={r['cosine_score']:.3f} "
                    f"mmd_sim={r['mmd_similarity']:.3f} "
                    f"combined={r['combined_score']:.3f}"
                ),
            }
            for r in ranked
        ]

        # ── Phase 8: Explain narrative ──────────────────────────────────────
        if mode == "deep" and self._state.llm_synthesizer and self._state.llm_synthesizer.available:
            try:
                explain = await self._state.llm_synthesizer.synthesize(
                    signal, window_events, causal_edges, similar_past, remediations
                )
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
                explain = self._template_explain(
                    signal, trigger_svc, window_events, ranked, remediations, drift_signals
                )
        else:
            explain = self._template_explain(
                signal, trigger_svc, window_events, ranked, remediations, drift_signals
            )

        confidence = ranked[0]["combined_score"] if ranked else 0.0
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            f"Reconstructed context for {trigger_svc} in {elapsed_ms:.1f}ms "
            f"(mode={mode}, events={len(window_events)}, matches={len(ranked)}, "
            f"causal_edges={len(causal_edges)}, drifts={len(drift_signals)})"
        )

        return {
            "related_events": window_events,
            "causal_chain": causal_edges,
            "similar_past_incidents": similar_past,
            "suggested_remediations": remediations,
            "confidence": confidence,
            "explain": explain,
        }

    def _template_explain(
        self,
        signal: dict,
        trigger_svc: str,
        window_events: list[dict],
        ranked: list[dict],
        remediations: list[dict],
        drift_signals: list[dict],
    ) -> str:
        """Fast template narrative (no LLM). Includes rename-robustness note."""
        n_matches = len(ranked)
        top_match = ranked[0]["incident_id"] if ranked else "none"
        top_sim = ranked[0]["combined_score"] if ranked else 0.0
        action = remediations[0].get("action", "unknown") if remediations else "unknown"
        conf = remediations[0].get("confidence", 0.0) if remediations else 0.0

        # Check for rename history
        cid = self._state.tracker.resolve(trigger_svc)
        all_names = self._state.tracker.get_all_names(cid)
        rename_note = ""
        if len(all_names) > 1:
            rename_note = (
                f" Service was previously named {all_names[0]}; "
                f"canonical identity preserved across rename."
            )

        drift_note = ""
        if drift_signals:
            drift_note = f" Behavioral drift detected in {len(drift_signals)} service(s)."

        return (
            f"Incident on {trigger_svc} with {len(window_events)} related events "
            f"in {config.WINDOW_SECONDS}s window. "
            f"Found {n_matches} similar past incidents; "
            f"best match: {top_match} (similarity={top_sim:.3f}).{rename_note}{drift_note} "
            f"Recommended action: {action} (confidence={conf:.2f})."
        )

    def _empty_context(self, signal: dict, trigger_svc: str) -> dict:
        """Return a minimal valid context when no window events found."""
        return {
            "related_events": [],
            "causal_chain": [],
            "similar_past_incidents": [],
            "suggested_remediations": [{
                "action": "rollback",
                "target": trigger_svc,
                "historical_outcome": "no_context",
                "confidence": 0.1,
            }],
            "confidence": 0.0,
            "explain": (
                f"No context events found for {trigger_svc} "
                f"in {config.WINDOW_SECONDS}s window before signal."
            ),
        }
