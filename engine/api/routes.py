"""
engine/api/routes.py — FastAPI route definitions for the Agurum engine.

Endpoints:
  POST /batch               ← Go gateway micro-batch ingest
  POST /reconstruct          ← Incident context reconstruction
  POST /remediation-feedback ← Human feedback for continuous learning
  GET  /health               ← Health check
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from engine.synthesis.episode_synthesizer import _event_to_string

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    """Health check — returns store, graph, and cache stats."""
    state = request.app.state
    return {
        "status": "ok",
        "episodes_indexed": len(state.index),
        "cache_entities": state.cache.entity_count(),
        "cache_events": state.cache.event_count(),
        "graph_nodes": state.graph.node_count(),
        "graph_edges": state.graph.edge_count(),
        "uptime_seconds": round(time.time() - state.start_time, 1),
    }


@router.post("/batch")
async def ingest_batch(request: Request) -> JSONResponse:
    """
    Receive a micro-batch of events from P1's Go gateway.

    Expected body: {"events": [list of event dicts]}

    Fan-out pipeline per event:
      1. EntityRegistry.resolve / register  → run_in_executor (SQLite I/O)
      2. OperationalGraph.upsert_edge       → run_in_executor (CPU-light, but locks)
      3. SlidingWindowCache.push            → await directly (asyncio.Lock, no executor)
      4. EventStore.append_event            → run_in_executor (DuckDB I/O)

    Throughput target: ≥1,000 events/s sustained.
    """
    state = request.app.state
    body = await request.json()

    # Accept both {"events": [...]} and a bare list (for flexibility)
    if isinstance(body, list):
        events = body
    else:
        events = body.get("events", [])

    if not events:
        return JSONResponse({"accepted": 0}, status_code=200)

    loop = asyncio.get_event_loop()
    ex = state.executor
    accepted = 0

    for event in events:
        event = dict(event)
        svc = event.get("service", "")
        ts = event.get("ts", "")
        kind = event.get("kind", "")

        # ── Keep AliasTracker + InMemoryStore in sync (for ContextReconstructor)
        state.tracker.process_event(event)
        state.store.append(event)

        # ── Handle topology rename events ─────────────────────────────────
        if kind == "topology" and event.get("change") == "rename":
            old_name = event.get("from_") or event.get("from", "")
            new_name = event.get("to", svc)
            if old_name and new_name and ts:
                try:
                    await loop.run_in_executor(
                        ex, state.registry.rename, old_name, new_name, ts
                    )
                    logger.info(f"Renamed {old_name!r} → {new_name!r} at {ts}")
                except KeyError:
                    # old_name not yet registered — register the new name fresh
                    await loop.run_in_executor(
                        ex, state.registry.register, new_name, ts
                    )
                    logger.warning(
                        f"Rename: {old_name!r} not found — registered {new_name!r} as new entity"
                    )
            event["canonical_id"] = new_name
            await state.cache.push(event)
            accepted += 1
            continue

        # ── Step 1: Resolve (or register) the service → canonical_id ──────
        cid = ""
        if svc and ts:
            try:
                cid = await loop.run_in_executor(
                    ex, state.registry.resolve, svc, ts
                )
            except KeyError:
                # First time we've seen this service — register it
                cid = await loop.run_in_executor(
                    ex, state.registry.register, svc, ts
                )
        event["canonical_id"] = cid

        # ── Step 2: Update OperationalGraph from topology/trace hints ─────
        caller = event.get("caller") or event.get("upstream")
        if caller and svc and caller != svc:
            try:
                caller_cid = await loop.run_in_executor(
                    ex, state.registry.resolve, caller, ts
                )
            except KeyError:
                caller_cid = await loop.run_in_executor(
                    ex, state.registry.register, caller, ts
                )
            rel = "calls" if kind in ("trace", "log", "metric") else "depends_on"
            await loop.run_in_executor(
                ex, state.graph.upsert_edge, caller_cid, cid, rel
            )

        # ── Step 3: Push into SlidingWindowCache (asyncio.Lock — safe direct)
        await state.cache.push(event)

        # ── Step 4: Persist to DuckDB EventStore (in executor) ────────────
        await loop.run_in_executor(
            ex, state.eventstore.append_event, event, cid
        )

        # ── Track per-incident events for episode synthesis ───────────────
        inc_id = event.get("incident_id")
        if inc_id:
            state.incidents.setdefault(inc_id, []).append(event)

        # ── Handle resolved remediations → synthesize episode ─────────────
        if kind == "remediation" and event.get("outcome") == "resolved" and inc_id:
            state.remediations[inc_id] = event
            state.learner.on_remediation(
                incident_id=inc_id,
                canonical_id=cid,
                action=event.get("action", "rollback"),
                outcome="resolved",
            )
            state.synthesizer.synthesize_all(state.incidents, state.remediations)

        accepted += 1

    return JSONResponse({"accepted": accepted}, status_code=200)



@router.post("/reconstruct")
async def reconstruct(request: Request) -> JSONResponse:
    """
    Reconstruct operational context for an incident signal.

    Expected body: IncidentSignal dict with:
      - incident_id: str
      - ts: str (ISO 8601)
      - trigger: str
      - service: str (preferred)
    
    Query params:
      - mode: "fast" (default) or "deep"
    """
    state = request.app.state
    signal = await request.json()
    mode = request.query_params.get("mode", "fast")

    t_start = time.perf_counter()

    try:
        context = await state.reconstructor.reconstruct(signal, mode=mode)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return JSONResponse({
            **context,
            "_meta": {
                "mode": mode,
                "latency_ms": round(elapsed_ms, 1),
            },
        })
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e), "explain": "Reconstruction failed — returning empty context."},
            status_code=500,
        )


@router.post("/remediation-feedback")
async def remediation_feedback(request: Request) -> JSONResponse:
    """
    Receive human feedback on a remediation suggestion.

    Expected body:
      - incident_id: str
      - was_helpful: bool
      - service: str (optional)
    """
    state = request.app.state
    body = await request.json()
    incident_id = body.get("incident_id", "")
    was_helpful = body.get("was_helpful", True)
    service = body.get("service")

    canonical_id = None
    if service:
        # Use AliasTracker (1-arg resolve), NOT EntityRegistry (2-arg)
        canonical_id = state.tracker.resolve(service)

    state.learner.on_feedback(
        incident_id=incident_id,
        was_helpful=was_helpful,
        canonical_id=canonical_id,
    )

    return JSONResponse({"status": "acknowledged", "incident_id": incident_id})
