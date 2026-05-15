"""
engine/main.py — FastAPI application entry point for the Agurum engine.

Startup sequence:
  1. Create ThreadPoolExecutor
  2. Load embedder via run_in_executor (~20s first time, cached after)
  3. Init AliasTracker, InMemoryStore
  4. Init NumpyBehavioralIndex, MMDDriftDetector, MMDReRanker
  5. Init EpisodeSynthesizer
  6. Init CausalEdgeExtractor, RemediationAdvisor
  7. Init LLMSynthesizer (optional — needs ANTHROPIC_API_KEY)
  8. Init ContinuousLearner
  9. Wire ContextReconstructor
  10. Include API routes + static demo

Server binds to UDS (/tmp/pce.sock) if PCE_UDS_PATH is set,
otherwise falls back to TCP (0.0.0.0:8000).
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from engine import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("agurum")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown."""
    logger.info("═" * 60)
    logger.info("  agurum Persistent Context Engine — starting up")
    logger.info("═" * 60)

    t_start = time.time()
    app.state.start_time = t_start

    # ── Step 1: ThreadPoolExecutor ────────────────────────────────────────
    app.state.executor = ThreadPoolExecutor(
        max_workers=config.EXECUTOR_MAX_WORKERS,
        thread_name_prefix="pce-worker",
    )
    logger.info(f"Thread pool: {config.EXECUTOR_MAX_WORKERS} workers")

    loop = asyncio.get_event_loop()

    # ── Step 2: Embedder (heavy — load in executor) ──────────────────────
    logger.info("Loading embedding model (this may take ~20s on first run)...")
    from engine.ml.embedder import get_embedder
    app.state.embedder = await loop.run_in_executor(
        app.state.executor, get_embedder
    )
    logger.info("Embedder loaded + warmed up ✓")

    # ── Step 3: P2's EntityRegistry (SQLite — load in executor) ──────────
    from engine.registry.entity_registry import EntityRegistry
    app.state.registry = await loop.run_in_executor(
        app.state.executor,
        EntityRegistry,
        config.SQLITE_PATH,
    )
    logger.info(f"EntityRegistry initialized (db={config.SQLITE_PATH!r}) ✓")

    # ── Step 3b: P3's AliasTracker + InMemoryStore (in-memory, fast) ─────
    # ContextReconstructor expects state.tracker (1-arg resolve) and
    # state.store (InMemoryStore with get_by_canonical_id). These run
    # in parallel with EntityRegistry — tracker for fast ML lookups,
    # registry for persistent SQLite-backed identity.
    from engine.registry.alias_tracker import AliasTracker
    from engine.store.in_memory_store import InMemoryStore
    app.state.tracker = AliasTracker()
    app.state.store = InMemoryStore()
    logger.info("AliasTracker + InMemoryStore initialized ✓")

    # ── Step 4: P2's OperationalGraph (NetworkX) ──────────────────────────
    from engine.graph.operational_graph import OperationalGraph
    app.state.graph = OperationalGraph()
    app.state.graph.set_registry(app.state.registry)  # enable role writeback
    logger.info("OperationalGraph initialized ✓")

    # ── Step 5: P2's SlidingWindowCache (asyncio.Lock — safe in loop) ────
    from engine.cache.sliding_window import SlidingWindowCache
    app.state.cache = SlidingWindowCache()
    logger.info("SlidingWindowCache initialized ✓")

    # ── Step 6: P3's EventStore (DuckDB — load in executor) ──────────────
    from engine.store.event_store import EventStore
    app.state.eventstore = await loop.run_in_executor(
        app.state.executor,
        EventStore,
        config.DUCKDB_PATH,
    )
    logger.info(f"EventStore initialized (db={config.DUCKDB_PATH!r}) ✓")

    # ── Step 7: ML pipeline components ───────────────────────────────────
    from engine.ml.numpy_index import NumpyBehavioralIndex
    from engine.ml.mmd_detector import MMDDriftDetector
    from engine.ml.mmd_reranker import MMDReRanker

    app.state.index = NumpyBehavioralIndex()
    app.state.mmd = MMDDriftDetector()
    app.state.reranker = MMDReRanker(mmd=app.state.mmd)
    logger.info("ML pipeline (Index + MMD + ReRanker) initialized ✓")

    # ── Step 8: EpisodeSynthesizer ───────────────────────────────────────
    from engine.synthesis.episode_synthesizer import EpisodeSynthesizer
    app.state.synthesizer = EpisodeSynthesizer(
        embedder=app.state.embedder,
        index=app.state.index,
        tracker=app.state.registry,   # EntityRegistry is the authoritative tracker
    )
    logger.info("EpisodeSynthesizer initialized ✓")

    # ── Step 9: CausalEdgeExtractor + RemediationAdvisor ─────────────────
    from engine.causal.causal_extractor import CausalEdgeExtractor
    from engine.remediation.remediation_advisor import RemediationAdvisor

    app.state.causal_extractor = CausalEdgeExtractor()
    app.state.remediation_advisor = RemediationAdvisor(tracker=app.state.registry)
    logger.info("CausalEdgeExtractor + RemediationAdvisor initialized ✓")

    # ── Step 10: LLMSynthesizer (optional) ───────────────────────────────
    from engine.llm.llm_synthesizer import LLMSynthesizer
    app.state.llm_synthesizer = LLMSynthesizer()
    if app.state.llm_synthesizer.available:
        logger.info("LLMSynthesizer initialized (deep mode available) ✓")
    else:
        logger.info("LLMSynthesizer: no API key — deep mode will use template ✓")

    # ── Step 11: ContinuousLearner ───────────────────────────────────────
    from engine.learning.continuous_learner import ContinuousLearner
    app.state.learner = ContinuousLearner(
        eventstore=app.state.eventstore,
        index=app.state.index
    )
    logger.info("ContinuousLearner initialized ✓")

    # ── Step 12: ContextReconstructor ────────────────────────────────────
    from engine.reconstruct.context_reconstructor import ContextReconstructor
    app.state.reconstructor = ContextReconstructor(state=app.state)
    logger.info("ContextReconstructor wired ✓")

    # ── Shared incident/remediation accumulators (used by /batch) ─────────
    app.state.incidents: dict[str, list[dict]] = {}
    app.state.remediations: dict[str, dict] = {}

    elapsed = time.time() - t_start
    logger.info(f"═══ Startup complete in {elapsed:.1f}s ═══")

    yield  # ── Application runs here ──

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down...")
    try:
        app.state.registry.close()
    except Exception:
        pass
    try:
        app.state.eventstore.close()
    except Exception:
        pass
    app.state.executor.shutdown(wait=True, cancel_futures=False)
    logger.info("Shutdown complete.")



# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="agurum – Persistent Context Engine",
    description="Operational memory engine for autonomous SRE",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow demo UI to call API from file:// or localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
from engine.api.routes import router  # noqa: E402
app.include_router(router)

# ── Static demo UI ────────────────────────────────────────────────────────────
_demo_dir = Path(__file__).parent / "demo"
if _demo_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    if config.UDS_PATH:
        logger.info(f"Binding to Unix Domain Socket: {config.UDS_PATH}")
        uvicorn.run(
            "engine.main:app",
            uds=config.UDS_PATH,
            log_level="info",
        )
    else:
        logger.info(f"Binding to TCP {config.HOST}:{config.PORT}")
        uvicorn.run(
            "engine.main:app",
            host=config.HOST,
            port=config.PORT,
            log_level="info",
        )
