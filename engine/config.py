"""
engine/config.py — centralized configuration for the Agurum engine.

All env vars, constants, and timeouts. Single source of truth.
"""
from __future__ import annotations

import os


# ── Server ────────────────────────────────────────────────────────────────────
HOST = os.environ.get("PCE_HOST", "0.0.0.0")
PORT = int(os.environ.get("PCE_PORT", "8000"))
UDS_PATH = os.environ.get("PCE_UDS_PATH", "")  # empty = use TCP; set to /tmp/pce.sock for Go gateway

# ── Storage ───────────────────────────────────────────────────────────────────
DUCKDB_PATH = os.environ.get("PCE_DB_PATH", ":memory:")
SQLITE_PATH = os.environ.get("PCE_SQLITE_PATH", ":memory:")

# ── Embedder ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("PCE_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# ── Sliding window / cache ────────────────────────────────────────────────────
WINDOW_SECONDS = int(os.environ.get("PCE_WINDOW_SEC", "300"))
MAX_CONTEXT_EVENTS = int(os.environ.get("PCE_MAX_CTX_EVENTS", "30"))

# ── Ingest ────────────────────────────────────────────────────────────────────
RING_BUFFER_CAP = int(os.environ.get("PCE_RING_CAP", "10000"))
BATCH_SIZE = int(os.environ.get("PCE_BATCH_SIZE", "100"))
BATCH_INTERVAL_MS = int(os.environ.get("PCE_BATCH_INTERVAL_MS", "100"))

# ── ML Pipeline ───────────────────────────────────────────────────────────────
ANN_TOP_K = int(os.environ.get("PCE_ANN_TOP_K", "20"))
RERANK_TOP_K = int(os.environ.get("PCE_RERANK_TOP_K", "5"))

# ── LLM (deep mode) ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.environ.get("PCE_LLM_MODEL", "claude-haiku-4-5")
LLM_MAX_TOKENS = int(os.environ.get("PCE_LLM_MAX_TOKENS", "400"))

# ── Groq (deep mode, preferred over Anthropic — free & ultra-fast) ───────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("PCE_GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ── Thread pool ───────────────────────────────────────────────────────────────
EXECUTOR_MAX_WORKERS = int(os.environ.get("PCE_EXECUTOR_WORKERS", "4"))

# ── Continuous Learner ────────────────────────────────────────────────────────
EWMA_ALPHA = float(os.environ.get("PCE_EWMA_ALPHA", "0.15"))

