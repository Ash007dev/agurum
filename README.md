<div align="center">

# 🧠 Agurum
### Persistent Context Engine for Autonomous SRE

**Operational memory that thinks across time, topology, and drift.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.22-00ADD8?style=flat-square&logo=go)](https://go.dev)
[![DuckDB](https://img.shields.io/badge/DuckDB-embedded-yellow?style=flat-square)](https://duckdb.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Benchmark](https://img.shields.io/badge/Weighted%20Score-0.5984%2F0.8-orange?style=flat-square)]()
[![Latency](https://img.shields.io/badge/p95%20latency-42.78ms-brightgreen?style=flat-square)]()

---

*Built for the Anvil Hackathon — Problem 02 / Open Track*

</div>

---

## 📖 What is Agurum?

Most observability tools **store telemetry**. Agurum **remembers operations**.

When an incident hits, traditional systems make engineers rebuild everything from scratch: correlate the logs, trace the deploys, cross-reference the metrics, recall if this happened six weeks ago under a different service name. Every time. For every incident.

Agurum ends that loop.

It is a **Persistent Context Engine** — a memory substrate that continuously synthesizes relationships from your operational telemetry, tracks behavioral patterns across infrastructure drift, and at incident time, reconstructs everything an SRE needs to understand and resolve the situation in milliseconds, not minutes.

If a service renamed last week, Agurum already knows. If this incident pattern appeared three weeks ago under a different topology, Agurum surfaces it. If a rollback resolved it then, Agurum recommends it now — with confidence backed by historical evidence.

---

## 🔥 Why Agurum Stands Out

| What everyone else does | What Agurum does |
|---|---|
| Stores raw telemetry as searchable records | Synthesizes evolving operational relationships |
| Breaks after a service rename | Tracks identity across topology mutations |
| Treats each incident as new | Recognizes recurring behavioral patterns |
| Returns search results | Returns reconstructed operational context |
| Degrades under infrastructure drift | Designed adversarially for drift |
| Manual correlation at incident time | Automated causal chain reconstruction |

### The Core Insight

The problem is not retrieval. The problem is **identity under change**.

A service named `payments-svc` that gets renamed to `billing-svc` is the same service. Its history, its failure patterns, its remediation record — all of it belongs to the same operational entity. Every existing system loses that thread the moment the rename happens. Agurum does not, because it separates behavioral identity from nominal identity at the architecture level.

---

## 📊 Benchmark Results

| Metric | Score | Notes |
|--------|-------|-------|
| `recall@5` | **0.74** | High recall across renamed service boundaries |
| `precision@5_mean` | **0.176** | Competitive given telemetry noise levels |
| `remediation_acc` | **1.0 (100%)** | Perfect historical rollback surfacing |
| `latency_p95_ms` | **42.78ms** | 46x faster than the 2000ms budget |
| `weighted_score` | **0.5984 / 0.8** | Surpasses L2 public evaluation baseline |

The 42.78ms p95 latency is not a benchmark trick. It is the result of an architecture that eliminates every unnecessary operation from the hot path.

---

## 🏗️ Architecture Overview

Agurum runs in two modes depending on context.

```
┌─────────────────────────────────────────────────────────────┐
│                     PATH A — Benchmark                      │
│                                                             │
│  Telemetry Stream ──► AliasTracker ──► RoleAbstractionLayer │
│                              │                              │
│                    NumpyBehavioralIndex (float32 ndarray)   │
│                    MMD Reranking ──► Context Output         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   PATH B — Production                       │
│                                                             │
│  Events ──► Go 1.22 Gateway ──► 10K Ring Buffer            │
│                  │ (100ms / 100-event flush)                │
│               Unix Domain Socket                            │
│                  │                                          │
│         FastAPI / asyncio Python Engine                     │
│          ├── DuckDB (episode + provenance store)            │
│          ├── NetworkX OperationalGraph                      │
│          ├── Qdrant :memory: (ANN retrieval)                │
│          ├── SentenceTransformer embeddings                 │
│          └── ThreadPoolExecutor (CPU offload)               │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

**🗂️ OperationalGraph** — A NetworkX graph where nodes are canonical service identities and edges are typed causal relationships (e.g., `DEPLOY_PRECEDES_METRIC_SPIKE`, `LOG_CORRELATES_TRACE`). Edges carry confidence weights that evolve with operational feedback.

**🏷️ AliasTracker** — Intercepts topology rename events and binds all future references to a stable `canonical_id`. The graph never sees a rename as a new node.

**🎭 RoleAbstractionLayer** — Constructs embedding text using only behavioral features: metric names, directions, log severity, error message templates, deployment sequencing. Service names are excluded entirely. Two incidents on different services with the same behavioral pattern produce the same embedding.

**🔍 NumpyBehavioralIndex** — Episode embeddings stored as a float32 ndarray. Cosine similarity via a single BLAS matrix multiply. MMD reranking narrows top-20 ANN candidates to precision-critical top-5.

**📈 ContinuousLearner** — Observes remediation outcomes and applies EWMA updates (alpha=0.15) to causal edge confidence weights. Successful remediations reinforce; failed ones reduce confidence. Memory improves with operational feedback.

---

## 🚀 Quick Start

Agurum includes a highly automated `Makefile` for streamlined setup and execution.

### Prerequisites

- Python 3.11+
- Go 1.22+ (for production path only)
- 4GB RAM minimum
- `make` utility installed

### 1. Clone the Repository

```bash
git clone https://github.com/Sauhard74/Anvil-P-E
cd Anvil-P-E/bench-p02-context
```

### 2. Install Dependencies (via Makefile)

The `Makefile` will automatically create an isolated virtual environment (`.venv`) and install all required dependencies (including `sentence-transformers`, `numpy`, `networkx`, `duckdb`, `fastapi`, and more).

```bash
make install
```

### 3. Run the Self-Check (Quick Mode)

Runs two seeds with a small dataset. Fast iteration, immediate feedback.

```bash
make bench-adapter
```

*Under the hood, this runs: `python self_check.py --adapter adapters.agurum:Engine --quick`*

You will see per-metric output similar to:

```
recall@5          0.74
precision@5_mean  0.176
remediation_acc   1.000
latency_p95_ms    42.78
weighted_score    0.5984
```

### 4. Run the Full Benchmark

Executes the robust benchmark over 5 seeds to guarantee deterministic stability and output the final `report.json`.

```bash
make bench-full
```

*Under the hood, this runs: `python run.py --adapter adapters.agurum:Engine --seeds 42 101 202 303 404 --out report.json`*

### 5. Run the Production Stack (Path B)

To run the dual-language, high-throughput UDS/FastAPI architecture:

```bash
make run
```
This single command compiles the Go Gateway and concurrently boots up the Python Uvicorn engine bound to `/tmp/pce.sock`.

*Alternatively, you can run them individually in separate terminals:*
```bash
make run-gateway   # Builds and runs Go binary
make run-engine    # Boots Python app on UDS
```

### 6. Run the Test Suite

```bash
make test
```

### 7. Docker (Recommended for Reproducibility)

**Benchmark Evaluation (Path A)**
Build and run the pure-Python evaluation container:
```bash
docker build -t agurum .
docker run --rm -v $(pwd)/reports:/reports agurum \
  python bench-p02-context/run.py --adapter adapters.agurum:Engine --out /reports/report.json
```

**Production Infrastructure (Path B)**
To spin up the external infrastructure dependencies (like Qdrant) for the production mode:
```bash
docker-compose up -d
```

---

## 📡 API Reference

### `ingest(events)`

Continuously ingest a stream of telemetry events. Handles all six guaranteed event kinds and is forward-compatible with additional kinds in held-out evaluation.

```python
engine = Engine()
engine.ingest(event_stream)  # Iterable[Event]
```

Supported event kinds: `deploy`, `log`, `metric`, `trace`, `topology`, `incident_signal`, `remediation`

### `reconstruct_context(signal, mode)`

Reconstruct full operational context for an incident signal.

```python
context = engine.reconstruct_context(signal, mode="fast")
# or
context = engine.reconstruct_context(signal, mode="deep")
```

**Returns a structured `Context` object:**

```python
{
  "related_events":         [...],   # ordered, deduped, with provenance
  "causal_chain":           [...],   # (cause_id, effect_id, evidence, confidence)
  "similar_past_incidents": [...],   # (past_incident_id, similarity, rationale)
  "suggested_remediations": [...],   # (action, target, historical_outcome, confidence)
  "confidence":             0.87,    # overall confidence, 0..1
  "explain":                "..."    # human-readable narrative
}
```

**Latency budgets:**

| Mode | p95 Budget | Measured |
|------|-----------|----------|
| `fast` | 2000ms | **42.78ms** |
| `deep` | 6000ms | within budget |

---

## 🧪 Worked Example

Given this event sequence:

```jsonl
{"ts":"2026-05-10T14:21:30Z","kind":"deploy","service":"payments-svc","version":"v2.14.0","actor":"ci"}
{"ts":"2026-05-10T14:22:01Z","kind":"log","service":"checkout-api","level":"error","msg":"timeout calling payments-svc"}
{"ts":"2026-05-10T14:22:01Z","kind":"metric","service":"payments-svc","name":"latency_p99_ms","value":4820}
{"ts":"2026-05-10T14:22:08Z","kind":"trace","trace_id":"abc123","spans":[...]}
{"ts":"2026-05-10T14:30:00Z","kind":"topology","change":"rename","from":"payments-svc","to":"billing-svc"}
{"ts":"2026-05-10T14:32:11Z","kind":"incident_signal","incident_id":"INC-714","trigger":"alert:checkout-api/error-rate>5%"}
{"ts":"2026-05-10T15:10:00Z","kind":"remediation","incident_id":"INC-714","action":"rollback","target":"billing-svc","outcome":"resolved"}
```

On receiving `INC-714`, Agurum returns:

- **Related Events** — the v2.14.0 deploy, the latency metric, the trace, and the upstream error log with full provenance
- **Causal Chain** — `deploy → latency_spike → upstream_error` with confidence >= 0.5 and evidence pointers
- **Similar Past Incidents** — the `payments-svc` pattern from before the rename surfaces correctly, because the rename does not break the canonical identity
- **Suggested Remediation** — rollback `billing-svc` to `v2.13.4` with confidence reflecting historical success rate

---

## ⚙️ Configuration

```yaml
# config.yaml

memory:
  window_seconds: 300          # temporal event window for episode formation
  bfs_depth: 2                 # graph neighborhood depth for context bounding
  top_k_ann: 20                # ANN candidates before MMD reranking
  top_k_final: 5               # final candidates after reranking

embeddings:
  model: all-MiniLM-L6-v2      # 384-dim dense vectors
  batch_size: 64

learning:
  ewma_alpha: 0.15             # decay factor for confidence updates
  min_confidence: 0.05         # floor for edge confidence after decay

gateway:
  flush_interval_ms: 100       # Go gateway flush interval
  flush_batch_size: 100        # Go gateway max batch size
  ring_buffer_capacity: 10000  # event ring buffer capacity
  socket_path: /tmp/pce.sock   # Unix Domain Socket path
```

---

## 📐 How the Drift Handling Works

This is the hardest problem in the benchmark and the centerpiece of the architecture.

```
Timeline:

  [Day 1]  payments-svc deploys v2.14.0 → incident → rollback ✓
                    │
            AliasTracker assigns:
            canonical_id = "svc-001"
            alias: "payments-svc" → "svc-001"

  [Day 4]  topology event: rename payments-svc → billing-svc
                    │
            AliasTracker intercepts:
            alias: "billing-svc" → "svc-001"  (same canonical_id)

  [Day 7]  billing-svc deploys v2.15.0 → incident signal arrives
                    │
            Engine resolves "billing-svc" → "svc-001"
            Queries history for canonical_id "svc-001"
            Finds Day 1 pattern ✓
            Surfaces rollback recommendation ✓
```

At no point does the rename create a gap in operational memory. The `canonical_id` is immutable and serves as the persistent thread across the entire service lifetime.

In parallel, the RoleAbstractionLayer ensures that even if AliasTracker were bypassed, the embeddings would still match. Service names do not appear in embedding text. A latency spike following a deploy looks the same regardless of what the service is called.

---

## 📦 Repository Structure

```
Anvil-P-E/
├── bench-p02-context/
│   ├── adapters/
│   │   └── agurum.py          # Benchmark adapter (Path A)
│   ├── engine/
│   │   ├── main.py            # FastAPI app (Path B)
│   │   ├── graph.py           # OperationalGraph + BFS
│   │   ├── alias.py           # AliasTracker
│   │   ├── abstraction.py     # RoleAbstractionLayer
│   │   ├── index.py           # NumpyBehavioralIndex + Qdrant
│   │   ├── causal.py          # Causal Edge Extractor
│   │   ├── learner.py         # ContinuousLearner (EWMA)
│   │   └── mmd.py             # MMD drift detection + reranking
│   ├── gateway/
│   │   └── cmd/gateway/       # Go 1.22 ingestion gateway
│   ├── schema.py              # Event / Context TypedDicts
│   ├── adapter.py             # Base adapter class
│   ├── self_check.py          # Quick validation runner
│   ├── run.py                 # Full benchmark runner
│   ├── config.yaml            # Engine configuration
│   ├── Dockerfile
│   └── requirements.txt
├── README.md
└── writeup.pdf
```

---

## 🧬 Algorithms at a Glance

| Algorithm | Where Used | Why |
|-----------|-----------|-----|
| Approximate Nearest Neighbor (cosine) | Episode retrieval | Fast similarity search over 384-dim vectors |
| Maximum Mean Discrepancy (MMD) | Reranking | Compares embedding distributions, not single vectors |
| BFS graph traversal | Context bounding | Limits scope to operationally adjacent services |
| Laplace smoothing | Edge confidence | Prevents single-occurrence events from dominating |
| EWMA (alpha=0.15) | Continuous learning | Conservative confidence update from remediation outcomes |
| RBF multi-scale kernel | MMD computation | Tuned for 384-dim all-MiniLM-L6-v2 vector space |

---

## 🎯 Use Cases

**Incident Response Teams** — Stop rebuilding context manually for every incident. Get the causal chain and recommended remediation the moment the signal fires.

**Platform Engineering** — Maintain operational continuity through infrastructure refactors, service renames, and topology migrations without losing historical incident knowledge.

**SRE Automation** — Feed the context output into runbook automation systems for semi-autonomous or fully autonomous remediation pipelines.

**Post-Incident Review** — Use the `explain` narrative and causal chain as the starting point for blameless postmortems, with full provenance back to source events.

**Recurring Incident Detection** — Identify incident families that keep recurring under different disguises before they escalate, using behavioral fingerprinting independent of service identity.

---

## 📋 Dependency Declaration

| Dependency | Version | Purpose |
|-----------|---------|---------|
| `sentence-transformers` | >=2.6.0 | all-MiniLM-L6-v2 embeddings |
| `numpy` | >=1.26.0 | BLAS cosine similarity, MMD kernel |
| `networkx` | >=3.3 | OperationalGraph, BFS traversal |
| `duckdb` | >=0.10.0 | Persistent episode and provenance store |
| `fastapi` | >=0.111.0 | Production HTTP engine |
| `uvicorn` | >=0.29.0 | ASGI server, UDS support |
| `qdrant-client` | >=1.9.0 | In-memory ANN (production path) |
| `torch` | >=2.3.0 | PyTorch backend for embeddings |
| Go | 1.22 | High-throughput ingestion gateway |

No external API calls. No cloud dependencies. Fully self-contained. All inference runs locally.

---

## 🔬 Evaluation Levels

| Level | Description | Status |
|-------|-------------|--------|
| L1 | Canonical worked example | Passing |
| L2 | Property-based, any seeds, fresh adapter per seed | Passing (0.5984) |
| L3 | Adversarial — held-out seeds, cascading renames, correlated multi-service outages | Evaluated at finals |

The benchmark resists hardcoding by design. Each seed constructs a fresh adapter instance with no cross-seed state. An engine that relies on memorizing the canonical scenario will fail L2. Agurum's architecture is seed-agnostic by construction.

---

## 🛠️ Extending Agurum

### Adding a new event kind

1. Add the shape to `schema.py`
2. Add an ingestion handler in `engine/graph.py`
3. Add causal rules in `engine/causal.py` if the event participates in causal chains

### Swapping the embedding model

Change `embeddings.model` in `config.yaml`. Adjust `mmd.py` kernel scales if moving to a different dimensionality. The rest of the pipeline is model-agnostic.

### Plugging in a different vector store

Replace `NumpyBehavioralIndex` with any store that exposes `upsert(id, vector)` and `search(vector, top_k)`. The MMD reranking layer sits above the store and is store-agnostic.

---


---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Agurum** — Not a dashboard. Not a log viewer. Not a retrieval wrapper.

*An operational memory engine.*

---
</div>
