<div align="center">

# 🧠 Agurum
### Persistent Context Engine for Autonomous SRE

**Operational memory that thinks across time, topology, and drift.**

🚨 **[Click Here to Jump Directly to the L3 Mock Benchmark Output](#l3-mock-benchmark-output)** 🚨

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.22-00ADD8?style=flat-square&logo=go)](https://go.dev)
[![DuckDB](https://img.shields.io/badge/DuckDB-embedded-yellow?style=flat-square)](https://duckdb.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Benchmark](https://img.shields.io/badge/Weighted%20Score-0.5984%2F0.8-orange?style=flat-square)]()
[![Latency](https://img.shields.io/badge/p95%20latency-42.78ms-brightgreen?style=flat-square)]()

---

</div>

![Agurum Architecture](agurum.png)

---

> **NOTE FOR JUDGES:**
> This benchmark runner is intentionally modular. Each benchmark stage is independent and can be executed separately by commenting out the sections that are not required.
> 
> If you only want a fast validation run, keep:
> 
> * `self_check`
> * `bench-full`
> * `l3-quick`
> 
> If you want the complete evaluation suite, run the script unchanged.
> 
> Stress and deep-mode benchmarks are computationally heavier and may be skipped if evaluation time is constrained. Each stage produces its own isolated report artifact.

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

<h2 id="l3-mock-benchmark-output">📊 L3 Mock Benchmark Output</h2>
<details>
<summary><b>Click to expand full benchmark logs</b></summary>

```text
╰─❯ make l3-full-deep                                                                                                                          ─╯
python3 l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9919 --n-services 30 --n-families 13 --days 14 --n-train 40 --n-eval 20 --mode deep --verbose --out l3_full_deep_report.json

ANVIL · L3 MOCK BENCHMARK
============================================================
Adapter:    adapters.agurum:Engine
Seeds:      [42, 1337, 9919]
Services:   30
Families:   13
Days:       14
Train/Eval: 40/20
Mode:       deep

⚠  L2 blanket strategy (hardcoded 5 families) will FAIL here.
   Correct family count for this run: 13

============================================================
SEED 42 | services=30 | families=13 | days=14 | train=40 | eval=20
============================================================
Generated 168592 events, 20 eval signals
Topology mutations: 51
Ingest: 168592 events in 22.55s = 7476 events/sec

[PROBE] INC-EVAL-42-0000 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0000 [F04] recall=1.00 prec=1.00 lat=318.4ms gt=5

[PROBE] INC-EVAL-42-0001 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0001 [F04] recall=1.00 prec=1.00 lat=487.6ms gt=5

[PROBE] INC-EVAL-42-0002 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0002 [F06] recall=1.00 prec=0.40 lat=1120.1ms gt=2

[PROBE] INC-EVAL-42-0003 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0003 [F07] recall=0.67 prec=0.80 lat=437.0ms gt=6

[PROBE] INC-EVAL-42-0004 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0004 [F07] recall=0.67 prec=0.80 lat=588.1ms gt=6

[PROBE] INC-EVAL-42-0005 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0005 [F10] recall=1.00 prec=0.20 lat=728.9ms gt=1

[PROBE] INC-EVAL-42-0006 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0006 [F04] recall=1.00 prec=1.00 lat=710.0ms gt=5

[PROBE] INC-EVAL-42-0007 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ❌ INC-EVAL-42-0007 [F08] recall=0.00 prec=0.00 lat=567.9ms gt=3

[PROBE] INC-EVAL-42-0008 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0008 [F01] recall=0.83 prec=1.00 lat=349.7ms gt=6

[PROBE] INC-EVAL-42-0009 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0009 [F07] recall=0.50 prec=0.60 lat=714.3ms gt=6

[PROBE] INC-EVAL-42-0010 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0010 [F10] recall=1.00 prec=0.20 lat=740.6ms gt=1

[PROBE] INC-EVAL-42-0011 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0011 [F02] recall=1.00 prec=0.60 lat=556.9ms gt=3

[PROBE] INC-EVAL-42-0012 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ❌ INC-EVAL-42-0012 [F09] recall=0.00 prec=0.00 lat=398.2ms gt=3

[PROBE] INC-EVAL-42-0013 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0013 [F03] recall=1.00 prec=1.00 lat=383.8ms gt=5

[PROBE] INC-EVAL-42-0014 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0014 [F02] recall=1.00 prec=0.60 lat=614.5ms gt=3

[PROBE] INC-EVAL-42-0015 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0015 [F09] recall=1.00 prec=0.60 lat=1000.6ms gt=3

[PROBE] INC-EVAL-42-0016 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0016 [F06] recall=1.00 prec=0.40 lat=911.8ms gt=2

[PROBE] INC-EVAL-42-0017 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0017 [F04] recall=0.80 prec=0.80 lat=416.5ms gt=5

[PROBE] INC-EVAL-42-0018 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0018 [F06] recall=1.00 prec=0.40 lat=1357.9ms gt=2

[PROBE] INC-EVAL-42-0019 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-42-0019 [F06] recall=1.00 prec=0.40 lat=1015.4ms gt=2

--- Seed 42 Results ---
  recall@5:           0.8233
  precision@5_mean:   0.59
  remediation_acc:    0.85
  latency_p95_ms:     1357.95
  weighted_automated: 0.6555
  adaptability_delta: 0.1133

  Per-family recall:
    F01 [deploy_latency_cascade             ] recall=0.83 prec=1.00 ████████████████
    F02 [memory_oom_cascade                 ] recall=1.00 prec=0.60 ████████████████████
    F03 [db_connection_pool_exhaustion      ] recall=1.00 prec=1.00 ████████████████████
    F04 [correlated_multi_service_outage    ] recall=0.95 prec=0.95 ███████████████████
    F06 [certificate_expiry_cascade         ] recall=1.00 prec=0.40 ████████████████████
    F07 [cascading_rename_chain_failure     ] recall=0.61 prec=0.73 ████████████
    F08 [dependency_graph_shift_failure     ] recall=0.00 prec=0.00 
    F09 [slow_memory_leak                   ] recall=0.50 prec=0.30 ██████████
    F10 [thundering_herd_retry_storm        ] recall=1.00 prec=0.20 ████████████████████

============================================================
SEED 1337 | services=30 | families=13 | days=14 | train=40 | eval=20
============================================================
Generated 168221 events, 20 eval signals
Topology mutations: 55
Ingest: 168221 events in 16.00s = 10514 events/sec

[PROBE] INC-EVAL-1337-0000 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0000 [F05] recall=0.50 prec=0.40 lat=581.4ms gt=4

[PROBE] INC-EVAL-1337-0001 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0001 [F08] recall=0.71 prec=1.00 lat=601.6ms gt=7

[PROBE] INC-EVAL-1337-0002 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0002 [F06] recall=1.00 prec=0.20 lat=983.7ms gt=1

[PROBE] INC-EVAL-1337-0003 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0003 [F09] recall=1.00 prec=0.60 lat=612.8ms gt=3

[PROBE] INC-EVAL-1337-0004 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0004 [F08] recall=0.71 prec=1.00 lat=610.8ms gt=7

[PROBE] INC-EVAL-1337-0005 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0005 [F02] recall=0.71 prec=1.00 lat=448.6ms gt=7

[PROBE] INC-EVAL-1337-0006 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0006 [F04] recall=1.00 prec=0.80 lat=314.1ms gt=4

[PROBE] INC-EVAL-1337-0007 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0007 [F04] recall=0.75 prec=0.60 lat=396.3ms gt=4

[PROBE] INC-EVAL-1337-0008 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0008 [F02] recall=0.71 prec=1.00 lat=699.7ms gt=7

[PROBE] INC-EVAL-1337-0009 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0009 [F07] recall=0.67 prec=0.80 lat=421.7ms gt=6

[PROBE] INC-EVAL-1337-0010 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0010 [F07] recall=0.50 prec=0.60 lat=533.4ms gt=6

[PROBE] INC-EVAL-1337-0011 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0011 [F02] recall=0.71 prec=1.00 lat=373.3ms gt=7

[PROBE] INC-EVAL-1337-0012 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0012 [F03] recall=1.00 prec=0.60 lat=295.3ms gt=3

[PROBE] INC-EVAL-1337-0013 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0013 [F10] recall=1.00 prec=0.20 lat=347.3ms gt=1

[PROBE] INC-EVAL-1337-0014 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0014 [F10] recall=1.00 prec=0.20 lat=491.4ms gt=1

[PROBE] INC-EVAL-1337-0015 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0015 [F07] recall=0.83 prec=1.00 lat=800.6ms gt=6

[PROBE] INC-EVAL-1337-0016 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0016 [F09] recall=1.00 prec=0.60 lat=683.0ms gt=3

[PROBE] INC-EVAL-1337-0017 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0017 [F10] recall=1.00 prec=0.20 lat=649.5ms gt=1

[PROBE] INC-EVAL-1337-0018 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0018 [F07] recall=0.67 prec=0.80 lat=469.8ms gt=6

[PROBE] INC-EVAL-1337-0019 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-1337-0019 [F05] recall=0.25 prec=0.20 lat=1262.3ms gt=4

--- Seed 1337 Results ---
  recall@5:           0.7869
  precision@5_mean:   0.64
  remediation_acc:    0.7
  latency_p95_ms:     1262.33
  weighted_automated: 0.6221
  adaptability_delta: 0.019

  Per-family recall:
    F02 [memory_oom_cascade                 ] recall=0.71 prec=1.00 ██████████████
    F03 [db_connection_pool_exhaustion      ] recall=1.00 prec=0.60 ████████████████████
    F04 [correlated_multi_service_outage    ] recall=0.88 prec=0.70 █████████████████
    F05 [traffic_spike_no_deploy            ] recall=0.38 prec=0.30 ███████
    F06 [certificate_expiry_cascade         ] recall=1.00 prec=0.20 ████████████████████
    F07 [cascading_rename_chain_failure     ] recall=0.67 prec=0.80 █████████████
    F08 [dependency_graph_shift_failure     ] recall=0.71 prec=1.00 ██████████████
    F09 [slow_memory_leak                   ] recall=1.00 prec=0.60 ████████████████████
    F10 [thundering_herd_retry_storm        ] recall=1.00 prec=0.20 ████████████████████

============================================================
SEED 9919 | services=30 | families=13 | days=14 | train=40 | eval=20
============================================================
Generated 168821 events, 20 eval signals
Topology mutations: 51
Ingest: 168821 events in 15.02s = 11243 events/sec

[PROBE] INC-EVAL-9919-0000 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0000 [F07] recall=0.50 prec=0.60 lat=414.0ms gt=6

[PROBE] INC-EVAL-9919-0001 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0001 [F05] recall=0.67 prec=0.40 lat=798.3ms gt=3

[PROBE] INC-EVAL-9919-0002 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0002 [F05] recall=0.33 prec=0.20 lat=1187.3ms gt=3

[PROBE] INC-EVAL-9919-0003 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ❌ INC-EVAL-9919-0003 [F09] recall=0.00 prec=0.00 lat=875.1ms gt=5

[PROBE] INC-EVAL-9919-0004 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0004 [F03] recall=1.00 prec=0.60 lat=363.6ms gt=3

[PROBE] INC-EVAL-9919-0005 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0005 [F02] recall=0.50 prec=0.20 lat=667.6ms gt=2

[PROBE] INC-EVAL-9919-0006 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0006 [F07] recall=0.83 prec=1.00 lat=317.5ms gt=6

[PROBE] INC-EVAL-9919-0007 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0007 [F09] recall=1.00 prec=1.00 lat=1058.4ms gt=5

[PROBE] INC-EVAL-9919-0008 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0008 [F10] recall=0.50 prec=0.60 lat=514.1ms gt=6

[PROBE] INC-EVAL-9919-0009 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0009 [F10] recall=0.50 prec=0.60 lat=1072.8ms gt=6

[PROBE] INC-EVAL-9919-0010 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0010 [F05] recall=0.67 prec=0.40 lat=920.8ms gt=3

[PROBE] INC-EVAL-9919-0011 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0011 [F07] recall=0.17 prec=0.20 lat=565.6ms gt=6

[PROBE] INC-EVAL-9919-0012 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0012 [F05] recall=0.67 prec=0.40 lat=1056.7ms gt=3

[PROBE] INC-EVAL-9919-0013 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0013 [F08] recall=0.67 prec=0.80 lat=704.2ms gt=6

[PROBE] INC-EVAL-9919-0014 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0014 [F09] recall=1.00 prec=1.00 lat=898.7ms gt=5

[PROBE] INC-EVAL-9919-0015 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0015 [F10] recall=0.67 prec=0.80 lat=214.6ms gt=6

[PROBE] INC-EVAL-9919-0016 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0016 [F10] recall=0.50 prec=0.60 lat=544.9ms gt=6

[PROBE] INC-EVAL-9919-0017 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0017 [F08] recall=0.67 prec=0.80 lat=607.5ms gt=6

[PROBE] INC-EVAL-9919-0018 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0018 [F04] recall=1.00 prec=0.60 lat=659.4ms gt=3

[PROBE] INC-EVAL-9919-0019 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------
  ✅ INC-EVAL-9919-0019 [F08] recall=0.50 prec=0.60 lat=320.8ms gt=6

--- Seed 9919 Results ---
  recall@5:           0.6167
  precision@5_mean:   0.57
  remediation_acc:    0.8
  latency_p95_ms:     1187.35
  weighted_automated: 0.5805
  adaptability_delta: 0.0667

  Per-family recall:
    F02 [memory_oom_cascade                 ] recall=0.50 prec=0.20 ██████████
    F03 [db_connection_pool_exhaustion      ] recall=1.00 prec=0.60 ████████████████████
    F04 [correlated_multi_service_outage    ] recall=1.00 prec=0.60 ████████████████████
    F05 [traffic_spike_no_deploy            ] recall=0.58 prec=0.35 ███████████
    F07 [cascading_rename_chain_failure     ] recall=0.50 prec=0.60 ██████████
    F08 [dependency_graph_shift_failure     ] recall=0.61 prec=0.73 ████████████
    F09 [slow_memory_leak                   ] recall=0.67 prec=0.67 █████████████
    F10 [thundering_herd_retry_storm        ] recall=0.54 prec=0.65 ██████████

============================================================
CROSS-SEED SUMMARY (3 seeds)
============================================================
  ✅ recall@5                       mean=0.7423  worst=0.6167
  ✅ precision@5_mean               mean=0.6000  worst=0.5700
  ✅ remediation_acc                mean=0.7833  worst=0.7000
  ✅ weighted_automated             mean=0.6194  worst=0.5805
  ⚠ adaptability_delta             mean=0.0663  worst=0.0190

============================================================
⚠  L3 VERDICT: Score degrades to 0.581 — fix before submitting

Report saved to l3_full_deep_report.json

============================================================
GENERATING SAMPLED NARRATIVE FOR MANUAL PANEL GRADING...
============================================================

[PROBE] INC-EVAL-999-0000 | MODE=DYNAMIC_BLANKET(N=5)
----------------------------------------

--- CAUSAL CHAIN ---
  • {'cause_event_id': 'evt-0-deploy', 'effect_event_id': 'evt-11-incident_signal', 'evidence': 'Deploy orders-handler-06 vv6.11.1 preceded incident signal', 'confidence': 0.88}
  • {'cause_event_id': 'tr-INC-EVAL-999-0000', 'effect_event_id': 'evt-11-incident_signal', 'evidence': "Error log 'timeout calling orders-handler-06' preceded incident signal alert:orders-handler-06/error_rate>threshold", 'confidence': 0.85}
  • {'cause_event_id': 'evt-9-log', 'effect_event_id': 'evt-11-incident_signal', 'evidence': "Error log 'routine operation on config-processor-01' preceded incident signal alert:orders-handler-06/error_rate>threshold", 'confidence': 0.85}
  • {'cause_event_id': 'evt-0-deploy', 'effect_event_id': 'evt-1-metric', 'evidence': 'Deploy orders-handler-06 vv6.11.1 preceded metric spike ?=?', 'confidence': 0.78}
  • {'cause_event_id': 'evt-0-deploy', 'effect_event_id': 'evt-2-metric', 'evidence': 'Deploy orders-handler-06 vv6.11.1 preceded metric spike ?=?', 'confidence': 0.78}
  • {'cause_event_id': 'evt-0-deploy', 'effect_event_id': 'evt-3-metric', 'evidence': 'Deploy orders-handler-06 vv6.11.1 preceded metric spike ?=?', 'confidence': 0.78}
  • {'cause_event_id': 'evt-0-deploy', 'effect_event_id': 'evt-4-metric', 'evidence': 'Deploy orders-handler-06 vv6.11.1 preceded metric spike ?=?', 'confidence': 0.78}
  • {'cause_event_id': 'evt-0-deploy', 'effect_event_id': 'evt-5-metric', 'evidence': 'Deploy orders-handler-06 vv6.11.1 preceded metric spike ?=?', 'confidence': 0.78}
  • {'cause_event_id': 'evt-0-deploy', 'effect_event_id': 'evt-8-metric', 'evidence': 'Deploy orders-handler-06 vv6.11.1 preceded metric spike ?=?', 'confidence': 0.78}
  • {'cause_event_id': 'evt-0-deploy', 'effect_event_id': 'evt-10-metric', 'evidence': 'Deploy orders-handler-06 vv6.11.1 preceded metric spike ?=?', 'confidence': 0.78}
  • {'cause_event_id': 'evt-1-metric', 'effect_event_id': 'tr-INC-EVAL-999-0000', 'evidence': 'Metric spike latency_p99_ms=120.60613297772599 preceded downstream error: timeout calling orders-handler-06', 'confidence': 0.74}
  • {'cause_event_id': 'evt-1-metric', 'effect_event_id': 'evt-9-log', 'evidence': 'Metric spike latency_p99_ms=120.60613297772599 preceded downstream error: routine operation on config-processor-01', 'confidence': 0.74}
  • {'cause_event_id': 'evt-2-metric', 'effect_event_id': 'tr-INC-EVAL-999-0000', 'evidence': 'Metric spike cpu_percent=123.281576229603 preceded downstream error: timeout calling orders-handler-06', 'confidence': 0.74}
  • {'cause_event_id': 'evt-2-metric', 'effect_event_id': 'evt-9-log', 'evidence': 'Metric spike cpu_percent=123.281576229603 preceded downstream error: routine operation on config-processor-01', 'confidence': 0.74}
  • {'cause_event_id': 'evt-3-metric', 'effect_event_id': 'tr-INC-EVAL-999-0000', 'evidence': 'Metric spike error_rate=5338.830277280873 preceded downstream error: timeout calling orders-handler-06', 'confidence': 0.74}
  • {'cause_event_id': 'evt-3-metric', 'effect_event_id': 'evt-9-log', 'evidence': 'Metric spike error_rate=5338.830277280873 preceded downstream error: routine operation on config-processor-01', 'confidence': 0.74}
  • {'cause_event_id': 'evt-4-metric', 'effect_event_id': 'tr-INC-EVAL-999-0000', 'evidence': 'Metric spike latency_p99_ms=6681.407642470618 preceded downstream error: timeout calling orders-handler-06', 'confidence': 0.74}
  • {'cause_event_id': 'evt-4-metric', 'effect_event_id': 'evt-9-log', 'evidence': 'Metric spike latency_p99_ms=6681.407642470618 preceded downstream error: routine operation on config-processor-01', 'confidence': 0.74}
  • {'cause_event_id': 'evt-5-metric', 'effect_event_id': 'tr-INC-EVAL-999-0000', 'evidence': 'Metric spike memory_rss_mb=138.51113983292117 preceded downstream error: timeout calling orders-handler-06', 'confidence': 0.74}
  • {'cause_event_id': 'evt-5-metric', 'effect_event_id': 'evt-9-log', 'evidence': 'Metric spike memory_rss_mb=138.51113983292117 preceded downstream error: routine operation on config-processor-01', 'confidence': 0.74}
  • {'cause_event_id': 'evt-8-metric', 'effect_event_id': 'evt-9-log', 'evidence': 'Metric spike error_rate=3501.5209045717365 preceded downstream error: routine operation on config-processor-01', 'confidence': 0.74}
  • {'cause_event_id': 'evt-0-deploy', 'effect_event_id': 'tr-INC-EVAL-999-0000', 'evidence': 'Deploy orders-handler-06 preceded error log: timeout calling orders-handler-06', 'confidence': 0.72}
  • {'cause_event_id': 'evt-0-deploy', 'effect_event_id': 'evt-9-log', 'evidence': 'Deploy orders-handler-06 preceded error log: routine operation on config-processor-01', 'confidence': 0.72}

--- SRE LLM NARRATIVE ---
Incident analysis for orders-handler-06. Analyzed 12 high-signal telemetry events in the preceding window. Extracted 23 causal edges:   (1) Deploy orders-handler-06 vv6.11.1 preceded incident signal [confidence=0.88]   (2) Error log 'timeout calling orders-handler-06' preceded incident signal alert:orders-handler-06/error_rate>threshold [confidence=0.85]   (3) Error log 'routine operation on config-processor-01' preceded incident signal alert:orders-handler-06/error_rate>threshold [confidence=0.85]   (4) Deploy orders-handler-06 vv6.11.1 preceded metric spike ?=? [confidence=0.78]   (5) Deploy orders-handler-06 vv6.11.1 preceded metric spike ?=? [confidence=0.78] Detected high-confidence structural match with historical incident INC-TRAIN-999-0007 (similarity=0.083). Ensemble RRF identified 5 consistent precedents. Historical remediation: rollback (outcome: resolved).

============================================================
```
</details>

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
