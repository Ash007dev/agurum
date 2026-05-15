# Agurum Implementation Plan: Path A & Path B

Our implementation separates into two distinct tracks—Path A and Path B—allowing us to meet raw benchmarking goals while building an asynchronous production-ready demo.

## Path A — The Benchmark Path

**Goal:** Provide a high-scoring, correct, purely synchronous implementation designed exclusively for the `bench-p02-context` harness. This is the path that objectively scores the submission.

**Implementation Details:**
- **Location:** `bench-p02-context/adapters/agurum.py`.
- **Synchronous Design:** Because the benchmark harness expects synchronous operations, the `Adapter` class (`Engine`) exposes plain synchronous methods `ingest(events)` and `reconstruct_context()`. Cannot utilize `asyncio`.
- **Lightweight Components:** It uses streamlined Python components without network, database daemons, or Go gateway overhead:
  - `AliasTracker`: A plain in-memory dictionary for entity identity.
  - `Embedder`: Singleton instance of `all-MiniLM-L6-v2`.
  - `NumpyBehavioralIndex`: In-memory ndarray for episode indexing.
  - `MMDDetector`, `MMDReRanker`, array-based algorithms, and `EpisodeSynthesizer`.
- **Constraints:** Because each benchmark seed uses a fresh adapter instance, external states cannot persist across seeds. Models must be pre-cached to disk effectively. Shipped prioritizing metric scores over high-throughput concurrency.

---

## Path B — The Production Demo

**Goal:** Deliver the high-throughput, integrated version of the system used for the final architectural demo.

**Implementation Details:**
- **Two-Language Architecture:** Utilizes a **Go Gateway** and an asynchronous **Python Engine**.
- **Go Gateway** (`gateway/`): Effectively handles high-throughput raw events, removing Python's GIL bottlenecks for I/O. It aggressively processes the stream and passes batches over a Unix Domain Socket (UDS) to Python.
- **FastAPI Engine** (`engine/main.py`): The active async receptor running via `uvicorn.run(app, uds="/tmp/pce.sock")`. Receives payload batches efficiently from Go.
- **Thread-safe Execution:** 
  - Operations like embeddings, MMD array math, and graph traversals are inherently synchronous and would starve the `asyncio` event loop.
  - All mathematical/ML/Graph CPU-bound operations are explicitly wrapped in standard `run_in_executor` threads.
- **Production Storage Components:**
  - `EntityRegistry`: Uses standard SQLite (3.45) for identity and alias resolution, queried securely.
  - `EventStore`: Leverages DuckDB (accessed via concurrent executor workers) for log persistence and complex analytical querying.
  - `OperationalGraph`: Implemented using NetworkX DiGraph for robust dependency tracking.
  - `QdrantBehavioralIndex`: In replacing simple Numpy matrices, this uses `QdrantClient(":memory:")`. The Rust backend is synchronous, so it works directly behind thread executors, combining dense vector searches with required payload metadata filtering.