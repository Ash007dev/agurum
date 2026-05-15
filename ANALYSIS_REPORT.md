# agurum Synchronization and Plan Feasibility Analysis

This report analyzes the synchronous bottlenecks and architectural flaws in the `AGURUM_PLAN.md` specification, identifying operations that will block the asyncio event loop and highlighting portions of the plan that are technically unfeasible as described.

## 1. Synchronous Operations in the Python Engine

The Python processing engine is built on `asyncio` (via FastAPI/aiohttp), which uses a single-threaded cooperative multitasking event loop. The following operations introduced by the plan are inherently synchronous and will block the event loop if executed directly within background tasks or route handlers:

1. **SentenceTransformers (`model.encode`)**
   - **Nature:** Heavy CPU-bound computation.
   - **Time:** ~14-20ms per batch.
   - **Impact:** While the model calculates the dense vectors, the entire Python process is stalled. The server cannot accept new HTTP UDS requests from the Go Gateway.

2. **NumPy BLAS MMD Computation (`compute_mmd_squared`)**
   - **Nature:** CPU-bound floating-point matrix operations.
   - **Time:** ~25ms for 20 candidates.
   - **Impact:** Synchronously blocks the event loop for the duration of the reranking phase. 

3. **NetworkX Traversals and Graph Mutations (`OperationalGraph`)**
   - **Nature:** CPU-bound memory operations.
   - **Time:** 1-5ms per BFS.
   - **Impact:** NetworkX is pure Python and strictly synchronous. Graph traversal or debounced recomputation (`_recompute_roles`) executing on the main thread will stall I/O.

4. **SQLite Operations (`EntityRegistry`)**
   - **Nature:** Synchronous disk I/O.
   - **Time:** <0.2ms per call.
   - **Impact:** The standard `sqlite3` library runs synchronously. Resolving every event concurrently over 1,000 req/sec will generate frequent, brief event loop pauses.

5. **Qdrant Local Memory Mode (`QdrantClient(":memory:")`)**
   - **Nature:** Rust FFI / CPU-bound index traversal.
   - **Time:** ~30ms per search.
   - **Impact:** Local execution via the Qdrant Rust wrapper is synchronous and does not yield control back to the asyncio loop.

6. **DuckDB Inserts/Queries**
   - **Nature:** Disk I/O and CPU-bound analytical queries.
   - **Impact:** The plan explicitly handles this via `run_in_executor`, which solves the loop blocking, but DuckDB operations themselves remain synchronous within their assigned thread workers.

---

## 2. What Won't Work in the Current Plan (Architectural Flaws)

Based on the analysis of framework constraints and native library behaviors, the following specific parts of the plan will fail or cause critical degradation:

### A. Qdrant Async + `:memory:` Mode Contradiction
- **The Plan States:** "Qdrant search -> qdrant-client async" AND "client = QdrantClient(':memory:')"
- **Why it won't work:** The official Python `qdrant-client`'s asynchronous client (`AsyncQdrantClient`) **only supports gRPC REST connections**. It does *not* support local `:memory:` mode. Using `:memory:` forces the use of the synchronous `QdrantClient`. Attempting to run synchronous Qdrant searches in an async route without an executor will severely block the event loop.

### B. Event Loop Starvation by ML & NetworkX
- **The Plan States:** `model.encode(list, batch_size=32)` and matrix math are executed. The plan only specifies `run_in_executor` for DuckDB.
- **Why it won't work:** At a target ingestion of 1,000 events/sec, the Go gateway pushes batches every 100ms. If the Python engine executes a 30ms ML embedding, a 25ms MMD pass, or a graph recomputation synchronously on the main thread, the UDS `/batch` endpoint will hang. The event loop will be starved, leading to HTTP timeouts on the Go side and dropped messages.

### C. Framework Fragmentation (FastAPI vs. aiohttp)
- **The Plan States:** In Section 3: "FastAPI app + startup sequence". In Section 2 (IPC): "Python aiohttp listens on the socket with `web.run_app(app, path='/tmp/pce.sock')`".
- **Why it won't work:** FastAPI operates on an ASGI server (like Uvicorn or Hypercorn). `aiohttp` uses its own internal HTTP server. You cannot natively start a FastAPI app directly using `aiohttp.web.run_app()`. Binding Uvicorn to a Unix Domain Socket requires different syntax (`uvicorn.run(app, uds="/tmp/pce.sock")`).

### D. Thread Locking in Async Context Contexts
- **The Plan States:** `SlidingWindowCache` uses `RLock safe` configurations.
- **Why it won't work:** In an `asyncio` application, standard threading locks (`threading.RLock`) are dangerous. If a coroutine acquires a thread lock and then execution yields via an `await`, another coroutine running on the same thread might attempt to acquire the lock and permanently deadlock the entire event loop. Async applications require `asyncio.Lock` unless the data structure is strictly accessed inside a synchronous `run_in_executor` worker.