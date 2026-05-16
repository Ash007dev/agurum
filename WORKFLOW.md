# Agurum — Implemented Product Workflow

Based on the actual codebase, the system operates in two distinct modes. They share the same underlying machine learning logic (like Embeddings and Maximum Mean Discrepancy checks) but are wired completely differently to satisfy different constraints.

---

## Part A: The Benchmark Path (Synchronous, Low-Latency)
**Goal:** To run within the isolated evaluation harness (`run.py` / `self_check.py`) and maximize scoring metrics (Recall, Latency).

### 1. How It Starts
You trigger this via `make bench-adapter` or `make bench-full`. The evaluation harness spins up a completely fresh instance of `bench-p02-context/adapters/agurum.py:Engine` for every single "seed" (scenario).

### 2. Ingestion Flow (Synchronous)
The harness feeds a stream of JSONL events via `Engine.ingest()`.
1. **Identity Resolution:** The `AliasTracker` strictly manages service identities natively in memory using Python dicts. If it sees a `topology` rename event, it maps the old name (`from_`) to the new name to ensure identity continuity.
2. **Storage:** Events are synchronously appended to an `InMemoryStore` (simple lists) for O(1) appending.
3. **Episode Synthesis:** If the event is a `remediation`, the `EpisodeSynthesizer` builds an "episode" (a chunk of historical context).
4. **Vector Indexing:** The episode is embedded using `sentence-transformers` and pushed into the `NumpyBehavioralIndex` using in-memory `ndarray` cosine similarity operations. *No external databases or async calls are used here to keep latency extremely low.*

### 3. Context Reconstruction Flow
When an `incident_signal` arrives, the harness calls `Engine.reconstruct_context(mode="fast")`.
1. **Signal Windowing:** Fetches all events within a 300s window around the incident using the `InMemoryStore`.
2. **Role Abstraction:** Instead of relying on exact text matches (which break if a service was renamed), events are normalized into "Role Embedding Strings" (e.g. tracking "databases" rather than "db-svc-v2").
3. **Candidate Retrieval:** The sequence of events is embedded, and the `NumpyBehavioralIndex` retrieves the Top-20 similar historical episodes.
4. **MMD Reranking:** The `MMDReRanker` applies Maximum Mean Discrepancy mathematically (via NumPy) to re-rank the candidates to the absolute most exact behavioral Top-5 matches.
5. **Context Generation:** Historic successful remediations are extracted from the matching episodes, and the final typed `Context` is returned synchronously.

---

## Part B: The Production Path (Asynchronous, High-Throughput)
**Goal:** To operate in a real-world microservice environment handling massive telemetry throughput without locking up the Python Global Interpreter Lock (GIL).

### 1. How It Starts
You run this via `make run` (or `make run-engine` and `make run-gateway` in separate terminals). It runs two separate processes simultaneously.

### 2. The Ingest Gateway (Go)
The Go application (`gateway/main.go`) acts as a massive shock-absorber for telemetry spikes.
1. **Listening:** It listens for raw JSONL strings on standard input (`stdin`) OR via an HTTP endpoint (`POST /inject` on port 8080).
2. **Buffering:** Validated events are pushed into a concurrent, mutex-locked `RingBuffer` with a capacity of 10,000 events.
3. **Flushing:** A dedicated goroutine (`Flusher`) wakes up every 100 milliseconds, grabs batches of up to 100 events, and fires them rapidly to the Python backend over a **Unix Domain Socket (`/tmp/pce.sock`)**.

### 3. The Python Engine (FastAPI + Uvicorn)
The Python application listens on the `/tmp/pce.sock` Unix Domain Socket using Uvicorn.
1. **Receiving Data (`POST /batch`):** FastAPI receives the batch of JSON events sent by the Go gateway.
2. **Event Loop Protection & Dispatch:** Because operations like Python `NetworkX` graphs, PyTorch models, and Database I/O freeze the async event loop, the engine offloads *everything* heavy to a ThreadPoolExecutor (`run_in_executor`).
3. **Heavy Storage (The 3 Databases):**
   * **EntityRegistry (SQLite):** Writes name aliases and topologies to disk.
   * **EventStore (DuckDB):** Inserts operational telemetry for heavy analytical querying later.
   * **QdrantBehavioralIndex:** Replaces NumPy for real production scale vector searching.
4. **Graph Updates:** The `OperationalGraph` (managed via `NetworkX`) updates relationships between infrastructure components in the background.

### 4. Context Reconstruction (`POST /reconstruct`)
An external system requests context for a live incident via the API.
1. **Neighborhood BFS:** The engine queries the `OperationalGraph` to calculate the blast radius of the affected service.
2. **Causal Chains:** Runs rules via `causal_extractor.py` to identify root causes (e.g., matching a deploy event right before a latency spike).
3. **Retrieval & Reranking:** Does an async Qdrant search inside the `ThreadPoolExecutor`, then reranks using the same NumPy MMD logic from Part A.
4. **Deep Synthesis (Optional):** If `mode="deep"` is requested, an LLM (`LLMSynthesizer`) generates a human-readable investigation narrative over the technical data before returning the Context payload.