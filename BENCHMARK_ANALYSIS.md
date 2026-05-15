# Benchmark Analysis: bench-p02-context

This document analyzes how the benchmarking process is designed and executed in the `bench-p02-context` directory.

## Overview
The benchmark harness is a pure Python, zero-dependency suite that evaluates the `Persistent Context Engine`. Its purpose is to test the quality, latency, and correctness of an adapter engine against deterministic synthetic telemetry.

## Execution
- **Self-check**: Developers can run `self_check.py` for rapid, local iteration to see aggregated metrics and indicative scores.
- **Full Run**: Executed via `run.py` to test against multiple seeds under specific parameters (number of services, days, etc.).

## Evaluation Layers (Anti-gaming mechanics)
The harness is designed to prevent "hardcoding" solutions through three layers of evaluation:
1. **L1 (Worked Example)**: A canonical trace testing basic functionality from the problem statement.
2. **L2 (Property-based Multi-seed)**: Generates random deterministic datasets for arbitrary seeds. It tests the engine against dynamic service landscapes, deploys, and topology mutations (such as component renames). Engines are fed a completely fresh instance per seed to eliminate state leakage.
3. **L3 (Held-out Adversarial)**: Private, complex scenarios (e.g., correlated outages, cascading renames) run by the council at final evaluation.

## Robustness & Metrics
- **State Isolation**: Each seed receives a fresh, cold-started adapter instance. In-memory caches must be reset.
- **Latency Protection**: Enforced against the worst-seed p95, skipping initial warmup queries to prevent cold-starts from poisoning latency results.
- **Core Metric (`recall@5`)**: Measures the system's ability to match incident families. Instead of checking for exact ID matches, it checks if a past incident from the *same family/signature* appears in the top 5 results.

## Adapter Architecture
To be evaluated, engines must implement a subclass of `Adapter` (e.g., inside `adapters/<your_team>.py`) with three core methods:
- `ingest(events)`: Synchronous log ingestion. No event loop.
- `reconstruct_context(signal, mode)`: Synchronous context generation based on incident signals.
- `close()`: Teardown logic.