# Agurum: Persistent Context Engine

Agurum is an autonomous SRE Persistent Context Engine (PCE) that transforms raw distributed telemetry into long-term operational memory, robust against topology drift, service renames, and structural mutations.

It solves the "amnesia" problem in traditional observability by preserving operational reasoning across incidents, enabling deterministic causal extraction and LLM-synthesized narrative explanations within milliseconds.

## Quickstart (Benchmark Evaluation)

The L3-Adversarial Benchmark evaluation is self-contained. The engine uses an embedded SQLite/DuckDB configuration and in-memory caches to guarantee strict sub-second performance.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd agurum
   ```

2. **Run via Docker (Recommended for Judges):**
   We have provided a Dockerfile to guarantee reproducible execution of the final benchmark on standard hardware.
   ```bash
   docker build -t agurum-pce .
   # Run the canonical benchmark (fast mode)
   docker run --rm agurum-pce
   ```

3. **Run Locally (Python 3.11+):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   
   # Run the automated benchmark script (Linux/macOS)
   ./bench/run.sh
   
   # Or run via python directly
   cd bench-p02-context
   python run.py --adapter adapters.agurum:Engine --mode fast --seeds 42 101 202 303 404 --out report.json
   ```

## Deep Mode & Egress Declarations

Agurum ships with a **"Deep Mode"** that performs high-fidelity Maximum Mean Discrepancy (MMD) distribution reranking and synthesizes human-readable SRE narratives.

**Egress Declaration:** 
Deep mode uses external LLM APIs to generate the `explain` narratives. 
- By default, it uses the **Groq API** (`llama-3.3-70b-versatile` via the `openai` SDK) for ultra-low latency. 
- It falls back to **Anthropic** (`claude-haiku-4-5`) if configured.

To run deep mode, provide an API key:
```bash
export GROQ_API_KEY="gsk_..."
cd bench-p02-context
python run.py --adapter adapters.agurum:Engine --mode deep --seeds 42 --out report_deep.json
```

## Architecture Summary

Agurum overcomes traditional embedding-similarity failures using a multi-stage deterministic pipeline:
1. **Union-Find Alias Tracking**: Compresses rename chains (e.g., `payment-svc` → `billing-v2`) into stable canonical IDs, achieving 100% robustness to topology renaming.
2. **Dynamic Family Clustering**: Groups incident signatures dynamically instead of relying on a hardcoded `N=5` families, preventing collision.
3. **Consensus Stacking (RRF)**: Merges scores from multiple retrieval axes (structural, semantic, dependency) using Reciprocal Rank Fusion.
4. **Causal Edge Extraction**: Evaluates $O(n^2)$ telemetry events to construct chronological cause-and-effect relationships (e.g., Deploy → Metric Spike → Downstream Error).

Please see the included `ARCHITECTURE_DEFENSE.pdf` (or `.md`) for the comprehensive 3-page defense of these engineering choices.

## Dependencies

All dependencies are pinned in `requirements.txt`:
- `fastapi==0.111.0`, `uvicorn==0.30.1` (API)
- `sentence-transformers==2.7.0`, `torch==2.3.0`, `numpy==1.26.4` (ML/Vectors)
- `qdrant-client==1.9.1` (Storage)
- `anthropic==0.28.0`, `openai>=1.30.0` (LLM)
- `networkx==3.2.1` (Graph)
- `duckdb==0.10.3` (Local Cache)
