# Agurum Benchmark Report (`bench-full`)

This report provides a detailed breakdown of the execution flow, algorithms, and metric values computed during the `make bench-full` run.

## 1. Execution Flow & Functions Called

When you run `make bench-full`, it executes the command:
`cd bench-p02-context && python run.py --adapter adapters.agurum:Engine --seeds 42 101 202 303 404 --out report.json`

The execution flow involves the following key components and function calls:

1.  **Harness Execution (`bench-p02-context/harness.py`)**:
    *   `run()` is invoked, iterating over the provided seeds (`42, 101, 202, 303, 404`).
    *   For each seed, it calls `_run_one_seed()`, which initializes the Agurum `Engine` adapter.
    *   `generate()` generates synthetic train and evaluation events.

2.  **Ingestion Phase (`adapters.agurum.Engine.ingest()`)**:
    *   The engine consumes `train_events` and `eval_events`.
    *   **Functions called**:   
        *   `self.tracker.process_event()`: Tracks service renames for alias resolution.
        *   `self.store.append()`: Stores raw events in memory.
        *   `self.synthesizer.synthesize_all()`: Converts resolved incidents and their remediations into dense episode vectors.

3.  **Context Reconstruction Phase (`adapters.agurum.Engine.reconstruct_context()`)**:
    *   For each evaluation incident signal, the engine attempts to reconstruct its context.
    *   **Functions called**:
        *   `self.tracker.resolve()`: Finds the canonical ID of the trigger service.
        *   `self._get_window_events()`: Fetches context events from the last 300 seconds.
        *   `self.embedder.encode_single()` & `self.embedder.encode_batch()`: Encodes text context into 384-dimensional dense vectors.
        *   `self.index.recall()`: Performs Approximate Nearest Neighbor (ANN) search over historical episodes.
        *   `self.reranker.rerank()`: Uses distribution similarity (MMD) to rerank the top candidates.

4.  **Evaluation & Scoring Phase (`bench-p02-context/metrics.py`)**:
    *   `score_match()` and `score_remediation()` evaluate the engine's predictions against the ground truth.
    *   `aggregate()` and `compute_score()` combine these to generate the final metric report.

---

## 2. Algorithms Used

The `Agurum Engine` leverages several specialized algorithms to achieve high recall and precision:

### A. Approximate Nearest Neighbor (ANN) Search
*   **Component**: `NumpyBehavioralIndex`
*   **Purpose**: Performs extremely fast and lightweight vector search over the episode embeddings.
*   **Metric Used**: Cosine similarity / Inner product between the query vector and historical episode vectors.

### B. Distribution Shift Detection & Reranking
*   **Component**: `MMDDriftDetector` and `MMDReRanker`
*   **Purpose**: Cosine similarity on single vectors can lose context. The reranker compares the *distribution* of event embeddings in the current window against the *distribution* of event embeddings in historical episodes to re-order the top results.

**How MMD (Maximum Mean Discrepancy) is computed:**
The algorithm calculates an **Unbiased MMD² Estimator** using a multi-scale RBF (Gaussian) kernel.

**Mathematical formulation:**
$$ MMD^2(P,Q) = E[k(x,x')] - 2 \cdot E[k(x,y)] + E[k(y,y')] $$
Where the kernel $k(x,y)$ is defined as:
$$ k(x,y) = \sum_i \exp\left(-\frac{||x-y||^2}{2\sigma_i^2}\right) $$
*   The scales used are tuned to 384-dimensional vector distances: $\sigma \in \{10.0, 50.0, 100.0, 200.0\}$.
*   *Unbiased constraint:* The algorithm explicitly zeros out the diagonal elements of the distance matrices ($K_{xx}$, $K_{yy}$) before summing to prevent self-bias.

### C. Alias Tracking & Entity Resolution
*   **Component**: `AliasTracker`
*   **Purpose**: Ensures that even if a service was renamed (e.g., `checkout-v1` to `checkout-v2`), the embedding and recall system can still accurately match historical incidents.

---

## 3. Metric Values Generated

The `make bench-full` run executed over **5 seeds**, generating **50 total signals**. Here are the aggregated values and the overall score from `report.json`:

### Aggregated Performance Metrics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **`recall@5`** | **0.74** | Fraction of incidents where the correct historical incident family was present in the top 5 suggestions. |
| **`precision@5_mean`** | **0.176** | The average precision at K=5 across all runs. |
| **`remediation_acc`** | **1.0** (100%) | Accuracy of the suggested remediation action matching the correct historical action. |
| **`latency_p95_ms`** | **42.78 ms** | The 95th percentile latency to reconstruct context (well within the fast budget of 2000 ms). |
| **`latency_mean_ms`** | **43.08 ms** | The average latency to reconstruct context. |
| **`ingest_ms` (avg)** | **~486 ms** | The time taken to process and synthesize the training and evaluation events per seed. |

### Weighted Score

The benchmark harness calculates an overall weighted score based on predefined metric weights. The maximum possible automated score is **0.80** (excluding manual grading axes).

*   **Final Weighted Score:** **0.5984 / 0.8000**
*   **Budget Alignment:** The latency perfectly matched the required fast operational constraints, scoring maximum points in its weighted tier.

---
*Note: This report is automatically generated based on the contents of `/bench-p02-context/report.json` and the source files within the `adapters.agurum` implementation.*
