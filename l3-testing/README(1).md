# L3 Mock Benchmark

Drop-in adversarial test harness that simulates held-out L3 conditions
before judges run them on your engine.

## Setup

```bash
# Copy the bench/ folder into your project root
cp -r bench/ your-project/l3-bench/

# Install no extra deps — pure Python stdlib only
# (your adapter's own deps still apply)
```

## Usage

```bash
# Quick smoke test — 7 families, 1 seed
make l3-quick ADAPTER=adapters.agurum:Engine

# Full L3 simulation — 10 families, 3 seeds
make l3-full ADAPTER=adapters.agurum:Engine

# Find out WHY families are confused
make diagnose ADAPTER=adapters.agurum:Engine

# Simulate the hidden chaos test
make chaos ADAPTER=adapters.agurum:Engine

# Expose hardcoded-5-family blanket assumption
make blanket-test ADAPTER=adapters.agurum:Engine
```

## What each tool reveals

| Tool | What it exposes |
|---|---|
| `l3-quick` | Does your engine hold with 7 families? Blanket of 5 scores 0.44 recall here |
| `l3-full` | Full L3 simulation. Score collapse here = submission risk |
| `l3-stress` | 50 services, 10 families, 21 days — finds latency and memory issues |
| `diagnose` | Prints confusion matrix + feature diffs for every family your engine misconfuses |
| `chaos` | Injects a topology shift MID-EVALUATION — measures recall delta |
| `blanket-test` | Sets families=7 — if your recall drops from 1.0 to ~0.7, you have a hardcoded blanket |

## Reading the blanket detection output

```
⚠  BLANKET DETECTED: recall≈1.0 + precision≈0.2 = hardcoded family count
   Your engine is exploiting L2's known 5-family structure.
   On L3 with 10 families, recall will drop to ~0.50.
```

If you see this, your engine is not generalizing — it is memorizing
the benchmark structure. Fix: infer family count dynamically or
use confidence-gated selection instead of a fixed blanket.

## L3 verdict thresholds

| Weighted score | Verdict |
|---|---|
| ≥ 0.68 | Engine holds — submission safe |
| 0.55–0.67 | Degrades — fix before submitting |
| < 0.55 | Collapses — architecture needs rework |
