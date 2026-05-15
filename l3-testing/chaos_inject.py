#!/usr/bin/env python3
"""
Chaos Injector — simulates the hidden chaos test from the judging mechanics.
"One hidden scenario, revealed at runtime: a topology shift injected mid-evaluation."

Usage:
    python chaos_inject.py --adapter adapters.agurum:Engine --seed 42

What this does:
    1. Ingest full training dataset
    2. Start eval queries
    3. MID-EVALUATION: inject a topology rename event
    4. Continue remaining eval queries
    5. Measure recall DROP across the chaos boundary
"""

import sys
import os
import json
import time
import argparse
import importlib

sys.path.insert(0, os.path.dirname(__file__))
from generator import L3Generator, FAMILY_SIGNATURES
from scorer import L3Scorer

def load_adapter(adapter_path):
    module_path, class_name = adapter_path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    AdapterClass = load_adapter(args.adapter)

    gen = L3Generator(seed=args.seed, n_services=20, days=10,
                      n_families=8, n_train_incidents=30, n_eval_incidents=20)
    events, eval_signals, ground_truth = gen.generate()

    adapter = AdapterClass()
    adapter.ingest(events)

    scorer = L3Scorer(ground_truth)
    results_pre = []
    results_post = []
    chaos_point = len(eval_signals) // 2

    print(f"\nCHAOS INJECTION TEST — seed={args.seed}")
    print("="*60)
    print(f"Eval signals: {len(eval_signals)}")
    print(f"Chaos injected after query #{chaos_point}")

    for i, signal in enumerate(eval_signals):
        if i == chaos_point:
            # CHAOS: inject rename of a random service mid-evaluation
            import random
            rng = random.Random(args.seed + 999)
            svc = rng.choice(gen.services)
            new_name = svc.split("-")[0] + "-chaossvc-99"
            chaos_event = {
                "ts": signal["ts"],
                "kind": "topology",
                "change": "rename",
                "from": svc,
                "to": new_name,
            }
            print(f"\n💥 CHAOS INJECTED: {svc} → {new_name}")
            adapter.ingest([chaos_event])
            print()

        t0 = time.perf_counter()
        ctx = adapter.reconstruct_context(signal, mode="fast")
        lat = (time.perf_counter() - t0) * 1000

        result = scorer.score_one(signal["incident_id"], ctx, lat)
        phase = "pre " if i < chaos_point else "post"
        marker = "✅" if result["recall@5"] > 0 else "❌"
        print(f"  [{phase}] {marker} {signal['incident_id'][:30]:<30} "
              f"recall={result['recall@5']:.2f} lat={lat:.1f}ms")

        if i < chaos_point:
            results_pre.append(result)
        else:
            results_post.append(result)

    import statistics
    pre_recall = statistics.mean([r["recall@5"] for r in results_pre]) if results_pre else 0
    post_recall = statistics.mean([r["recall@5"] for r in results_post]) if results_post else 0
    delta = pre_recall - post_recall

    print(f"\n{'='*60}")
    print(f"PRE-CHAOS  recall@5: {pre_recall:.4f}")
    print(f"POST-CHAOS recall@5: {post_recall:.4f}")
    print(f"DELTA:               {delta:.4f}  (lower = more resilient)")

    if delta < 0.05:
        print("✅ Engine is chaos-resilient")
    elif delta < 0.15:
        print("⚠  Engine partially degrades under mid-eval chaos")
    else:
        print("❌ Engine collapses under mid-eval topology shift")

    try:
        adapter.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
