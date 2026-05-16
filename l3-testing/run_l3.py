#!/usr/bin/env python3
"""
L3 Mock Benchmark Runner
Usage:
    python run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9999 --out l3_report.json

Simulates held-out L3 adversarial conditions:
  - 30 services (vs 12 in L2)
  - 10 incident families (vs 5 in L2)
  - 15 topology mutations (vs 8 in L2)
  - Cascading rename chains
  - Correlated multi-service outages
  - Dependency graph shifts mid-incident
  - 40 train + 20 eval incidents (vs 24+10 in L2)
  - ~400 background events/service/day (vs ~200 in L2)
"""

import sys
import os
import json
import time
import argparse
import importlib
from typing import List

sys.path.insert(0, os.path.dirname(__file__))
from generator import L3Generator, FAMILY_SIGNATURES
from scorer import L3Scorer


def load_adapter(adapter_path: str):
    module_path, class_name = adapter_path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def run_seed(
    AdapterClass,
    seed: int,
    n_services: int,
    days: int,
    n_families: int,
    n_train: int,
    n_eval: int,
    mode: str,
    verbose: bool,
) -> dict:
    print(f"\n{'='*60}")
    print(f"SEED {seed} | services={n_services} | families={n_families} | "
          f"days={days} | train={n_train} | eval={n_eval}")
    print(f"{'='*60}")

    # Generate dataset
    gen = L3Generator(
        seed=seed,
        n_services=n_services,
        days=days,
        n_families=n_families,
        n_train_incidents=n_train,
        n_eval_incidents=n_eval,
    )
    events, eval_signals, ground_truth = gen.generate()
    print(f"Generated {len(events)} events, {len(eval_signals)} eval signals")
    print(f"Topology mutations: {len(gen.rename_log)}")

    # Fresh adapter per seed (no cross-seed leakage)
    adapter = AdapterClass()

    # Ingest phase — measure throughput
    t0 = time.perf_counter()
    adapter.ingest(events)
    ingest_time = time.perf_counter() - t0
    eps = len(events) / ingest_time if ingest_time > 0 else float("inf")
    print(f"Ingest: {len(events)} events in {ingest_time:.2f}s = {eps:.0f} events/sec")

    # Eval phase — query each signal
    scorer = L3Scorer(ground_truth)
    results = []

    for signal in eval_signals:
        inc_id = signal["incident_id"]
        t_start = time.perf_counter()
        try:
            context = adapter.reconstruct_context(signal, mode=mode)
        except Exception as e:
            print(f"  ERROR on {inc_id}: {e}")
            context = {
                "related_events": [],
                "causal_chain": [],
                "similar_past_incidents": [],
                "suggested_remediations": [],
                "confidence": 0.0,
                "explain": "",
            }
        latency_ms = (time.perf_counter() - t_start) * 1000

        result = scorer.score_one(inc_id, context, latency_ms)
        results.append(result)

        if verbose:
            fam = ground_truth.get(inc_id, {}).get("family_id", "?")
            gt_count = len(result["true_similar"])
            hit = "✅" if result["recall@5"] > 0 else "❌"
            print(f"  {hit} {inc_id} [{fam}] "
                  f"recall={result['recall@5']:.2f} "
                  f"prec={result['precision@5']:.2f} "
                  f"lat={latency_ms:.1f}ms "
                  f"gt={gt_count}")

    # Close adapter
    try:
        adapter.close()
    except Exception:
        pass

    # Aggregate scores
    agg = scorer.aggregate(results)
    agg["seed"] = seed
    agg["ingest_eps"] = round(eps, 1)

    print(f"\n--- Seed {seed} Results ---")
    print(f"  recall@5:           {agg['recall@5']}")
    print(f"  precision@5_mean:   {agg['precision@5_mean']}")
    print(f"  remediation_acc:    {agg['remediation_acc']}")
    print(f"  latency_p95_ms:     {agg['latency_p95_ms']}")
    print(f"  weighted_automated: {agg['weighted_automated']}")
    print(f"  adaptability_delta: {agg['adaptability_delta']}")
    print(f"\n  Per-family recall:")
    for fam, rec in sorted(agg["per_family_recall"].items()):
        prec = agg["per_family_precision"].get(fam, 0)
        sig_name = FAMILY_SIGNATURES.get(fam, {}).get("name", "?")
        bar = "█" * int(rec * 20)
        print(f"    {fam} [{sig_name:35s}] recall={rec:.2f} prec={prec:.2f} {bar}")

    return agg


def main():
    parser = argparse.ArgumentParser(description="L3 Mock Benchmark")
    parser.add_argument("--adapter", required=True,
                        help="module.path:ClassName")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 9999],
                        help="Random seeds (each runs a fresh adapter)")
    parser.add_argument("--n-services", type=int, default=30,
                        help="Number of services (L2=12, L3=30+)")
    parser.add_argument("--days", type=int, default=14,
                        help="Simulated days (L2=7, L3=14)")
    parser.add_argument("--n-families", type=int, default=10,
                        help="Incident families (L2=5, L3=10)")
    parser.add_argument("--n-train", type=int, default=40,
                        help="Training incidents (L2=24, L3=40)")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Eval incidents (L2=10, L3=20)")
    parser.add_argument("--mode", default="fast",
                        choices=["fast", "deep"])
    parser.add_argument("--out", default="l3_report.json",
                        help="Output JSON report path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-query results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed, 15 services, 5 days, 5 families")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [args.seeds[0]]
        args.n_services = 15
        args.days = 7
        args.n_families = 7
        args.n_train = 20
        args.n_eval = 10
        print("QUICK MODE: 1 seed, 15 services, 7 days, 7 families")

    # Load adapter
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench-p02-context"))
    AdapterClass = load_adapter(args.adapter)

    print(f"\nANVIL · L3 MOCK BENCHMARK")
    print(f"{'='*60}")
    print(f"Adapter:    {args.adapter}")
    print(f"Seeds:      {args.seeds}")
    print(f"Services:   {args.n_services}")
    print(f"Families:   {args.n_families}")
    print(f"Days:       {args.days}")
    print(f"Train/Eval: {args.n_train}/{args.n_eval}")
    print(f"Mode:       {args.mode}")
    print(f"\n⚠  L2 blanket strategy (hardcoded 5 families) will FAIL here.")
    print(f"   Correct family count for this run: {args.n_families}")

    all_seed_results = []
    for seed in args.seeds:
        result = run_seed(
            AdapterClass=AdapterClass,
            seed=seed,
            n_services=args.n_services,
            days=args.days,
            n_families=args.n_families,
            n_train=args.n_train,
            n_eval=args.n_eval,
            mode=args.mode,
            verbose=args.verbose,
        )
        all_seed_results.append(result)

    # Cross-seed summary
    print(f"\n{'='*60}")
    print(f"CROSS-SEED SUMMARY ({len(args.seeds)} seeds)")
    print(f"{'='*60}")

    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0

    metrics = ["recall@5", "precision@5_mean", "remediation_acc",
               "weighted_automated", "adaptability_delta"]
    for m in metrics:
        vals = [r.get(m, 0) for r in all_seed_results]
        worst = min(vals)
        label = "⚠ " if worst < 0.5 else "✅ "
        print(f"  {label}{m:30s} mean={mean(vals):.4f}  worst={worst:.4f}")

    # L3 verdict
    worst_weighted = min(r.get("weighted_automated", 0) for r in all_seed_results)
    print(f"\n{'='*60}")
    if worst_weighted >= 0.68:
        print(f"✅ L3 VERDICT: Engine holds at {worst_weighted:.3f} — submission safe")
    elif worst_weighted >= 0.55:
        print(f"⚠  L3 VERDICT: Score degrades to {worst_weighted:.3f} — fix before submitting")
    else:
        print(f"❌ L3 VERDICT: Score collapses to {worst_weighted:.3f} — architecture needs rework")

    # Blanket detection
    all_precisions = [r.get("precision@5_mean", 0) for r in all_seed_results]
    all_recalls = [r.get("recall@5", 0) for r in all_seed_results]
    if mean(all_recalls) >= 0.95 and mean(all_precisions) <= 0.25:
        print(f"\n⚠  BLANKET DETECTED: recall≈1.0 + precision≈0.2 = hardcoded family count")
        print(f"   Your engine is exploiting L2's known 5-family structure.")
        print(f"   On L3 with {args.n_families} families, recall will drop to "
              f"~{5/args.n_families:.2f}.")

    # Save report
    report = {
        "config": {
            "adapter": args.adapter,
            "seeds": args.seeds,
            "n_services": args.n_services,
            "n_families": args.n_families,
            "days": args.days,
            "mode": args.mode,
        },
        "seed_results": all_seed_results,
        "summary": {m: mean([r.get(m, 0) for r in all_seed_results]) for m in metrics},
        "worst_weighted": worst_weighted,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.out}")


    if args.mode == "deep":
        print("\n" + "="*60)
        print("GENERATING SAMPLED NARRATIVE FOR MANUAL PANEL GRADING...")
        print("="*60)
        os.environ["AGURUM_SAMPLE_NARRATIVE"] = "1"
        try:
            gen = L3Generator(seed=999, n_services=15, days=7, n_families=5, n_train_incidents=15, n_eval_incidents=1)
            events, eval_signals, gt = gen.generate()
            adapter = AdapterClass()
            adapter.ingest(events)
            ctx = adapter.reconstruct_context(eval_signals[0], mode="deep")
            
            print("\n--- CAUSAL CHAIN ---")
            for edge in ctx.get("causal_chain", []):
                print(f"  • {edge}")
                
            print("\n--- SRE LLM NARRATIVE ---")
            print(ctx.get("explain", "No explanation generated."))
            adapter.close()
            print("\n" + "="*60)
        except Exception as e:
            print(f"Failed to generate sampled narrative: {e}")

if __name__ == "__main__":
    main()
