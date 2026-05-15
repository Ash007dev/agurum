#!/usr/bin/env python3
"""
Twin Family Diagnostic
Run this to understand WHY your engine confuses incident families.
It generates a dataset, runs your engine, then prints a feature comparison
for every family pair your engine confused.

Usage:
    python diagnose_twins.py --adapter adapters.agurum:Engine --seed 42
"""

import sys
import os
import json
import time
import argparse
import importlib
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from generator import L3Generator, FAMILY_SIGNATURES

def load_adapter(adapter_path):
    module_path, class_name = adapter_path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-families", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    AdapterClass = load_adapter(args.adapter)

    gen = L3Generator(seed=args.seed, n_services=20, days=10,
                      n_families=args.n_families, n_train_incidents=30, n_eval_incidents=15)
    events, eval_signals, ground_truth = gen.generate()

    adapter = AdapterClass()
    adapter.ingest(events)

    print(f"\nFAMILY TWIN DIAGNOSTIC — seed={args.seed}")
    print("="*70)

    confusion = defaultdict(lambda: defaultdict(int))
    family_features = {}

    for signal in eval_signals:
        inc_id = signal["incident_id"]
        meta = ground_truth.get(inc_id, {})
        true_family = meta.get("family_id", "?")
        true_similar = set(meta.get("ground_truth_family_ids", []))

        ctx = adapter.reconstruct_context(signal, mode="fast")
        returned = [m.get("incident_id") or m.get("past_incident_id")
                    for m in (ctx.get("similar_past_incidents") or [])]

        # What families did we return?
        returned_families = []
        for rid in returned:
            rfam = ground_truth.get(rid, {}).get("family_id", "?")
            returned_families.append(rfam)

        # Count confusions
        for rfam in returned_families:
            if rfam != true_family:
                confusion[true_family][rfam] += 1

        # Store family signature features for comparison
        if true_family not in family_features:
            sig = FAMILY_SIGNATURES.get(true_family, {})
            family_features[true_family] = {
                "metrics": sig.get("metrics_spiked", []),
                "depth": sig.get("causal_depth", 0),
                "deploy_delta": sig.get("deploy_to_spike_s"),
                "remediation": sig.get("remediation"),
                "multi_service": sig.get("multi_service", False),
            }

    print("\nCONFUSION MATRIX (true_family → mistaken_for → count)")
    print("-"*70)
    for true_fam, mistakes in sorted(confusion.items()):
        if mistakes:
            print(f"\n  True: {true_fam} [{FAMILY_SIGNATURES.get(true_fam,{}).get('name','?')}]")
            for wrong_fam, count in sorted(mistakes.items(), key=lambda x: -x[1]):
                print(f"    Confused with: {wrong_fam} "
                      f"[{FAMILY_SIGNATURES.get(wrong_fam,{}).get('name','?')}] "
                      f"× {count}")

                # Feature diff
                tf = family_features.get(true_fam, {})
                wf = FAMILY_SIGNATURES.get(wrong_fam, {})
                diffs = []
                if tf.get("metrics") != wf.get("metrics_spiked"):
                    diffs.append(f"metrics: {tf.get('metrics')} vs {wf.get('metrics_spiked')}")
                if tf.get("depth") != wf.get("causal_depth"):
                    diffs.append(f"depth: {tf.get('depth')} vs {wf.get('causal_depth')}")
                if tf.get("remediation") != wf.get("remediation"):
                    diffs.append(f"remediation: {tf.get('remediation')} vs {wf.get('remediation')}")
                if tf.get("deploy_delta") != wf.get("deploy_to_spike_s"):
                    diffs.append(f"deploy_delta: {tf.get('deploy_delta')} vs {wf.get('deploy_to_spike_s')}")

                if diffs:
                    print(f"    DISTINGUISHING FEATURES (use these to discriminate):")
                    for d in diffs:
                        print(f"      → {d}")
                else:
                    print(f"    ⚠ NO OBVIOUS FEATURE DIFF — families may be structurally identical")

    print("\n" + "="*70)
    print("FAMILY FEATURE TABLE")
    print(f"{'Family':<6} {'Metrics':<35} {'Depth':<6} {'DeployDelta':<20} {'Remediation':<15}")
    print("-"*70)
    for fam_id in sorted(family_features.keys()):
        f = family_features[fam_id]
        sig = FAMILY_SIGNATURES.get(fam_id, {})
        print(f"{fam_id:<6} {str(f['metrics']):<35} {f['depth']:<6} "
              f"{str(f['deploy_delta']):<20} {f['remediation']:<15}")

    try:
        adapter.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
