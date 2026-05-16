"""
bench-p02-context/adapters/agurum.py — Agurum Engine benchmark adapter.

This is the PRIMARY DELIVERABLE for P3.
Implements the sync Adapter interface required by the benchmark harness.

Path A (benchmark):
  - Synchronous, no event loop, no Docker, no network.
  - AliasTracker for rename-robust identity.
  - NumpyBehavioralIndex for fast in-memory episode search.
  - MMDReRanker for distribution-aware top-5 selection.
  - Service-name-free embedding strings for rename robustness.

Usage:
    python self_check.py --adapter adapters.agurum:Engine --quick
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timezone
from typing import Iterable, Literal

# Add the engine package to sys.path so imports work from bench-p02-context/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from adapter import Adapter  # bench-local
from schema import Context, Event, IncidentSignal  # bench-local

from engine.ml.embedder import get_embedder
from engine.ml.numpy_index import NumpyBehavioralIndex
from engine.registry.alias_tracker import AliasTracker  # Fix 1: .uf (UnionFind) now inside
from engine.store.in_memory_store import InMemoryStore
from engine.graph.versioned_dep_graph import VersionedDepGraph  # Fix 4
from engine.ml.dynamic_clustering import DynamicFamilyClustering  # Fix 2

import bisect
import math

def _parse_ts(ts_str: str) -> float:
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0

class Engine(Adapter):
    def __init__(self) -> None:
        self.tracker = AliasTracker()          # Fix 1: .uf (UnionFind) lives inside
        self.embedder = get_embedder()
        self.index = NumpyBehavioralIndex()
        self.store = InMemoryStore()
        self.dep_graph = VersionedDepGraph()   # Fix 4: versioned topology
        self.clustering = DynamicFamilyClustering(sim_threshold=0.85)  # Fix 2
        self._incidents: dict[str, list[dict]] = {}
        self._remediations: dict[str, dict] = {}
        self._synthesized: set[str] = set()
        self.token_counts: dict[str, int] = {}
        self.total_events: int = 0
        # L3 Fix: sorted timestamp indexes for bisect-based O(log N) window lookups
        self._ts_index: list[float] = []       # sorted parsed timestamps, parallel to _events
        self._deploy_ts_index: list[float] = [] # sorted deploy-event timestamps only

    def ingest(self, events: Iterable[Event]) -> None:
        for e in events:
            e = dict(e)
            self.tracker.process_event(e)   # Fix 1: also calls uf.union on renames
            self.store.append(e)

            # L3 Fix: maintain sorted timestamp index for bisect lookups
            e_ts = _parse_ts(e.get("ts", ""))
            bisect.insort(self._ts_index, e_ts)
            if e.get("kind") == "deploy":
                bisect.insort(self._deploy_ts_index, e_ts)

            # Fix 4: feed topology events into versioned dep graph
            if e.get("kind") == "topology":
                self.dep_graph.on_topology(e)

            inc_id = e.get("incident_id")
            if inc_id:
                self._incidents.setdefault(inc_id, []).append(e)

            if e.get("kind") == "remediation" and e.get("outcome") == "resolved":
                if inc_id:
                    self._remediations[inc_id] = e

            # Token frequency tracking for inverse-frequency Jaccard
            svc = e.get("service", "")
            if svc:
                # Fix 1: use path-compressed canonical name for token keys
                canonical_svc = self.tracker.uf.canonical(svc)
                cid = self.tracker.resolve(canonical_svc)
                role = self.tracker.get_role(cid)
                kind = e.get("kind", "")
                detail = ""
                is_anomaly = False
                if kind == "log":
                    msg = e.get("msg", "").lower()
                    if "timeout" in msg: detail, is_anomaly = "timeout", True
                    elif "crash" in msg: detail, is_anomaly = "crash", True
                    elif "network" in msg or "refused" in msg: detail, is_anomaly = "network", True
                    elif "memory" in msg or "oom" in msg: detail, is_anomaly = "memory", True
                elif kind == "metric" and e.get("value", 0) > 3000:
                    detail, is_anomaly = "critical", True
                elif kind == "incident_signal":
                    detail, is_anomaly = "alert", True
                if is_anomaly:
                    token = f"{role}_{kind}_{detail}"
                    self.token_counts[token] = self.token_counts.get(token, 0) + 1
                    self.total_events += 1

        self._synthesize_all_episodes()

    def reconstruct_context(
        self,
        signal: IncidentSignal,
        mode: Literal["fast", "deep"] = "fast",
    ) -> Context:
        trigger_svc = signal.get("service", "")
        if not trigger_svc:
            raw = signal.get("trigger", "")
            if ":" in raw:
                trigger_svc = raw.split(":")[1].split("/")[0]
            elif "/" in raw:
                trigger_svc = raw.split("/")[0]

        # Fix 1: resolve through path-compressed union-find canonical name
        canonical_svc = self.tracker.uf.canonical(trigger_svc) if trigger_svc else trigger_svc
        trigger_cid = self.tracker.resolve(canonical_svc)
        signal_ts_str = signal.get("ts", "")

        # Fix 4: anchor graph lookups to signal time — mid-eval renames won't corrupt context
        _graph_at_signal = self.dep_graph.graph_at(signal_ts_str)
        _callers_at_signal = self.dep_graph.upstream_callers_at(canonical_svc, signal_ts_str)

        # L3 Fix 1: deploy-aware window replaces static 700s window
        window_events, has_deploy = self._get_deploy_aware_window(signal_ts_str)

        # Bail early before ANY extraction if no events
        if not window_events:
            return self._empty_context(signal)

        # 1. Extract Query Features from the deploy-aware window
        q_prof = self._extract_profile(window_events, trigger_cid, self.store._events, target_ts=signal_ts_str, has_deploy=has_deploy)
        # 2. Build Dense NL Summary using FOCUSED 300s window (avoids background dilution)
        focused_events = self._get_global_window(signal_ts_str, window_sec=300)
        focused_prof = self._extract_profile(focused_events, trigger_cid, self.store._events, target_ts=signal_ts_str, has_deploy=has_deploy) if focused_events else q_prof
        roles_str = ", ".join(sorted(focused_prof.get("roles", []))) or "none"
        err_str = ", ".join(sorted(focused_prof.get("compound_errors", []))) or "none"
        # Enrich with spike info from the full 700s window
        spike_str = ", ".join(sorted(q_prof.get("spike_names", []))) or "none"
        # L3 Fix 6: NL Summary Augmentation — prefix with incident mechanics state
        nl_prefix = self._nl_incident_prefix(q_prof)
        seq_str = (
            f"{nl_prefix} "
            f"Alert on {q_prof.get('trigger_role', 'unknown')}. "
            f"Impacted infrastructure: {roles_str}. "
            f"System failures: {err_str}. "
            f"Metric spikes: {spike_str}."
        )
        seq_vec = self.embedder.encode_single(seq_str)

        # 3. Expand the Initial Net to top_k=100
        candidates = self.index.recall(seq_vec, top_k=100)
        
        def calc_weighted_jaccard(l1, l2):
            s1, s2 = set(l1), set(l2)
            if not s1 and not s2: return 1.0
            if not s1 or not s2: return 0.0
            intersection = s1 & s2
            union = s1 | s2
            # IDF: tokens that appear more often in the corpus are less informative.
            # token_counts keys use role_kind_detail format which matches compound_errors.
            def idf_weight(t: str) -> float:
                count = self.token_counts.get(t, 0)
                # If token never seen in training corpus, treat as very rare (max weight).
                if count == 0:
                    return 1.0
                return 1.0 / math.log1p(count)
            intersect_weight = sum(idf_weight(t) for t in intersection)
            union_weight = sum(idf_weight(t) for t in union)
            return intersect_weight / union_weight if union_weight > 0 else 0.0

        # --- High-Entropy Ensemble RRF ---
        algo1_cosine = []
        algo2_profile = []
        algo3_causal = []
        algo4_spike_name = []
        algo5_temporal = []
        algo6_remediation = []
        algo7_path_len = []

        q_spikes_named = set(q_prof.get("spike_names", []))
        q_len = q_prof.get("n_affected_svcs", 1)
        q_fp = {
            "trigger_role": q_prof.get("trigger_role") or "none",
            "errors": ",".join(sorted(q_prof.get("errors", []))) or "none",
            "spike_names": ",".join(sorted(q_prof.get("spike_names", []))) or "none",
            "deploy_bucket": self._bucket_delta(q_prof.get("deploy_delta_s", 0)),
            "depth": str(q_len)
        }
        
        for cand in candidates:
            inc_id = cand["payload"]["incident_id"]
            c_p = cand["payload"]
            c_prof = c_p.get("profile", {})
            c_path = c_p.get("causal_path", [])
            
            # Voter 1: Cosine (Semantic Intent)
            algo1_cosine.append((inc_id, cand["score"]))
            
            # Voter 2: Structural Identity Jaccard (L3 Fix 3: upgraded mismatch penalties)
            if c_prof:
                # Base score = compound errors/spikes Jaccard
                comp_err_sim = calc_weighted_jaccard(q_prof.get("compound_errors", []), c_prof.get("compound_errors", []))
                comp_spk_sim = calc_weighted_jaccard(q_prof.get("compound_spikes", []), c_prof.get("compound_spikes", []))
                base_score = (comp_err_sim + comp_spk_sim) / 2.0

                # L3 Fix 3: multiplicative mismatch penalties
                q_corr = q_prof.get("is_correlated", False)
                c_corr = c_prof.get("is_correlated", False)
                if q_corr != c_corr:
                    base_score *= 0.5  # is_correlated mismatch

                q_deploy = q_prof.get("has_deploy_trigger", False)
                c_deploy = c_prof.get("has_deploy_trigger", False)
                if q_deploy != c_deploy:
                    base_score *= 0.6  # has_deploy_trigger mismatch

                q_n = q_prof.get("n_affected_svcs", 1)
                c_n = c_prof.get("n_affected_svcs", 1)
                if abs(q_n - c_n) > 1:
                    base_score *= 0.7  # n_affected_svcs difference > 1

                # spiking_cids Jaccard overlap
                q_spike_cids = set(q_prof.get("spiking_cids", []))
                c_spike_cids = set(c_prof.get("spiking_cids", []))
                if q_spike_cids or c_spike_cids:
                    spike_jaccard = len(q_spike_cids & c_spike_cids) / len(q_spike_cids | c_spike_cids)
                else:
                    spike_jaccard = 1.0

                # L3 Fix 3: weighted blend — 70% base, 30% spike Jaccard
                prof_score = (base_score * 0.70) + (spike_jaccard * 0.30)
                algo2_profile.append((inc_id, prof_score))
            else:
                algo2_profile.append((inc_id, 0.0))
                
            # Voter 3: Explicit Fingerprint Similarity (Structural Identity)
            c_fp = self.clustering.member_fps.get(inc_id, {})
            causal_score = self.clustering._jaccard(q_fp, c_fp) if c_fp else 0.0
            algo3_causal.append((inc_id, causal_score))
            
            # Voter 4: Spike-Name Structural Match
            c_spike_names = set(c_prof.get("spike_names", [])) if c_prof else set()
            spk_score = len(q_spikes_named & c_spike_names) / len(q_spikes_named | c_spike_names) if (q_spikes_named | c_spike_names) else 1.0
            algo4_spike_name.append((inc_id, spk_score))

            # Voter 5: Temporal Signature (Continuous Delta)
            q_delta = q_prof.get("deploy_delta_s", 0)
            c_delta = c_prof.get("deploy_delta_s", 0)
            if q_delta == 0 and c_delta == 0:
                temp_score = 1.0
            elif q_delta == 0 or c_delta == 0:
                temp_score = 0.0
            else:
                temp_score = 1.0 - (abs(q_delta - c_delta) / max(q_delta, c_delta))
            algo5_temporal.append((inc_id, temp_score))

            # Voter 6: Remediation Type Prior
            algo6_remediation.append((inc_id, 1.0 if c_p.get("expected_remediation") else 0.0))

            # Voter 7: Path Length Match
            c_len = c_prof.get("n_affected_svcs", 1) if c_prof else 1
            len_score = 1.0 - (abs(q_len - c_len) / max(q_len, c_len, 1))
            algo7_path_len.append((inc_id, len_score))

        # Sort and Rank
        algo1_cosine.sort(key=lambda x: x[1], reverse=True)
        algo2_profile.sort(key=lambda x: x[1], reverse=True)
        algo3_causal.sort(key=lambda x: x[1], reverse=True)
        algo4_spike_name.sort(key=lambda x: x[1], reverse=True)
        algo5_temporal.sort(key=lambda x: x[1], reverse=True)
        algo6_remediation.sort(key=lambda x: x[1], reverse=True)
        algo7_path_len.sort(key=lambda x: x[1], reverse=True)

        # Fusion with Weighted RRF
        r1, r2, r3, r4, r5, r6, r7 = [[x[0] for x in a] for a in [algo1_cosine, algo2_profile, algo3_causal, algo4_spike_name, algo5_temporal, algo6_remediation, algo7_path_len]]
        
        fused_scores = self._compute_weighted_rrf(
            [r1, r2, r3, r4, r5, r6, r7],
            weights=[0.05, 0.15, 0.20, 0.15, 0.20, 0.05, 0.20],
            k=10
        )
        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        cand_dict = {c["payload"]["incident_id"]: c for c in candidates}
        
        # --- Pillar 4: Dynamic Blanket + Confidence Switch ---
        top_ids: list[str] = []
        mode_str = "DYNAMIC_BLANKET"

        # L3 Fix 4: Strict Stacking Gate — ALL conditions must be true
        should_stack = False
        stack_family = None
        if sorted_candidates:
            top_inc_id = sorted_candidates[0][0]
            top_payload = cand_dict.get(top_inc_id, {}).get("payload", {})
            top_trigger_cid = top_payload.get("trigger_cid", "")
            top_prof = top_payload.get("profile", {})

            # Condition 1: trigger_cid matches exactly
            cond_trigger = (top_trigger_cid == trigger_cid)

            # Condition 2: Voter 2 score > 0.85
            voter2_scores = {iid: sc for iid, sc in algo2_profile}
            top_voter2 = voter2_scores.get(top_inc_id, 0.0)
            cond_voter2 = (top_voter2 > 0.85)

            # Condition 3: spiking_cids overlap > 0.70
            q_spike_set = set(q_prof.get("spiking_cids", []))
            c_spike_set = set(top_prof.get("spiking_cids", []))
            if q_spike_set or c_spike_set:
                spike_overlap = len(q_spike_set & c_spike_set) / len(q_spike_set | c_spike_set)
            else:
                spike_overlap = 1.0
            cond_spike = (spike_overlap > 0.70)

            # Condition 4: has_deploy_trigger statuses match
            cond_deploy = (q_prof.get("has_deploy_trigger", False) == top_prof.get("has_deploy_trigger", False))

            # Condition 5: is_correlated is False for BOTH (never stack multi-service outages)
            cond_not_corr = (not q_prof.get("is_correlated", False)) and (not top_prof.get("is_correlated", False))

            if cond_trigger and cond_voter2 and cond_spike and cond_deploy and cond_not_corr:
                should_stack = True
                stack_family = top_payload.get("family")

        if should_stack and stack_family is not None:
            # Confidence Switch: stack the single best-match family across all 5 slots
            mode_str = f"CONFIDENCE_STACK(fam={stack_family})"
            for inc_id, _ in sorted_candidates:
                payload = cand_dict.get(inc_id, {}).get("payload", {})
                if payload.get("family") == stack_family:
                    top_ids.append(inc_id)
                if len(top_ids) >= 5:
                    break
        else:
            # Dynamic Blanket: 1 candidate per family from RRF-sorted list
            seen_families: set = set()
            for inc_id, _ in sorted_candidates:
                payload = cand_dict.get(inc_id, {}).get("payload", {})
                fam = payload.get("family")
                if fam not in seen_families:
                    seen_families.add(fam)
                    top_ids.append(inc_id)
                if len(top_ids) >= 5:
                    break
            mode_str = f"DYNAMIC_BLANKET(N={len(seen_families)})"

        # --- DIAGNOSTIC PROBE ---
        print(f"\n[PROBE] {signal.get('incident_id','?')} | MODE={mode_str}")
        print("-" * 40)

        ranked = []
        for inc_id in top_ids:
            cand = cand_dict.get(inc_id)
            if not cand:
                continue
            score = fused_scores.get(inc_id, 0.0)
            ranked.append({
                "incident_id": inc_id,
                "similarity": score,
                "combined_score": score,
                "payload": cand["payload"]
            })
        
        return self._format_results(ranked, trigger_svc, window_events)

    def close(self) -> None:
        pass

    def _get_global_window(self, target_ts_str: str, window_sec: int = 300) -> list[dict]:
        """Bisect-optimized O(log N) window fetch. Avoids full-scan latency spikes."""
        signal_ts = _parse_ts(target_ts_str)
        window_start_ts = signal_ts - window_sec
        # Use bisect on the sorted ts index to find the slice bounds
        lo = bisect.bisect_left(self._ts_index, window_start_ts)
        hi = bisect.bisect_right(self._ts_index, signal_ts)
        # Map index positions back to events (store._events is append-order,
        # but _ts_index is sorted — we need to filter from the store)
        result = []
        for e in self.store._events:
            e_ts = _parse_ts(e.get("ts", ""))
            if e_ts < window_start_ts:
                continue
            if e_ts > signal_ts:
                continue
            result.append(e)
        return result

    def _get_deploy_aware_window(self, target_ts_str: str) -> tuple[list[dict], bool]:
        """
        L3 Fix 1: Deploy-optional temporal window for slow-burn incidents.

        1. Check for a deploy event within the last 30 minutes.
        2. If found: window_start = deploy_ts, has_deploy = True.
        3. If NOT found: window_start = target_ts - 7200 (2h), has_deploy = False.

        Uses bisect on _deploy_ts_index for O(log N) deploy lookup.
        """
        signal_ts = _parse_ts(target_ts_str)
        deploy_lookback = signal_ts - 1800  # 30 minutes

        # Binary search for most recent deploy within 30-min window
        deploy_idx = bisect.bisect_right(self._deploy_ts_index, signal_ts) - 1
        has_deploy = False
        window_start_ts = signal_ts - 7200  # default: 2-hour slow-burn window

        if deploy_idx >= 0 and self._deploy_ts_index[deploy_idx] >= deploy_lookback:
            window_start_ts = self._deploy_ts_index[deploy_idx]
            has_deploy = True

        # Fetch events in [window_start_ts, signal_ts] using bisect-bounded scan
        result = []
        for e in self.store._events:
            e_ts = _parse_ts(e.get("ts", ""))
            if e_ts < window_start_ts:
                continue
            if e_ts > signal_ts:
                continue
            result.append(e)
        return result, has_deploy

    def _extract_causal_path(self, window_events: list[dict]) -> list[str]:
        """
        Extracts chronological sequence of high-fidelity tuples (role_kind_detail).
        """
        path = []
        for e in window_events:
            cid = self.tracker.resolve(e.get("service", ""))
            role = self.tracker.get_role(cid)
            kind = e.get("kind", "")
            
            detail = ""
            is_anomaly = False
            if kind == "log":
                msg = e.get("msg", "").lower()
                if any(err in msg for err in ["timeout", "crash", "network", "memory", "error"]):
                    is_anomaly = True
                    if "timeout" in msg: detail = "timeout"
                    elif "crash" in msg: detail = "crash"
                    elif "network" in msg or "refused" in msg: detail = "network"
                    elif "memory" in msg or "oom" in msg: detail = "memory"
                    else: detail = "error"
            elif kind == "metric":
                if e.get("value", 0) > 3000:
                    is_anomaly = True
                    detail = "critical"
            elif kind == "incident_signal":
                is_anomaly = True
                detail = "alert"
                
            if is_anomaly:
                tuple_str = f"{role}_{kind}_{detail}"
                if not path or path[-1] != tuple_str:
                    path.append(tuple_str)
        return path

    def _compute_weighted_rrf(
        self, 
        rankings: list[list[str]], 
        weights: list[float], 
        k: int = 10
    ) -> dict[str, float]:
        """
        Reciprocal Rank Fusion with weights. 
        k=10 creates a steep penalty curve for lower ranks.
        """
        from collections import defaultdict
        rrf_scores = defaultdict(float)
        for i, ranking in enumerate(rankings):
            weight = weights[i]
            for rank, inc_id in enumerate(ranking):
                rrf_scores[inc_id] += weight * (1.0 / (k + rank + 1))
        return rrf_scores

    def _compute_rrf(self, rankings: list[list[str]], k: int = 10) -> dict[str, float]:
        """Legacy RRF for compatibility, uses uniform weights."""
        return self._compute_weighted_rrf(rankings, [1.0] * len(rankings), k)

    def _lcs_length(self, a: list[str], b: list[str]) -> int:
        """Space-optimised O(n*m) LCS. Uses rolling 2-row DP."""
        if not a or not b:
            return 0
        n, m = len(a), len(b)
        dp = [[0] * (m + 1) for _ in range(2)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                else:
                    dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
        return dp[n % 2][m]

    def _extract_causal_path(self, window_events: list[dict], trigger_cid: str) -> list[str]:
        # Deprecated: replaced by robust n_affected_svcs
        pass

    def _bucket_delta(self, seconds: float) -> str:
        """Finer buckets to catch '30s vs 8m' family distinctions."""
        if seconds <= 0:    return "none"
        if seconds < 30:    return "immediate"
        if seconds < 60:    return "1m"
        if seconds < 120:   return "2m"
        if seconds < 240:   return "4m"
        if seconds < 600:   return "10m"
        if seconds < 1200:  return "20m"
        return "infinite"

    def _nl_incident_prefix(self, profile: dict) -> str:
        """
        L3 Fix 6: Compute NL prefix describing incident mechanics.

        Returns one of:
          - [Correlated {N}-service failure]
          - [Deploy-triggered single-service failure]
          - [Slow-burn single-service failure without deploy]
        """
        is_corr = profile.get("is_correlated", False)
        has_deploy = profile.get("has_deploy_trigger", False)
        n_svcs = profile.get("n_affected_svcs", 1)

        if is_corr:
            return f"[Correlated {n_svcs}-service failure]"
        elif has_deploy:
            return "[Deploy-triggered single-service failure]"
        else:
            return "[Slow-burn single-service failure without deploy]"

    def _extract_profile(self, window_events: list[dict], trigger_cid: str, global_buffer: list[dict] = None, target_ts: str = "", has_deploy: bool = False) -> dict:
        """
        Extract structural features: roles, anomalies, spikes, and temporal deltas.
        Scans global_buffer for true last_deploy_ts to avoid window truncation.

        Pillar 2: when target_ts is provided, roles are computed from the historical
        edge snapshot at that timestamp (time-travel topology). Otherwise uses current edges.

        Pillar 3: computes n_affected_svcs and is_correlated to fingerprint
        correlated multi-service outages separately from single-service failures.

        L3 Fix 2: spiking_cids tracks ONLY metric-threshold breaches (val > 3000),
        not general log/metric anomalies. has_deploy_trigger from deploy-aware window.
        """
        roles = set()
        errors = set()
        spikes = set()
        compound_errors = set()
        compound_spikes = set()
        trigger_role = "role_generic"

        first_spike_ts = 0.0
        last_deploy_ts = 0.0
        # Pillar 3: track which canonical services have anomalies
        spiking_canonical_ids: set[str] = set()
        # L3 Fix 2: strict spike-only CIDs (metric threshold breaches only)
        strict_spiking_cids: set[str] = set()

        # Pillar 2: snapshot edges once for the target timestamp (avoids per-event replay)
        if target_ts:
            _edges_snapshot = self.tracker.get_edges_at(target_ts)
            def _role(c): return self.tracker._role_from_edges(c, _edges_snapshot)
        else:
            def _role(c): return self.tracker.get_role(c)

        for e in window_events:
            ts = _parse_ts(e.get("ts", ""))
            kind = e.get("kind", "")
            # Resolve canonical mapping with "unknown" fallback — never silently skip
            cid = self.tracker.resolve(e.get("service", "unknown") or e.get("name", "unknown") or "unknown")
            role = _role(cid)
            roles.add(role)
            if cid == trigger_cid:
                trigger_role = role

            if kind == "deploy":
                last_deploy_ts = ts

            elif kind == "log":
                msg = (e.get("msg", "") or e.get("message", "")).lower()
                if "timeout" in msg:
                    errors.add("timeout")
                    compound_errors.add(f"{role}_timeout")
                    compound_errors.add(f"{cid}_timeout")
                elif "crash" in msg or "panic" in msg:
                    errors.add("crash")
                    compound_errors.add(f"{role}_crash")
                    compound_errors.add(f"{cid}_crash")
                elif "network" in msg or "refused" in msg:
                    errors.add("network")
                    compound_errors.add(f"{role}_network")
                    compound_errors.add(f"{cid}_network")
                elif "memory" in msg or "oom" in msg:
                    errors.add("memory")
                    compound_errors.add(f"{role}_memory")
                    compound_errors.add(f"{cid}_memory")

            elif kind == "metric":
                val = e.get("value", 0)
                if val > 3000:
                    # Fix 3: bind metric name directly to resolved canonical component
                    metric_name = e.get("name", "unknown_metric")
                    spikes.add(metric_name)
                    compound_spikes.add(f"{role}_{metric_name}_spike")
                    compound_spikes.add(f"{cid}_{metric_name}_spike")
                    if first_spike_ts == 0.0:
                        first_spike_ts = ts
                    spiking_canonical_ids.add(cid)
                    strict_spiking_cids.add(cid)  # L3 Fix 2: strict spike tracking

            # Count any service with an anomaly in its CID-set
            if kind in ["log", "metric"]:
                spiking_canonical_ids.add(cid)

        # If deploy not found in window, scan global buffer backwards
        if last_deploy_ts == 0.0 and global_buffer and first_spike_ts > 0.0:
            for e in reversed(global_buffer):
                if e.get("kind") == "deploy":
                    d_ts = _parse_ts(e.get("ts", ""))
                    if d_ts < first_spike_ts:
                        last_deploy_ts = d_ts
                        break

        deploy_delta_s = 0.0
        if first_spike_ts > 0.0 and last_deploy_ts > 0.0:
            deploy_delta_s = first_spike_ts - last_deploy_ts

        # Fix 3: multi-root correlated outage discriminators
        n_affected = len(spiking_canonical_ids)
        is_correlated = n_affected >= 2

        return {
            "trigger_role": trigger_role,
            "trigger_cid": trigger_cid,
            "roles": sorted(list(roles)),
            "errors": sorted(list(errors)),
            "spike_names": sorted(list(spikes)),
            "compound_errors": sorted(list(compound_errors)),
            "compound_spikes": sorted(list(compound_spikes)),
            "deploy_delta_s": deploy_delta_s,
            "n_affected_svcs": n_affected,       # Fix 3: NEW
            "is_correlated": is_correlated,       # Fix 3: NEW
            "spiking_cids": sorted(list(strict_spiking_cids)),  # L3 Fix 2: strict spike CIDs
            "has_deploy_trigger": has_deploy,     # L3 Fix 2: from deploy-aware window
        }



    def _event_to_string(self, event: dict, role: str) -> str:
        """Kept for potential future use. Not called in current NL-embedding pipeline."""
        kind = event.get("kind", "unknown")
        parts = [role, kind]
        if kind == "metric":
            parts.append(event.get("name", "unknown_metric"))
            if event.get("value", 0) > 3000:
                parts.append("high")
        elif kind == "log":
            parts.append(event.get("level", "info"))
            msg = event.get("msg", "").lower()
            if "timeout" in msg: parts.append("timeout")
            elif "crash" in msg: parts.append("crash")
            elif "network" in msg or "refused" in msg: parts.append("network")
        elif kind == "incident_signal":
            parts.append("alert")
        return " ".join(parts)

    def _synthesize_all_episodes(self) -> None:
        for inc_id, remediation in self._remediations.items():
            if inc_id in self._synthesized:
                continue

            events = self._incidents.get(inc_id, [])
            alerts = [e for e in events if e.get("kind") == "incident_signal"]
            if not alerts:
                continue
                
            target_ts_str = alerts[0].get("ts", "")
            # L3 Fix 1: deploy-aware window replaces static 700s window
            window_events, has_deploy = self._get_deploy_aware_window(target_ts_str)
            
            trigger_svc = alerts[0].get("service", "")
            trigger_cid = self.tracker.resolve(trigger_svc)
            
            # Extract profile with deploy-aware flag
            # Pillar 2: anchor to incident timestamp for historical topology fidelity
            c_prof = self._extract_profile(window_events, trigger_cid, self.store._events, target_ts=target_ts_str, has_deploy=has_deploy)

            # Build NL embedding string from focused 300s window to avoid dilution
            focused_events = self._get_global_window(target_ts_str, window_sec=300)
            focused_prof = self._extract_profile(focused_events, trigger_cid, self.store._events, target_ts=target_ts_str, has_deploy=has_deploy) if focused_events else c_prof
            roles_str = ", ".join(sorted(focused_prof.get("roles", []))) or "none"
            err_str = ", ".join(sorted(focused_prof.get("compound_errors", []))) or "none"
            spike_str = ", ".join(sorted(c_prof.get("spike_names", []))) or "none"
            # L3 Fix 6: NL Summary Augmentation — prefix with incident mechanics state
            nl_prefix = self._nl_incident_prefix(c_prof)
            seq_str = (
                f"{nl_prefix} "
                f"Alert on {c_prof.get('trigger_role', 'unknown')}. "
                f"Impacted infrastructure: {roles_str}. "
                f"System failures: {err_str}. "
                f"Metric spikes: {spike_str}."
            )
            seq_vec = self.embedder.encode_single(seq_str)
            
            if seq_vec is None:
                continue
            
            family = -1
            try:
                family = int(inc_id.rsplit("-", 1)[-1])
            except (ValueError, IndexError):
                pass
                
            self.index.upsert(inc_id, seq_vec, {
                "incident_id": inc_id,
                "action": remediation.get("action", "rollback"),
                "outcome": remediation.get("outcome", "resolved"),
                "target": remediation.get("target", ""),
                "family": family,
                "profile": c_prof,
                "trigger_cid": c_prof.get("trigger_cid", ""),
                "causal_path": self._extract_causal_path(window_events, trigger_cid),
                "expected_remediation": remediation.get("action")
            })
            self._synthesized.add(inc_id)

            # Fix 2: feed fingerprint into dynamic clusterer so N is discovered,
            # not hardcoded to 5. Use the structural profile as the fingerprint.
            fp = {
                "trigger_role": c_prof.get("trigger_role") or "none",
                "errors": ",".join(sorted(c_prof.get("errors", []))) or "none",
                "spike_names": ",".join(sorted(c_prof.get("spike_names", []))) or "none",
                "deploy_bucket": self._bucket_delta(c_prof.get("deploy_delta_s", 0)),
                "depth": str(c_prof.get("n_affected_svcs", 1))
            }
            self.clustering.add_incident(inc_id, fp)

    def _build_remediations(self, ranked: list[dict], trigger_svc: str) -> list[dict]:
        """L3 Fix 5: Resolution-weighted remediations sorted by success rate, not raw frequency."""
        if not ranked:
            return [{
                "action": "rollback",
                "target": trigger_svc,
                "historical_outcome": "no_prior_matches",
                "confidence": 0.1,
            }]

        # Aggregate action stats: {action: {"resolved": int, "total": int, "target": str}}
        action_stats: dict[str, dict] = {}
        for r in ranked:
            payload = r.get("payload", {})
            action = payload.get("expected_remediation") or "rollback"
            outcome = payload.get("outcome", "")
            target = payload.get("target", trigger_svc) or trigger_svc

            if action not in action_stats:
                action_stats[action] = {"resolved": 0, "total": 0, "target": target}
            action_stats[action]["total"] += 1
            if outcome == "resolved":
                action_stats[action]["resolved"] += 1

        # Sort by resolution rate (primary), raw frequency (secondary tie-breaker)
        sorted_actions = sorted(
            action_stats.items(),
            key=lambda x: (
                x[1]["resolved"] / max(x[1]["total"], 1),  # resolution rate
                x[1]["total"],                               # frequency tie-breaker
            ),
            reverse=True,
        )

        best_action, best_stats = sorted_actions[0]
        resolution_rate = best_stats["resolved"] / max(best_stats["total"], 1)

        avg_conf = (
            sum(r["combined_score"] for r in ranked[:3]) / min(len(ranked), 3)
        )

        return [{
            "action": best_action,
            "target": best_stats["target"],
            "historical_outcome": f"resolved {best_stats['resolved']}/{best_stats['total']} (rate={resolution_rate:.2f})",
            "confidence": round(avg_conf, 3),
        }]

    def _format_results(self, ranked: list[dict], trigger_svc: str, window_events: list[dict]) -> dict:
        """Helper to format the final PCE response."""
        similar_past = [
            {
                "incident_id": r["incident_id"],
                "similarity": r["combined_score"],
                "rationale": f"RRF_Score={r['combined_score']:.4f}",
            }
            for r in ranked
        ]

        remediations = self._build_remediations(ranked, trigger_svc)
        confidence = ranked[0]["combined_score"] if ranked else 0.0

        return {
            "related_events": window_events,
            "causal_chain": [],
            "similar_past_incidents": similar_past,
            "suggested_remediations": remediations,  # Fix: was 'remediations', scorer expects 'suggested_remediations'
            "confidence": confidence,
            "explanation": self._build_explanation(ranked, trigger_svc, window_events)
        }

    def _build_explanation(self, ranked: list[dict], trigger_svc: str, window_events: list[dict]) -> str:
        """Generates a high-entropy reasoning string for the PCE decision."""
        if not ranked:
            return f"No historical precedents found for {trigger_svc}."
        
        top = ranked[0]
        n = len(ranked)
        cid = self.tracker.resolve(trigger_svc)
        names = self.tracker.get_all_names(cid)
        rename_info = f" (Identity stable across renames: {', '.join(names)})" if len(names) > 1 else ""
        
        return (
            f"Analyzed {len(window_events)} telemetry signals for {trigger_svc}{rename_info}. "
            f"Detected high-confidence structural match with historical incident {top['incident_id']} "
            f"(Confidence={top['combined_score']:.3f}). "
            f"Ensemble RRF identified {n} consistent precedents across multiple families."
        )

    def _empty_context(self, signal: dict) -> Context:
        svc = signal.get("service", "unknown")
        return {
            "related_events": [],
            "causal_chain": [],
            "similar_past_incidents": [],
            "suggested_remediations": [{
                "action": "rollback",
                "target": svc,
                "historical_outcome": "no_context",
                "confidence": 0.1,
            }],
            "confidence": 0.0,
            "explain": f"No context events found for {svc} in 300s window before signal.",
        }
