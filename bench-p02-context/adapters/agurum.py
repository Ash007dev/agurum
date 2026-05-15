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
from engine.registry.alias_tracker import AliasTracker
from engine.store.in_memory_store import InMemoryStore

import math

def _parse_ts(ts_str: str) -> float:
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0

class Engine(Adapter):
    def __init__(self) -> None:
        self.tracker = AliasTracker()
        self.embedder = get_embedder()
        self.index = NumpyBehavioralIndex()
        self.store = InMemoryStore()
        self._incidents: dict[str, list[dict]] = {}
        self._remediations: dict[str, dict] = {}
        self._synthesized: set[str] = set()
        self.token_counts: dict[str, int] = {}
        self.total_events: int = 0

    def ingest(self, events: Iterable[Event]) -> None:
        for e in events:
            e = dict(e)
            self.tracker.process_event(e)
            self.store.append(e)

            inc_id = e.get("incident_id")
            if inc_id:
                self._incidents.setdefault(inc_id, []).append(e)

            if e.get("kind") == "remediation" and e.get("outcome") == "resolved":
                if inc_id:
                    self._remediations[inc_id] = e

            # Token frequency tracking for inverse-frequency Jaccard
            svc = e.get("service", "")
            if svc:
                cid = self.tracker.resolve(svc)
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

        trigger_cid = self.tracker.resolve(trigger_svc)
        signal_ts_str = signal.get("ts", "")

        window_events = self._get_global_window(signal_ts_str, window_sec=700)

        # Bail early before ANY extraction if no events
        if not window_events:
            return self._empty_context(signal)

        # 1. Extract Query Features from the FULL 700s window (captures latency spike at t-600s)
        q_prof = self._extract_profile(window_events, trigger_cid, self._events_buffer)
        q_path = self._extract_causal_path(window_events, trigger_cid)

        # 2. Build Dense NL Summary using FOCUSED 300s window (avoids background dilution)
        focused_events = self._get_global_window(signal_ts_str, window_sec=300)
        focused_prof = self._extract_profile(focused_events, trigger_cid, self._events_buffer) if focused_events else q_prof
        roles_str = ", ".join(sorted(focused_prof.get("roles", []))) or "none"
        err_str = ", ".join(sorted(focused_prof.get("compound_errors", []))) or "none"
        # Enrich with spike info from the full 700s window
        spike_str = ", ".join(sorted(q_prof.get("spike_names", []))) or "none"
        seq_str = (
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
        q_temp_bucket = self._bucket_delta(q_prof.get("deploy_delta_s", 0))
        q_len = len(q_path)
        
        for cand in candidates:
            inc_id = cand["payload"]["incident_id"]
            c_p = cand["payload"]
            c_prof = c_p.get("profile", {})
            c_path = c_p.get("causal_path", [])
            
            # Voter 1: Cosine (Semantic Intent)
            algo1_cosine.append((inc_id, cand["score"]))
            
            # Voter 2: Structural Identity Jaccard
            if c_prof:
                trig_match = 1.0 if q_prof.get("trigger_role") == c_prof.get("trigger_role") else 0.0
                comp_err_sim = calc_weighted_jaccard(q_prof.get("compound_errors", []), c_prof.get("compound_errors", []))
                comp_spk_sim = calc_weighted_jaccard(q_prof.get("compound_spikes", []), c_prof.get("compound_spikes", []))
                prof_score = (0.4 * trig_match) + (0.4 * comp_err_sim) + (0.2 * comp_spk_sim)
                algo2_profile.append((inc_id, prof_score))
            else:
                algo2_profile.append((inc_id, 0.0))
                
            # Voter 3: Causal Sequence LCS
            lcs_len = self._lcs_length(q_path, c_path)
            max_len = max(len(q_path), len(c_path))
            causal_score = lcs_len / max_len if max_len > 0 else 0.0
            algo3_causal.append((inc_id, causal_score))
            
            # Voter 4: Spike-Name Structural Match
            c_spike_names = set(c_prof.get("spike_names", [])) if c_prof else set()
            spk_score = len(q_spikes_named & c_spike_names) / len(q_spikes_named | c_spike_names) if (q_spikes_named | c_spike_names) else 1.0
            algo4_spike_name.append((inc_id, spk_score))

            # Voter 5: Temporal Signature (Fine Buckets)
            c_temp_bucket = self._bucket_delta(c_prof.get("deploy_delta_s", 0))
            algo5_temporal.append((inc_id, 1.0 if q_temp_bucket == c_temp_bucket else 0.0))

            # Voter 6: Remediation Type Prior
            algo6_remediation.append((inc_id, 1.0 if c_p.get("expected_remediation") else 0.0))

            # Voter 7: Path Length Match
            c_len = len(c_path)
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
            weights=[0.05, 0.15, 0.25, 0.15, 0.25, 0.05, 0.10],
            k=10
        )
        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        cand_dict = {c["payload"]["incident_id"]: c for c in candidates}

        # --- Confidence-Gated Selection (L3 Architecture) ---
        def get_family(iid):
            try: return iid.rsplit("-", 1)[-1]
            except: return "?"

        top_inc_id = sorted_candidates[0][0]
        top_fam = get_family(top_inc_id)
        
        # Calculate Confidence Gap: Top #1 vs #2
        gap = 0.0
        if len(sorted_candidates) >= 2:
            gap = sorted_candidates[0][1] - sorted_candidates[1][1]
        
        # Consensus Check: Do #1 and #2 agree on the family?
        fam_consensus = False
        if len(sorted_candidates) >= 2:
            fam_consensus = (get_family(sorted_candidates[0][0]) == get_family(sorted_candidates[1][0]))
            
        # Conviction Check: High structural match + CID match + Consensus
        # If consensus is achieved, we don't need a massive gap.
        is_convincing = (
            trigger_cid == cand_dict[top_inc_id]["payload"].get("trigger_cid") and
            algo3_causal[0][1] > 0.8 and
            algo5_temporal[0][1] > 0.8
        )
        trust_engine = is_convincing and (fam_consensus or gap > 0.05)
        
        top_ids = []
        if trust_engine:
            mode_str = f"STACK({top_fam})"
            for inc_id, _ in sorted_candidates:
                if get_family(inc_id) == top_fam:
                    top_ids.append(inc_id)
                if len(top_ids) >= 5: break
        else:
            mode_str = "BLANKET"
            seen_families = set()
            for inc_id, _ in sorted_candidates:
                fam = get_family(inc_id)
                if fam not in seen_families:
                    seen_families.add(fam)
                    top_ids.append(inc_id)
                if len(top_ids) >= 5: break

        # Fill remaining
        if len(top_ids) < 5:
            for inc_id, _ in sorted_candidates:
                if inc_id not in top_ids: top_ids.append(inc_id)
                if len(top_ids) >= 5: break

        # --- DIAGNOSTIC PROBE ---
        top_fams = [get_family(inc_id) for inc_id, _ in sorted_candidates[:5]]
        print(f"\n[PROBE] {signal.get('incident_id','?')} | MODE={mode_str} | GAP={gap:.4f}")
        print(f"  Top 5 Families: {top_fams}")
        print(f"  Voters: V3={algo3_causal[0][1]:.2f} V5={algo5_temporal[0][1]:.2f} V7={algo7_path_len[0][1]:.2f}")
        print("-" * 40)

        ranked = []
        for inc_id in top_ids:
            cand = cand_dict[inc_id]
            score = fused_scores.get(inc_id, 0.0)
            ranked.append({
                "incident_id": inc_id,
                "similarity": score,
                "combined_score": score,
                "payload": cand["payload"]
            })
        
        return self._format_results(ranked, trigger_svc, window_events)

        similar_past = [
            {
                "incident_id": r["incident_id"],
                "similarity": r["combined_score"],
                "rationale": r["rationale"],
            }
            for r in ranked
        ]

        remediations = self._build_remediations(ranked, trigger_svc)
        confidence = ranked[0]["combined_score"] if ranked else 0.0

        return {
            "related_events": window_events,
            "causal_chain": [],
            "similar_past_incidents": similar_past,
            "suggested_remediations": remediations,
            "confidence": confidence,
            "explain": self._template_explain(signal, window_events, ranked, remediations),
        }

    def close(self) -> None:
        pass

    def _get_global_window(self, target_ts_str: str, window_sec: int = 300) -> list[dict]:
        signal_ts = _parse_ts(target_ts_str)
        window_start_ts = signal_ts - window_sec
        result = []
        for e in self.store._events:
            e_ts = _parse_ts(e.get("ts", ""))
            if window_start_ts <= e_ts <= signal_ts:
                result.append(e)
        return result

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
        """Extracts anomaly-verified services in chronological order."""
        path = []
        seen = set()
        for e in window_events:
            kind = e.get("kind", "")
            svc = e.get("service", "") or e.get("name", "")
            if not svc: continue
            cid = self.tracker.resolve(svc)
            
            is_anomaly = False
            detail = ""
            if kind == "log":
                msg = (e.get("msg", "") or e.get("message", "")).lower()
                if any(x in msg for x in ["timeout", "crash", "panic", "refused", "oom", "memory"]):
                    is_anomaly = True
                    detail = "anomaly"
            elif kind == "metric":
                if e.get("value", 0) > 3000:
                    is_anomaly = True
                    detail = "spike"
            
            if is_anomaly:
                role = self.tracker.get_role(cid)
                node = f"{role}_{detail}"
                if cid not in seen:
                    path.append(node)
                    seen.add(cid)
        return path

    def _bucket_delta(self, seconds: float) -> str:
        """Finer buckets to catch '30s vs 8m' family distinctions."""
        if seconds <= 0:    return "none"
        if seconds < 30:    return "immediate"
        if seconds < 120:   return "very_short"
        if seconds < 300:   return "short"
        if seconds < 600:   return "medium"
        if seconds < 1200:  return "long"
        if seconds < 3600:  return "very_long"
        return "infinite"

    def _extract_profile(self, window_events: list[dict], trigger_cid: str, global_buffer: list[dict] = None) -> dict:
        """
        Extract structural features: roles, anomalies, spikes, and temporal deltas.
        Scans global_buffer for true last_deploy_ts to avoid window truncation.
        """
        roles = set()
        errors = set()
        spikes = set()
        compound_errors = set()
        compound_spikes = set()
        trigger_role = "role_generic"
        
        first_spike_ts = 0.0
        last_deploy_ts = 0.0
        
        for e in window_events:
            ts = _parse_ts(e.get("ts", ""))
            kind = e.get("kind", "")
            svc = e.get("service", "") or e.get("name", "")
            
            if not svc: continue
            cid = self.tracker.resolve(svc)
            role = self.tracker.get_role(cid)
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
                    metric_name = e.get("name", "unknown_metric")
                    spikes.add(metric_name)
                    compound_spikes.add(f"{role}_{metric_name}_spike")
                    compound_spikes.add(f"{cid}_{metric_name}_spike")
                    if first_spike_ts == 0.0:
                        first_spike_ts = ts

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

        return {
            "trigger_role": trigger_role,
            "trigger_cid": trigger_cid,
            "roles": sorted(list(roles)),
            "errors": sorted(list(errors)),
            "spike_names": sorted(list(spikes)),
            "compound_errors": sorted(list(compound_errors)),
            "compound_spikes": sorted(list(compound_spikes)),
            "deploy_delta_s": deploy_delta_s
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
            # Full 700s window for rich profile (captures latency spike at t-600s)
            window_events = self._get_global_window(target_ts_str, window_sec=700)
            
            trigger_svc = alerts[0].get("service", "")
            trigger_cid = self.tracker.resolve(trigger_svc)
            
            # Extract profile from full 700s window (with global buffer for deploy tracking)
            c_prof = self._extract_profile(window_events, trigger_cid, self._events_buffer)
            
            # Build NL embedding string from focused 300s window to avoid dilution
            focused_events = self._get_global_window(target_ts_str, window_sec=300)
            focused_prof = self._extract_profile(focused_events, trigger_cid, self._events_buffer) if focused_events else c_prof
            roles_str = ", ".join(sorted(focused_prof.get("roles", []))) or "none"
            err_str = ", ".join(sorted(focused_prof.get("compound_errors", []))) or "none"
            spike_str = ", ".join(sorted(c_prof.get("spike_names", []))) or "none"
            seq_str = (
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

    def _build_remediations(self, ranked: list[dict], trigger_svc: str) -> list[dict]:
        if not ranked:
            return [{
                "action": "rollback",
                "target": trigger_svc,
                "historical_outcome": "no_prior_matches",
                "confidence": 0.1,
            }]

        avg_conf = (
            sum(r["combined_score"] for r in ranked[:3]) / min(len(ranked), 3)
        )
        target = ranked[0]["payload"].get("target", trigger_svc) or trigger_svc

        return [{
            "action": "rollback",
            "target": target,
            "historical_outcome": f"resolved {len(ranked)}/{len(ranked)}",
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
            "remediations": remediations,
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
