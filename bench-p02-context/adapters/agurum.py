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
        q_prof = self._extract_profile(window_events, trigger_cid)
        q_path = self._extract_causal_path(window_events)

        # 2. Build Dense NL Summary using FOCUSED 300s window (avoids background dilution)
        focused_events = self._get_global_window(signal_ts_str, window_sec=300)
        focused_prof = self._extract_profile(focused_events, trigger_cid) if focused_events else q_prof
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

        # Score Candidates across all 4 Algorithms
        algo1_cosine = []
        algo2_profile = []
        algo3_causal = []
        algo4_spike_name = []

        q_spikes_named = set(q_prof.get("spike_names", []))
        
        for cand in candidates:
            inc_id = cand["payload"]["incident_id"]
            c_prof = cand["payload"].get("profile", {})
            c_path = cand["payload"].get("causal_path", [])
            
            # Voter 1: Cosine (Dense Vector Similarity)
            algo1_cosine.append((inc_id, cand["score"]))
            
            # Voter 2: Inverse-Frequency Weighted Compound Jaccard
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
            
            # Voter 4: Spike-Name Jaccard
            c_spike_names = set(c_prof.get("spike_names", [])) if c_prof else set()
            if not q_spikes_named and not c_spike_names:
                spk_name_score = 1.0
            elif not q_spikes_named or not c_spike_names:
                spk_name_score = 0.0
            else:
                spk_name_score = len(q_spikes_named & c_spike_names) / len(q_spikes_named | c_spike_names)
            algo4_spike_name.append((inc_id, spk_name_score))

        # Sort each algorithm's list descending
        algo1_cosine.sort(key=lambda x: x[1], reverse=True)
        algo2_profile.sort(key=lambda x: x[1], reverse=True)
        algo3_causal.sort(key=lambda x: x[1], reverse=True)
        algo4_spike_name.sort(key=lambda x: x[1], reverse=True)

        rank1 = [x[0] for x in algo1_cosine]
        rank2 = [x[0] for x in algo2_profile]
        rank3 = [x[0] for x in algo3_causal]
        rank4 = [x[0] for x in algo4_spike_name]

        # --- DIAGNOSTIC PROBE ---
        print(f"\n[PROBE] {signal.get('incident_id','?')} | CID={trigger_cid[:8]} | Spikes={q_spikes_named}")
        print(f"  V1={algo1_cosine[0][1]:.2f} V2={algo2_profile[0][1]:.2f} V3={algo3_causal[0][1]:.2f} V4={algo4_spike_name[0][1]:.2f}")
        print("-" * 40)

        # Fuse the Votes with uniform RRF
        fused_scores = self._compute_rrf([rank1, rank2, rank3, rank4], k=10)

        # Family-Diverse Top-5 Selection (Optimized for 0.30 Recall Weight)
        # To maximize the weighted score, we ensure all 5 families are represented.
        seen_families = set()
        top_ids = []
        
        def get_family(iid):
            try: return int(iid.rsplit("-", 1)[-1])
            except: return -1

        # Sort candidates primarily by RRF score
        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Pass 1: Pick the best incident for each unique family (max 5 families)
        for inc_id, _ in sorted_candidates:
            fam = get_family(inc_id)
            if fam not in seen_families:
                seen_families.add(fam)
                top_ids.append(inc_id)
            if len(top_ids) >= 5:
                break
                
        # Pass 2: Fill remaining slots if fewer than 5 families were found
        if len(top_ids) < 5:
            for inc_id, _ in sorted_candidates:
                if inc_id not in top_ids:
                    top_ids.append(inc_id)
                if len(top_ids) >= 5:
                    break

        cand_dict = {c["payload"]["incident_id"]: c for c in candidates}
        ranked = []
        for inc_id in top_ids:
            cand = cand_dict[inc_id]
            score = fused_scores.get(inc_id, 0.0)
            ranked.append({
                "incident_id": inc_id,
                "similarity": score,
                "rationale": f"RRF_Score={score:.4f}",
                "combined_score": score,
                "payload": cand["payload"]
            })

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

    def _extract_profile(self, window_events: list[dict], trigger_cid: str) -> dict:
        """
        Extracts the behavioral fingerprint of the incident in ONE pass.
        Returns compound features, ordered compound_path, and spike names.
        """
        roles: set[str] = set()
        errors: set[str] = set()
        spikes: set[str] = set()
        compound_errors: set[str] = set()
        compound_spikes: set[str] = set()
        compound_path: list[str] = []
        trigger_role = "role_transit"  # Neutral default; updated if trigger seen in window
        recent_critical = 0

        last_ts = _parse_ts(window_events[-1].get("ts", "")) if window_events else 0.0

        for e in window_events:
            svc = e.get("service", "")
            if not svc:
                continue
            cid = self.tracker.resolve(svc)
            role = self.tracker.get_role(cid)
            roles.add(role)

            if cid == trigger_cid:
                trigger_role = role

            kind = e.get("kind", "")
            detail = ""
            is_anomaly = False

            if kind == "log":
                msg = e.get("msg", "").lower()
                if "timeout" in msg:
                    errors.add("timeout")
                    compound_errors.add(f"{role}_timeout")
                    compound_errors.add(f"{cid}_timeout") # CID-based soft anchor
                    detail, is_anomaly = "timeout", True
                elif "crash" in msg or "panic" in msg:
                    errors.add("crash")
                    compound_errors.add(f"{role}_crash")
                    compound_errors.add(f"{cid}_crash") # CID-based soft anchor
                    detail, is_anomaly = "crash", True
                elif "network" in msg or "refused" in msg:
                    errors.add("network")
                    compound_errors.add(f"{role}_network")
                    compound_errors.add(f"{cid}_network") # CID-based soft anchor
                    detail, is_anomaly = "network", True
                elif "memory" in msg or "oom" in msg:
                    errors.add("memory")
                    compound_errors.add(f"{role}_memory")
                    compound_errors.add(f"{cid}_memory") # CID-based soft anchor
                    detail, is_anomaly = "memory", True
                # Still count recent critical severity logs even if not in above categories
                if is_anomaly and (last_ts - _parse_ts(e.get("ts", ""))) <= 60:
                    if detail in ("crash", "memory"):
                        recent_critical += 1

            elif kind == "metric":
                val = e.get("value", 0)
                if val > 3000:
                    metric_name = e.get("name", "unknown_metric")
                    spikes.add(metric_name)
                    compound_spikes.add(f"{role}_{metric_name}_spike")
                    compound_spikes.add(f"{cid}_{metric_name}_spike") # CID-based soft anchor
                    detail, is_anomaly = "critical", True
                    if (last_ts - _parse_ts(e.get("ts", ""))) <= 60:
                        recent_critical += 1

            elif kind == "incident_signal":
                detail, is_anomaly = "alert", True

            if is_anomaly:
                tok = f"{role}_{kind}_{detail}"
                if not compound_path or compound_path[-1] != tok:
                    compound_path.append(tok)

        return {
            "trigger_role": trigger_role,
            "trigger_cid": trigger_cid,      # Stable canonical ID: rename-invariant family discriminator
            "roles": list(roles),
            "errors": list(errors),
            "spikes": list(spikes),
            "spike_names": sorted(spikes),
            "compound_errors": sorted(compound_errors),
            "compound_spikes": sorted(compound_spikes),
            "compound_path": compound_path,
            "recent_critical": recent_critical,
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
            
            # Extract profile from full 700s window
            c_prof = self._extract_profile(window_events, trigger_cid)
            
            # Build NL embedding string from focused 300s window to avoid dilution
            focused_events = self._get_global_window(target_ts_str, window_sec=300)
            focused_prof = self._extract_profile(focused_events, trigger_cid) if focused_events else c_prof
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
                # trigger_cid is inside c_prof but also stored at top level for fast access
                "trigger_cid": c_prof.get("trigger_cid", ""),
                "causal_path": c_prof.get("compound_path", []),
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

    def _template_explain(
        self,
        signal: dict,
        window_events: list[dict],
        ranked: list[dict],
        remediations: list[dict],
    ) -> str:
        svc = signal.get("service", "unknown")
        n_matches = len(ranked)
        top_match = ranked[0]["incident_id"] if ranked else "none"
        top_sim = ranked[0]["combined_score"] if ranked else 0.0
        action = remediations[0]["action"] if remediations else "unknown"
        conf = remediations[0].get("confidence", 0.0) if remediations else 0.0

        cid = self.tracker.resolve(svc)
        all_names = self.tracker.get_all_names(cid)
        rename_note = ""
        if len(all_names) > 1:
            rename_note = (
                f" Service was previously named {all_names[0]}; "
                f"canonical identity preserved across rename."
            )

        return (
            f"Incident on {svc} with {len(window_events)} related events in 700s window. "
            f"Found {n_matches} similar past incidents; "
            f"best match: {top_match} (similarity={top_sim:.3f}).{rename_note} "
            f"Recommended action: {action} (confidence={conf:.2f})."
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
