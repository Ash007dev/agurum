"""
L3 Mock Scenario Generator
Simulates adversarial conditions the held-out L3 bench uses.

L3 conditions injected:
  - More services (configurable, default 30)
  - More incident families (configurable, default 10)
  - Cascading rename chains (A->B->C->D)
  - Correlated multi-service outages (2-3 services fail together)
  - Families morphed across rename AND dependency-graph shifts
  - Denser background noise (~400 events/service/day)
  - Novel service topologies not seen in training
"""

import random
import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any

BASE_TS = datetime(2026, 1, 1, 0, 0, 0)

# ─── Incident family definitions ───────────────────────────────────────────
# Each family has a signature: which metrics spike, causal depth,
# time-delta profile, remediation type. These are the TRUE discriminators.
# L3 uses 10 families vs L2's 5 — your blanket of 5 will miss 5 families.

FAMILY_SIGNATURES = {
    "F01": {
        "name": "deploy_latency_cascade",
        "metrics_spiked": ["latency_p99_ms"],
        "causal_depth": 2,
        "deploy_to_spike_s": (45, 90),       # (min, max)
        "spike_to_incident_s": (120, 300),
        "remediation": "rollback",
        "upstream_errors": True,
        "multi_service": False,
    },
    "F02": {
        "name": "memory_oom_cascade",
        "metrics_spiked": ["memory_rss_mb", "gc_pause_ms"],
        "causal_depth": 3,
        "deploy_to_spike_s": (300, 600),
        "spike_to_incident_s": (600, 1200),
        "remediation": "restart",
        "upstream_errors": True,
        "multi_service": False,
    },
    "F03": {
        "name": "db_connection_pool_exhaustion",
        "metrics_spiked": ["db_pool_active", "latency_p99_ms"],
        "causal_depth": 2,
        "deploy_to_spike_s": (60, 120),
        "spike_to_incident_s": (60, 180),
        "remediation": "config_change",
        "upstream_errors": False,
        "multi_service": False,
    },
    "F04": {
        "name": "correlated_multi_service_outage",   # L3 SPECIAL
        "metrics_spiked": ["error_rate", "latency_p99_ms"],
        "causal_depth": 4,
        "deploy_to_spike_s": (30, 60),
        "spike_to_incident_s": (30, 90),
        "remediation": "rollback",
        "upstream_errors": True,
        "multi_service": True,     # 2-3 services fail simultaneously
    },
    "F05": {
        "name": "traffic_spike_no_deploy",
        "metrics_spiked": ["qps", "latency_p99_ms", "cpu_percent"],
        "causal_depth": 1,
        "deploy_to_spike_s": None,  # no deploy trigger
        "spike_to_incident_s": (300, 900),
        "remediation": "scale_out",
        "upstream_errors": False,
        "multi_service": False,
    },
    "F06": {
        "name": "certificate_expiry_cascade",
        "metrics_spiked": ["tls_handshake_ms", "error_rate"],
        "causal_depth": 3,
        "deploy_to_spike_s": None,
        "spike_to_incident_s": (1800, 3600),
        "remediation": "cert_rotate",
        "upstream_errors": True,
        "multi_service": False,
    },
    "F07": {
        "name": "cascading_rename_chain_failure",    # L3 SPECIAL
        "metrics_spiked": ["latency_p99_ms"],
        "causal_depth": 5,
        "deploy_to_spike_s": (90, 180),
        "spike_to_incident_s": (180, 420),
        "remediation": "rollback",
        "upstream_errors": True,
        "multi_service": False,
        "rename_chain_length": 4,   # A->B->C->D within incident window
    },
    "F08": {
        "name": "dependency_graph_shift_failure",    # L3 SPECIAL
        "metrics_spiked": ["error_rate"],
        "causal_depth": 3,
        "deploy_to_spike_s": (120, 240),
        "spike_to_incident_s": (240, 600),
        "remediation": "rollback",
        "upstream_errors": True,
        "multi_service": True,
        "dep_graph_mutates": True,   # dependency rewired mid-incident
    },
    "F09": {
        "name": "slow_memory_leak",
        "metrics_spiked": ["memory_rss_mb", "latency_p99_ms"],
        "causal_depth": 2,
        "deploy_to_spike_s": (3600, 7200),   # very slow — hours not minutes
        "spike_to_incident_s": (3600, 7200),
        "remediation": "restart",
        "upstream_errors": False,
        "multi_service": False,
    },
    "F10": {
        "name": "thundering_herd_retry_storm",
        "metrics_spiked": ["qps", "error_rate", "latency_p99_ms"],
        "causal_depth": 4,
        "deploy_to_spike_s": (30, 120),
        "spike_to_incident_s": (120, 300),
        "remediation": "circuit_break",
        "upstream_errors": True,
        "multi_service": True,
    },
}


class L3Generator:
    def __init__(self, seed: int, n_services: int = 30, days: int = 14,
                 n_families: int = 10, n_train_incidents: int = 40,
                 n_eval_incidents: int = 20):
        self.rng = random.Random(seed)
        self.seed = seed
        self.n_services = n_services
        self.days = days
        self.n_families = min(n_families, len(FAMILY_SIGNATURES))
        self.n_train = n_train_incidents
        self.n_eval = n_eval_incidents

        self.families = list(FAMILY_SIGNATURES.keys())[:self.n_families]
        self.alias_map: Dict[str, str] = {}
        self.rename_log: List[Dict] = []
        self.dep_graph: Dict[str, List[str]] = {}
        self.events: List[Dict] = []
        self.ground_truth: Dict[str, Dict] = {}
        self.services = self._generate_services()


    # ── Service and topology generation ────────────────────────────────────

    def _generate_services(self) -> List[str]:
        prefixes = ["auth","payments","billing","orders","cart","inventory",
                    "gateway","search","notify","recommend","catalog","shipping",
                    "fraud","analytics","reporting","cache","queue","scheduler",
                    "config","identity","ledger","reconcile","webhook","audit"]
        suffixes = ["svc","api","service","worker","processor","handler"]
        svcs = []
        for i in range(self.n_services):
            p = self.rng.choice(prefixes)
            s = self.rng.choice(suffixes)
            name = f"{p}-{s}-{i:02d}"
            svcs.append(name)
            self.alias_map[name] = name
        return svcs

    def _ts(self, offset_s: float) -> str:
        t = BASE_TS + timedelta(seconds=offset_s)
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _rename_service(self, svc: str, offset_s: float) -> str:
        """Rename a service, updating alias map."""
        suffixes = ["svc","api","service","worker","processor"]
        new_name = svc.split("-")[0] + "-" + self.rng.choice(suffixes) + \
                   "-" + str(self.rng.randint(10, 99))
        canonical = self.alias_map.get(svc, svc)
        self.alias_map[new_name] = canonical
        self.alias_map[svc] = canonical
        ev = {
            "ts": self._ts(offset_s),
            "kind": "topology",
            "change": "rename",
            "from": svc,
            "to": new_name,
        }
        self.events.append(ev)
        self.rename_log.append({"from": svc, "to": new_name, "at": offset_s})
        return new_name

    def _cascade_rename(self, svc: str, offset_s: float,
                        chain_len: int = 4) -> str:
        """Rename A->B->C->D in quick succession (L3 adversarial)."""
        current = svc
        for i in range(chain_len):
            current = self._rename_service(current, offset_s + i * 30)
        return current

    def _shift_dependency(self, svc: str, new_dep: str, offset_s: float):
        """Rewire dependency graph mid-incident (L3 adversarial)."""
        if svc in self.dep_graph:
            old_deps = self.dep_graph[svc]
        else:
            old_deps = []
        self.dep_graph[svc] = [new_dep] + [d for d in old_deps if d != new_dep]
        self.events.append({
            "ts": self._ts(offset_s),
            "kind": "topology",
            "change": "dependency_shift",
            "from": svc,
            "to": new_dep,
        })

    # ── Background noise generation ─────────────────────────────────────────

    def _generate_background(self):
        """~400 events per service per day of normal operation."""
        total_seconds = self.days * 86400
        for svc in self.services:
            n_events = int(self.rng.gauss(400 * self.days, 50))
            for _ in range(n_events):
                offset = self.rng.uniform(0, total_seconds)
                kind = self.rng.choices(
                    ["metric", "log", "trace"],
                    weights=[0.5, 0.3, 0.2]
                )[0]
                if kind == "metric":
                    self.events.append({
                        "ts": self._ts(offset),
                        "kind": "metric",
                        "service": svc,
                        "name": self.rng.choice(
                            ["latency_p99_ms","qps","error_rate","cpu_percent","memory_rss_mb"]),
                        "value": self.rng.gauss(100, 20),
                    })
                elif kind == "log":
                    self.events.append({
                        "ts": self._ts(offset),
                        "kind": "log",
                        "service": svc,
                        "level": "info",
                        "msg": f"routine operation on {svc}",
                    })
                else:
                    caller = self.rng.choice(self.services)
                    self.events.append({
                        "ts": self._ts(offset),
                        "kind": "trace",
                        "trace_id": f"bg-{self.rng.randint(0,999999):06d}",
                        "spans": [
                            {"svc": caller, "dur_ms": self.rng.randint(5,50)},
                            {"svc": svc, "dur_ms": self.rng.randint(2,30)},
                        ],
                    })

    # ── Incident generation ─────────────────────────────────────────────────

    def _generate_incident(self, incident_id: str, family_id: str,
                           base_offset_s: float, is_eval: bool = False) -> Dict:
        """Generate one incident with full telemetry matching the family signature."""
        sig = FAMILY_SIGNATURES[family_id]
        svc = self.rng.choice(self.services)
        current_name = svc
        deploy_id = f"ev-deploy-{incident_id}"
        ev_ids = []

        # Optional deploy trigger
        if sig["deploy_to_spike_s"] is not None:
            deploy_offset = base_offset_s
            version = f"v{self.rng.randint(1,9)}.{self.rng.randint(0,99)}.{self.rng.randint(0,9)}"
            deploy_ev = {
                "_id": deploy_id,
                "ts": self._ts(deploy_offset),
                "kind": "deploy",
                "service": current_name,
                "version": version,
                "actor": "ci",
            }
            self.events.append(deploy_ev)
            ev_ids.append(deploy_id)

            d_min, d_max = sig["deploy_to_spike_s"]
            spike_offset = deploy_offset + self.rng.uniform(d_min, d_max)
        else:
            spike_offset = base_offset_s + self.rng.uniform(300, 900)

        # L3 special: cascading rename chain DURING incident
        if sig.get("rename_chain_length"):
            current_name = self._cascade_rename(
                current_name, spike_offset - 60, sig["rename_chain_length"])

        # L3 special: dependency graph shift during incident
        if sig.get("dep_graph_mutates") and len(self.services) > 2:
            new_dep = self.rng.choice([s for s in self.services if s != svc])
            self._shift_dependency(current_name, new_dep, spike_offset - 30)

        # Metric spikes
        metric_ids = []
        for metric_name in sig["metrics_spiked"]:
            mid = f"ev-metric-{incident_id}-{metric_name}"
            self.events.append({
                "_id": mid,
                "ts": self._ts(spike_offset),
                "kind": "metric",
                "service": current_name,
                "name": metric_name,
                "value": self.rng.uniform(3000, 9000),
            })
            metric_ids.append(mid)
            ev_ids.append(mid)

        # Multi-service correlated outage (L3 adversarial)
        if sig.get("multi_service") and len(self.services) > 2:
            extra_svcs = self.rng.sample(
                [s for s in self.services if s != svc],
                min(2, len(self.services) - 1)
            )
            for esvc in extra_svcs:
                mid2 = f"ev-metric-{incident_id}-{esvc}-corr"
                self.events.append({
                    "_id": mid2,
                    "ts": self._ts(spike_offset + self.rng.uniform(5, 30)),
                    "kind": "metric",
                    "service": esvc,
                    "name": sig["metrics_spiked"][0],
                    "value": self.rng.uniform(2000, 7000),
                })
                ev_ids.append(mid2)

        # Upstream caller errors
        if sig["upstream_errors"] and len(self.services) > 1:
            caller = self.rng.choice([s for s in self.services if s != svc])
            trace_id = f"tr-{incident_id}"
            lid = f"ev-log-{incident_id}"
            self.events.append({
                "_id": lid,
                "ts": self._ts(spike_offset + 10),
                "kind": "log",
                "service": caller,
                "level": "error",
                "msg": f"timeout calling {current_name}",
                "trace_id": trace_id,
            })
            self.events.append({
                "ts": self._ts(spike_offset + 15),
                "kind": "trace",
                "trace_id": trace_id,
                "spans": [
                    {"svc": caller, "dur_ms": self.rng.randint(4000, 8000)},
                    {"svc": current_name, "dur_ms": self.rng.randint(3500, 7500)},
                ],
            })
            ev_ids.append(lid)

        # Incident signal
        i_min, i_max = sig["spike_to_incident_s"]
        incident_offset = spike_offset + self.rng.uniform(i_min, i_max)
        signal_ev = {
            "_id": f"ev-signal-{incident_id}",
            "ts": self._ts(incident_offset),
            "kind": "incident_signal",
            "incident_id": incident_id,
            "trigger": f"alert:{current_name}/{sig['metrics_spiked'][0]}>threshold",
        }
        self.events.append(signal_ev)

        # Remediation (for training incidents only)
        if not is_eval:
            rem_offset = incident_offset + self.rng.uniform(300, 900)
            self.events.append({
                "ts": self._ts(rem_offset),
                "kind": "remediation",
                "incident_id": incident_id,
                "action": sig["remediation"],
                "target": current_name,
                "historical_outcome": "resolved",
                "outcome": "resolved",
            })

        return {
            "incident_id": incident_id,
            "family_id": family_id,
            "service_canonical": self.alias_map.get(svc, svc),
            "is_eval": is_eval,
            "signal_ts": self._ts(incident_offset),
            "signal": signal_ev,
            "ground_truth_family_ids": [
                inc_id for inc_id, meta in self.ground_truth.items()
                if meta["family_id"] == family_id and not meta["is_eval"]
            ],
        }

    # ── Topology mutations (renames between incidents) ──────────────────────

    def _inject_topology_mutations(self, n_mutations: int = 15):
        """Rename services at random times — denser than L2's 8."""
        total_s = self.days * 86400
        for _ in range(n_mutations):
            svc = self.rng.choice(self.services)
            offset = self.rng.uniform(total_s * 0.1, total_s * 0.8)
            new_name = self._rename_service(svc, offset)
            # Update services list with new name
            if svc in self.services:
                idx = self.services.index(svc)
                self.services[idx] = new_name

    # ── Full dataset generation ─────────────────────────────────────────────

    def generate(self) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Returns:
            train_events: all events to ingest before eval
            eval_signals: list of IncidentSignal dicts to query
            ground_truth: {incident_id: {family_id, similar_incident_ids}}
        """
        total_s = self.days * 86400

        # Generate background noise
        self._generate_background()

        # Inject topology mutations (denser than L2)
        self._inject_topology_mutations(n_mutations=15)

        # Generate TRAIN incidents spread across first 70% of timeline
        train_window = total_s * 0.70
        train_incidents = []
        for i in range(self.n_train):
            family = self.rng.choice(self.families)
            inc_id = f"INC-TRAIN-{self.seed}-{i:04d}"
            offset = self.rng.uniform(0, train_window)
            meta = self._generate_incident(inc_id, family, offset, is_eval=False)
            meta["is_eval"] = False
            self.ground_truth[inc_id] = meta
            train_incidents.append(meta)

        # Generate EVAL incidents in last 30% of timeline
        eval_window_start = total_s * 0.70
        eval_incidents = []
        for i in range(self.n_eval):
            family = self.rng.choice(self.families)
            inc_id = f"INC-EVAL-{self.seed}-{i:04d}"
            offset = self.rng.uniform(eval_window_start, total_s)
            meta = self._generate_incident(inc_id, family, offset, is_eval=True)
            meta["is_eval"] = True
            # Ground truth: which TRAIN incidents are in same family?
            meta["ground_truth_family_ids"] = [
                t["incident_id"] for t in train_incidents
                if t["family_id"] == family
            ]
            self.ground_truth[inc_id] = meta
            eval_incidents.append(meta)

        # Sort all events by timestamp
        self.events.sort(key=lambda e: e["ts"])

        # Strip internal _id fields before returning (keep in ground truth only)
        clean_events = [{k: v for k, v in e.items() if k != "_id"}
                        for e in self.events]

        eval_signals = [m["signal"] for m in eval_incidents]

        return clean_events, eval_signals, self.ground_truth
