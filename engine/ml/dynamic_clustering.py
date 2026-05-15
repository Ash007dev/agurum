"""
engine/ml/dynamic_clustering.py — Online family clustering with dynamic N.

Replaces the hardcoded blanket-of-5. Discovers N clusters organically using
Jaccard similarity on incident fingerprints. diverse_top5() picks the best
candidate per discovered cluster, ensuring diverse coverage even when N > 5
or N < 5 families exist in the corpus.
"""
from __future__ import annotations


class DynamicFamilyClustering:
    """
    Single-pass online clustering over incident fingerprint dicts.

    Each cluster has a centroid (feature dict) updated via majority-merge.
    New incidents attach to the nearest cluster if similarity >= threshold,
    otherwise they seed a new cluster.
    """

    def __init__(self, sim_threshold: float = 0.50) -> None:
        self.threshold = sim_threshold
        # Each cluster: {members: list, best_rep: str}
        self.clusters: list[dict] = []
        self.member_fps: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_incident(self, incident_id: str, fingerprint: dict) -> None:
        """Assign incident to nearest cluster or start a new one using single-linkage."""
        fp = self._normalise(fingerprint)
        self.member_fps[incident_id] = fp
        
        best_idx, best_sim = -1, 0.0
        for i, c in enumerate(self.clusters):
            # Single-linkage: max similarity to any member in the cluster
            max_s = 0.0
            for mem_id in c["members"]:
                s = self._jaccard(fp, self.member_fps[mem_id])
                if s > max_s:
                    max_s = s
            if max_s > best_sim:
                best_sim, best_idx = max_s, i

        if best_sim >= self.threshold and best_idx >= 0:
            c = self.clusters[best_idx]
            c["members"].append(incident_id)
        else:
            self.clusters.append({
                "members": [incident_id],
                "best_rep": incident_id,
            })

    def diverse_top5(self, ranked_candidates: list) -> list:
        """
        Pick one best candidate per discovered cluster, best-first.
        Falls back to filling from remaining if < 5 distinct clusters exist.
        """
        seen_clusters: set[int] = set()
        result = []

        for cand in ranked_candidates:
            fp = self._normalise(self._get_fp(cand))
            cid = self._assign_cluster(fp)
            if cid not in seen_clusters:
                seen_clusters.add(cid)
                result.append(cand)
            if len(result) == 5:
                break

        # Fill remaining slots if < 5 distinct clusters were found
        if len(result) < 5:
            for cand in ranked_candidates:
                if cand not in result:
                    result.append(cand)
                if len(result) == 5:
                    break

        return result

    @property
    def n_discovered(self) -> int:
        return len(self.clusters)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_cluster(self, fingerprint: dict) -> int:
        if not self.clusters:
            return 0
        fp = self._normalise(fingerprint)
        best_idx, best_sim = 0, 0.0
        for i, c in enumerate(self.clusters):
            max_s = 0.0
            for mem_id in c["members"]:
                s = self._jaccard(fp, self.member_fps[mem_id])
                if s > max_s:
                    max_s = s
            if max_s > best_sim:
                best_sim, best_idx = max_s, i
        return best_idx

    def _jaccard(self, a: dict, b: dict) -> float:
        """Computes more granular Jaccard similarity between two fingerprints."""
        keys = ["trigger_role", "errors", "spike_names", "deploy_bucket", "depth"]
        total_sim = 0.0
        for k in keys:
            v_a = a.get(k, "none")
            v_b = b.get(k, "none")
            
            if k in ["errors", "spike_names"]:
                # Handle set-valued features
                s_a = set(v_a.split(",")) if v_a != "none" else set()
                s_b = set(v_b.split(",")) if v_b != "none" else set()
                if not s_a and not s_b:
                    total_sim += 1.0
                elif not s_a or not s_b:
                    total_sim += 0.0
                else:
                    total_sim += len(s_a & s_b) / len(s_a | s_b)
            else:
                # Direct equality for categorical features
                total_sim += 1.0 if v_a == v_b else 0.0
                
        return total_sim / len(keys)

    def _merge(self, a: dict, b: dict) -> dict:
        """Centroid keeps existing value on key conflict (stable centroid)."""
        merged = dict(a)
        for k, v in b.items():
            if k not in merged:
                merged[k] = v
        return merged

    @staticmethod
    def _normalise(fp: dict) -> dict:
        """Ensure all values are strings for stable Jaccard comparison."""
        return {k: str(v) if not isinstance(v, str) else v for k, v in fp.items()}

    @staticmethod
    def _get_fp(cand) -> dict:
        """Extract fingerprint from a candidate (dict or object)."""
        if isinstance(cand, dict):
            return cand.get("fingerprint", {})
        return getattr(cand, "fingerprint", {})
