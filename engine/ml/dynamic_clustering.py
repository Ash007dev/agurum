"""
engine/ml/dynamic_clustering.py — Online family clustering with flat-token dynamic drift.

Fix 2: Replaces the single-linkage member-scan approach with flat unrolled token sets.
Compound fingerprint tuples (errors, spike_names) are expanded into discrete k:v tokens
before Jaccard comparison to prevent structural evaluation collapse on multi-value fields.

Cluster centroids accumulate tokens via features.update() (dynamic drift), so the cluster
grows to absorb the vocabulary of every admitted member — boosting recall for drifting
incident families over time.
"""
from __future__ import annotations


class DynamicFamilyClustering:
    """
    Single-pass online clustering over incident fingerprint dicts.

    Each cluster stores a flat token set (centroid) updated via dynamic drift.
    New incidents attach to the nearest cluster if Jaccard(flat_feats, centroid)
    >= threshold, otherwise they seed a new cluster.

    Fix 2: _to_flat_set unrolls compound tuples/lists into discrete k:item tokens
    so Jaccard operates on atomic tokens rather than opaque compound strings.
    """

    def __init__(self, sim_threshold: float = 0.50) -> None:
        self.threshold = sim_threshold
        # Each cluster: {"features": set(), "members": list()}
        self.clusters: list[dict] = []
        self.member_fps: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_incident(self, incident_id: str, fingerprint: dict) -> None:
        """
        Assign incident to nearest cluster or start a new one.

        Uses flat token Jaccard against the cluster centroid pool, then
        accumulates the incident's tokens into the winning centroid (dynamic drift).
        """
        fp = self._normalise(fingerprint)
        self.member_fps[incident_id] = fp

        flat_feats = self._to_flat_set(fp)
        best_idx, best_sim = -1, -1.0

        for i, c in enumerate(self.clusters):
            intersection = len(flat_feats & c["features"])
            union = len(flat_feats | c["features"])
            sim = intersection / union if union > 0 else 0.0
            if sim > best_sim:
                best_sim, best_idx = sim, i

        if best_sim >= self.threshold:
            self.clusters[best_idx]["members"].append(incident_id)
            # Dynamic Drift: accumulate new signature tokens into the cluster pool
            self.clusters[best_idx]["features"].update(flat_feats)
        else:
            self.clusters.append({"features": flat_feats, "members": [incident_id]})

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

    def _to_flat_set(self, fingerprint: dict) -> set:
        """
        Unroll compound tuple/list values into discrete k:item tokens.

        Example: {"errors": "timeout,crash"} → {"errors:timeout", "errors:crash"}
        This prevents Jaccard from treating multi-value strings as opaque atoms.
        """
        flat: set = set()
        for k, v in fingerprint.items():
            if isinstance(v, (tuple, list)):
                for item in v:
                    flat.add(f"{k}:{item}")
            elif isinstance(v, str) and "," in v:
                # Normalised CSV fields (e.g. "timeout,crash") → unroll
                for item in v.split(","):
                    item = item.strip()
                    if item:
                        flat.add(f"{k}:{item}")
            else:
                flat.add(f"{k}:{v}")
        return flat

    def _assign_cluster(self, fingerprint: dict) -> int:
        """Return index of the closest cluster by flat-token Jaccard."""
        if not self.clusters:
            return 0
        fp = self._normalise(fingerprint)
        flat_feats = self._to_flat_set(fp)
        best_idx, best_sim = 0, -1.0
        for i, c in enumerate(self.clusters):
            intersection = len(flat_feats & c["features"])
            union = len(flat_feats | c["features"])
            sim = intersection / union if union > 0 else 0.0
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return best_idx

    def _jaccard(self, a: dict, b: dict) -> float:
        """
        Flat-token Jaccard between two normalised fingerprint dicts.
        Used by the RRF voter (Voter 3) in agurum.py.
        """
        flat_a = self._to_flat_set(self._normalise(a))
        flat_b = self._to_flat_set(self._normalise(b))
        if not flat_a and not flat_b:
            return 1.0
        union = len(flat_a | flat_b)
        return len(flat_a & flat_b) / union if union > 0 else 0.0

    @staticmethod
    def _normalise(fp: dict) -> dict:
        """Ensure all values are strings for stable comparison."""
        return {k: str(v) if not isinstance(v, str) else v for k, v in fp.items()}

    @staticmethod
    def _get_fp(cand) -> dict:
        """Extract fingerprint from a candidate (dict or object)."""
        if isinstance(cand, dict):
            return cand.get("fingerprint", {})
        return getattr(cand, "fingerprint", {})
