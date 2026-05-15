"""
engine/registry/alias_tracker.py — lightweight rename tracker for the benchmark adapter.

No SQLite, no async — just dicts. Boot time ~0ms.
Handles the from_ field (with underscore) for topology rename events.
"""
from __future__ import annotations

import uuid


class AliasTracker:
    """
    Tracks service name → canonical_id mappings across renames.
    Used in the benchmark adapter (Path A). No dependencies beyond stdlib.

    Critical: rename events use from_ (with underscore), NOT from.
    """

    def __init__(self) -> None:
        self._name_to_cid: dict[str, str] = {}          # current name → canonical_id
        self._cid_to_names: dict[str, list[str]] = {}   # canonical_id → all names (history)

    def process_event(self, event: dict) -> None:
        """Call on every event. Handles registration + rename in one pass."""
        svc = event.get("service")
        if svc and svc not in self._name_to_cid:
            self._register(svc)

        # CRITICAL: rename events use from_ (with underscore)
        if event.get("kind") == "topology" and event.get("change") == "rename":
            old = event.get("from_")   # ← underscore, NOT "from"
            new = event.get("to")
            if old and new:
                self._rename(old, new)

    def resolve(self, name: str) -> str:
        """Return canonical_id for name. Registers if not seen."""
        if not name:
            return self._register("__unknown__")
        if name not in self._name_to_cid:
            self._register(name)
        return self._name_to_cid[name]

    def get_all_names(self, canonical_id: str) -> list[str]:
        """Return all names (including historical) for a canonical_id."""
        return self._cid_to_names.get(canonical_id, [])

    def get_current_name(self, canonical_id: str) -> str:
        """Return the most recent name for a canonical_id."""
        names = self._cid_to_names.get(canonical_id, [])
        return names[-1] if names else canonical_id

    def _register(self, name: str) -> str:
        cid = str(uuid.uuid4())
        self._name_to_cid[name] = cid
        self._cid_to_names[cid] = [name]
        return cid

    def _rename(self, old: str, new: str) -> None:
        if old not in self._name_to_cid:
            # Register old first so history is preserved
            self._register(old)
        cid = self._name_to_cid[old]
        self._name_to_cid[new] = cid
        self._cid_to_names[cid].append(new)
        # old name still valid for historical lookups — don't delete
