"""
Lightweight rename tracker for the benchmark adapter (Path A).

No SQLite, no async, no external dependencies — plain dicts only.
Boot time: ~0ms (just dict allocation).

CRITICAL: rename events use "from_" (with underscore), NOT "from".
Python reserves "from" as a keyword. The benchmark generator outputs
events with key "from_". Reading event.get("from") returns None and
silently breaks all renames.
"""
from __future__ import annotations

import uuid


class AliasTracker:
    """
    Maps service names (current and historical) to stable canonical UUIDs.

    A rename from "payments-svc" to "billing-svc" makes both names resolve
    to the same canonical_id. Old names are NEVER deleted — historical
    lookups across the full benchmark window must work.

    Thread safety: NOT required. Used only in synchronous benchmark path.
    """

    def __init__(self) -> None:
        self._name_to_cid: dict[str, str] = {}
        self._cid_to_names: dict[str, list[str]] = {}

    def process_event(self, event: dict) -> None:
        """
        Call on every event in ingest(). Two jobs:
        1. Register unseen services.
        2. Handle rename topology events (reads from_, NOT from).

        Never raises. Silently skips malformed events.
        """
        # Job 1: register unseen services
        svc = event.get("service")
        if svc and svc not in self._name_to_cid:
            self._register(svc)

        # Job 2: handle rename events
        # CRITICAL: "from_" with underscore — NOT "from"
        if event.get("kind") == "topology" and event.get("change") == "rename":
            old = event.get("from_")   # ← UNDERSCORE
            new = event.get("to")
            if old and new:
                self._rename(old, new)

    def resolve(self, name: str) -> str:
        """
        Returns canonical_id for name. Auto-registers if not seen.
        NEVER raises. Always returns a string.
        """
        if name not in self._name_to_cid:
            return self._register(name)
        return self._name_to_cid[name]

    def get_family_id(self, name: str) -> str:
        """Alias for resolve(). Used by callers who prefer this name."""
        return self.resolve(name)

    def get_current_name(self, canonical_id: str) -> str:
        """Return the most recent name for a canonical_id."""
        return self._cid_to_names.get(canonical_id, ["unknown"])[-1]

    def get_all_names(self, canonical_id: str) -> list[str]:
        """Return all names (historical + current) for a canonical_id."""
        return self._cid_to_names.get(canonical_id, []).copy()

    def known_services(self) -> list[str]:
        """Return all known service names (current and historical)."""
        return list(self._name_to_cid.keys())

    def _register(self, name: str) -> str:
        """Register a new service name with a fresh canonical UUID."""
        cid = str(uuid.uuid4())
        self._name_to_cid[name] = cid
        self._cid_to_names[cid] = [name]
        return cid

    def _rename(self, old: str, new: str) -> None:
        """
        Map new name to same canonical_id as old name.
        DO NOT delete old name — historical lookups must still work.
        """
        if old not in self._name_to_cid:
            # Old name never seen — register it first, then rename
            self._register(old)
        cid = self._name_to_cid[old]
        self._name_to_cid[new] = cid   # new name → same cid
        # Track name history (ordered: first=original, last=current)
        if new not in self._cid_to_names.get(cid, []):
            self._cid_to_names[cid].append(new)
