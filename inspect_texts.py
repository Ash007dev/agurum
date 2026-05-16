import json
import numpy as np
from engine.store.in_memory_store import InMemoryStore
from engine.registry.alias_tracker import AliasTracker
from engine.synthesis.episode_synthesizer import _event_to_string

store = InMemoryStore()
tracker = AliasTracker()

with open("bench-p02-context/data/train_events.jsonl", "r") as f:
    for line in f:
        e = json.loads(line)
        tracker.process_event(e)
        store.append(e)

with open("bench-p02-context/data/eval_events.jsonl", "r") as f:
    for line in f:
        e = json.loads(line)
        tracker.process_event(e)
        store.append(e)

def get_strings_for_inc(inc_id):
    signals = [e for e in store._events if e.get("incident_id") == inc_id and e.get("kind") == "incident_signal"]
    if not signals: return []
    signal = signals[0]
    
    ts = float(signal["ts"].replace("Z", "+00:00").replace("T", " ").split()[1].replace(":", "")) # rough
    from datetime import datetime
    ts = datetime.fromisoformat(signal["ts"].replace("Z", "+00:00")).timestamp()
    
    window_events = []
    for e in store._events:
        try:
            e_ts = datetime.fromisoformat(e["ts"].replace("Z", "+00:00")).timestamp()
            if ts - 300 <= e_ts <= ts:
                window_events.append(e)
        except:
            pass
            
    signal_svc = signal.get("service", "")
    trigger_cid = tracker.resolve(signal_svc)
    local_topology = {trigger_cid: "node_trigger"}
    peer_counter = 1
    
    strings = []
    for e in window_events[-30:]:
        cid = tracker.resolve(e.get("service", ""))
        if cid not in local_topology:
            local_topology[cid] = f"node_peer_{peer_counter}"
            peer_counter += 1
        local_id = local_topology[cid]
        strings.append(_event_to_string(e, local_node_id=local_id))
    return strings

print("INC-18301-0:")
for s in get_strings_for_inc("INC-18301-0"): print("  ", s)

print("\nINC-89426-1:")
for s in get_strings_for_inc("INC-89426-1"): print("  ", s)

