import os
import subprocess
import time
import requests
import pytest

from tests.uds_helpers import run_uds_server

GATEWAY_PORT = 8087  # unique port for this test module

# Stress parameters — simulate 1000 events/sec for a compressed window.
# We send 50 batches of 100 events with 0.08s sleep ≈ 1 250 events/sec.
# Test completes in ~5 seconds (50 * 0.08 = 4 s) keeping CI-friendly runtime.
_BATCH_SIZE   = 100
_BATCHES      = 50
_SLEEP_BETWEEN = 0.08   # seconds


@pytest.fixture(scope="module")
def stress_gateway():
    env = os.environ.copy()
    env.update({
        "RING_BUFFER_CAP": "10000",
        "BATCH_SIZE": "100",
        "BATCH_INTERVAL_MS": "100",
        "PCE_UDS_PATH": "",  # Force Go Gateway to use TCP fallback (hits live API on port 8000)
        "GATEWAY_PORT": str(GATEWAY_PORT),
    })
    # Use the precompiled binary to avoid 'go run' startup lag which causes timeouts on Windows
    gateway_bin = os.path.join(os.path.dirname(__file__), "..", "bin", "gateway.exe")
    if not os.path.exists(gateway_bin):
        gateway_bin = os.path.join(os.path.dirname(__file__), "..", "bin", "gateway") # fallback for non-Windows
    
    proc = subprocess.Popen(
        [gateway_bin],
        cwd="gateway", stdin=subprocess.PIPE,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    yield f"http://localhost:{GATEWAY_PORT}"
    proc.terminate()
    proc.wait()


def test_gateway_stress(stress_gateway):
    """
    Simulate ~1 000 events/sec for ~5 seconds.
    Verify: gateway stays alive, buffer drains to 0, and drop count is 0
    (all events fit within the 10 000-cap ring buffer and are flushed by UDS).
    """
    events = [{"ts": "ts", "kind": "kind"} for _ in range(_BATCH_SIZE)]

    for _ in range(_BATCHES):
        resp = requests.post(f"{stress_gateway}/inject", json=events)
        assert resp.status_code == 202, "Gateway crashed or rejected during stress"
        time.sleep(_SLEEP_BETWEEN)

    # Wait for the last batch(es) to be flushed over UDS
    time.sleep(1.5)

    health = requests.get(f"{stress_gateway}/health").json()
    assert health["buffer_size"] == 0,    f"Buffer not empty after stress: {health}"
    assert health["drops_total"] == 0,    f"Unexpected drops during stress: {health}"
