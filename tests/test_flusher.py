import os
import socket
import threading
import subprocess
import time
import requests
import pytest


def _wait_for_gateway(url: str, retries: int = 40, delay: float = 0.3):
    """Poll /health until the gateway is up, or raise after timeout."""
    for _ in range(retries):
        try:
            r = requests.get(f"{url}/health", timeout=1)
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(delay)
    raise RuntimeError(f"Gateway at {url} did not start within {retries * delay:.1f}s")

from tests.uds_helpers import run_uds_server

GATEWAY_PORT = 8083  # unique port for this test module


@pytest.fixture(scope="module")
def flusher_gateway():
    """
    Spin up a UDS mock server + gateway configured with BATCH_SIZE=10 and
    BATCH_INTERVAL_MS=300 so we can test both flush triggers.
    """
    sock_path = "/tmp/test_flusher.sock"
    srv = run_uds_server(sock_path)
    env = os.environ.copy()
    env.update({
        "RING_BUFFER_CAP": "100",
        "BATCH_SIZE": "10",
        "BATCH_INTERVAL_MS": "300",
        "UDS_PATH": sock_path,
        "GATEWAY_PORT": str(GATEWAY_PORT),
    })
    proc = subprocess.Popen(
        ["go", "run", "."],
        cwd="gateway", stdin=subprocess.PIPE,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://localhost:{GATEWAY_PORT}"
    _wait_for_gateway(base_url)
    yield base_url
    proc.terminate()
    proc.wait()
    srv.close()
    if os.path.exists(sock_path):
        os.unlink(sock_path)


def test_flusher_by_size(flusher_gateway):
    """Injecting exactly BATCH_SIZE events must trigger an immediate flush."""
    events = [{"ts": "1", "kind": "test"} for _ in range(10)]
    requests.post(f"{flusher_gateway}/inject", json=events)
    # Give the flusher goroutine a moment to detect size >= batch and flush
    time.sleep(0.5)
    health = requests.get(f"{flusher_gateway}/health").json()
    assert health["buffer_size"] == 0


def test_flusher_by_time(flusher_gateway):
    """Fewer than BATCH_SIZE events must be flushed after the interval elapses."""
    events = [{"ts": "1", "kind": "test"} for _ in range(5)]
    requests.post(f"{flusher_gateway}/inject", json=events)

    # Before interval fires the events should still be buffered
    health = requests.get(f"{flusher_gateway}/health").json()
    assert health["buffer_size"] == 5

    # Wait for the 300ms interval to fire
    time.sleep(0.6)
    health = requests.get(f"{flusher_gateway}/health").json()
    assert health["buffer_size"] == 0
