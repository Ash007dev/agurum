import os
import socket
import threading
import subprocess
import time
import requests
import pytest

from tests.uds_helpers import run_uds_server

GATEWAY_PORT = 8086  # unique port for this test module


@pytest.fixture(scope="module")
def uds_server():
    sock_path = "/tmp/test_uds_roundtrip.sock"
    srv = run_uds_server(sock_path)
    yield sock_path
    srv.close()
    if os.path.exists(sock_path):
        os.unlink(sock_path)


@pytest.fixture(scope="module")
def gateway_process(uds_server):
    env = os.environ.copy()
    env.update({
        "RING_BUFFER_CAP": "100",
        "BATCH_SIZE": "10",
        "BATCH_INTERVAL_MS": "100",
        "UDS_PATH": uds_server,
        "GATEWAY_PORT": str(GATEWAY_PORT),
    })
    proc = subprocess.Popen(
        ["go", "run", "."],
        cwd="gateway", stdin=subprocess.PIPE,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    yield f"http://localhost:{GATEWAY_PORT}"
    proc.terminate()
    proc.wait()


def test_uds_roundtrip_flush(gateway_process):
    """
    Inject exactly BATCH_SIZE events — the flusher must POST them over UDS
    and the buffer must be empty afterwards.
    """
    events = [{"ts": "1", "kind": "k"} for _ in range(10)]
    resp = requests.post(f"{gateway_process}/inject", json=events)
    assert resp.status_code == 202
    # Give flusher time to detect batch full and POST over UDS
    time.sleep(0.5)
    health = requests.get(f"{gateway_process}/health").json()
    assert health["buffer_size"] == 0
