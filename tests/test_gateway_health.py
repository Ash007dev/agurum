import os
import subprocess
import time
import requests
import pytest

from tests.uds_helpers import run_uds_server

GATEWAY_PORT = 8084  # unique port for this test module


@pytest.fixture(scope="module")
def health_gateway():
    """Gateway with a long flush interval so events stay buffered during assertions."""
    env = os.environ.copy()
    env.update({
        "RING_BUFFER_CAP": "100",
        "BATCH_SIZE": "10",
        "BATCH_INTERVAL_MS": "5000",
        "UDS_PATH": "/tmp/test_health.sock",
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


def test_gateway_health_status_ok(health_gateway):
    """GET /health must return status=ok with buffer_size and drops_total fields."""
    resp = requests.get(f"{health_gateway}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "buffer_size" in data
    assert "drops_total" in data


def test_gateway_health_buffer_size(health_gateway):
    """Injecting 3 events must be reflected in buffer_size."""
    payload = {"ts": "ts", "kind": "test"}
    requests.post(f"{health_gateway}/inject", json=[payload, payload, payload])
    time.sleep(0.2)
    data = requests.get(f"{health_gateway}/health").json()
    assert data["status"] == "ok"
    assert data["buffer_size"] >= 3
    assert data["drops_total"] == 0
