import os
import subprocess
import time
import requests
import pytest

GATEWAY_PORT = 8081  # unique port for this test module


@pytest.fixture(scope="module")
def gateway_process():
    # Start Go gateway with tiny ring buffer to exercise capacity/drop logic.
    # BATCH_INTERVAL_MS is very high so the flusher never fires during test.
    env = os.environ.copy()
    env.update({
        "RING_BUFFER_CAP": "10",
        "BATCH_SIZE": "100",
        "BATCH_INTERVAL_MS": "5000",
        "UDS_PATH": "/tmp/test_ring_buffer.sock",
        "GATEWAY_PORT": str(GATEWAY_PORT),
    })
    subprocess.run(["go", "build", "-o", "test_bin", "."], cwd="gateway", check=True)
    proc = subprocess.Popen(
        ["./test_bin"],
        cwd="gateway", stdin=subprocess.PIPE,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://localhost:{GATEWAY_PORT}"
    for _ in range(50):
        try:
            if requests.get(f"{url}/health").status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    yield url
    proc.terminate()
    proc.wait()


def test_ring_buffer_capacity_and_drops(gateway_process):
    """Inject 15 events into a capacity-10 buffer; expect 10 buffered, 5 dropped."""
    events = [{"ts": f"100{i}", "kind": "test"} for i in range(15)]

    resp = requests.post(f"{gateway_process}/inject", json=events)
    assert resp.status_code == 202

    health = requests.get(f"{gateway_process}/health").json()
    assert health["buffer_size"] == 10
    assert health["drops_total"] == 5
