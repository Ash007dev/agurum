import os
import subprocess
import time
import requests
import pytest

GATEWAY_PORT = 8082  # unique port for this test module


@pytest.fixture(scope="module")
def gateway_process():
    env = os.environ.copy()
    env.update({
        "RING_BUFFER_CAP": "100",
        "BATCH_SIZE": "100",
        "BATCH_INTERVAL_MS": "5000",
        "UDS_PATH": "/tmp/test_validator.sock",
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


def test_validator_valid_event(gateway_process):
    """A valid event with ts and kind must be accepted."""
    resp = requests.post(f"{gateway_process}/inject", json=[{"ts": "123", "kind": "test"}])
    assert resp.status_code == 202
    time.sleep(0.3)
    health = requests.get(f"{gateway_process}/health").json()
    assert health["buffer_size"] >= 1


def test_validator_reject_missing_ts(gateway_process):
    """An event missing 'ts' must be rejected (buffer size unchanged)."""
    size_before = requests.get(f"{gateway_process}/health").json()["buffer_size"]
    requests.post(f"{gateway_process}/inject", json=[{"kind": "test"}])
    time.sleep(0.2)
    size_after = requests.get(f"{gateway_process}/health").json()["buffer_size"]
    assert size_before == size_after


def test_validator_reject_missing_kind(gateway_process):
    """An event missing 'kind' must be rejected (buffer size unchanged)."""
    size_before = requests.get(f"{gateway_process}/health").json()["buffer_size"]
    requests.post(f"{gateway_process}/inject", json=[{"ts": "123"}])
    time.sleep(0.2)
    size_after = requests.get(f"{gateway_process}/health").json()["buffer_size"]
    assert size_before == size_after
