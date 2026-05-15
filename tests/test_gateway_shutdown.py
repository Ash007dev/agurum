import os
import signal
import subprocess
import tempfile
import time
import requests
import pytest

from tests.uds_helpers import run_uds_server

GATEWAY_PORT = 8085  # unique port for this test module

GATEWAY_DIR = os.path.join(os.path.dirname(__file__), "..", "gateway")


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


def test_gateway_graceful_shutdown():
    """
    Verify SIGTERM causes the gateway to drain its ring buffer via a final
    flush before exiting with returncode 0.

    We compile the gateway binary first with `go build` so that SIGTERM is
    delivered directly to the gateway process (not to a `go run` wrapper).
    When `go run` receives SIGTERM it terminates immediately without giving
    the child process a chance to run its signal handler, yielding exit
    code -15.  Running the pre-built binary avoids this indirection.
    """
    sock_path = "/tmp/test_shutdown.sock"
    srv = run_uds_server(sock_path)

    # Pre-compile the gateway binary into a temp file
    bin_path = tempfile.mktemp(prefix="gateway_shutdown_test_")
    build_result = subprocess.run(
        ["go", "build", "-o", bin_path, "."],
        cwd=GATEWAY_DIR,
        capture_output=True,
        text=True,
    )
    assert build_result.returncode == 0, (
        f"go build failed:\n{build_result.stderr}"
    )

    env = os.environ.copy()
    env.update({
        "RING_BUFFER_CAP": "100",
        "BATCH_SIZE": "100",       # high batch size — won't auto-flush mid-test
        "BATCH_INTERVAL_MS": "5000",  # long interval — won't auto-flush mid-test
        "UDS_PATH": sock_path,
        "GATEWAY_PORT": str(GATEWAY_PORT),
    })

    proc = subprocess.Popen(
        [bin_path],
        stdin=subprocess.PIPE,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    base_url = f"http://localhost:{GATEWAY_PORT}"
    try:
        _wait_for_gateway(base_url)

        # Inject 50 events — they should stay buffered (interval=5000ms, batch=100)
        events = [{"ts": "1", "kind": "test"} for _ in range(50)]
        requests.post(f"{base_url}/inject", json=events)

        health = requests.get(f"{base_url}/health").json()
        assert health["buffer_size"] == 50, (
            f"Expected 50 buffered events, got {health}"
        )

        # Send SIGTERM — gateway must drain and exit cleanly
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)

        # Process must exit with code 0
        assert proc.returncode == 0, (
            f"Expected exit code 0, got {proc.returncode} "
            f"(negative means killed by signal -{proc.returncode})"
        )

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        if os.path.exists(bin_path):
            os.unlink(bin_path)

