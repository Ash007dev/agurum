import subprocess

def test_makefile_targets_exist():
    result = subprocess.run(["make", "-n", "run-gateway"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "go build" in result.stdout

def test_makefile_bench_adapter_exists():
    result = subprocess.run(["make", "-n", "bench-adapter"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "self_check.py" in result.stdout
