#!/usr/bin/env bash
# bench/run.sh — Mandatory evaluation script for the L3-Adversarial Benchmark
#
# This script is invoked by the judges' automated evaluation harness.
# It ingests the published sample, runs the canonical scenario,
# and emits a JSON report matching the SDK schema.
#
# Usage: ./bench/run.sh

set -e

echo "================================================================"
echo " Starting Agurum Persistent Context Engine Evaluation"
echo "================================================================"

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Check if the python virtual environment exists, if not, attempt to use system python
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python"
    # Fallback for Windows environments
    if [ ! -f "$PYTHON_BIN" ]; then
        PYTHON_BIN=".venv/Scripts/python.exe"
    fi
else
    PYTHON_BIN="python3"
fi

echo "[*] Using Python: $PYTHON_BIN"

# --- Evaluator API Key Fallback ---
# We inject the Groq API key directly so the judges can evaluate the
# Deep Mode SRE Narratives without needing to set up their own accounts.
export GROQ_API_KEY="${GROQ_API_KEY:-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}"

# Execute all benchmark sequences (L2 and L3)
echo "[*] Running bench-adapter (quick check)..."
cd bench-p02-context
$PYTHON_BIN self_check.py --adapter adapters.agurum:Engine --quick

echo "[*] Running bench-full (canonical Path A benchmark)..."
$PYTHON_BIN run.py --adapter adapters.agurum:Engine --mode fast --seeds 42 101 202 303 404 --out ../report.json

echo "[*] Running bench-deep (deep mode Path A benchmark)..."
$PYTHON_BIN run.py --adapter adapters.agurum:Engine --mode deep --seeds 42 101 202 303 404 --out ../report_deep.json
cd ..

export PYTHONPATH="$PWD/bench-p02-context:$PYTHONPATH"

echo "[*] Running l3-quick..."
$PYTHON_BIN l3-testing/run_l3.py --adapter adapters.agurum:Engine --quick

echo "[*] Running blanket-test..."
$PYTHON_BIN l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 --n-services 20 --n-families 7 --days 10 --n-train 30 --n-eval 15 --verbose --out l3_blanket_test.json

echo "[*] Running diagnose..."
$PYTHON_BIN l3-testing/diagnose_twins.py --adapter adapters.agurum:Engine --seed 42 --n-families 12

echo "[*] Running chaos..."
$PYTHON_BIN l3-testing/chaos_inject.py --adapter adapters.agurum:Engine --seed 42

echo "[*] Running l3-full..."
$PYTHON_BIN l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9919 --n-services 30 --n-families 13 --days 14 --n-train 40 --n-eval 20 --verbose --out l3_full_report.json

echo "[*] Running l3-full-deep..."
$PYTHON_BIN l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9919 --n-services 30 --n-families 13 --days 14 --n-train 40 --n-eval 20 --mode deep --verbose --out l3_full_deep_report.json

echo "[*] Running l3-stress..."
$PYTHON_BIN l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9999 31415 27182 --n-services 50 --n-families 15 --days 21 --n-train 60 --n-eval 30 --out l3_stress_report.json

echo "[*] Running l3-stress-deep..."
$PYTHON_BIN l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9999 31415 27182 --n-services 50 --n-families 15 --days 21 --n-train 60 --n-eval 30 --mode deep --out l3_stress_deep_report.json

echo "[*] All benchmarks complete! Reports written to root."
echo "================================================================"
