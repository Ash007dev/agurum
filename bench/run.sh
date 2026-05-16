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

# Execute the canonical benchmark sequence required by the SDK
echo "[*] Running canonical Path A benchmark..."

cd bench-p02-context
$PYTHON_BIN run.py \
    --adapter adapters.agurum:Engine \
    --mode fast \
    --seeds 42 101 202 303 404 \
    --out ../report.json

cd ..
echo "[*] Benchmark complete. Results written to report.json."
echo "================================================================"
