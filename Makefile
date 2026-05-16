.PHONY: bench-adapter bench-full bench-gateway run-engine run-gateway run test help

VENV_DIR := .venv

ifeq ($(OS),Windows_NT)
PYTHON ?= python
VENV_PY := $(VENV_DIR)\Scripts\python.exe
VENV_PIP := $(VENV_DIR)\Scripts\pip.exe
else
PYTHON ?= python3
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
endif

.PHONY: venv install

# Create project venv (no-op if exists)
venv:
	@echo "Creating virtualenv at $(VENV_DIR) if missing..."
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "venv exists"; \
	else \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi

# Install requirements into project venv
install: venv
	@echo "Installing requirements into $(VENV_DIR)..."
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

# ── Path A — Benchmark ────────────────────────────────────────────────────────


bench-adapter:   ## Run benchmark adapter quick-check (Path A)
	$(VENV_PY) bench-p02-context/self_check.py --adapter adapters.agurum:Engine --quick

bench-full:      ## Full benchmark run with 5 seeds (Path A)
	cd bench-p02-context && $(PYTHON) run.py --adapter adapters.agurum:Engine \
		--seeds 42 101 202 303 404 --out report.json

bench-deep:      ## Full benchmark run with 5 seeds in deep mode (Path A)
	cd bench-p02-context && $(PYTHON) run.py --adapter adapters.agurum:Engine --mode deep \
		--seeds 42 101 202 303 404 --out report_deep.json

# ── Path B — Production stack ─────────────────────────────────────────────────

run-engine:      ## Start Python engine on UDS (Path B)
	$(VENV_PY) -m uvicorn engine.main:app --uds /tmp/pce.sock

run-gateway:     ## Build and run Go gateway (Path B)
	mkdir -p bin
	cd gateway && go build -o ../bin/gateway . && ../bin/gateway

run: run-engine run-gateway   ## Start full Path B stack (engine + gateway)

# ── Gateway stress / bench ────────────────────────────────────────────────────


bench-gateway:   ## Stress-test gateway: 1 000 events/sec for 30 s, zero drops
	@echo "==> Building gateway binary..."
	mkdir -p bin
	cd gateway && go build -o ../bin/gateway .
	@echo "==> Running gateway stress test..."
	$(VENV_PY) -m pytest tests/test_gateway_stress.py -v

# ── Test suite ────────────────────────────────────────────────────────────────

test:            ## Run full test suite
	$(VENV_PY) -m pytest tests/ -v

# ── Help ──────────────────────────────────────────────────────────────────────

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── P3 Checkpoint sequence ────────────────────────────────────────────────────

l3-quick:        ## Fast self-check across 1 seed, 7 families
	$(PYTHON) l3-testing/run_l3.py --adapter adapters.agurum:Engine --quick

blanket-test:    ## Expose hardcoded-5-family blanket assumption
	$(PYTHON) l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 --n-services 20 --n-families 7 --days 10 --n-train 30 --n-eval 15 --verbose --out l3_blanket_test.json

diagnose:        ## Find out WHY families are confused
	$(PYTHON) l3-testing/diagnose_twins.py --adapter adapters.agurum:Engine --seed 42 --n-families 12

chaos:           ## Simulate the hidden chaos test
	$(PYTHON) l3-testing/chaos_inject.py --adapter adapters.agurum:Engine --seed 42

l3-full:         ## Full L3 simulation — 10 families, 3 seeds
	$(PYTHON) l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9919 --n-services 30 --n-families 13 --days 14 --n-train 40 --n-eval 20 --verbose --out l3_full_report.json

l3-full-deep:    ## Full L3 simulation in Deep Mode (LLM Narrative + Causal Chain)
	$(PYTHON) l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9919 --n-services 30 --n-families 13 --days 14 --n-train 40 --n-eval 20 --mode deep --verbose --out l3_full_deep_report.json

l3-stress:       ## Stress test (max params)
	$(PYTHON) l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9999 31415 27182 --n-services 50 --n-families 15 --days 21 --n-train 60 --n-eval 30 --out l3_stress_report.json

l3-stress-deep:  ## Stress test in Deep Mode (max params)
	$(PYTHON) l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9999 31415 27182 --n-services 50 --n-families 15 --days 21 --n-train 60 --n-eval 30 --mode deep --out l3_stress_deep_report.json
