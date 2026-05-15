.PHONY: bench-adapter bench-full bench-gateway run-engine run-gateway run test help

# ── Path A — Benchmark ────────────────────────────────────────────────────────

bench-adapter:   ## Run benchmark adapter quick-check (Path A)
	cd bench-p02-context && python3 self_check.py --adapter adapters.agurum:Engine --quick

bench-full:      ## Full benchmark run with 5 seeds (Path A)
	cd bench-p02-context && python3 run.py --adapter adapters.agurum:Engine \
		--seeds 42 101 202 303 404 --out report.json

# ── Path B — Production stack ─────────────────────────────────────────────────

run-engine:      ## Start Python engine on UDS (Path B)
	python3 -m uvicorn engine.main:app --uds /tmp/pce.sock

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
	python3 -m pytest tests/test_gateway_stress.py -v

# ── Test suite ────────────────────────────────────────────────────────────────

test:            ## Run full test suite
	python3 -m pytest tests/ -v

# ── Help ──────────────────────────────────────────────────────────────────────

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── P3 Checkpoint sequence ────────────────────────────────────────────────────

l3-quick:        ## Fast self-check across 2 seeds
	cd bench-p02-context && python3 self_check.py --adapter adapters.agurum:Engine --quick

blanket-test:    ## Verify dynamic N clustering (same as l3-quick right now)
	cd bench-p02-context && python3 self_check.py --adapter adapters.agurum:Engine --quick

diagnose:        ## (Placeholder) run diagnostic tests
	@echo "Running diagnostic probe..."
	cd bench-p02-context && python3 self_check.py --adapter adapters.agurum:Engine --quick

chaos:           ## (Placeholder) run chaos engineering tests
	@echo "Running chaos graph tests..."
	cd bench-p02-context && python3 self_check.py --adapter adapters.agurum:Engine --quick

l3-full:         ## Full 5-seed L3 evaluation
	cd bench-p02-context && python3 run.py --adapter adapters.agurum:Engine \
		--seeds 42 101 202 303 404 --out report.json
