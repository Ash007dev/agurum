.PHONY: bench-adapter bench-full bench-gateway run-engine run-gateway run test help

# ── Path A — Benchmark ────────────────────────────────────────────────────────

bench-adapter:   ## Run benchmark adapter quick-check (Path A)
	cd bench-p02-context && python self_check.py --adapter adapters.agurum:Engine --quick

bench-full:      ## Full benchmark run with 5 seeds (Path A)
	cd bench-p02-context && python run.py --adapter adapters.agurum:Engine \
		--seeds 42 101 202 303 404 --out report.json

# ── Path B — Production stack ─────────────────────────────────────────────────

run-engine:      ## Start Python engine on UDS (Path B)
	python -m uvicorn engine.main:app --uds /tmp/pce.sock

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
	python -m pytest tests/test_gateway_stress.py -v

# ── Test suite ────────────────────────────────────────────────────────────────

test:            ## Run full test suite
	python -m pytest tests/ -v

# ── Help ──────────────────────────────────────────────────────────────────────

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
