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

l3-quick:        ## Fast self-check across 1 seed, 7 families
	PYTHONPATH=$$PWD/bench-p02-context:$$PYTHONPATH python3 l3-testing/run_l3.py --adapter adapters.agurum:Engine --quick

blanket-test:    ## Expose hardcoded-5-family blanket assumption
	PYTHONPATH=$$PWD/bench-p02-context:$$PYTHONPATH python3 l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 --n-services 20 --n-families 7 --days 10 --n-train 30 --n-eval 15 --verbose --out l3_blanket_test.json

diagnose:        ## Find out WHY families are confused
	PYTHONPATH=$$PWD/bench-p02-context:$$PYTHONPATH python3 l3-testing/diagnose_twins.py --adapter adapters.agurum:Engine --seed 42 --n-families 10

chaos:           ## Simulate the hidden chaos test
	PYTHONPATH=$$PWD/bench-p02-context:$$PYTHONPATH python3 l3-testing/chaos_inject.py --adapter adapters.agurum:Engine --seed 42

l3-full:         ## Full L3 simulation — 10 families, 3 seeds
	PYTHONPATH=$$PWD/bench-p02-context:$$PYTHONPATH python3 l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9999 --n-services 30 --n-families 13 --days 14 --n-train 40 --n-eval 20 --verbose --out l3_full_report.json

l3-stress:       ## Stress test (max params)
	PYTHONPATH=$$PWD/bench-p02-context:$$PYTHONPATH python3 l3-testing/run_l3.py --adapter adapters.agurum:Engine --seeds 42 1337 9999 31415 27182 --n-services 50 --n-families 10 --days 21 --n-train 60 --n-eval 30 --out l3_stress_report.json
