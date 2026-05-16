# L3 Mock Benchmark Makefile
# Drop this into your project root alongside your bench-p02-context/ folder

ADAPTER ?= adapters.agurum:Engine
BENCH   = bench/

# Quick L3 smoke test (7 families, 1 seed, fast)
l3-quick:
	cd $(BENCH) && python run_l3.py \
		--adapter $(ADAPTER) \
		--quick \
		--verbose \
		--out ../l3_quick_report.json

# Full L3 test (10 families, 3 seeds — simulates held-out conditions)
l3-full:
	cd $(BENCH) && python run_l3.py \
		--adapter $(ADAPTER) \
		--seeds 42 1337 9999 \
		--n-services 30 \
		--n-families 10 \
		--days 14 \
		--n-train 40 \
		--n-eval 20 \
		--verbose \
		--out ../l3_full_report.json

# Stress test (max params — see how far your engine scales)
l3-stress:
	cd $(BENCH) && python run_l3.py \
		--adapter $(ADAPTER) \
		--seeds 42 1337 9999 31415 27182 \
		--n-services 50 \
		--n-families 10 \
		--days 21 \
		--n-train 60 \
		--n-eval 30 \
		--out ../l3_stress_report.json

# Diagnose which families your engine confuses
diagnose:
	cd $(BENCH) && python diagnose_twins.py \
		--adapter $(ADAPTER) \
		--seed 42 \
		--n-families 10

# Simulate the hidden chaos test from the judging mechanics
chaos:
	cd $(BENCH) && python chaos_inject.py \
		--adapter $(ADAPTER) \
		--seed 42

# Blanket detection: run L3 with 7 families to expose hardcoded-5 assumption
blanket-test:
	cd $(BENCH) && python run_l3.py \
		--adapter $(ADAPTER) \
		--seeds 42 1337 \
		--n-services 20 \
		--n-families 7 \
		--days 10 \
		--n-train 30 \
		--n-eval 15 \
		--verbose \
		--out ../l3_blanket_test.json

.PHONY: l3-quick l3-full l3-stress diagnose chaos blanket-test
