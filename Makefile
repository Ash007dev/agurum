bench-adapter:   ## Run benchmark adapter (Path A)
	cd bench-p02-context && python self_check.py --adapter adapters.agurum:Engine --quick

bench-full:      ## Full benchmark run
	cd bench-p02-context && python run.py --adapter adapters.agurum:Engine

run-engine:      ## Start Python engine on UDS (Path B)
	python -m uvicorn engine.main:app --uds /tmp/pce.sock

run-gateway:     ## Build and run Go gateway
	cd gateway && go build -o ../bin/gateway . && ../bin/gateway

run: run-engine run-gateway   ## Start full stack
