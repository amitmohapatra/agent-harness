# Universal Agent Harness — development tasks.
UV ?= uv
PY ?= .venv/bin/python
PYTEST ?= $(PY) -m pytest

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Create the venv and install everything (uv)
	$(UV) venv --python 3.12 .venv
	$(UV) sync --all-extras

.PHONY: test
test:  ## Run every test except the benchmarks
	$(PYTEST) -q -m "not performance"

.PHONY: test-unit
test-unit:  ## Unit tests only
	$(PYTEST) tests/unit -q

.PHONY: test-contract
test-contract:  ## Port/contract conformance
	$(PYTEST) tests/contract -q

.PHONY: test-integration
test-integration:  ## Integration tests
	$(PYTEST) tests/integration -q

.PHONY: test-e2e
test-e2e:  ## End-to-end, including the real Memory Service SDK (HTTP mocked)
	$(PYTEST) tests/e2e -q

.PHONY: test-langgraph
test-langgraph:  ## LangGraph adapter tests
	$(PYTEST) integrations/langgraph/tests -q

.PHONY: test-compat
test-compat:  ## Compatibility matrix (rewrites compatibility-matrix.json)
	$(PYTEST) tests/compatibility -q

.PHONY: test-live
test-live:  ## End-to-end against a running Memory Service (MEMORY_SERVICE_URL)
	MEMORY_SERVICE_URL=$${MEMORY_SERVICE_URL:-http://localhost:8080} \
	MEMORY_API_KEY=$${MEMORY_API_KEY:-dev-key} \
	$(PYTEST) tests/e2e/test_live_memory_service.py -q -s

.PHONY: test-live-full
test-live-full:  ## Every feature against a running Memory Service, with database checks
	MEMORY_SERVICE_URL=$${MEMORY_SERVICE_URL:-http://localhost:8080} \
	MEMORY_API_KEY=$${MEMORY_API_KEY:-dev-key} \
	$(PYTEST) tests/e2e/test_live_full_surface.py -q -s

.PHONY: bench
bench:  ## Harness overhead benchmark (writes benchmark-results.json)
	$(PYTEST) tests/performance -m performance -q -s

.PHONY: bench-throughput
bench-throughput:  ## Turns/second against a running Memory Service (throughput-results.json)
	$(PYTEST) tests/performance/test_throughput.py -m performance -q -s

.PHONY: lint
lint:  ## Ruff
	.venv/bin/ruff check .

.PHONY: format
format:  ## Ruff autofix
	.venv/bin/ruff check --fix .

.PHONY: typecheck
typecheck:  ## Pyright
	.venv/bin/pyright

.PHONY: check
check: lint typecheck test  ## Everything a release gate runs
	@echo "all gates passed"

.PHONY: coverage
coverage:  ## Test coverage for the core package
	$(PYTEST) -q -m "not performance" --cov=universal_agent_harness --cov-report=term-missing

.PHONY: examples
examples:  ## Run the runnable examples
	$(PY) examples/plain_python.py >/dev/null && echo "plain_python ok"
	$(PY) examples/langgraph_agent.py >/dev/null && echo "langgraph_agent ok"
	$(PY) examples/reorder_workflow.py >/dev/null && echo "reorder_workflow ok"
	$(PY) examples/memory_tour.py >/dev/null && echo "memory_tour ok"

.PHONY: clean
clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .hypothesis **/__pycache__
