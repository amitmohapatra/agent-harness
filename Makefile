# trellis-harness — development tasks.
UV ?= uv
PY ?= .venv/bin/python
PYTEST ?= $(PY) -m pytest
#: the matrix runs in this many shards at once (MATRIX_SHARD=i/n each): its cells mostly
#: wait (time limits, surfaces), so more shards than cores
MATRIX_SHARDS ?= 8

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Create the venv with every extra (uv)
	$(UV) sync

.PHONY: typings
typings:  ## Link trellis.contracts, trellis.memory and trellis.runs where pyright looks (see pyproject)
	@mkdir -p typings/trellis
	@$(PY) -c "import os, trellis.contracts as c, trellis.memory as m, trellis.runs as r; \
	[os.path.lexists(d) or os.symlink(s, d) for s, d in ((p.__path__[0], 'typings/trellis/' + os.path.basename(p.__path__[0])) for p in (c, m, r))]"

.PHONY: test
test:  ## Every test except the benchmark and the live ones, at 100% line and branch coverage
	$(PYTEST) -q -m "not performance and not live and not matrix" --cov=trellis.harness --cov-branch --cov-report=term-missing:skip-covered --cov-fail-under=100

.PHONY: test-live
test-live:  ## Opt-in tests against running services (BIFROST_URL, MEMORY_URL, RUNS_URL, TRELLIS_API_KEY)
	$(PYTEST) -q -m live

.PHONY: bench
bench:  ## Harness overhead vs the committed baseline (writes build/benchmark-results.json)
	$(PYTEST) tests/performance -m performance -q -s

.PHONY: matrix
matrix:  ## The generated feature matrix (feature x adapter x way x mode x selection), sharded, and its report (build/matrix.md)
	@mkdir -p build && rm -f build/matrix-*.xml build/matrix-*.log
	@pids=""; for i in $$(seq 1 $(MATRIX_SHARDS)); do \
	  env -u BIFROST_URL -u MEMORY_URL -u RUNS_URL -u TRELLIS_API_KEY -u OTEL_EXPORTER_OTLP_ENDPOINT \
	    MATRIX_SHARD=$$i/$(MATRIX_SHARDS) $(PYTEST) -q -m matrix tests/matrix -p no:cacheprovider \
	    --junitxml=build/matrix-$$i.xml > build/matrix-$$i.log 2>&1 & pids="$$pids $$!"; \
	done; status=0; for p in $$pids; do wait $$p || status=1; done; \
	for i in $$(seq 1 $(MATRIX_SHARDS)); do tail -n 1 build/matrix-$$i.log; done; \
	$(PY) -m tests.matrix.report build/matrix-*.xml --out build/matrix.md; \
	if [ $$status -ne 0 ]; then grep -h "^FAILED\|^ERROR" build/matrix-*.log; fi; exit $$status

.PHONY: lint
lint:  ## Ruff
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

.PHONY: typecheck
typecheck: typings  ## Pyright
	.venv/bin/pyright

.PHONY: check
check: lint typecheck test  ## Everything CI runs, except the benchmark
	@echo "all gates passed"

.PHONY: examples
examples:  ## Run every example offline (scripted model, memory, gateway), several at once
	$(PY) scripts/run_examples.py

.PHONY: examples-live
examples-live:  ## Run every example with the environment as it is (the real services that are set)
	$(PY) scripts/run_examples.py --live

.PHONY: docs-check
docs-check:  ## Every link and anchor resolves; every snippet parses and names only the real API
	$(PY) scripts/check_docs.py

.PHONY: docs-mermaid
docs-mermaid:  ## Every Mermaid diagram parses (node; MERMAID_MODULES: a node_modules with mermaid, jsdom)
	node scripts/check_mermaid.mjs $$(git ls-files --cached --others --exclude-standard '*.md' | grep -v '^\.claude/')

.PHONY: clean
clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache build typings **/__pycache__
