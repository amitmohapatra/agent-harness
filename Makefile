# trellis-harness — development tasks.
UV ?= uv
PY ?= .venv/bin/python
PYTEST ?= $(PY) -m pytest

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Create the venv with every extra (uv)
	$(UV) sync

.PHONY: typings
typings:  ## Link trellis.contracts and trellis.memory where pyright looks (see pyproject)
	@mkdir -p typings/trellis
	@$(PY) -c "import os, trellis.contracts as c, trellis.memory as m; \
	[os.path.lexists(d) or os.symlink(s, d) for s, d in ((p.__path__[0], 'typings/trellis/' + os.path.basename(p.__path__[0])) for p in (c, m))]"

.PHONY: test
test:  ## Every test except the benchmark and the live ones, at 100% line and branch coverage
	$(PYTEST) -q -m "not performance and not live" --cov=trellis.harness --cov-branch --cov-report=term-missing:skip-covered --cov-fail-under=100

.PHONY: test-live
test-live:  ## Opt-in tests against running services (BIFROST_URL, MEMORY_URL, RUNS_URL, TRELLIS_API_KEY)
	$(PYTEST) -q -m live

.PHONY: bench
bench:  ## Harness overhead vs the committed baseline (writes build/benchmark-results.json)
	$(PYTEST) tests/performance -m performance -q -s

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
examples:  ## Run every example with no services
	@for f in examples/*.py; do case $$f in */_*) continue;; esac; \
	  env -u BIFROST_URL -u MEMORY_URL -u RUNS_URL -u TRELLIS_API_KEY -u OTEL_EXPORTER_OTLP_ENDPOINT \
	    $(PY) $$f >/dev/null && echo "$$f ok" || exit 1; done

.PHONY: clean
clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache build typings **/__pycache__
