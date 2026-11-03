.PHONY: test test-local lockcheck audit lint types fmt clean help

help:
	@echo "Targets:"
	@echo "  make test            - run full pytest suite"
	@echo "  make test-local      - lint + types + test (the gate before any push)"
	@echo "  make audit           - known vulnerabilities in uv.lock and the vendored browser libraries (needs the network)"
	@echo "  make lint            - ruff check"
	@echo "  make types           - mypy"
	@echo "  make fmt             - ruff format"
	@echo "  make clean           - remove caches"

# Ignore list mirrors ci.yml. These are script style checks that need a
# running app, live provider creds, or a populated DB. Run them by hand.
test:
	uv run pytest tests/ -v --tb=short \
		--ignore=tests/test_vector_index.py \
		--ignore=tests/test_full_system.py \
		--ignore=tests/test_dashboard.py \
		--ignore=tests/test_receipt_upload.py \
		--ignore=tests/test_cache_threshold.py

test-local: lockcheck lint types test
	@echo ""
	@echo "✓ Local test suite passed."
	@echo "  REMINDER: multi-provider smoke is still required before push."
	@echo "  Boot the app against each of watsonx, openai, aws, gemini, ollama"
	@echo "  and run scripts/coach/assert_nudges.py (see README, Testing)."

lockcheck:
	@# A pyproject change without a re-lock leaves uv.lock drifting: the
	@# next uv run rewrites it AFTER the push and the tree looks dirty
	@# out of nowhere. Fail here, before anything gets committed.
	uv lock --check

audit:
	@# Known vulnerabilities in every version uv.lock pins, every extra,
	@# dependency group and Python version included, through the OSV feed
	@# (see scripts/audit_lock.py). It needs the network, so it is its own
	@# target instead of part of test-local. Run it before any push.
	uv run --no-project python scripts/audit_lock.py
	uv run --no-project --with truststore python scripts/audit_vendor.py

lint:
	uv run ruff check banko_ai/ tests/

# Advisory for now, same as CI (continue-on-error). The codebase carries
# about 200 findings that predate the July 2026 dep landing and need a
# dedicated typing pass, not a drive-by fix.
types:
	-uv run mypy banko_ai/

fmt:
	uv run ruff format banko_ai/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
