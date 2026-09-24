.PHONY: install format lint typecheck test coverage audit check docs hello tour ui-demo examples clean changelog

install:
	uv sync

format:
	uv run ruff format

lint:
	uv run ruff check

lint-fix:
	uv run ruff check --fix

typecheck:
	uv run pyright

test:
	uv run pytest

coverage:
	uv run pytest --cov=runa --cov-report=term-missing

# Audits the resolved lockfile rather than the declared ranges, so it reports what a user
# actually installs. CI runs pypa/gh-action-pip-audit over the same export; this target is the
# local equivalent, and needs a working `ensurepip` since pip-audit builds its own environment.
audit:
	uv export --format requirements-txt --no-emit-project --all-extras --quiet \
		| uv tool run pip-audit -r /dev/stdin

check:
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

docs:
	uv run zensical build --strict

changelog:
	uv tool run git-cliff -o CHANGELOG.md

hello:
	uv run python examples/00_quickstart/hello.py

tour:
	uv run python examples/applications/tour.py

ui-demo:
	uv run python examples/applications/seed_ui_demo.py
	cd ui_demo && ../.venv/bin/runa ui

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .pyright
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info