.PHONY: install format lint typecheck test check docs hello tour examples clean changelog

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

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .pyright
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info