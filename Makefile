.PHONY: install format lint typecheck test coverage audit check dist-check docs hello tour ui-demo examples clean changelog release

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
# actually installs, transitive dependencies included. CI runs this same target, advisory-only.
# The export is already pinned and hashed, so `--disable-pip` keeps pip-audit from building an
# environment to re-resolve what it was handed.
audit:
	uv export --format requirements-txt --no-emit-project --all-extras --quiet \
		| uv tool run pip-audit --disable-pip -r /dev/stdin

check:
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

# Everything above runs against the working tree, where a file that was never committed imports
# as happily as one that was. This unpacks HEAD the way a clone, a release tag and the published
# sdist all get it -- `git archive`, nothing read from disk -- and imports every module in it, so
# a file left out of a commit fails here instead of in someone else's checkout.
dist-check:
	rm -rf build/head && mkdir -p build/head
	git archive HEAD | tar -x -C build/head
	uv run --project build/head python -c \
		"import importlib, pkgutil, runa; \
		[importlib.import_module(m.name) for m in pkgutil.walk_packages(runa.__path__, 'runa.')]"

docs:
	uv run zensical build --strict

# Regenerates the whole file. Pass the version being released so its section is headed with
# that version instead of "Unreleased": the tag is created after this commit, so git-cliff
# cannot infer it. `make changelog` alone (no VERSION) renders the pending work as Unreleased.
changelog:
	uv tool run git-cliff $(if $(VERSION),--tag v$(VERSION),) -o CHANGELOG.md

# `make release VERSION=0.6.0`. The version bump, the lockfile and the changelog land in one
# `feat: release X.Y.Z` commit, and the tag points at it. Pushing that tag is what builds and
# publishes to PyPI (.github/workflows/release.yml), so it stays a separate command typed by a
# human: everything this target does is still local and undoable.
release:
	@test -n "$(VERSION)" || { echo "usage: make release VERSION=0.6.0"; exit 1; }
	@test -z "$$(git status --porcelain)" \
		|| { echo "working tree is dirty: a tag has to match what it ships"; exit 1; }
	@git rev-parse -q --verify refs/tags/v$(VERSION) >/dev/null \
		&& { echo "v$(VERSION) is already tagged"; exit 1; } || true
	$(MAKE) check
	uv version $(VERSION)
	$(MAKE) changelog VERSION=$(VERSION)
	git add CHANGELOG.md pyproject.toml uv.lock
	git commit -m "feat: release $(VERSION)"
	git tag v$(VERSION)
	@echo "tagged v$(VERSION). to publish: git push origin main && git push origin v$(VERSION)"

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