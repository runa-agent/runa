# CLAUDE.md

Runa is an opinionated, Rails-inspired Python framework for agentic AI.
Conventions: @RUNA.md

## Commands
- `make install`: uv sync
- `make format` / `make lint-fix`: ruff
- `make typecheck`: pyright
- `make test`: pytest (single test: `uv run pytest path/to/test.py::test_name`)
- `make check`: format + lint + typecheck + test

## Workflow
- Run `make check` before calling anything done.
- Commits: one line, `<type>: <summary>`. Type is feat, fix, docs, refactor, or test.

## Design rules
- Convention over configuration: a feature must work with zero config. Add an option only when no convention can cover the case.
- Every convention has an explicit override. Example: <one real case from Runa>.
- Omakase: use the chosen stack (uv, ruff, pyright, pytest). Ask before adding a dependency.
- Design from the call site: write the usage you want first, then the implementation.
- Organize code by responsability.
- Few deep modules over many shallow ones. No pass-through wrappers.
- Build only what the current task needs.

## Code style
- Types carry the contract. Docstrings are one summary line, plus only what types cannot say.