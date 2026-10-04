# CLAUDE.md

Runa is an opinionated, Rails-inspired Python framework for agentic AI.
Conventions: @RUNA.md

## Verify
- `make check` (format, lint, typecheck, test) must pass before work is done.
- Single test: `uv run pytest path/to/test.py::test_name`
- Other targets: see the Makefile.

## Rules
- A feature must work with zero config. Add an option only when no convention can cover the case.
- Every convention has an explicit override. Example: <one real case from Runa>.
- Ask before adding a dependency or a top-level module.
- Write the call site first, then the implementation.
- Few deep modules over many shallow ones. No pass-through wrappers.
- Build only what the task needs.
- Commits: one line, `<type>: <summary>`. Type is feat, fix, docs, refactor, or test.

## Agent skills

### Issue tracker

GitHub Issues in `runa-agent/runa`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, used verbatim as label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` plus `docs/adr/` at the repo root. See `docs/agents/domain.md`.