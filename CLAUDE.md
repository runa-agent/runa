# CLAUDE.md

Runa is an opinionated Python framework for agentic AI.

Conventions: [Runa.md](./RUNA.md)

## Commands

* `make install`: uv sync
* `make format`: ruff format
* `make lint` / `make lint-fix`: ruff check
* `make typecheck`: pyright
* `make test`: pytest
* `make check`: format + lint + test

## Zen of Python

- `uv run python -c "import this"`
- Less is more.
- Important things come first.
- Let Ruff keep code simple.
- Let types speak for themselves.
- Let docstrings explain what types cannot.
- Code that does not pass test is not done.
- Give oneliner commit message: `feat`, `fix`, `docs`, `refactor`, `test`
- Lint rules: `E`, `F`, `I`, `B`, `SIM`, `UP`, `D`

## Development principles

- Optimize for Developer Happiness
- Convention Over Configuration
- Don't Repeat Yourself
- Keep it Simple
- You Aren't Gonna Need it
- Give Escape Hatch
- Organize code by conventions and responsibility