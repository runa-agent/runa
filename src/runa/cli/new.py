"""cli/new.py: scaffold a new Runa application.

Establishes the conventional project layout so a fresh project has somewhere obvious to put
agents, tools, prompts, and guardrails (under `app/`), plus tests, eval cases, shared config, the
SQLite database, and developer docs (at the project root) without any configuration. Given a
`name`, scaffolds into `root/name`; without one, scaffolds `root` itself in place.
"""

from __future__ import annotations

from pathlib import Path

_APP_SUBDIRS = ("agents", "guardrails", "prompts", "tools")
_ROOT_PACKAGE_SUBDIRS = ("tests", "evals", "config")
_ROOT_PLAIN_SUBDIRS = ("db", "docs")
_TOP_LEVEL_ENTRIES = (
    "app",
    *_ROOT_PACKAGE_SUBDIRS,
    *_ROOT_PLAIN_SUBDIRS,
    "pyproject.toml",
    "main.py",
    "Dockerfile",
    ".gitignore",
    ".env",
)

_PYPROJECT_TEMPLATE = """[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["runa-ai", "python-dotenv"]

[project.optional-dependencies]
serve = ["runa-ai[serve]"]
"""

_MAIN_TEMPLATE = '''"""main.py: the application entry point.

Loads `.env` so every `runa` command (and this file, run directly) picks up
whichever API key(s) the agents below need, without exporting anything into
the shell. There's no separate configuration step beyond that: a model is a
per-`Agent` class attribute (see `app/agents/`), and Runa resolves it to the
right provider (OpenAI, Anthropic, Google, Meta, DeepSeek, or Alibaba) from
its name, so nothing here wires up a model or provider globally.
"""

from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    # from app.agents import ExampleAgent
    #
    # print(ExampleAgent().run_sync("...").output)
    pass
'''

_ENV_TEMPLATE = """# Loaded by main.py via load_dotenv(). Fill in the API key for whichever
# model(s) your agents use (see app/agents/), then never commit this file.
# OPENAI_API_KEY (gpt-*) / ANTHROPIC_API_KEY (claude-*) / GEMINI_API_KEY (gemini-*)
# LLAMA_API_KEY (llama-*) / DEEPSEEK_API_KEY (deepseek-*) / DASHSCOPE_API_KEY (qwen-*)
OPENAI_API_KEY=

# The token clients must send to `runa serve` as `Authorization: Bearer <token>`. Required in
# production; `runa serve --no-auth` is the local escape hatch.
RUNA_API_KEY=

# Uncomment to share sessions, traces and eval history across replicas instead of keeping them
# in this process's db/runa.db. Needed for any deployment running more than one instance.
# RUNA_POSTGRES_DSN=postgresql://user:password@host:5432/runa
"""

_GITIGNORE_TEMPLATE = """__pycache__/
*.pyc
db/runa.db
.env
"""

_DOCKERFILE_TEMPLATE = """FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir uv && uv sync --frozen --extra serve

EXPOSE 8000

# `runa serve` puts this app's agents behind an HTTP API: POST /agents/<name>/runs, plus
# /runs/stream and an unauthenticated /health for the load balancer. It needs RUNA_API_KEY set
# (clients send it as `Authorization: Bearer <token>`) and refuses to start without one; pass
# --no-auth if something in front of this container is already doing authentication.
CMD ["uv", "run", "runa", "serve", "--host", "0.0.0.0", "--port", "8000"]
"""


class ProjectAlreadyExists(Exception):
    """Raised when `runa new` targets a directory that already exists."""


def scaffold_project(name: str | None, *, root: Path) -> Path:
    """Create `root/name` with the conventional Runa project layout.

    Without a `name`, scaffolds `root` itself (used for `runa new` with no argument), so
    `root`'s own existence isn't grounds for `ProjectAlreadyExists` -- only an entry this
    function would otherwise write into is.
    """
    project_dir = root / name if name else root
    if name:
        if project_dir.exists():
            raise ProjectAlreadyExists(f"{project_dir} already exists")
    else:
        for entry in _TOP_LEVEL_ENTRIES:
            if (project_dir / entry).exists():
                raise ProjectAlreadyExists(f"{project_dir / entry} already exists")

    app_dir = project_dir / "app"
    for subdir in _APP_SUBDIRS:
        package_dir = app_dir / subdir
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("")
    (app_dir / "__init__.py").write_text("")

    for subdir in _ROOT_PACKAGE_SUBDIRS:
        package_dir = project_dir / subdir
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("")

    for subdir in _ROOT_PLAIN_SUBDIRS:
        (project_dir / subdir).mkdir()

    (project_dir / "pyproject.toml").write_text(
        _PYPROJECT_TEMPLATE.format(name=name or project_dir.resolve().name)
    )
    (project_dir / "main.py").write_text(_MAIN_TEMPLATE)
    (project_dir / "Dockerfile").write_text(_DOCKERFILE_TEMPLATE)
    (project_dir / ".gitignore").write_text(_GITIGNORE_TEMPLATE)
    (project_dir / ".env").write_text(_ENV_TEMPLATE)

    return project_dir
