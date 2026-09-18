# Deployment for a Runa agent served over HTTP: api.py (SQLite, single-worker only) or
# api_postgres.py (Postgres/Redis, safe for --workers N or multiple replicas) -- see
# docker-compose.yml, which overrides CMD below to pick one.
#
# Assumes a `runa new` project layout with api.py added at the root, next to main.py (see
# api.py/api_postgres.py in this directory for what that file looks like). A real app's
# pyproject.toml would list fastapi/uvicorn (and runa[postgres]/runa[redis], if using those
# backends) as plain dependencies, so `--no-dev` below excludes only actual dev tooling
# (pytest/ruff/pyright) -- this repo's own pyproject keeps them in `dev` instead, purely so
# `runa` itself stays dependency-free by default, which is why this Dockerfile (built directly
# from this repo, for the example) syncs every group rather than excluding dev.
FROM python:3.14-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen

COPY . .

RUN mkdir -p db

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
