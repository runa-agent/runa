"""cli/serve.py: `runa serve`, the production entry point for this app's agents.

`fastapi`/`uvicorn`/`runa.serve` are imported lazily inside `serve_agents`, so a plain `runa`
install (no `serve` extra) still works for every other command, the same arrangement `cli/ui.py`
uses for the dashboard.
"""

from __future__ import annotations

import os
from pathlib import Path

API_KEY_ENV = "RUNA_API_KEY"


class MissingAPIKey(Exception):
    """Raised when `runa serve` starts with neither `RUNA_API_KEY` nor an explicit `--no-auth`."""


def resolve_api_key(*, no_auth: bool) -> str | None:
    """The token `runa serve` will require, or `None` for an open server.

    Refuses to start unauthenticated by accident: an agent endpoint costs money per call, so
    "nobody set the variable" has to be an error rather than a silently open door. `--no-auth`
    makes the same choice explicit and is then perfectly fine for local use.

    Lives here, not in `runa/serve.py`, so `cli/main.py` can catch `MissingAPIKey` without
    importing FastAPI: a plain install has to be able to parse `runa serve --help` and to print
    a clean error, neither of which should need the `serve` extra.
    """
    if no_auth:
        return None
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise MissingAPIKey(
            f"{API_KEY_ENV} is not set. Set it to the token clients must send as "
            "`Authorization: Bearer <token>`, or pass --no-auth to serve without authentication."
        )
    return api_key


def serve_agents(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    no_auth: bool = False,
    workers: int = 1,
) -> None:
    """Serve `root`'s agents over HTTP until interrupted."""
    api_key = resolve_api_key(no_auth=no_auth)  # before importing anything heavy

    import uvicorn

    from runa.serve import create_app

    app = create_app(root, api_key=api_key)

    auth = "no auth" if api_key is None else "bearer auth"
    print(f"runa serve running at http://{host}:{port} ({auth})")
    uvicorn.run(app, host=host, port=port, workers=workers, log_level="info")


__all__ = ["API_KEY_ENV", "MissingAPIKey", "resolve_api_key", "serve_agents"]
