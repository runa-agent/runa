"""cli/serve.py: `runa serve`, the short way to put this app's agents behind HTTP.

The server itself is `runa.serve`: this file only turns flags into its two arguments and hands the
app to uvicorn, the same shape `cli/ui.py` has for the dashboard. `uvicorn`/`runa.serve` are
imported inside `serve_agents`, so a plain install (no `serve` extra) still runs every other
command; `cli/main.py` turns the `ModuleNotFoundError` that a missing extra raises into the
one-line `uv add "runa-ai[serve]"` instruction.
"""

from pathlib import Path


def serve_agents(
    root: Path, *, host: str = "127.0.0.1", port: int = 8000, no_auth: bool = False
) -> None:
    """Serve `root`'s agents over HTTP, in this one process, until interrupted.

    One process is the whole process model this command has, and the reason is the app object
    above: `create_app(root, api_key=...)` takes arguments, so there is no import string for a
    supervisor to re-import per worker -- and an import string is what uvicorn requires to fork
    (it refuses an app object with `workers > 1`). A deployment that wants several workers has an
    address already: the `asgi.py` that `runa new` scaffolds, run as `uvicorn asgi:app
    --workers N`. That's the seam for a different process model, not a flag here.
    """
    import uvicorn

    from runa.serve import create_app, resolve_api_key

    api_key = resolve_api_key(no_auth=no_auth)  # before binding the port
    app = create_app(root, api_key=api_key)

    auth = "no auth" if api_key is None else "bearer auth"
    print(f"runa serve running at http://{host}:{port} ({auth})")
    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["serve_agents"]
