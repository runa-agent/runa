"""cli/serve.py: `runa serve`, the short way to put this app's agents behind HTTP.

The server itself is `runa.serve`: this file only turns flags into its two arguments and hands the
app to uvicorn, the same shape `cli/ui.py` has for the dashboard. `uvicorn`/`runa.serve` are
imported inside `serve_agents`, so a plain install (no `serve` extra) still runs every other
command; `cli/main.py` turns the `ModuleNotFoundError` that a missing extra raises into the
one-line `uv add "runa-ai[serve]"` instruction.
"""

from pathlib import Path


def serve_agents(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    no_auth: bool = False,
    workers: int = 1,
) -> None:
    """Serve `root`'s agents over HTTP until interrupted."""
    import uvicorn

    from runa.serve import create_app, resolve_api_key

    api_key = resolve_api_key(no_auth=no_auth)  # before binding the port
    app = create_app(root, api_key=api_key)

    auth = "no auth" if api_key is None else "bearer auth"
    print(f"runa serve running at http://{host}:{port} ({auth})")
    uvicorn.run(app, host=host, port=port, workers=workers, log_level="info")


__all__ = ["serve_agents"]
