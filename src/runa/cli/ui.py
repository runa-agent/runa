"""cli/ui.py: `runa ui`, a local read-only web dashboard over this app's `db/runa.db`.

`fastapi`/`uvicorn`/`runa.web` are imported lazily inside `serve_ui`, so a plain `runa` install
(no `ui` extra) still works for every other command; only running `runa ui` itself needs them.
"""

from pathlib import Path


def serve_ui(root: Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the `runa ui` dashboard, serving `root`'s Agents/Sessions/Traces/Evaluations."""
    import uvicorn

    from runa.web.app import create_app

    print(f"runa ui running at http://{host}:{port}")
    uvicorn.run(create_app(root), host=host, port=port, log_level="warning")


__all__ = ["serve_ui"]
