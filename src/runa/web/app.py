"""web/app.py: the `runa ui` FastAPI app -- Agents, Sessions, Evaluations.

Read-only except for one write: "Add to evals" on a trace page appends a case to `evals/`.

Every route calls straight into one `web/<page>.py`'s render function; no route does its own
data-fetching or HTML-building. `create_app(root)` closes over the app's directory, the same way
every `cli/*.py` command takes `root` as a parameter instead of assuming `cwd`.

No standalone Traces tab: `web/sessions.py`'s merged timeline already shows a session's traces in
context, and `/traces/{trace_id}` stays routable (unlinked from the nav) as the "open trace" target
from a session's trace card and for a session-less trace (an eval run, a one-off `run_sync()`).
"""

from pathlib import Path
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from runa.cli._project import AppLoadError, NotARunaProject
from runa.cli.eval import TraceHasNoInput, add_trace_to_evals
from runa.cli.traces import TraceNotFound
from runa.web import agents as agents_page
from runa.web import evaluations as evaluations_page
from runa.web import sessions as sessions_page
from runa.web import traces as traces_page
from runa.web._html import empty, page


def _error_page(active: str, message: str) -> str:
    return page(title="Error", active=active, body=f"<h1>Error</h1>{empty(message)}")


def create_app(root: Path) -> FastAPI:
    """Build the `runa ui` app for the project at `root`."""
    app = FastAPI(title="runa ui", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/agents")

    @app.get("/agents", response_class=HTMLResponse, include_in_schema=False)
    def agents_list() -> str:
        return agents_page.render(root=root)

    @app.get("/sessions", response_class=HTMLResponse, include_in_schema=False)
    def sessions_list() -> str:
        return sessions_page.render_list(root=root)

    @app.get("/sessions/{session_id}", response_class=HTMLResponse, include_in_schema=False)
    def session_detail(session_id: str) -> HTMLResponse:
        try:
            return HTMLResponse(sessions_page.render_detail(session_id, root=root))
        except sessions_page.SessionNotFound:
            return HTMLResponse(_error_page("Sessions", "Session not found."), status_code=404)

    @app.get("/traces/{trace_id}", response_class=HTMLResponse, include_in_schema=False)
    def trace_detail(trace_id: str, added: bool = False) -> HTMLResponse:
        try:
            return HTMLResponse(traces_page.render_detail(trace_id, root=root, added=added))
        except traces_page.TraceNotFound:
            return HTMLResponse(_error_page("", "Trace not found."), status_code=404)

    @app.post("/traces/{trace_id}/eval", include_in_schema=False, response_model=None)
    async def trace_add_to_evals(
        trace_id: str, request: Request
    ) -> HTMLResponse | RedirectResponse:
        form = parse_qs((await request.body()).decode())
        expected = form.get("expected", [""])[0].strip() or None
        try:
            add_trace_to_evals(trace_id, root=root, expected=expected)
        except TraceNotFound, TraceHasNoInput:
            return HTMLResponse(_error_page("", "Trace has no input to add."), status_code=404)
        return RedirectResponse(f"/traces/{quote(trace_id)}?added=true", status_code=303)

    @app.get("/evaluations", response_class=HTMLResponse, include_in_schema=False)
    def evaluations_list() -> str:
        return evaluations_page.render_list(root=root)

    @app.get("/evaluations/{run_id}", response_class=HTMLResponse, include_in_schema=False)
    def evaluation_detail(run_id: int) -> HTMLResponse:
        try:
            return HTMLResponse(evaluations_page.render_detail(run_id, root=root))
        except evaluations_page.EvalRunNotFound:
            return HTMLResponse(
                _error_page("Evaluations", "Evaluation run not found."), status_code=404
            )

    @app.exception_handler(NotARunaProject)
    def _not_a_project(_request: Request, _exc: NotARunaProject) -> HTMLResponse:
        return HTMLResponse(_error_page("Agents", "Not a runa project."), status_code=400)

    @app.exception_handler(AppLoadError)
    def _app_load_error(_request: Request, _exc: AppLoadError) -> HTMLResponse:
        return HTMLResponse(_error_page("Agents", "Failed to load application."), status_code=500)

    return app
