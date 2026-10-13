"""web/app.py: the `runa ui` FastAPI app -- Agents, Sessions, Evaluations.

Read-only except for one write: "Add to evals" on a trace page appends a case to `evals/`.

Every route calls straight into one `web/<page>.py`'s render function; no route does its own
data-fetching or HTML-building. `create_app(root)` takes the app's directory rather than assuming
`cwd`, and hands it to `project.use` so every store this dashboard reads resolves to the same
file; the two renderers that still take `root` want the filesystem, not the database.

No standalone Traces tab: `web/sessions.py`'s merged timeline already shows a session's traces in
context, and `/traces/{trace_id}` stays routable (unlinked from the nav) as the "open trace" target
from a session's trace card and for a session-less trace (an eval run, a one-off `run_sync()`).

The id routes take `:path`, not a single segment, because the ids are the application's: `user/42`
is a natural session id, and an ASGI server percent-decodes the path before this router sees it, so
the `%2F` the link carries arrives as a `/` and a one-segment route would 404 on an id its own
listing page just linked to. Nothing is nested under either prefix for the greedier match to
shadow. Where a redirect goes is `_html.py`'s `trace_path` and only its -- percent-encoding the id
whole is what keeps a crafted one a path on this app rather than another site (CWE-601).
"""

from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from runa import project
from runa.eval.corpus import CaseAlreadyInEvals, TraceHasNoInput, add_trace_to_evals
from runa.exceptions import OperatorError
from runa.tracing import TraceNotFound
from runa.web import agents as agents_page
from runa.web import evaluations as evaluations_page
from runa.web import sessions as sessions_page
from runa.web import traces as traces_page
from runa.web._html import empty, page, trace_path


def _error_page(active: str, message: str) -> str:
    return page(title="Error", active=active, body=f"<h1>Error</h1>{empty(message)}")


def create_app(root: Path) -> FastAPI:
    """Build the `runa ui` app for the project at `root`.

    Which project this dashboard reads is said here, once, rather than in each route: it reads six
    concerns out of `root/db/runa.db` and a route that forwarded `root` to a reader could only
    move the four that took it. `project.use` is the whole of that -- the app's data and its
    import -- and the Agents page is the one that needs the import, which `project.use` leaves
    until something reads the app's modules. So a dashboard still serves Sessions, Traces and
    Evaluations for an app whose `main.py` is broken, and only `/agents` reports it; and the
    import still happens once however many page views arrive, since a view that re-ran the
    developer's `main.py` would re-register whatever it registers (an exporter, a client) behind
    the page it rendered.

    `root` reaching a route at all is the filesystem half of the same answer: the two renderers
    that take it want files (`evals/`), not the database.
    """
    project.use(root)
    app = FastAPI(title="runa ui", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/agents")

    @app.get("/agents", response_class=HTMLResponse, include_in_schema=False)
    def agents_list() -> str:
        return agents_page.render(root=root)

    @app.get("/sessions", response_class=HTMLResponse, include_in_schema=False)
    def sessions_list() -> str:
        return sessions_page.render_list()

    @app.get("/sessions/{session_id:path}", response_class=HTMLResponse, include_in_schema=False)
    def session_detail(session_id: str) -> HTMLResponse:
        try:
            return HTMLResponse(sessions_page.render_detail(session_id))
        except sessions_page.SessionNotFound:
            return HTMLResponse(_error_page("Sessions", "Session not found."), status_code=404)

    @app.get("/traces/{trace_id:path}", response_class=HTMLResponse, include_in_schema=False)
    def trace_detail(trace_id: str) -> HTMLResponse:
        try:
            return HTMLResponse(traces_page.render_detail(trace_id, root=root))
        except TraceNotFound:
            return HTMLResponse(_error_page("", "Trace not found."), status_code=404)

    @app.post("/traces/{trace_id:path}/eval", include_in_schema=False, response_model=None)
    async def trace_add_to_evals(
        trace_id: str, request: Request
    ) -> HTMLResponse | RedirectResponse:
        form = parse_qs((await request.body()).decode())
        expected = form.get("expected", [""])[0].strip() or None
        try:
            add_trace_to_evals(trace_id, root=root, expected=expected)
        except CaseAlreadyInEvals:
            pass  # e.g. a double submit: the trace page already shows where the case is
        except TraceNotFound, TraceHasNoInput:
            return HTMLResponse(_error_page("", "Trace has no input to add."), status_code=404)
        return RedirectResponse(trace_path(trace_id), status_code=303)

    @app.get("/evaluations", response_class=HTMLResponse, include_in_schema=False)
    def evaluations_list() -> str:
        return evaluations_page.render_list()

    @app.get("/evaluations/{run_id}", response_class=HTMLResponse, include_in_schema=False)
    def evaluation_detail(run_id: int) -> HTMLResponse:
        try:
            return HTMLResponse(evaluations_page.render_detail(run_id))
        except evaluations_page.EvalRunNotFound:
            return HTMLResponse(
                _error_page("Evaluations", "Evaluation run not found."), status_code=404
            )

    @app.exception_handler(project.NotARunaProject)
    def _not_a_project(_request: Request, _exc: project.NotARunaProject) -> HTMLResponse:
        return HTMLResponse(_error_page("Agents", "Not a runa project."), status_code=400)

    @app.exception_handler(project.AppLoadError)
    def _app_load_error(_request: Request, _exc: project.AppLoadError) -> HTMLResponse:
        return HTMLResponse(_error_page("Agents", "Failed to load application."), status_code=500)

    @app.exception_handler(OperatorError)
    def _operator_error(_request: Request, _exc: OperatorError) -> HTMLResponse:
        """Catch-all 400 for an operator error no route above gave a better answer.

        Unlike `cli/main.py`, this app can't reduce every operator error to one response: a
        missing session is a 404 on a page that says so, and the routes above keep naming their
        own types to say which. This is the floor under them, so an `OperatorError` from a new
        surface renders a page instead of a 500, not a replacement for them.
        """
        return HTMLResponse(_error_page("", "Bad request."), status_code=400)

    return app
