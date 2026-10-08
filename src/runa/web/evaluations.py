"""web/evaluations.py: the Evaluations pages -- `agent.evaluate()` runs, from the eval store.

Data comes from the `EvalStore` `runa.db.evals()` hands back; this module only turns
`EvalRun`/`EvalCaseRow` into HTML. A case links to the trace of its run, and one
that passed in the agent's previous run but failed here is marked regressed.
"""

from datetime import datetime
from urllib.parse import quote

from runa import db
from runa.eval.store import EvalCaseRow, EvalRun
from runa.exceptions import OperatorError
from runa.web._html import back_link, chip, empty, empty_hint, escape, page, pre

__all__ = ["EvalRunNotFound", "render_detail", "render_list"]


class EvalRunNotFound(OperatorError):
    """Raised when `render_detail` names a run id this deployment has no record of.

    The same shape as `SessionNotFound`/`TraceNotFound`: an id off the request names nothing.
    Only the dashboard can reach it today, since no command takes a run id, but whose fault it
    is doesn't depend on which surface asked.
    """


def _fmt_timestamp(value: str) -> str:
    """`EvalRun.created_at`'s `datetime.isoformat()` form, trimmed to match `agent_sessions`'s."""
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _score_bar(pass_rate: float) -> str:
    pct = round(pass_rate * 100)
    return (
        f'<div style="display:flex; align-items:center; gap:8px">'
        f'<div class="score-bar"><div style="width:{pct}%"></div></div>'
        f'<span class="meta">{pct}%</span></div>'
    )


def _case_result_chip(result: dict) -> str:
    status = str(result.get("status", "")).lower()
    kind = "ok" if status == "pass" else ("default" if status == "skipped" else "error")
    return chip(f"{result.get('metric', '?')}: {result.get('status', '?')}", kind)


def _case_card(case: EvalCaseRow, *, regressed: bool) -> str:
    verdict = chip("passed", "ok") if case.passed else chip("failed", "error")
    if regressed:
        verdict = chip("regressed", "error") + " " + verdict
    trace_link = (
        f' <a class="meta" href="/traces/{escape(quote(case.trace_id))}">open trace ›</a>'
        if case.trace_id
        else ""
    )
    results = "".join(_case_result_chip(result) for result in case.results)
    reasons = "".join(
        f'<div class="field-label" style="margin-top:8px">{escape(result.get("metric", ""))}'
        f"</div><div>{escape(result.get('reason', ''))}</div>"
        for result in case.results
    )
    return f"""<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:center">
    <strong>case {case.index}{trace_link}</strong> <span>{verdict}</span>
  </div>
  <div class="field-label" style="margin-top:10px">Input</div>{pre(case.input)}
  {f'<div class="field-label">Output</div>{pre(case.output)}' if case.output else ""}
  <div class="chips" style="margin-top:10px">{results}</div>
  <details><summary>reasons</summary>{reasons}</details>
</div>"""


def render_list() -> str:
    """Render `/evaluations`: the most recent `agent.evaluate()` runs, newest first."""
    runs = db.evals().list(limit=100)
    if not runs:
        body = empty_hint("no evaluation runs yet, run", "runa eval")
    else:
        rows = "".join(
            f'<a class="row" href="/evaluations/{run.id}">'
            f'<span class="primary">{escape(run.agent_name)}</span>'
            f"{_score_bar(run.pass_rate)}"
            f'<span class="meta">{escape(_fmt_timestamp(run.created_at))}</span></a>'
            for run in runs
        )
        body = f'<div class="list">{rows}</div>'
    return page(
        title="Evaluations",
        active="Evaluations",
        body=f'<h1>Evaluations</h1><p class="subtitle">Every agent.evaluate() run.</p>{body}',
    )


def render_detail(run_id: int) -> str:
    """Render `/evaluations/{run_id}`: that run's summary plus every case it graded."""
    store = db.evals()
    run: EvalRun | None = store.get(run_id)
    if run is None:
        raise EvalRunNotFound(f"no eval run found with id {run_id!r}")
    baseline = store.baseline(run.agent_name, before=run.id) or {}
    last_run = f" · last run {sum(baseline.values())}/{len(baseline)} passed" if baseline else ""
    header = (
        f"<h1>{escape(run.agent_name)}</h1>"
        f'<p class="subtitle">{escape(_fmt_timestamp(run.created_at))} · score {run.score:.2f} · '
        f"{round(run.pass_rate * 100)}% passed{last_run}</p>"
    )
    cases = "".join(
        _case_card(case, regressed=not case.passed and baseline.get(case.input, False))
        for case in run.cases
    ) or empty("no cases in this run")
    body = back_link("/evaluations", "evaluations") + header + cases
    return page(title=f"{run.agent_name} eval", active="Evaluations", body=body)
