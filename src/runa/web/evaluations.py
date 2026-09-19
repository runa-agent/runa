"""web/evaluations.py: the Evaluations pages -- `agent.evaluate()` runs, from `db/runa.db`.

Data comes from `runa.eval.storage` (`list_eval_runs`/`get_eval_run`); this module only turns
`EvalRun`/`EvalCaseRow` into HTML.
"""

from datetime import datetime
from pathlib import Path

from runa.cli._project import resolve_db_path
from runa.eval.storage import EvalCaseRow, EvalRun, get_eval_run, list_eval_runs
from runa.web._html import back_link, chip, empty, empty_hint, escape, page, pre

__all__ = ["EvalRunNotFound", "render_detail", "render_list"]


class EvalRunNotFound(Exception):
    """Raised when `render_detail` names an `eval_runs.id` `db/runa.db` has no record of."""


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


def _case_card(case: EvalCaseRow) -> str:
    verdict = chip("passed", "ok") if case.passed else chip("failed", "error")
    results = "".join(_case_result_chip(result) for result in case.results)
    reasons = "".join(
        f'<div class="field-label" style="margin-top:8px">{escape(result.get("metric", ""))}'
        f"</div><div>{escape(result.get('reason', ''))}</div>"
        for result in case.results
    )
    return f"""<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:center">
    <strong>case {case.index}</strong> {verdict}
  </div>
  <div class="field-label" style="margin-top:10px">Input</div>{pre(case.input)}
  {f'<div class="field-label">Output</div>{pre(case.output)}' if case.output else ""}
  <div class="chips" style="margin-top:10px">{results}</div>
  <details><summary>reasons</summary>{reasons}</details>
</div>"""


def render_list(*, root: Path) -> str:
    """Render `/evaluations`: the most recent `agent.evaluate()` runs, newest first."""
    runs = list_eval_runs(limit=100, db_path=resolve_db_path(root))
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


def render_detail(run_id: int, *, root: Path) -> str:
    """Render `/evaluations/{run_id}`: that run's summary plus every case it graded."""
    run: EvalRun | None = get_eval_run(run_id, db_path=resolve_db_path(root))
    if run is None:
        raise EvalRunNotFound(f"no eval run found with id {run_id!r}")
    header = (
        f"<h1>{escape(run.agent_name)}</h1>"
        f'<p class="subtitle">{escape(_fmt_timestamp(run.created_at))} · score {run.score:.2f} · '
        f"{round(run.pass_rate * 100)}% passed</p>"
    )
    cases = "".join(_case_card(case) for case in run.cases) or empty("no cases in this run")
    body = back_link("/evaluations", "evaluations") + header + cases
    return page(title=f"{run.agent_name} eval", active="Evaluations", body=body)
