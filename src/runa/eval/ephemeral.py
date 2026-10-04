"""eval/ephemeral.py: `EphemeralEvalStore`, eval history that dies with the process.

What `runa.db.evals()` resolves to under `RUNA_DATABASE_URL=memory://`. Rows are kept marshalled,
as `eval/store.py`'s `case_values` produces them and `to_run` reads them back, so a case's
`passed` and its `results` come back through the same mapping the SQL adapters use rather than
being handed back as the objects that went in.
"""

from datetime import UTC, datetime
from typing import Any

from runa.eval.report import Report
from runa.eval.store import CASE_COLUMNS, EvalRun, case_values, to_run

_runs: list[dict[str, Any]] = []
_cases: dict[int, list[dict[str, Any]]] = {}


def reset() -> None:
    """Drop every stored run. `runa.db.reset_ephemeral()` is how a test reaches this."""
    _runs.clear()
    _cases.clear()


class EphemeralEvalStore:
    """The in-process `EvalStore`: this process's eval history, gone when it exits.

    A handle, not a container: every instance reads and writes the one module-level history, so
    the run `agent.evaluate()` just saved is the baseline the next one compares against.
    """

    def save(self, report: Report) -> int:
        """Persist `report` as one run plus one row per case, returning the new run's id."""
        run_id = len(_runs) + 1  # ids count up from 1, as both SQL backends' sequences do
        _runs.append(
            {
                "id": run_id,
                "agent_name": report.agent_name,
                "created_at": _now(),
                "score": report.score,
                "pass_rate": report.pass_rate,
            }
        )
        _cases[run_id] = [
            dict(zip(CASE_COLUMNS, case_values(run_id, case), strict=True)) for case in report.cases
        ]
        return run_id

    def get(self, run_id: int) -> EvalRun | None:
        """Look up one run by id, with every case it graded, or `None` if it doesn't exist."""
        row = next((run for run in _runs if run["id"] == run_id), None)
        if row is None:
            return None
        return to_run(row, sorted(_cases.get(run_id, []), key=lambda case: case["case_index"]))

    def list(self, *, limit: int = 50) -> list[EvalRun]:
        """Return the most recent `limit` runs, newest first, without their cases."""
        return [to_run(row, []) for row in reversed(_runs[-limit:])]

    def baseline(self, agent_name: str, *, before: int | None = None) -> dict[str, bool] | None:
        """Map each input of `agent_name`'s latest run to whether it passed."""
        candidates = [
            run
            for run in _runs
            if run["agent_name"] == agent_name and (before is None or run["id"] < before)
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda run: run["id"])
        return {case["input"]: bool(case["passed"]) for case in _cases.get(latest["id"], [])}


def _now() -> str:
    """`created_at` in the `datetime.isoformat()` form both SQL adapters store."""
    return datetime.now(UTC).isoformat()


__all__ = ["EphemeralEvalStore", "reset"]
