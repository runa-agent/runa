"""eval/case.py: `Case`, one eval input plus what "good" looks like for it.

Deliberately minimal: `input` is required, everything else is optional.
`context` is what the case's run is given; `expected`, `expected_tool` and
`passages` are signals `evaluate_agent()` (see `eval/evaluate.py`) uses to
decide which metrics apply. Adding any of them never requires touching an
existing `Case`.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Case:
    """One eval case: an input, plus optional evidence of what a good run looks like.

    `context` is passed to the run as `Agent.run(context=...)`, for an agent
    whose instructions or tools need one. `expected` enables answer-correctness
    grading; `expected_tool` enables the deterministic "was it called"
    (tool-correctness) check; `passages` (retrieval passages the agent should
    be grounded in) enables faithfulness grading. Any combination is valid,
    including none of them, in which case only task completion and answer
    relevance run.
    """

    input: str
    context: Any = None
    expected: str | None = None
    expected_tool: str | None = None
    passages: list[str] | None = None
    metadata: dict[str, Any] | None = None
