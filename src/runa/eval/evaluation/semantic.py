"""eval/evaluation/semantic.py: how each judge-graded metric reaches a score.

One `grade_*` pipeline per judged metric, each a `(score, reason)` for one case. Which of them
run, what they're called, and what score counts as a pass are not here: that's the metric's
identity, declared once in `eval/evaluation/metrics.py`, which pairs each entry with its pipeline
and iterates them. What's here is the prompts, which are a metric's content.

A pipeline is several judge calls rather than a single "grade this" prompt: an extraction step
pulls out what needs grading (statements, claims, a task/outcome pair), a verdict step grades each
extracted piece, and the score is a plain aggregate of those verdicts. This mirrors how LLM-judge
frameworks get more reliable scores than a single holistic call would.

A pipeline raises rather than scoring a case it couldn't grade; `Metric.evaluate` turns that into
an `ERROR` result.
"""

import asyncio

from runa.eval.case import Case
from runa.eval.judge import JudgeModel, extract_json
from runa.run import Run

_CORRECTNESS_CRITERIA = (
    "Determine whether the actual output is substantively correct relative to the expected "
    "output: it may be worded differently, but must not contradict or omit its key claims."
)


def _format_tool_calls(run: Run) -> str:
    if not run._tool_calls:
        return "(no tools were called)"
    return "\n".join(
        f"- {call.name}(arguments={call.arguments!r}) -> {call.output!r}"
        for call in run._tool_calls
    )


async def grade_task_completion(judge: JudgeModel, case: Case, run: Run) -> tuple[float, str]:
    """Score how completely the outcome the run achieved matches the task the input asked for."""
    extracted = extract_json(
        await judge.ask(
            "Given a user's request, the tools an AI called while handling it, and the AI's "
            "final response, identify (1) the task the user wanted done and (2) the factual "
            "outcome the AI actually achieved, derived strictly from what happened, with no "
            "judgment of quality.\n\n"
            f"Request:\n{case.input}\n\n"
            f"Tools called:\n{_format_tool_calls(run)}\n\n"
            f"Response:\n{run.output}\n\n"
            'Reply with JSON only: {"task": "...", "outcome": "..."}'
        )
    )
    verdict = extract_json(
        await judge.ask(
            "Given a task and the outcome actually achieved, score how completely the outcome "
            "achieves the task, from 0 (not at all) to 1 (perfectly).\n\n"
            f"Task:\n{extracted['task']}\n\n"
            f"Outcome:\n{extracted['outcome']}\n\n"
            'Reply with JSON only: {"verdict": <float 0-1>, "reason": "..."}'
        )
    )
    return float(verdict["verdict"]), str(verdict["reason"])


async def grade_answer_relevance(judge: JudgeModel, case: Case, run: Run) -> tuple[float, str]:
    """Score the fraction of the output's statements that address the input."""
    statements = extract_json(
        await judge.ask(
            "Break the following text down into a list of the individual statements it makes. "
            "An ambiguous fragment counts as its own statement.\n\n"
            f"Text:\n{run.output}\n\n"
            'Reply with JSON only: {"statements": ["...", ...]}'
        )
    )["statements"]
    if not statements:
        return 1.0, "The response made no distinct statements to evaluate."

    verdicts = extract_json(
        await judge.ask(
            "For each statement below, judge whether it is relevant to addressing the given "
            "input: 'yes' (relevant), 'no' (irrelevant), or 'idk' (ambiguous, but possibly "
            "supporting information). Give a reason only for 'no' or 'idk'. Return exactly one "
            "verdict per statement, in the same order.\n\n"
            f"Input:\n{case.input}\n\n"
            f"Statements:\n{statements}\n\n"
            "Reply with JSON only: "
            '{"verdicts": [{"verdict": "yes|no|idk", "reason": "..."}, ...]}'
        )
    )["verdicts"]
    irrelevant = [v.get("reason", "") for v in verdicts if v["verdict"].strip().lower() == "no"]
    score = sum(1 for v in verdicts if v["verdict"].strip().lower() != "no") / len(verdicts)

    reason = extract_json(
        await judge.ask(
            "Given an answer-relevancy score and the reasons any statements were judged "
            "irrelevant to the input, write one concise sentence justifying the score. If there "
            "are no irrelevant statements, say something positive instead.\n\n"
            f"Score: {score:.2f}\n\n"
            f"Input:\n{case.input}\n\n"
            f"Irrelevant statements:\n{irrelevant}\n\n"
            'Reply with JSON only: {"reason": "..."}'
        )
    )["reason"]
    return score, reason


async def grade_answer_correctness(judge: JudgeModel, case: Case, run: Run) -> tuple[float, str]:
    """Score the output against `case.expected`, on steps derived from the criteria."""
    steps = extract_json(
        await judge.ask(
            "Given the evaluation criteria below, write 3-4 concise steps for judging how well "
            "an actual output matches an expected output.\n\n"
            f"Criteria:\n{_CORRECTNESS_CRITERIA}\n\n"
            'Reply with JSON only: {"steps": ["...", ...]}'
        )
    )["steps"]
    graded = extract_json(
        await judge.ask(
            "Following the evaluation steps below, score the actual output from 0 (no "
            "alignment) to 10 (strong alignment) against the expected output. Ground your "
            "reasoning in specific details, without stating the score itself in it.\n\n"
            f"Evaluation steps:\n{steps}\n\n"
            f"Input:\n{case.input}\n\n"
            f"Actual output:\n{run.output}\n\n"
            f"Expected output:\n{case.expected}\n\n"
            'Reply with JSON only: {"score": <int 0-10>, "reason": "..."}'
        )
    )
    return float(graded["score"]) / 10, str(graded["reason"])


async def grade_faithfulness(judge: JudgeModel, case: Case, run: Run) -> tuple[float, str]:
    """Score the fraction of the output's claims that `case.context` doesn't contradict."""
    context = "\n\n".join(case.context or [])
    truths_reply, claims_reply = await asyncio.gather(
        judge.ask(
            "List the factual, undisputed truths that can be inferred from the following "
            "context. Include a truth even if you can't verify it's actually correct, only that "
            "the context asserts it.\n\n"
            f"Context:\n{context}\n\n"
            'Reply with JSON only: {"truths": ["...", ...]}'
        ),
        judge.ask(
            "List the factual claims made in the following AI output, taken at face value. "
            "Each claim should keep the full context it was made in, not be cherry-picked.\n\n"
            f"Output:\n{run.output}\n\n"
            'Reply with JSON only: {"claims": ["...", ...]}'
        ),
    )
    truths = extract_json(truths_reply)["truths"]
    claims = extract_json(claims_reply)["claims"]
    if not claims:
        return 1.0, "The response made no factual claims to check against the context."

    verdicts = extract_json(
        await judge.ask(
            "For each claim below, judge whether it contradicts the given context: 'yes' "
            "(agrees), 'no' (contradicts), or 'idk' (the context doesn't say either way). Give a "
            "reason only for 'no' or 'idk', and correct the claim using the context where you "
            "can. Return exactly one verdict per claim, in the same order.\n\n"
            f"Context:\n{'\n\n'.join(truths)}\n\n"
            f"Claims:\n{claims}\n\n"
            "Reply with JSON only: "
            '{"verdicts": [{"verdict": "yes|no|idk", "reason": "..."}, ...]}'
        )
    )["verdicts"]
    contradictions = [v.get("reason", "") for v in verdicts if v["verdict"].strip().lower() == "no"]
    score = sum(1 for v in verdicts if v["verdict"].strip().lower() != "no") / len(verdicts)

    reason = extract_json(
        await judge.ask(
            "Given a faithfulness score and the contradictions found between an AI's output and "
            "its context, write one concise sentence justifying the score. If there are no "
            "contradictions, say something positive instead.\n\n"
            f"Score: {score:.2f}\n\n"
            f"Contradictions:\n{contradictions}\n\n"
            'Reply with JSON only: {"reason": "..."}'
        )
    )["reason"]
    return score, reason


__all__ = [
    "grade_answer_correctness",
    "grade_answer_relevance",
    "grade_faithfulness",
    "grade_task_completion",
]
