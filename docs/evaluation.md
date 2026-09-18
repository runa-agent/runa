# Evaluating Agents

Runa distinguishes two kinds of checks. **Tests** verify invariants with a plain `assert`.
**Evals** grade behavior, including with a judge model, against a dataset of cases.

`evals/` holds datasets of `Case`s, graded against an agent:

```bash
runa generate evaluation support_agent
```

```python
# evals/support_agent_eval.py
from runa import Case

from app.agents import SupportAgent

agent = SupportAgent()

dataset = [
    Case(
        input="Where's my order #4821?",
        expected="Asks for or looks up the order status",
    ),
]
```

```bash
runa eval
```

A module must declare module-level `agent` and `dataset`. `runa eval` imports every module under
`evals/` and calls `agent.evaluate(dataset)` on each, the same path production evaluation runs
through.

Pass the Agent's declared `name` to run just that one's module, for example
`runa eval support_agent`.

## What a `Case` Can Carry

Only `input` is required. Everything else is optional evidence that decides which metrics run:

| Field | Enables |
|---|---|
| `expected` | Answer-correctness grading |
| `expected_tool` | A deterministic "was it called" tool-correctness check |
| `context` | Faithfulness grading against retrieval passages |
| `metadata` | Arbitrary data carried through to the report |

With none of them, task completion and answer relevance still run. Every case runs to completion
even if an earlier one errors, and the finished `Report` is persisted to `runa.db`.

## Choosing a Judge

Semantic metrics (task completion, answer correctness, answer relevance, faithfulness, tool
correctness) are graded by a judge model, by default the agent's own `model`:

```python
report = await agent.evaluate(dataset, judge="gpt-5.4")
```

Override a pass threshold globally or per metric:

```python
await agent.evaluate(dataset, threshold=0.8)
await agent.evaluate(dataset, thresholds={"faithfulness": 0.9})
```

## Example

```python
--8<--"examples/13_eval/case_dataset.py"
```

Tests are a separate, deterministic check, a bare `test_*` function with a plain `assert`:

```python
--8<--"examples/12_test/test_example.py"
```
