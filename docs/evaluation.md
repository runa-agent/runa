# Evaluating Agents

Runa distinguishes two kinds of checks. **Tests** verify invariants with a plain `assert`.
**Evals** grade behavior, including with a judge model, against a dataset of cases.

An eval is one file, `evals/<agent_name>.jsonl`, with one case per line. `runa generate agent`
creates it with every agent; for an agent written by hand, run
`runa generate evaluation support_agent`.

```json
{"input": "Where's my order #4821?", "expected": "Asks for or looks up the order status"}
{"input": "Cancel order A100", "expected_tool": "cancel_order"}
```

```bash
runa eval
```

The filename is the Agent's declared `name`, the same one `runa chat` takes, so there is nothing
to register. Each line's keys are `Case` fields, and a line only carries the ones it needs. A
bare string is shorthand for an input-only case: `"Where's my order?"`. `runa eval` calls `agent.evaluate(dataset)` for every file, the same path production evaluation
runs through.

Pass the Agent's `name` to run just its dataset, for example `runa eval support_agent`.

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

## Turning Real Runs Into Cases

The best cases come from runs that went wrong. Every run is traced, so add a bad one straight from
its trace:

```bash
runa eval --add TRACE_ID --expected "Looks up the order status"
```

Or open the trace in `runa ui` and use "Add to evals". Either way the run's input is appended to
its agent's `evals/<agent_name>.jsonl`, with the trace id kept in `metadata`. `--expected` is
optional, and can be filled in later in the file. An input already in the file isn't added twice.

## Catching Regressions

Each `runa eval` compares against the agent's previous run, matching cases by input. A case that
passed last time and fails now is a regression:

```
3 passed
1 failed
1 regressed (last run: 4/4 passed)

Failures
────────────────────────────────
case_2  regressed  task_completion: Didn't look up the order
```

`report.regressions` lists them in Python. `runa eval` still exits with 1 on any failure, so CI
fails on every broken case, not only new ones.

In `runa ui`, an evaluation run marks the same regressions, and every case links to the trace of
its run, so a failure opens straight onto the tool calls and model turns behind it.

## Building Cases in Python

When cases need code, for example a configured agent or generated inputs, write a module instead.
It declares module-level `agent` and `dataset`, any iterable of `Case`:

```python
# evals/support_agent_eval.py
from runa import Case

from app.agents import SupportAgent

agent = SupportAgent(model="gpt-5.4-nano")

dataset = [Case(input=f"Where's my order #{n}?") for n in range(4800, 4810)]
```

A module can still keep its cases in a file, and a `.jsonl` sharing its stem belongs to it rather
than running on its own:

```python
dataset = Dataset.from_jsonl(Path(__file__).with_suffix(".jsonl"))
```

## Choosing a Judge

Semantic metrics (task completion, answer correctness, answer relevance, faithfulness, tool
correctness) are graded by a judge model, by default the agent's own `model`:

```python
report = await agent.evaluate(dataset, judge="gpt-5.4")
```

Cases run 8 at a time. Pass `concurrency=1` for an agent whose tools can't run in parallel:

```python
await agent.evaluate(dataset, concurrency=1)
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

The same dataset, loaded from JSONL:

```python
--8<--"examples/13_eval/jsonl_dataset.py"
```

More in [`examples/13_eval/`](https://github.com/Benybrahim/runa/tree/main/examples/13_eval).

Tests are a separate, deterministic check, a bare `test_*` function with a plain `assert`:

```python
--8<--"examples/12_test/test_example.py"
```

More in [`examples/12_test/`](https://github.com/Benybrahim/runa/tree/main/examples/12_test).
