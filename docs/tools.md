# Tools

A tool is a plain Python function. `@tool` derives its JSON schema from the function's signature
and docstring. There is nothing else to declare.

```python
from runa import tool


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    ...
```

* The docstring becomes the tool's description.
* Type hints become the JSON schema. `str`, `int`, `float`, `bool`, `list[T]`, `dict`,
  `Literal[...]`, `Enum` subclasses, and `T | None` are all understood. Anything else falls back
  to an unconstrained schema.
* A parameter with no default is required. One with a default is optional.

Attach tools to an agent:

```python
class WeatherAgent(Agent):
    name = "weather_agent"
    instructions = "You answer questions about the weather."
    tools = [get_weather]
```

The model decides on its own when to call a tool. You never invoke it directly.

## Async Tools

`@tool` works on `async def` functions the same way:

```python
@tool
async def fetch_price(symbol: str) -> float:
    """Look up a stock's current price."""
    ...
```

## Reserved Parameters

Name a parameter `ctx` or `call_id` to receive the run's `RunContextWrapper` or the tool call's
id instead of a model-supplied argument. Neither appears in the tool's schema:

```python
@tool
def whoami(ctx) -> str:
    """Return the current user's name from run context."""
    return ctx.context.user_name
```

## Gating a Tool Call

A tool call can be checked before it runs, or paused for a human to approve:

* [Guardrails](guardrails.md): a predicate that stops the run if it trips.
* [Human Approval](approval.md): a predicate that pauses the run for a person to decide.

## Example

```python
--8<--"examples/02_tool/basic_tool.py"
```

```python
--8<--"examples/02_tool/async_tool.py"
```

More in [`examples/02_tool/`](https://github.com/Benybrahim/runa/tree/main/examples/02_tool).
