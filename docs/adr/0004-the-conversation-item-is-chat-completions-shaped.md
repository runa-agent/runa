# ADR-0004: the conversation item is chat-completions-shaped, and says so

- Status: accepted
- Date: 2026-10-07

## Context

Every turn of conversation Runa passes around -- `agent.history`, what a
`session` stores, what a `Compactor` trims, what a serialized `RunState`
carries, what a `Model` is handed -- is one `dict[str, Any]`. The alias was
called `TResponseInputItem`, inherited from the `openai-agents` SDK Runa
replaced, and `_types.py` introduced it as one of "the provider-neutral
request/response shapes Runa's own runtime is built on".

Neutral is not what it is. The shape is OpenAI's chat-completions wire format:
`message["tool_calls"]`, `call["function"]["arguments"]` as a JSON *string*,
`{"role": "tool", "tool_call_id": ...}` for a result. The backends say so
plainly by their size -- `chat_completions.py` is largely passthrough,
`anthropic.py` is real translation -- so the docstring was the only place
claiming otherwise, and it is the kind of claim a contributor builds on.

Two costs followed from the shape being an unannotated dict rather than an owned
one. Reading an item was guesswork repeated locally: `content` is a string on
one turn and a list of parts on the next, so `guardrail._latest_text`,
`session/store.message_text`, `anthropic._text_content` and
`memory.remember_from_conversation` each had their own flattening, two of them
wrong for content shapes the other two handled. Parsing a call's arguments was
repeated five times (`tool_execution`, `tool`, `handoff`, `mcp`, `guardrail`),
each with its own answer for empty, malformed, and not-an-object. Adding a
content kind -- reasoning blocks, citations, audio -- meant finding all of them,
and missing one degraded to `""` or `str(item)` rather than failing.

## Decision

Name the shape for what it is and give it an owner. `runa/_items.py` holds the
`ConversationItem` alias, the argument for the chat-completions shape, and every
reader of one: `content_text`, `item_text`, `role_and_text`, `latest_text`,
`latest_user_index`/`latest_user_text`, `parsed_arguments`. Nothing else in Runa
reaches into `content` or re-parses `function.arguments`; `_types.py` keeps the
shapes that are Runa's own (`Usage`, `ModelSettings`, `RunContextWrapper`) and
points at `_items` for the one that isn't.

The shape itself stays, and stays chat-completions. There is no neutral format
to pick, only three real ones and the option of inventing a fourth. Adopting the
format two of the three major APIs already speak costs exactly one translating
backend; a Runa-native item would cost every backend a translation, including
the two that would otherwise need none, and it would still be one vendor's
model of a conversation wearing different field names.

`TResponseInputItem` is gone rather than aliased to the new name. It is exported
from `runa/__init__.py`, so this is a breaking rename for code that imported it
-- in practice a custom `Session` or `Compactor` annotation, since nothing is
constructed from it. Progress over stability: a vestigial name from a removed
dependency, kept as a second way to spell the public one, is precisely what
RUNA.md's closing rule says to close.

Rejected alternatives:

- **A typed item hierarchy** (`UserMessage`/`AssistantMessage`/`ToolResult`
  dataclasses, or a Pydantic union). This is the change the four parsers argue
  for, and it is the wrong one here. The item's plainness is load-bearing in
  three public places: `RunState.to_json` round-trips it, every `Session`
  stores it as a JSON blob, and application code reads `agent.history` without
  importing a Runa type. A hierarchy buys exhaustiveness at the cost of a
  serializer, a deserializer, and a vocabulary an app has to learn to read its
  own history -- and the parsers were never the real problem, their *number*
  was.
- **A genuinely provider-neutral internal format.** See above: a fourth shape,
  translated by every backend instead of one.
- **Keeping the name, fixing only the docstring.** Cheaper and not a breaking
  change. Rejected because `TResponse*` names a response type from an SDK that
  is no longer a dependency, and a name that has to be explained in prose is the
  defect, not its documentation.
- **Putting the accessors in `_types.py`.** One fewer module. Rejected because
  `_types.py` is Runa's own vocabulary and the item is the one shape in it that
  is a provider's; the split is what lets each module's docstring tell the truth
  about the shapes it holds.

## Consequences

- One place to change when a provider adds a content kind: `content_text`. A
  reader that wants an item's words calls `item_text` and cannot disagree with
  the next reader about what they are.
- Three fixed by consolidation rather than by intent: `memory`'s extraction read
  `output[0]["content"]` directly and saw `""` for a parts-list reply;
  `guardrail`'s flattening returned the string `"None"` for a contentless item;
  `tool`/`handoff`/`mcp` accepted a non-object JSON body and failed later,
  inside argument binding, instead of as a tool error the model can read.
- `runa.ConversationItem` replaces `runa.TResponseInputItem`, with no alias. An
  out-of-tree `Session` or `Compactor` updates one import; neither's runtime
  behavior changes, since the value is still the same plain dict.
- `session/store.py` no longer owns "how a stored message is flattened to text"
  -- one of the three rules its docstring existed to state. It owns the two that
  are its own (how a session id matches an agent, how a timestamp renders) and
  delegates the third, because a transcript reads an item rather than defining
  one.
- `parsed_arguments` raises `ValueError` and leaves the policy to its caller:
  the turn loop turns it into that call's result so the model can retry, a
  backend translating history lets it raise. The error strings the model sees
  are unchanged.
- **Not fixed by this ADR:** the item is still `dict[str, Any]`, so nothing
  stops a new call site from reading `item["content"]` itself. The accessors
  make the right thing the easy thing; they cannot make the wrong thing a type
  error. That is the price of the plainness the decision keeps.
