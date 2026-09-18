"""`knowledge = "auto"`: retrieving the application's own documents before every turn.

See RUNA.md #8 and docs/knowledge.md.

Unlike Memory, Knowledge isn't learned from conversations and isn't scoped to a user -- its
source of truth is a directory of files. Put Markdown/PDF/text/CSV files under a directory and
`.search` ingests them lazily on first use, no manual `.ingest()` call needed.

Run it:

    uv run python examples/08_knowledge/auto_knowledge.py
"""

from pathlib import Path

from runa import Agent, Knowledge

_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_KNOWLEDGE_DIR.mkdir(exist_ok=True)
(_KNOWLEDGE_DIR / "refund_policy.md").write_text(
    "# Refund Policy\n\nRefunds are issued within 3 business days for orders delayed 7+ days.\n"
)


class SupportAgent(Agent):
    """Answers policy questions, retrieving matching chunks from `_KNOWLEDGE_DIR` automatically."""

    name = "support_agent"
    instructions = "You are a helpful support assistant."
    knowledge = Knowledge(_KNOWLEDGE_DIR)


run = SupportAgent().run_sync("What's your refund policy for delayed orders?")
print(run.output)
