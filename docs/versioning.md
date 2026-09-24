# Versioning and Stability

Runa follows [semantic versioning](https://semver.org). Before `1.0`, a minor bump may contain a
breaking change; from `1.0` on, only a major one will.

## What is public

Anything importable from the top-level `runa` package, plus the documented submodules:

```python
from runa import Agent, tool, guardrail, approval, Run, ModelSettings   # public
from runa.session import SQLiteSession                                  # public
from runa.tracing import observe, trace, list_traces                    # public
from runa.memory import Memory
from runa.knowledge import Knowledge
from runa.eval import Case, Report
from runa.db.prune import prune
from runa.serve import create_app
```

A leading underscore means private, and that applies to whole modules as much as to names:
`runa._models`, `runa._types`, `runa.run_internal` and `runa._graph` are implementation. They
change without notice and without a deprecation cycle. If you find yourself importing from one,
that is worth an issue: either the thing you need should be public, or there is a public way to
do it that is not obvious enough.

The CLI is public too. A command's name, its flags, and its exit codes are part of the contract,
because they end up in CI pipelines and Dockerfiles. Its human-readable output is not: the exact
wording and layout of `runa traces list` may change in any release, so parse the database or the
HTTP API rather than the terminal.

## Supported Python

Runa supports the three most recent stable Python versions, currently **3.12, 3.13 and 3.14**,
and every one of them is tested in CI. Dropping a version is a breaking change and follows the
rule above.

## Deprecation

Something public is removed in two steps, never one:

1. It keeps working and warns. A `DeprecationWarning` names what to use instead, and the release
   notes say the same thing. The warning lands at least one minor release before removal.
2. It is removed, in the next release that is allowed to break (a minor before `1.0`, a major
   after).

An exception exists for security: something actively unsafe may be changed immediately, called
out in the release notes.

## What a release contains

`CHANGELOG.md` is generated from commit messages, so every user-visible change is in it. Commits
are prefixed `feat`, `fix`, `docs`, `refactor` or `test`, and the first two are what show up as
changes you can act on.
