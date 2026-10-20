---
name: reviewing-architecture
description: Reviews a codebase's architecture, or one module or diff of it, against John Ousterhout's A Philosophy of Software Design, checked against the project's own documented design intent. Use when asked to review architecture, module boundaries, abstractions, layering, coupling, information hiding, change cost or maintainability. Read-only. Produces a short, evidence-backed HTML report with a few ranked findings and redesign priorities.
argument-hint: "[path | diff]"
allowed-tools: Read, Grep, Glob, Write, Agent, Bash(git log:*), Bash(git diff:*), Bash(git show:*), Bash(git ls-files:*), Bash(git rev-parse:*), Bash(ls:*), Bash(wc:*), Bash(open:*), Bash(xdg-open:*)
---

# Reviewing Architecture

Find where design adds complexity and name the few changes that would remove the most of it.
Ousterhout's book is the lens; the repository is the subject. Don't summarize the book.

**Read-only.** Don't edit, refactor, format, commit or reconfigure anything in the repository.
The one file written is the report, and it goes outside the repository (see [Output](#output)).

## Scope

`$ARGUMENTS` picks what's reviewed:

* empty: the whole repository
* a path: that module or package, plus the boundaries it shares with the rest
* `diff`: the design impact of `git diff HEAD` (new or moved boundaries, leaked knowledge,
  repeated decisions), judged against the code it touches

A narrow scope still reads enough of the surrounding code to judge its boundaries.

## Workflow

### 1. Read the intended design first

Read `README`, `CLAUDE.md`, `AGENTS.md`, architecture docs, ADRs and any convention docs
(for example `RUNA.md`). They record **deliberate decisions**. Two consequences:

* Check the code against them. Where code and docs diverge, that's a finding.
* Don't report a documented, deliberate trade-off as a defect. Challenge it only by arguing
  against the decision itself, named as such, with the cost it imposes in this repository.

### 2. Map the system

Identify entrypoints, major modules, dependency direction, persistence, external integrations
and generated code. Directory layout is a hint, not the architecture: follow imports.

For a large repository, map it with 2 to 4 parallel `Explore` agents, each given one region, and
keep the judging yourself.

### 3. Trace 3 to 5 flows

Pick flows that cross the most boundaries (request to storage, CLI to external call, event to
side effect). For each, note which module owns each decision, where data changes representation,
and where errors are handled. These traces are the evidence for everything after.

### 4. Measure change amplification from history

Hypothetical "what if X changed" probes are weak evidence; real co-change is strong. Run:

```bash
git log --since="6 months ago" --name-only --pretty=format:"--%h %s" -- <scope>
```

Look for files that keep changing together across module boundaries, and for one kind of
change (a new provider, a new setting, a renamed concept) that touches many modules each time.
Quote commits as evidence. Use a hypothetical probe only where history is too short.

### 5. Judge each important boundary

For each major module ask:

* **Depth.** How much does it hide behind how small an interface? Classify deep, moderately
  deep, shallow or unclear. Line count alone proves nothing.
* **Leakage.** What implementation knowledge (representation, protocol, vendor type, storage
  id, ordering rule) has escaped into callers?
* **Pass-through.** Does this layer offer a different abstraction from its neighbors, or just
  rename and forward?
* **Pull down.** Do several callers handle complexity the module could absorb, and does that
  complexity belong to the module conceptually?
* **Generality.** Is the interface shaped around one caller, or, the opposite failure, generic
  for reuse nobody needs?
* **Together or apart.** Do pieces that share knowledge or always change together live apart,
  or do unrelated pieces share a module?
* **Errors.** Could a special case be defined out of existence instead of handled by every caller?
* **Obviousness and consistency.** Can behavior be predicted from local reading? Does the same
  concept behave the same way everywhere? A hard-to-name abstraction is a signal worth checking.
* **Duplicated decisions.** Is one rule (validation, retry, auth, a state transition) known in
  several places? Repeated knowledge matters more than repeated lines.

Tests are evidence too: heavy mocking or awkward construction points at a leaky boundary
worth investigating, not proof of one.

### 6. Design it twice

For each finding you'll report, sketch one plausible alternative and say concretely what
complexity it removes and what it costs.

### 7. Verify, then rank

Re-read every cited location and confirm the claim before it goes in the report. Drop anything
that rests only on naming, directory structure or taste.

**Severity** rates a finding:

* **Critical**: widespread complexity, or an important class of change is disproportionately hard
* **High**: a major boundary leaks or causes real change amplification
* **Medium**: a local issue with limited system-wide reach
* **Low**: a smell worth fixing opportunistically

**Priority** ranks the fixes: **P0** do first, **P1** do next, **P2** optional. A Low finding is
never P0. Don't inflate either.

## Evidence standard

Every finding cites `path:line` and quotes the few lines that show it. When evidence is partial,
say so and lower the confidence (High / Medium / Low).

Don't recommend, by reflex: splitting a class because it's large, an interface with one
implementation, an extra layer, a design pattern, generality for hypothetical callers, or
removing all duplication. A little duplication can beat an abstraction that couples unrelated
things, and one larger module can beat several shallow ones. The goal is less complexity, not
more abstraction.

## Output

Fill `assets/report-template.html` (read it when you reach this step) and write it to
`/tmp/architecture-review-<repo-name>-<YYYY-MM-DD>.html`. Sections, in order:

1. **Verdict.** Broadly aligned / Partially aligned / Materially misaligned, with the biggest
   strength, biggest weakness, highest-leverage redesign, biggest risk and overall confidence.
   Understandable on its own.
2. **Architecture at a glance.** A compact diagram of boundaries and dependency direction,
   annotated only with the most important leaks.
3. **Findings.** The 5 to 7 highest-impact, ordered by impact. Each: severity and title,
   principle, location, evidence (quoted), why it matters, alternative and recommendation,
   trade-off, confidence.
4. **What works.** 3 to 5 decisions that genuinely reduce complexity, each with evidence. No
   generic praise.
5. **Change-amplification hotspots.** 2 to 4 real scenarios from step 4: change, modules touched
   now, cause, better boundary, expected touch count after.
6. **Redesign priorities.** P0/P1/P2, the smallest set that addresses the main risk, each saying
   what complexity disappears. Not a backlog.
7. **What not to change.** Sound boundaries to keep, and tempting refactors that would add
   indirection without removing complexity.
8. **Scorecard.** Strong / Good / Mixed / Weak / Poor for: complexity management, module depth,
   information hiding, abstraction quality, layering, dependency direction, change
   amplification, error complexity, consistency, obviousness. No numeric score unless asked.

Open the report in the default browser straight after writing it (`open <path>` on macOS,
`xdg-open <path>` on Linux), without asking.

Then reply in chat with the verdict, the top three findings in one line each, and the report's
path. Don't paste the whole report. Offer to apply the P0 change as a follow-up; that is the
only path to editing code from here.

## Quality gate

Before writing the report, confirm:

* every finding has a verified `path:line` and quoted evidence
* no finding contradicts a documented, deliberate decision without saying so
* each recommendation removes complexity rather than adding abstraction by default
* strengths are recognized as specifically as weaknesses
* uncertain conclusions carry a lower confidence
