---
name: reviewing-software-design
description: Reviews a software codebase's architecture against the principles in John Ousterhout's A Philosophy of Software Design. Use when evaluating architecture, module boundaries, abstractions, dependencies, layering, complexity, information hiding, or maintainability. Produces evidence-based findings, severity-ranked issues, and concrete redesign recommendations without modifying the codebase.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Reviewing Software Design

# Purpose

Evaluate whether a codebase's architecture follows the principles from John Ousterhout's *A Philosophy of Software Design*, with particular emphasis on reducing complexity and creating deep, well-encapsulated modules.

The skill performs an evidence-based architecture review and produces a **standalone HTML report** containing the findings, architecture map, scorecard, recommendations, and overall assessment.

This is an **analysis and review skill**. Do not modify the repository unless the user explicitly asks for remediation after the review.


The review should answer:

1. Where is architectural complexity located?
2. Which module boundaries successfully hide complexity?
3. Where does complexity leak across boundaries?
4. Which abstractions are shallow, overly specialized, or misplaced?
5. Which dependencies, layers, or interfaces create unnecessary cognitive load?
6. What concrete architectural changes would reduce complexity?

---

# Core principles to evaluate

Use these principles as the review framework.

## 1. Complexity is the primary enemy

Look for:

* large numbers of dependencies
* change amplification
* duplicated decisions
* hidden coupling
* special cases
* surprising behavior
* knowledge spread across many modules
* abstractions that require understanding many implementation details

Distinguish between:

* **intrinsic complexity**: complexity inherent to the domain
* **accidental complexity**: complexity introduced by the software design

Prefer findings that identify accidental complexity.

---

## 2. Deep modules

A strong module provides substantial functionality behind a relatively simple interface.

Look for:

* small/simple public APIs hiding substantial implementation
* modules that absorb complexity for callers
* cohesive responsibilities
* high implementation-to-interface value

Warning signs:

* large public APIs
* wrappers that mostly forward calls
* many getters/setters exposing internals
* callers needing to understand implementation details
* abstractions with little functionality behind them

Do not judge modules by line count alone.

A large implementation with a simple interface may be a good deep module.

A small class with a complicated interface may be a bad shallow module.

---

## 3. Information hiding

For each important module, determine:

* What design decisions does it own?
* What representation details does it hide?
* What assumptions must callers know?
* Which internal decisions appear in other modules?

Look specifically for **information leakage**:

* shared knowledge of internal representations
* callers constructing internal objects directly
* duplicate logic around the same implementation detail
* APIs exposing data structures because they are convenient internally
* configuration or state that must be coordinated across modules

A module boundary is valuable when it allows one side to change without forcing knowledge changes on the other side.

---

## 4. General-purpose versus special-purpose abstractions

Prefer abstractions that are general enough to eliminate unnecessary coupling and duplication.

Look for overly specialized interfaces such as:

* APIs designed around one current caller
* abstractions named after a single use case
* duplicated abstractions with nearly identical semantics
* domain-independent infrastructure polluted with application-specific assumptions

Do not recommend generalization merely for theoretical reuse.

Generalize when doing so:

* simplifies callers
* hides more complexity
* eliminates duplication
* creates a stronger abstraction boundary

---

## 5. Different layer, different abstraction

Inspect the architecture from top to bottom.

For each layer, ask:

> What does this layer know or provide that the adjacent layer does not?

Identify:

* pass-through functions
* pass-through classes
* duplicated representations
* layers that merely rename concepts
* wrappers with no meaningful abstraction
* orchestration leaking into low-level components
* low-level implementation details leaking upward

A layering violation is especially important when a higher layer must know about several lower-level implementation details.

---

## 6. Pull complexity downward

When multiple callers independently handle complexity that could be encapsulated below them, flag it.

Look for:

* repeated validation
* repeated error handling
* repeated state transitions
* repeated protocol knowledge
* repeated serialization rules
* repeated database/storage assumptions
* callers branching on lower-level implementation details

Prefer designs where the lower-level abstraction absorbs complexity and exposes a simpler contract.

However, do not blindly move business rules downward. The complexity must belong conceptually to that lower-level abstraction.

---

## 7. Together or apart

Evaluate whether related responsibilities are correctly grouped.

Consider combining code when:

* it shares important knowledge
* separating it creates unnecessary interfaces
* callers repeatedly coordinate both pieces
* the two pieces change together
* separation causes duplicated assumptions

Consider separating code when:

* responsibilities have different knowledge
* they evolve independently
* one abstraction is general-purpose and the other is application-specific
* the combined module becomes difficult to reason about
* unrelated users depend on unrelated portions

Do not use class size or "single responsibility" slogans as the deciding rule.

Judge the effect on system complexity.

---

## 8. Define errors out of existence

Inspect common error paths and edge cases.

Ask:

> Can this design make the situation ordinary rather than exceptional?

Look for:

* repeated defensive checks
* callers handling the same special case
* null/absence handling spread across many modules
* duplicated validation
* avoidable invalid intermediate states
* APIs that force callers into complicated branching

Prefer designs that make invalid or exceptional states harder to represent when appropriate.

Do not recommend hiding genuinely important errors.

---

## 9. Design it twice

For significant architectural decisions, consider at least one plausible alternative.

The review should not assume the current design is optimal simply because it works.

For important findings, briefly compare:

* current design
* plausible alternative
* why the alternative reduces complexity

Prioritize high-leverage architectural alternatives rather than cosmetic refactoring.

---

## 10. Comments and names as architectural signals

Although this review is architecture-first, comments and names can reveal architectural problems.

Inspect important public interfaces and boundaries for:

* vague names
* misleading names
* comments that explain implementation instead of intent
* comments compensating for confusing abstractions
* interfaces whose semantics require extensive explanation
* names that expose accidental implementation details

A difficult-to-name abstraction may indicate a difficult-to-understand abstraction.

Do not criticize comments merely because they are absent. Judge whether important design intent is communicated.

---

## 11. Consistency

Look for consistent:

* abstractions
* naming
* dependency directions
* error semantics
* lifecycle management
* data access patterns
* layering conventions
* API behavior

Inconsistency matters because it forces developers to repeatedly learn exceptions.

Do not demand consistency when consistency would preserve a bad abstraction.

---

## 12. Obviousness

Ask whether a competent developer can predict behavior from local inspection.

Look for:

* surprising side effects
* hidden global state
* implicit dependencies
* magic behavior
* non-obvious lifecycle rules
* APIs with inconsistent semantics
* behavior that depends on distant code

Prefer designs where the obvious interpretation is usually correct.

---

# Review workflow

Follow this sequence.

## Phase 1 — Establish the architecture

First inspect the repository without judging it.

Identify:

* language(s)
* frameworks
* build system
* executable/service entrypoints
* major packages/modules
* domain boundaries
* infrastructure boundaries
* persistence/data access
* external integrations
* configuration
* tests
* generated code
* deployment/runtime boundaries

Read the repository's existing architecture documentation when available:

* README
* architecture/design docs
* ADRs
* package/module documentation
* contributor documentation
* configuration documentation

Treat documentation as a hypothesis to verify against the implementation.

Do not assume directory structure equals architecture.

---

## Phase 2 — Trace the important flows

Trace several representative flows through the system.

Prefer flows that cross multiple architectural boundaries, such as:

* request → application/service → domain → persistence
* event → handler → business logic → external system
* CLI/API → orchestration → domain → infrastructure
* background job → queue → worker → storage

For each flow, record:

* entrypoint
* major modules
* dependencies
* transformations
* ownership of important decisions
* error handling
* state ownership

This provides concrete evidence for later architectural judgments.

---

## Phase 3 — Identify module boundaries

For every major module/package/component, determine:

1. What does it provide?
2. What does it hide?
3. What does it depend on?
4. Who depends on it?
5. What knowledge does it force onto callers?
6. What reasons would require changing it?
7. What complexity would callers have without it?

Classify important modules as approximately:

* **deep**
* **moderately deep**
* **shallow**
* **unclear**

Explain the classification with evidence.

---

## Phase 4 — Analyze dependencies

Build a conceptual dependency graph.

Look for:

* cycles
* fan-out
* fan-in
* unstable dependencies
* infrastructure leaking into domain/application code
* upper layers depending on implementation details
* cross-module knowledge of representations
* dependency directions that make change expensive

Do not report dependency count by itself as a problem.

Explain why a dependency creates complexity.

---

## Phase 5 — Analyze change amplification

Use likely future changes as probes.

Ask:

> “If this requirement changed, how many places would probably need to change?”

Useful probes include:

* changing persistence technology
* changing an external API
* adding a new implementation
* changing a business rule
* changing a data representation
* adding another consumer
* changing authentication/authorization
* introducing another UI/API client
* changing an error policy

Flag designs where small conceptual changes require coordinated edits across many unrelated modules.

---

## Phase 6 — Look for information leakage

For each major boundary, ask:

> “What implementation knowledge has escaped this module?”

Examples:

* database schemas leaking into domain logic
* HTTP concepts leaking into domain objects
* vendor SDK types spreading through the codebase
* storage-specific IDs exposed everywhere
* serialization formats becoming domain concepts
* protocol details replicated across callers

Information leakage is a high-value finding.

---

## Phase 7 — Look for shallow modules and pass-through layers

Search for structural symptoms such as:

* methods that simply delegate
* classes that only wrap another class
* interfaces implemented by one trivial adapter
* packages that add naming but no abstraction
* repetitive DTO/model transformations
* endpoints containing large amounts of business logic
* service classes that orchestrate many low-level details

Do not automatically recommend deleting wrappers.

Explain whether the boundary actually hides complexity.

---

## Phase 8 — Look for duplicated decisions

Search for places where multiple modules independently know the same rule.

Examples:

* authorization rules
* validation rules
* retries
* caching policies
* lifecycle rules
* serialization rules
* state transitions
* business invariants

Repeated knowledge is often more important than duplicated lines of code.

---

## Phase 9 — Evaluate error complexity

Find error handling that is:

* repeated
* inconsistent
* overly defensive
* coupled to implementation details
* forced onto many callers

Identify candidates where the abstraction could absorb the complexity.

---

## Phase 10 — Rank architectural findings

Classify findings by severity.

### Critical

The architecture creates widespread complexity or makes an important class of changes disproportionately difficult.

### High

A major boundary leaks complexity or causes significant change amplification.

### Medium

A local architectural issue increases complexity but has limited system-wide impact.

### Low

A design smell or inconsistency worth addressing opportunistically.

Do not inflate severity.

---

# Evidence standard

Every significant finding must include concrete evidence from the repository.

A finding should contain:

**Principle**
Which Ousterhout principle is involved.

**Location**
Relevant files, modules, classes, packages, or dependency relationships.

**Evidence**
What the code actually does.

**Complexity impact**
Why this increases cognitive load, change amplification, dependency, or obscurity.

**Recommendation**
A concrete architectural change.

**Trade-off**
What becomes more complex, less flexible, or more constrained as a result.

**Confidence**
High / Medium / Low.

Never claim that a problem exists solely from naming conventions or directory structure.

When evidence is incomplete, say so explicitly.

---

# Avoid false positives

Do not recommend:

* splitting classes merely because they are large
* adding interfaces merely for abstraction's sake
* creating more layers because "clean architecture" says so
* making every module generic
* eliminating all duplication
* eliminating all dependencies
* hiding every error
* pushing all logic downward
* adding design patterns automatically
* optimizing architecture for hypothetical future requirements

The goal is **less complexity**, not maximum abstraction.

A small amount of duplication may be preferable to an abstraction that couples unrelated concepts.

A dependency may be healthy when it represents an appropriate ownership relationship.

A larger module may be better than several shallow modules.

---

# Architecture scorecard

After the detailed analysis, provide a qualitative scorecard.

Use:

* **Strong**
* **Good**
* **Mixed**
* **Weak**
* **Poor**

Evaluate:

| Dimension             | Assessment |
| --------------------- | ---------- |
| Complexity management |            |
| Module depth          |            |
| Information hiding    |            |
| Abstraction quality   |            |
| Layering              |            |
| Dependency direction  |            |
| Change amplification  |            |
| Error complexity      |            |
| Consistency           |            |
| Obviousness           |            |

Do not compute a single numeric score unless the user explicitly requests one.

The purpose is diagnosis, not pretending architectural quality is precisely measurable.

---

# Final report format

Use this structure:

## Executive summary

Give the 3–7 most important conclusions.

State whether the architecture is:

* broadly aligned
* partially aligned
* materially misaligned

Explain the main reason.

## Architecture map

Describe the major modules/layers and dependency direction.

Use a compact ASCII diagram when useful.

Example:

```text
API
 ↓
Application
 ↓
Domain
 ↓
Persistence
 ↓
Database
```

Annotate important leakage or coupling.

## Strong aspects

Identify architectural decisions that successfully reduce complexity.

For each:

* principle
* evidence
* why it works

## Findings

For each finding:

```text
### [HIGH] Shallow boundary between X and Y

Principle:
Deep modules / Information hiding

Evidence:
...

Why it matters:
...

Recommendation:
...

Trade-off:
...

Confidence:
High
```

Order findings from highest architectural impact to lowest.

## Change-amplification hotspots

Describe changes that currently require edits across many modules.

## Information-leakage hotspots

Identify representations, implementation details, or decisions leaking across boundaries.

## Recommended redesign priorities

Give the smallest set of high-leverage changes.

Prefer recommendations that:

1. simplify many callers
2. hide knowledge in one place
3. remove dependencies
4. eliminate repeated decisions
5. reduce future change amplification

Do not provide a long refactoring backlog.

## What not to change

Explicitly identify parts of the architecture that should remain as they are, especially when a tempting refactor would add abstraction without reducing complexity.

## Overall assessment

Finish with:

* strongest architectural principle currently demonstrated
* weakest principle
* highest-leverage redesign
* biggest architectural risk
* confidence in the overall assessment

---

# Important operating rules

## Be repository-specific

Do not produce a generic summary of Ousterhout's book.

The book is the evaluation framework; the repository is the subject.

## Prefer architectural evidence over style opinions

A finding is valuable when it explains why a developer will struggle to understand or change the system.

## Follow dependencies, not just files

The important unit of analysis is often the **knowledge dependency**, not the source file.

## Look for concentrated knowledge

Good architecture often means one module owns a difficult decision instead of many modules knowing fragments of it.

## Consider both directions

Ask both:

> “What complexity does this module hide?”

and:

> “What complexity does this module force onto its callers?”

## Treat tests as architectural evidence

Tests can reveal:

* leaked implementation details
* difficult construction
* excessive mocking
* hidden dependencies
* unclear ownership
* inappropriate module boundaries

Do not use test difficulty as automatic proof of bad architecture; use it as evidence to investigate.

## Treat documentation as evidence, not truth

Compare architecture documentation with implementation.

If they diverge, report the divergence as a potential maintenance problem.

## Do not modify the repository

This skill performs review only.

Do not:

* edit files
* refactor code
* reformat code
* generate commits
* change configuration

unless the user explicitly asks for a follow-up implementation after the review.

---

# Quick trigger checklist

Before concluding, verify that the review considered:

* [ ] Where is complexity concentrated?
* [ ] Which modules are deep?
* [ ] Which modules are shallow?
* [ ] What information leaks across boundaries?
* [ ] Which layers merely pass through?
* [ ] Where is complexity duplicated across callers?
* [ ] Which abstractions are overly specialized?
* [ ] Which abstractions are unnecessarily generic?
* [ ] Where does change amplify across modules?
* [ ] Which errors could potentially be defined out of existence?
* [ ] Which dependencies create cognitive load?
* [ ] Where is behavior non-obvious?
* [ ] Where is inconsistency forcing developers to learn exceptions?
* [ ] Which architectural decisions are genuinely strong?
* [ ] What are the few highest-leverage redesigns?

The final review should be **specific enough that another engineer could use it as an architecture discussion document**.

# HTML REPORT OUTPUT

The final review must be delivered as a **standalone HTML report**, not only as Markdown or plain text.

## Required output

After completing the repository analysis:

1. Generate a complete HTML document.
2. Save it as:
   `architecture-review.html`
3. Return the generated HTML file to the user as a downloadable artifact.
4. The HTML must contain the complete review; do not require the user to copy Markdown into an HTML file manually.
5. Do not return only a description of the report or an HTML snippet.

## HTML structure

The generated report should contain:

* a clear report title
* executive summary
* architecture map
* strong aspects
* findings
* change-amplification hotspots
* information-leakage hotspots
* recommended redesign priorities
* what not to change
* architecture scorecard
* overall assessment
* review scope / confidence notes

Use semantic HTML elements such as:

* `<header>`
* `<main>`
* `<section>`
* `<article>`
* `<table>`
* `<footer>`

## Presentation requirements

The report should be professional and easy to scan.

Include:

* responsive layout for desktop and mobile
* clear heading hierarchy
* readable typography
* visually distinct severity levels
* styled scorecard
* tables for structured findings
* code blocks / diagrams where useful
* callout boxes for important conclusions
* sufficient spacing and visual hierarchy

Use CSS in the same HTML file.

The report should work when opened directly in a browser without requiring a build step.

## Self-contained requirement

Prefer a **single self-contained HTML file**.

Do not require:

* a JavaScript framework
* a CSS build system
* a bundler
* a server
* external assets

Avoid external CDN dependencies unless the user explicitly requests them.

## Content fidelity

The HTML report must preserve the architecture-review reasoning defined by this skill.

Do not replace the repository-specific analysis with a generic explanation of Ousterhout's principles.

Every significant finding must retain:

* Principle
* Location
* Evidence
* Complexity impact / Why it matters
* Recommendation
* Trade-off
* Confidence

The qualitative scorecard must use:

* Strong
* Good
* Mixed
* Weak
* Poor

Do not calculate a numeric architecture score unless the user explicitly requests one.

## Output behavior

The final assistant response should:

* briefly summarize the completed review
* provide the generated `.html` file as a downloadable artifact
* mention any important limitations or incomplete evidence

Do not paste the entire HTML source into the chat unless the user explicitly asks for the source code.

## Important

The HTML generation step happens **after** the repository review is complete.

The workflow is:

Repository
→ establish architecture
→ trace flows
→ analyze boundaries and dependencies
→ identify findings
→ rank findings
→ produce recommendations
→ generate final HTML report
→ return `architecture-review.html`
