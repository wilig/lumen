---
name: Lumen is a general-purpose language
description: Lumen has evolved from a short-lived AI-agent scripting language to a general-purpose language. Design decisions should no longer assume programs are short-lived or exit quickly. Long-running services, actors, and large data structures are in scope.
type: project
---

Lumen is no longer scoped as a "weekend prototype" or "short-lived programs for AI agents."
It is a general-purpose statically-typed language targeting native via Cranelift.

**Why:** The language surface (types, match, Result/Option, effects, FFI, GC) has grown
past the original "AI writes, human audits, program exits" use case. The next phase
(actors, concurrency, real GC) requires treating it as a long-running systems language.

**How to apply:** Don't use "programs are short-lived" as a justification for cutting
corners on memory management, data structure efficiency, or runtime costs. Every design
decision should hold up for a server that runs for weeks.
