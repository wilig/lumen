# Lumen

A statically-typed, AI-first programming language compiling to WebAssembly. Targets Wasmtime.

**Status:** planning / pre-implementation.

See:
- [`docs/design.md`](docs/design.md) — design rationale and decisions
- [`docs/grammar.ebnf`](docs/grammar.ebnf) — formal grammar
- [`docs/fluency-validation.md`](docs/fluency-validation.md) — LLM fluency check and findings
- Beads issues (`bd list`) — tracked work

## Core idea

Most Lumen code will be written by AI agents. Humans will spend their time reading and auditing. That inverts the usual language-design tradeoffs:

- Few constructs, one way to do everything
- Token-efficient but readable
- Strong typing, no nulls, no exceptions
- Errors carry automatic provenance (source location + argument values)
- Looks like TypeScript so LLMs are already fluent
