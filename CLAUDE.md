# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
cargo build              # Build the Lumen compiler
cargo test               # Run all tests (lexer, parser, type checker)
cargo run -- build FILE  # Compile a .lm file to a native binary

# Examples
cargo run -- build examples/hello.lm && ./examples/hello
cargo run -- build examples/bricks.lm && ./examples/bricks  # requires raylib
```

Raylib is optional. On Linux it looks in `/usr/lib/odin/vendor/raylib/linux`.
On macOS it uses `brew --prefix raylib`. Programs that `import std/raylib`
link against it; programs without the import don't need it installed.

## Architecture Overview

Lumen is a statically-typed language that compiles to native code via Cranelift.

**Compiler pipeline:** `src/lexer.rs` -> `src/parser.rs` -> `src/types.rs` -> `src/native.rs` -> object file -> system linker (`cc`)

**Runtime** (`runtime/rt.c`): Actor message queue, green threads (ucontext + epoll/kqueue), TCP/HTTP helpers, dynamic list operations, RC alloc helpers.

**FFI bridge** (`runtime/raylib_bridge.c`): Converts Lumen's f64/i64 values to raylib's f32 structs/Color types. Linked only when `import std/raylib` is present.

**Memory model:** Reference counting with magic sentinel (0x4C554D45). `rc_alloc` uses malloc, stores rc+magic in an 8-byte header before the payload. Lists manage their own memory via realloc (treated as scalar for RC).

**Key types:** All values are 8 bytes at runtime (i64/f64/PTR). Structs and tuples are heap-allocated behind pointers. Lists store elements contiguously after a ListHeader. Strings are `[len:i32 | data:bytes]`.

## Conventions & Patterns

- **Naming:** snake_case for functions/variables, CamelCase for types/actors
- **Imports:** `import std/io`, `import std/raylib`, `import std/math`, etc.
- **Return:** Non-unit functions require explicit `return`
- **Effects:** Functions are `pure` by default; add `io` after return type for I/O
- **No closures:** Lambdas are non-capturing (`fn(x: i32): i32 { return x + 1 }`)
- **Tests:** Unit tests live in `#[cfg(test)] mod tests` in each source file
- **RC convention:** Fresh allocations have rc=1. Borrowing expressions (variable reads, field access) need `rc_incr`. Scope cleanup does `rc_decr`.
