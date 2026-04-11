# Lumen — Design Document (v0.1, draft)

> A statically-typed, AI-first programming language compiling to WebAssembly, targeting Wasmtime.

## Purpose

Lumen exists because the primary author of code is now an AI agent, not a human. Human time is spent reading and auditing, not typing. That inverts the usual ergonomic tradeoffs:

- **Few constructs** — agents waste tokens deciding between equivalent options. Remove the choice.
- **One way to do everything** — makes diffs legible and review mechanical.
- **Token-efficient** — context window is a budget. Every keyword earns its place.
- **Strongly typed, strongly checked** — agents benefit from mechanical feedback loops more than humans do.
- **Errors with provenance** — "root cause" should be reachable by walking a data structure, not reading a stack trace.

## Non-goals

- Native compilation (Wasmtime only)
- FFI, foreign libraries (closed world for v1)
- Performance parity with C (Wasm overhead is acceptable)
- Novel syntax (we deliberately look like TypeScript so LLMs are fluent)
- Concurrency, async, actors (v1 is single-threaded)

## Fix list: what we fix from established languages

1. **No null/undefined/nil.** `Option<T>` only.
2. **No exceptions.** `Result<T, E>` with `?` propagation.
3. **No implicit numeric conversions.** Five int types, explicit `as`.
4. **No operator overloading, no macros, no reflection.** What you read is what runs.
5. **No inheritance.** Structs + functions + sum types + composition.
6. **No `this`/`self`.** Methods are functions; first-arg style (UFCS).
7. **No shadowing.** Reusing a name in scope is an error.
8. **No ambient globals.** Top-level `let` = constants. State flows through values.
9. **Errors carry structured provenance** (source location + argument values) automatically at `?` sites.
10. **No untyped escape hatch.** No `any`.

## Key decisions

| Area | Decision |
|---|---|
| Host language (compiler) | Rust |
| Target runtime | Wasmtime only, via WASI |
| Syntax base | TypeScript-shaped (LLM fluency) |
| Memory | v1: bump allocator in linear memory, no free. Eventually GC. |
| Integers | `i32`, `i64`, `u32`, `u64`, `f64` — explicit `as` conversions |
| Effects | `pure` / `io`, surfaced in signatures, checked |
| Loops | `for x in iter` only. Recursion for unbounded. No `while`. No `break`/`continue` in v1. |
| Errors | `Result<T, E>` + `?`. Frames auto-captured on Err path. |
| Generics | v1: none for users. `Option` and `Result` are built-ins, specialized at call site. |
| Concurrency | None in v1. |
| Modules | `import std/foo` only. No user modules in v1, no relative imports, no star imports. |

## Keyword budget (target: 18–22)

`fn`, `let`, `type`, `import`, `if`, `else`, `match`, `for`, `in`, `return`, `as`, `io`, `pure`, `true`, `false`, `unit`

Constructors (`Ok`, `Err`, `Some`, `None`) are stdlib identifiers, not keywords. `Result`, `Option` are built-in type names, not keywords. That's **16 keywords**.

## The error-frame design (the novel piece)

At every `?` propagation site, the compiler inserts a call that wraps the error value with a **frame**:

```
struct Frame {
  file:      string  // compile-time literal
  line:      i32     // compile-time literal
  col:       i32     // compile-time literal
  fn_name:   string  // compile-time literal
  args:      Debug[] // captured at runtime — stretch goal
  next:      Option<Frame>  // previous frame in chain
}
```

The Ok path is untouched. Only the Err branch walks the frame-push code, so the success case pays nothing. When `main` returns `Err`, the runtime walks the chain and prints each frame.

**Why this matters for agents:** when a test fails, the error already contains everything needed to diagnose it. The agent doesn't need to re-run with prints or attach a debugger. "Look at the innermost frame, check its arguments, read the error type" is a mechanical recipe.

## Memory layout (v1)

- Linear memory, single page to start, grown as needed
- Bump pointer in a global; no free
- String: `(ptr: i32, len: i32)` pair (the pair lives on the Wasm stack as two i32s)
- Struct: pointer to a contiguous block in linear memory
- Sum type: `(tag: i32, payload: ...)` — payload layout is variant-specific; overall size is max of all variants
- `Option<T>`: `tag: i32` (0=None, 1=Some) + `value: T`
- `Result<T, E>`: `tag: i32` (0=Ok, 1=Err) + union of T and E

## Compilation pipeline

```
.lm source
  → Lexer (hand-written)
  → Parser (recursive descent)
  → AST (with spans)
  → Type checker (bidirectional, monomorphic)
  → Codegen (wasm-encoder)
  → .wasm binary
```

No IR, no optimizer. Wasmtime's JIT handles optimization. Total compiler target size: ~3000–4000 LOC of Rust.

## Open questions (deferred past v1)

- Generics for user types (huge; needs monomorphization pass)
- A real GC (Wasm GC proposal is maturing but uneven across runtimes)
- Concurrency model (if added: structured `parallel { ... }` block, not colored functions)
- Module system for user code
- LSP / tooling
- How to capture argument values in frames for non-`Copy` types

## See also

- `grammar.ebnf` — formal grammar
- `fluency-validation.md` — empirical check that LLMs can write valid Lumen
