# LLM Fluency Validation

## Purpose

The biggest risk in a weekend prototype of a new language is that LLMs can't write it on the first shot. Since Lumen is AI-first, low generation quality kills the value prop entirely. This document records an empirical (if informal) check.

## Method

I (Claude Opus 4.6) picked 5 small programming problems, and for each one wrote what I would naturally emit given only:
- `docs/design.md`
- `docs/grammar.ebnf`
- The one `load_user` example from the design doc

No further prompting, no iteration. Then I compared the output against the grammar and flagged any construct that would fail to parse or typecheck under the v1 spec.

This is not as rigorous as actually prompting a separate model cold. But as a self-introspective smoke test it tells us where the language surface creates friction with LLM priors formed on TypeScript/Rust/Python.

## Problem 1 — Hello world

**Prompt:** "Write a program that prints 'hello, world'."

**What I'd emit:**
```
import std/io

fn main(): Result<unit, unit> io {
    io.println("hello, world")
    Ok(unit)
}
```

**Verdict:** ✅ Parses cleanly. Typechecks under the spec.

**Friction noted:** I hesitated over the return type of `main`. TS/Rust conventions differ. The design should commit to one — probably `Result<unit, Error>` with a built-in top-level `Error` type. Minor.

---

## Problem 2 — Count lines in a file

**Prompt:** "Read a file named by argv[1] and print the number of lines."

**What I'd emit:**
```
import std/io
import std/string
import std/list

fn main(): Result<unit, IoError> io {
    let path = "input.txt"
    let contents = io.read_file(path)?
    let lines = string.split(contents, "\n")
    io.println("lines: " + list.len(lines).to_string())
    Ok(unit)
}
```

**Verdict:** ⚠️ Mostly parses, but **three friction points**:

1. **`argv`** — I wouldn't reach for it because the design doc doesn't mention it. I'd hardcode a path. The v1 spec needs to decide: hardcoded paths, or a WASI-backed `std/env` module. I recommend: `std/env.args(): List<string>` in the stdlib so `main` can receive arguments.

2. **`list.len(lines).to_string()`** — I instinctively reached for a method-style `.to_string()`. The grammar supports dotted calls (UFCS postfix), so this parses. But there's no `to_string()` defined in the stdlib yet. The stdlib task should add `int.to_string(n): string` for all five numeric types.

3. **String concatenation with `+`** — the grammar allows `+` only in `add_expr`, which I've left as unrestricted. Binding `+` for strings needs to be explicit: either operator overloading (which we forbid!) or desugaring `string + string` in the typechecker, or a `string.concat(a, b)` function. I recommend: **typechecker special-cases `string + string` to call `string.concat`**. This keeps the surface clean without generalized overloading.

**Takeaway:** Stdlib task needs `std/env`, `int.to_string`, and the typechecker needs to special-case `string + string`.

---

## Problem 3 — Sum of squares from 1 to N

**Prompt:** "Return the sum of squares of integers from 1 to N."

**What I'd emit:**
```
fn sum_of_squares(n: i32): i32 {
    let total: i32 = 0
    for i in range(1, n + 1) {
        total = total + i * i
    }
    total
}
```

**Verdict:** ❌ **Does not work.** Critical issue.

**The problem:** `let` in Lumen declares an immutable binding (like Rust `let` without `mut`), but I wrote `total = total + i * i` as a mutation. The grammar doesn't have an assignment statement, and "no shadowing" forbids re-binding `let total = ...`.

**This is a real design hole.** I fell into it on the first test case. An LLM will fall into it constantly.

**Options:**
1. **Add `mut` + assignment statement.** Costs: one more keyword, one more statement form, fights the "one way" principle.
2. **Force functional style with fold.** Costs: agents have to know fold exists; more complex stdlib.
3. **Allow `let` to re-bind in a controlled way (like Rust `let mut`).** Same cost as option 1.
4. **Provide a `mut_let`-ish single form:** `var total: i32 = 0` then `total = ...`. Adds `var` keyword.
5. **Force recursion for accumulation.** Cost: bad ergonomics, LLMs don't reach for this.

**My recommendation:** bite the bullet and add **`var` declarations + assignment statements**. This adds 1 keyword (`var`) and 1 statement form (`IDENT = expr`). It's the honest solution because accumulation is ubiquitous. A fold-only language is purer but fights LLM muscle memory extremely hard.

Updated keyword count would still be within budget (17).

**With `var`, my emitted code would become:**
```
fn sum_of_squares(n: i32): i32 {
    var total: i32 = 0
    for i in range(1, n + 1) {
        total = total + i * i
    }
    total
}
```

**This is the single most important finding of the validation.**

---

## Problem 4 — Parse integer, handle errors

**Prompt:** "Parse a user-supplied string as i32 and return a clear error if it's not a valid integer."

**What I'd emit:**
```
import std/int

type ParseFail = | NotANumber(input: string)

fn parse_count(s: string): Result<i32, ParseFail> {
    match int.parse_i32(s) {
        Ok(n) => Ok(n),
        Err(_) => Err(NotANumber(input: s)),
    }
}
```

**Verdict:** ✅ Parses and typechecks cleanly. This is the sweet spot of the design — `Result`, `match`, sum-type constructors with named fields, and `?` all work as expected.

**Friction noted:** I reached for `int.parse_i32` but the design doc wrote `i32.parse`. **Dotted-method on type names** (`i32.parse(s)`) is unusual — it implies type names can be used as namespaces, which is a small departure from TS/Rust. It probably works (type names are just identifiers at parse time), but we should decide: is `i32` a module-like namespace, or does parsing live in `std/int.parse_i32`? I recommend the latter: all stdlib functions live in proper modules; primitive type names are not namespaces. This avoids an LLM confusion point.

**Takeaway:** stdlib design should put parsing in `std/int` as `parse_i32`, `parse_i64`, etc., not as methods on the type. The design doc should be corrected.

---

## Problem 5 — Shape sum type with area

**Prompt:** "Define a sum type for Circle, Rectangle, and Triangle, and a function that computes area."

**What I'd emit:**
```
type Shape =
    | Circle(radius: f64)
    | Rectangle(width: f64, height: f64)
    | Triangle(base: f64, height: f64)

fn area(s: Shape): f64 {
    match s {
        Circle(radius: r) => 3.14159 * r * r,
        Rectangle(width: w, height: h) => w * h,
        Triangle(base: b, height: h) => 0.5 * b * h,
    }
}
```

**Verdict:** ✅ Parses and typechecks cleanly under the grammar. Exhaustiveness holds.

**Friction noted:**
1. `Circle(radius: r)` — I used named subpatterns like the grammar specifies. This is a small departure from TS/Rust (where you'd write `Circle { radius: r }` or `Circle(r)`). An LLM unfamiliar with Lumen's field-named pattern syntax might default to the Rust-style `Circle { radius }` or TS-style object destructuring. **Worth considering**: allow *both* `Circle(radius: r)` and `Circle { radius: r }` as synonyms? Or pick whichever is closer to what LLMs emit by default? **My instinct: use `{}` for struct-like patterns**, matching Rust, because LLMs default to that.
2. `3.14159` — no built-in `math.pi`. Stdlib should have `std/math`.

---

## Summary

| Problem | Verdict | Key finding |
|---|---|---|
| 1. Hello world | ✅ | None |
| 2. Count lines | ⚠️ | Needs `std/env.args`, `int.to_string`, `+` for strings |
| 3. Sum of squares | ❌ | **Needs `var` + assignment** |
| 4. Parse integer | ✅ | Move parsing into `std/int` module |
| 5. Shape sum type | ✅ | Use `{}` not `()` for struct-like variant patterns |

**Hypothesis holds, with three required fixes before starting implementation:**

1. **Add `var` declarations and assignment statements.** This is the big one. Without it, accumulation loops don't work, and LLMs will fall into this trap on every numeric program.
2. **Use `{}` for struct-like variant patterns and constructors.** Matches Rust and LLM muscle memory.
3. **Special-case `string + string` in the typechecker.** No general operator overloading.

**Plus two stdlib clarifications:**

- `std/env.args(): List<string>` for command-line arguments
- `std/int.parse_i32(s): Result<i32, ParseError>` and `.to_string(n): string` for all numeric types
- Put parsing in proper modules, not as methods on type names

**Updated keyword count:** 17 (added `var`). Still well within Lua→Go budget.

**Updated grammar items:**
- `var_decl = "var" , IDENT , [ ":" , type ] , "=" , expr ;`
- `assign_stmt = IDENT , "=" , expr ;`
- Variant patterns change from `IDENT(pat_field_list)` to `IDENT{pat_field_list}`
- Variant constructors: same shift from `(...)` to `{...}` for named fields

Out of 5 problems, 3 came out clean and 2 revealed real issues. That's consistent with "LLMs are fluent, but the spec has a couple of traps that need fixing before building." The hypothesis is validated; the spec needs a small revision pass before Phase 1.
