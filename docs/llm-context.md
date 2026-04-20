# Writing Lumen — Context for LLM Sessions

Load this file into context before writing Lumen. It covers the points
where Lumen looks like TypeScript or Rust but isn't. Examples are
authoritative — when in doubt, read `examples/*.lm` and the matching
module in `std/`.

Build and run:
```
cargo run -- build path/to/foo.lm && ./path/to/foo
```

## Effects

Every `fn` signature declares its effect after the return type. Two
effects exist: the implicit `pure` (default) and explicit `io`.

```lumen
fn add(a: i32, b: i32): i32 { return a + b }         // pure
fn greet(): unit io { io.println("hi") }              // io
```

A `pure` fn cannot call an `io` fn. If you touch `io.*`, `fs.*`,
`net.*`, `http.*`, or `debug.*`, the caller needs `io`. `main` almost
always ends with `io`.

## Bindings

`let` is immutable. `var` is mutable. This inverts TypeScript, where
`let` is the mutable default.

```lumen
let x: i32 = 1
var y: i32 = 1
y = y + 1        // ok
// x = x + 1     // error: cannot assign to `let`
```

Type annotations are usually inferred; supply them when the RHS is
ambiguous (e.g. `list.new()` returns `List<Error>` until pinned).

## Integers

Integer literals default to `i32`. Suffix with `_i64`, `_u32`, `_u64`
when needed. `list.len` / `string.len` return `i32`.

## Calls: no method dispatch on ordinary values

Ordinary values don't have methods. All stdlib calls are module-qualified.

```lumen
list.map(xs, f)        // yes
xs.map(f)              // no — parse error or unknown method
string.len(s)          // yes
s.len()                // no
```

The `.` token does appear in two non-call contexts that are easy to
mistake for method calls:

- **Actor messages** — `send h.method(args)` and `ask h.method(args)`
  are concurrency operators, not calls. They queue a message across a
  scheduling boundary (actors run on green threads) and have their own
  keywords. The `.` here selects which message handler, not a method.
- **Module-qualified calls** — `io.println(x)`, `list.map(xs, f)`.
  Here `io` and `list` are modules, not values.

So any time you're tempted to write `value.method(...)`, it won't
work. `send`/`ask`/`debug` are language forms with their own syntax.

## String interpolation

`"hello \{name}"` — note the backslash. `${name}` and bare `{name}`
don't parse. Escape literal `\{` with `\\{`.

```lumen
io.println("n=\{n} sum=\{1 + 2 + 3}")
```

Interpolated expressions can use any type: `io.println` and the
interpolation formatter accept any value via universal print.

## Sum types and match

```lumen
type Shape =
    | Circle { radius: i32 }
    | Rectangle { width: i32, height: i32 }
    | Triangle(i32, i32, i32)
```

- Variants are named bare: `Circle { radius: 5 }`, not `Shape.Circle`.
- Payloads come in three shapes: none (`| Empty`), named
  (`| Circle { radius: i32 }`), or positional (`| Triangle(i32, i32, i32)`).
- Match arms are comma-separated. Blocks as arm bodies also need a
  trailing comma.

```lumen
let area = match s {
    Circle { radius: r } => 3 * r * r,
    Rectangle { width: w, height: h } => w * h,
    Triangle(a, b, c) => (a + b + c) / 2,
}
```

Pattern fields **bind new names**: `Circle { radius: r }` binds `r`,
not field-accesses `.radius`. There is no `{ radius }` shorthand —
write the binding explicitly. `_` is the wildcard.

## Result / Option / `?`

`Result<T, E>` and `Option<T>` are built in. Constructors: `Ok(x)`,
`Err(e)`, `Some(x)`, `None`.

```lumen
fn divide(a: i32, b: i32): Result<i32, string> {
    return if b == 0 { Err("div by zero") } else { Ok(a / b) }
}

fn compute(): Result<i32, string> {
    let x = divide(100, 5)?     // unwraps Ok, propagates Err
    let y = divide(x, 4)?
    return Ok(x + y)
}
```

`?` returns early on `Err`/`None`; the containing fn must return a
compatible `Result`/`Option`.

## Generics

`fn id<T>(x: T): T`, `type Pair<A, B> = { fst: A, snd: B }`. Generics
are monomorphized per call-site type.

T usually flows from arguments. When no argument carries it — e.g.
`fn nothing<T>(): Maybe<T>` — you **must annotate** the receiver so
the typechecker can infer T from context:

```lumen
let x: Maybe<i32> = nothing()    // ok — T inferred from annotation
// let x = nothing()             // error — T is ambiguous
```

The same applies to `list.new()` / `map.new()`.

## Lambdas and fn pointers

Lambdas are non-capturing. You can only reference the params and
globals, not outer locals. The standard pattern is to hoist the
function to a named top-level `fn` and pass its name as a fn pointer:

```lumen
fn is_even(n: i32): bool { return n % 2 == 0 }

fn main(): unit io {
    var xs: List<i32> = list.new()
    xs = list.push(xs, 1)
    xs = list.push(xs, 2)
    io.println(list.filter(xs, is_even))   // pass fn by name
}
```

Inline lambdas work where a `fn(T): U` is expected, but they still
don't capture.

## String vs bytes

I/O boundaries (`fs.read_file`, `fs.write_file`, network) use `bytes`.
`string` is for text you interpolate or compare. Convert with
`string.from_bytes(b)` / `bytes.from_string(s)`. This is deliberate —
don't wrap it away.

## List and Map

`List<T>` is dynamic; it reallocates on `push`. `Map<K, V>` is
insertion-ordered. `K` can only be `string` today.

```lumen
var xs: List<i32> = list.new()
xs = list.push(xs, 42)              // returns the (possibly relocated) list
let n = list.get(xs, 0)             // no bounds check; callers check list.len

var ages: Map<string, i32> = map.new()
ages = map.set(ages, "alice", 30)
match map.get(ages, "alice") {
    Some(age) => io.println(age),
    None => io.println("missing"),
}
```

`list.push` / `map.set` return the updated container — reassign with
`var`, don't mutate in place.

## What doesn't exist

- No traits / impl / interfaces. Pluggable behavior = fn-pointer arg.
- No closures (lambdas don't capture).
- No `async` / `await`. Use actors for concurrency.
- Only `for i in range(a, b)` loops — no `for x in xs`, no `while`.
  Iterate collections by index:
  ```lumen
  for i in range(0, list.len(xs)) { io.println(list.get(xs, i)) }
  ```
- No methods on primitives — no `42.to_string()`, use
  `int.to_string_i32(42)`.
- No `null` / `undefined`. Use `Option<T>`.
- No early `throw`. Use `Result<T, E>` + `?`.
- No module import aliases beyond the default (`import std/io` exposes
  `io.println`). Don't invent paths.

## Actors

```lumen
actor Counter { count: i32 }

msg Counter.add(self, n: i32): Counter {
    return Counter { count: self.count + n }
}
msg Counter.get(self): i32 { return self.count }

fn main(): i32 io {
    let c = spawn Counter { count: 0 }
    send c.add(5)                 // fire-and-forget
    let n = ask c.get()           // blocking request/response
    io.println(n)
    return 0
}
```

An `msg` handler receives `self` plus arguments; return the new state
(or a scalar for `ask`). `send` doesn't wait; `ask` does.

## Common traps

- `let` forgotten where `var` is needed: assignment errors point at
  the binding, not the usage.
- Expected a method, used a method: `xs.map(f)` compiles only in your
  head.
- Interpolation: `${name}` silently parses as a literal `${` followed
  by a name.
- Bare variant assumed generic: `let m = Just { value: 7 }` without
  an annotation can confuse inference for generic sum types —
  annotate `let m: Maybe<i32> = ...`.
- Trailing commas in match arms matter; missing one is a parse error
  pointing at the next arm.
- `split` currently returns `i64` (a list pointer), not `List<string>` —
  iterate by index and call `string.len` / `string.substring` on each
  element.
