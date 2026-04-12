//! Lumen — a statically-typed, AI-first language compiling to Wasm (Wasmtime/WASI).
//!
//! Pipeline: `.lm` source → lexer → parser → AST → type checker → codegen → `.wasm`.
//! See `docs/design.md` and `docs/grammar.ebnf`.

pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod native;
pub mod parser;
pub mod span;
pub mod types;
pub mod wasm;
