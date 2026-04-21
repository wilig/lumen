//! Lumen — a statically-typed, AI-first language compiling to native code via Cranelift.
//!
//! Pipeline: `.lm` source → lexer → parser → AST → type checker → native codegen → executable.
//! See `docs/design.md` and `docs/grammar.ebnf`.

pub mod ast;
pub mod dwarf;
pub mod fmt;
pub mod imports;
pub mod lexer;
pub mod lsp;
pub mod native;
pub mod parser;
pub mod span;
pub mod types;
