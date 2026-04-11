//! Lumen AST definitions. See `lumen-2s7`.
//!
//! Every node carries a `Span` so the type checker and codegen can emit
//! source-located errors and the error-frame runtime (`lumen-x4a`) can
//! attribute `?` sites to their originating line/col.

pub use crate::span::Span;
