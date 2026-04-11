//! Lumen AST definitions. See `lumen-2s7`.
//!
//! Every node carries a `Span` so the type checker and codegen can emit
//! source-located errors and the error-frame runtime (`lumen-x4a`) can
//! attribute `?` sites to their originating line/col.

/// Byte range in the source, plus 1-based line/column of the start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
    pub line: u32,
    pub col: u32,
}
