//! Source locations. Shared by lexer, parser, typechecker, and the
//! error-frame runtime so every downstream tool can report locations
//! in a consistent way.

/// Byte range in the source, plus 1-based line/column of the first byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
    pub line: u32,
    pub col: u32,
}

impl Span {
    pub const DUMMY: Span = Span {
        start: 0,
        end: 0,
        line: 0,
        col: 0,
    };

    pub fn new(start: u32, end: u32, line: u32, col: u32) -> Self {
        Self {
            start,
            end,
            line,
            col,
        }
    }
}
