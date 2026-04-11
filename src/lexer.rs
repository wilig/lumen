//! Hand-written lexer. See `lumen-afh`.
//!
//! Produces a stream of tokens, each carrying a span (byte offsets + line/col),
//! from a UTF-8 source string. No lexer generator.
