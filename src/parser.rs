//! Recursive-descent parser. See `lumen-2s7` and `docs/grammar.ebnf`.
//!
//! Consumes a token stream from `lexer` and produces an `ast::Module`.
//! Every AST node carries a span for downstream error reporting.
