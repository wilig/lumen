//! Hand-written lexer. See `lumen-afh`.
//!
//! Produces a stream of [`Token`]s with source [`Span`]s from a UTF-8 source
//! string. No lexer generator.
//!
//! The lexer is byte-oriented (source is ASCII-dominant) and only steps into
//! UTF-8 decoding inside string literals, where non-ASCII bytes are legal.
//! Identifiers, keywords, and operators are ASCII-only; any non-ASCII byte
//! outside a string or comment is a lex error.

use crate::span::Span;

/// A lexed token.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// The 17 keywords plus all literals, identifiers, and punctuation that can
/// appear in a Lumen source file.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // --- Keywords (15 true keywords) ---------------------------------------
    // The 17 reserved words in the spec include `io` and `pure`, but those
    // are treated as *contextual* keywords by the parser — they still lex
    // as plain identifiers so they can be used as module names (`std/io`)
    // and stdlib identifiers (`io.println(...)`). The parser recognizes them
    // positionally after a fn return type.
    Fn,
    Let,
    Var,
    Type,
    Import,
    Extern,
    Actor,
    Msg,
    Spawn,
    Send,
    Ask,
    If,
    Else,
    Match,
    For,
    In,
    Return,
    As,
    True,
    False,
    Unit,

    // --- Identifiers & literals -------------------------------------------
    Ident(String),
    /// Integer literal. `value` holds the unsigned magnitude; the parser
    /// (or unary-minus handling) is responsible for sign. `suffix` is the
    /// explicit type tag if the source wrote one, e.g. `42i64`.
    IntLit { value: u64, suffix: Option<IntSuffix> },
    FloatLit(f64),
    /// Decoded string — escape sequences have been resolved.
    StringLit(String),

    // --- Punctuation ------------------------------------------------------
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semi,
    Dot,
    Question,
    Pipe,
    Arrow,    // ->
    FatArrow, // =>

    // --- Operators --------------------------------------------------------
    Eq,      // =
    PlusEq,  // +=
    MinusEq, // -=
    StarEq,  // *=
    EqEq,    // ==
    NotEq,   // !=
    Lt,
    Gt,
    LtEq,
    GtEq,
    AndAnd,
    OrOr,
    Bang,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    Eof,
}

/// Explicit numeric suffix on an integer literal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntSuffix {
    I32,
    I64,
    U32,
    U64,
}

/// A lex-time failure with source location and a human-readable message.
#[derive(Clone, Debug)]
pub struct LexError {
    pub span: Span,
    pub message: String,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.span.line, self.span.col, self.message)
    }
}

impl std::error::Error for LexError {}

/// Convenience: lex an entire source file into a `Vec<Token>` ending in
/// [`TokenKind::Eof`]. Errors out at the first illegal construct.
pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    let mut lx = Lexer::new(src);
    let mut out = Vec::new();
    loop {
        let tok = lx.next_token()?;
        let is_eof = matches!(tok.kind, TokenKind::Eof);
        out.push(tok);
        if is_eof {
            break;
        }
    }
    Ok(out)
}

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    #[inline]
    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.src.get(self.pos + offset).copied()
    }

    /// Advance one byte, updating line/column. Only safe when the current
    /// byte is ASCII (lead bytes of a UTF-8 sequence must go through the
    /// string-literal path, which handles multi-byte chars as a unit).
    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(b)
    }

    fn here(&self) -> (u32, u32, u32) {
        (self.pos as u32, self.line, self.col)
    }

    fn span_from(&self, start: u32, line: u32, col: u32) -> Span {
        Span::new(start, self.pos as u32, line, col)
    }

    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') => {
                    self.bump();
                }
                Some(b'/') if self.peek_at(1) == Some(b'/') => {
                    // Line comment: consume everything up to but not
                    // including the terminating newline.
                    while let Some(c) = self.peek() {
                        if c == b'\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                Some(b'/') if self.peek_at(1) == Some(b'*') => {
                    // Block comment: /* ... */ (supports nesting).
                    self.bump(); // /
                    self.bump(); // *
                    let mut depth = 1u32;
                    while depth > 0 {
                        match self.peek() {
                            None => break, // unterminated — lexer will hit EOF
                            Some(b'/') if self.peek_at(1) == Some(b'*') => {
                                self.bump(); self.bump();
                                depth += 1;
                            }
                            Some(b'*') if self.peek_at(1) == Some(b'/') => {
                                self.bump(); self.bump();
                                depth -= 1;
                            }
                            _ => { self.bump(); }
                        }
                    }
                }
                _ => return,
            }
        }
    }

    fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_trivia();
        let (start, line, col) = self.here();

        let Some(b) = self.peek() else {
            return Ok(Token {
                kind: TokenKind::Eof,
                span: self.span_from(start, line, col),
            });
        };

        if b.is_ascii_alphabetic() || b == b'_' {
            return Ok(self.lex_ident_or_keyword(start, line, col));
        }
        if b.is_ascii_digit() {
            return self.lex_number(start, line, col);
        }
        if b == b'"' {
            return self.lex_string(start, line, col);
        }

        // Single- and two-char punctuation/operators.
        self.bump();
        let kind = match b {
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            b'[' => TokenKind::LBracket,
            b']' => TokenKind::RBracket,
            b',' => TokenKind::Comma,
            b':' => TokenKind::Colon,
            b';' => TokenKind::Semi,
            b'.' => TokenKind::Dot,
            b'?' => TokenKind::Question,
            b'|' => {
                if self.peek() == Some(b'|') {
                    self.bump();
                    TokenKind::OrOr
                } else {
                    TokenKind::Pipe
                }
            }
            b'&' => {
                if self.peek() == Some(b'&') {
                    self.bump();
                    TokenKind::AndAnd
                } else {
                    return Err(LexError {
                        span: self.span_from(start, line, col),
                        message: "unexpected `&` (did you mean `&&`?)".into(),
                    });
                }
            }
            b'=' => match self.peek() {
                Some(b'=') => {
                    self.bump();
                    TokenKind::EqEq
                }
                Some(b'>') => {
                    self.bump();
                    TokenKind::FatArrow
                }
                _ => TokenKind::Eq,
            },
            b'!' => match self.peek() {
                Some(b'=') => {
                    self.bump();
                    TokenKind::NotEq
                }
                _ => TokenKind::Bang,
            },
            b'<' => match self.peek() {
                Some(b'=') => {
                    self.bump();
                    TokenKind::LtEq
                }
                _ => TokenKind::Lt,
            },
            b'>' => match self.peek() {
                Some(b'=') => {
                    self.bump();
                    TokenKind::GtEq
                }
                _ => TokenKind::Gt,
            },
            b'+' => match self.peek() {
                Some(b'=') => { self.bump(); TokenKind::PlusEq }
                _ => TokenKind::Plus,
            },
            b'-' => match self.peek() {
                Some(b'>') => { self.bump(); TokenKind::Arrow }
                Some(b'=') => { self.bump(); TokenKind::MinusEq }
                _ => TokenKind::Minus,
            },
            b'*' => match self.peek() {
                Some(b'=') => { self.bump(); TokenKind::StarEq }
                _ => TokenKind::Star,
            },
            b'/' => TokenKind::Slash,
            b'%' => TokenKind::Percent,
            _ => {
                return Err(LexError {
                    span: self.span_from(start, line, col),
                    message: format!("unexpected character `{}`", escape_byte(b)),
                });
            }
        };

        Ok(Token {
            kind,
            span: self.span_from(start, line, col),
        })
    }

    fn lex_ident_or_keyword(&mut self, start: u32, line: u32, col: u32) -> Token {
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.bump();
            } else {
                break;
            }
        }
        // Safe: we only consumed ASCII bytes.
        let text = std::str::from_utf8(&self.src[start as usize..self.pos]).unwrap();
        let kind = match text {
            "fn" => TokenKind::Fn,
            "let" => TokenKind::Let,
            "var" => TokenKind::Var,
            "type" => TokenKind::Type,
            "import" => TokenKind::Import,
            "extern" => TokenKind::Extern,
            "actor" => TokenKind::Actor,
            "msg" => TokenKind::Msg,
            "spawn" => TokenKind::Spawn,
            "send" => TokenKind::Send,
            "ask" => TokenKind::Ask,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "match" => TokenKind::Match,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "return" => TokenKind::Return,
            "as" => TokenKind::As,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "unit" => TokenKind::Unit,
            other => TokenKind::Ident(other.to_string()),
        };
        Token {
            kind,
            span: self.span_from(start, line, col),
        }
    }

    fn lex_number(&mut self, start: u32, line: u32, col: u32) -> Result<Token, LexError> {
        // Integer part.
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }

        let mut is_float = false;

        // Fractional part: only consume `.` if followed by a digit, so that
        // `3.foo()` lexes as `3 . foo ( )` (field/method access on an int).
        if self.peek() == Some(b'.')
            && matches!(self.peek_at(1), Some(c) if c.is_ascii_digit())
        {
            is_float = true;
            self.bump(); // .
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }

        // Exponent.
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.bump();
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.bump();
            }
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(LexError {
                    span: self.span_from(start, line, col),
                    message: "float literal exponent has no digits".into(),
                });
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }

        if is_float {
            let text = std::str::from_utf8(&self.src[start as usize..self.pos]).unwrap();
            let v: f64 = text.parse().map_err(|_| LexError {
                span: self.span_from(start, line, col),
                message: format!("invalid float literal `{text}`"),
            })?;
            return Ok(Token {
                kind: TokenKind::FloatLit(v),
                span: self.span_from(start, line, col),
            });
        }

        // Integer — check for a type suffix before closing the token.
        let digits_end = self.pos;
        let suffix = self.lex_int_suffix();

        let text = std::str::from_utf8(&self.src[start as usize..digits_end]).unwrap();
        let value: u64 = text.parse().map_err(|_| LexError {
            span: self.span_from(start, line, col),
            message: format!("integer literal `{text}` out of range"),
        })?;

        Ok(Token {
            kind: TokenKind::IntLit { value, suffix },
            span: self.span_from(start, line, col),
        })
    }

    fn lex_int_suffix(&mut self) -> Option<IntSuffix> {
        let lead = self.peek()?;
        if lead != b'i' && lead != b'u' {
            return None;
        }
        let slice = self.src.get(self.pos..self.pos + 3)?;
        let suf = match slice {
            b"i32" => IntSuffix::I32,
            b"i64" => IntSuffix::I64,
            b"u32" => IntSuffix::U32,
            b"u64" => IntSuffix::U64,
            _ => return None,
        };
        // Consume 3 ASCII bytes.
        self.pos += 3;
        self.col += 3;
        Some(suf)
    }

    fn lex_string(&mut self, start: u32, line: u32, col: u32) -> Result<Token, LexError> {
        self.bump(); // opening "
        let mut out = String::new();
        loop {
            let b = match self.peek() {
                None => {
                    return Err(LexError {
                        span: self.span_from(start, line, col),
                        message: "unterminated string literal".into(),
                    });
                }
                Some(b) => b,
            };

            if b == b'"' {
                self.bump();
                return Ok(Token {
                    kind: TokenKind::StringLit(out),
                    span: self.span_from(start, line, col),
                });
            }

            if b == b'\n' {
                return Err(LexError {
                    span: self.span_from(start, line, col),
                    message: "unterminated string literal (newline in string)".into(),
                });
            }

            if b == b'\\' {
                self.bump();
                match self.peek() {
                    Some(b'n') => {
                        self.bump();
                        out.push('\n');
                    }
                    Some(b't') => {
                        self.bump();
                        out.push('\t');
                    }
                    Some(b'r') => {
                        self.bump();
                        out.push('\r');
                    }
                    Some(b'\\') => {
                        self.bump();
                        out.push('\\');
                    }
                    Some(b'"') => {
                        self.bump();
                        out.push('"');
                    }
                    Some(b'0') => {
                        self.bump();
                        out.push('\0');
                    }
                    Some(c) => {
                        let (s, l, co) = self.here();
                        self.bump();
                        return Err(LexError {
                            span: Span::new(s, self.pos as u32, l, co),
                            message: format!("unknown escape sequence `\\{}`", escape_byte(c)),
                        });
                    }
                    None => {
                        return Err(LexError {
                            span: self.span_from(start, line, col),
                            message: "unterminated string literal (trailing backslash)".into(),
                        });
                    }
                }
                continue;
            }

            // Regular char — may be multi-byte UTF-8. Read the whole code
            // point as a unit so col tracking stays right.
            let char_len = utf8_char_len(b);
            let end = self.pos + char_len;
            if end > self.src.len() {
                return Err(LexError {
                    span: self.span_from(start, line, col),
                    message: "invalid UTF-8 in string literal".into(),
                });
            }
            let slice = &self.src[self.pos..end];
            match std::str::from_utf8(slice) {
                Ok(s) => out.push_str(s),
                Err(_) => {
                    return Err(LexError {
                        span: self.span_from(start, line, col),
                        message: "invalid UTF-8 in string literal".into(),
                    });
                }
            }
            self.pos = end;
            self.col += 1;
        }
    }
}

#[inline]
fn utf8_char_len(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead < 0xC0 {
        // Stray continuation byte. Treat as a 1-byte "bad" step; the
        // surrounding UTF-8 decode will fail and emit a helpful error.
        1
    } else if lead < 0xE0 {
        2
    } else if lead < 0xF0 {
        3
    } else {
        4
    }
}

fn escape_byte(b: u8) -> String {
    if b.is_ascii_graphic() {
        (b as char).to_string()
    } else {
        format!("\\x{:02x}", b)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex(src)
            .expect("lex should succeed")
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    fn lex_ok(src: &str) -> Vec<Token> {
        lex(src).unwrap_or_else(|e| panic!("lex failed: {e}"))
    }

    #[test]
    fn empty_source_is_just_eof() {
        assert_eq!(kinds(""), vec![TokenKind::Eof]);
    }

    #[test]
    fn all_true_keywords() {
        // `io` and `pure` are contextual — they lex as idents. The 15 true
        // keywords below have dedicated token kinds.
        let src = "fn let var type import if else match for in return as true false unit";
        assert_eq!(
            kinds(src),
            vec![
                TokenKind::Fn,
                TokenKind::Let,
                TokenKind::Var,
                TokenKind::Type,
                TokenKind::Import,
                TokenKind::If,
                TokenKind::Else,
                TokenKind::Match,
                TokenKind::For,
                TokenKind::In,
                TokenKind::Return,
                TokenKind::As,
                TokenKind::True,
                TokenKind::False,
                TokenKind::Unit,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn io_and_pure_lex_as_identifiers() {
        assert_eq!(
            kinds("io pure"),
            vec![
                TokenKind::Ident("io".into()),
                TokenKind::Ident("pure".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn identifiers_and_underscores() {
        assert_eq!(
            kinds("foo _bar baz_42 _"),
            vec![
                TokenKind::Ident("foo".into()),
                TokenKind::Ident("_bar".into()),
                TokenKind::Ident("baz_42".into()),
                TokenKind::Ident("_".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn builtin_type_names_lex_as_idents() {
        // i32, i64, u32, u64, f64, bool, string are not keywords — they are
        // regular identifiers resolved by the parser/typechecker.
        assert_eq!(
            kinds("i32 i64 u32 u64 f64 bool string"),
            vec![
                TokenKind::Ident("i32".into()),
                TokenKind::Ident("i64".into()),
                TokenKind::Ident("u32".into()),
                TokenKind::Ident("u64".into()),
                TokenKind::Ident("f64".into()),
                TokenKind::Ident("bool".into()),
                TokenKind::Ident("string".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn integer_literals_default_and_suffixed() {
        assert_eq!(
            kinds("0 42 7i32 7i64 7u32 7u64"),
            vec![
                TokenKind::IntLit {
                    value: 0,
                    suffix: None
                },
                TokenKind::IntLit {
                    value: 42,
                    suffix: None
                },
                TokenKind::IntLit {
                    value: 7,
                    suffix: Some(IntSuffix::I32)
                },
                TokenKind::IntLit {
                    value: 7,
                    suffix: Some(IntSuffix::I64)
                },
                TokenKind::IntLit {
                    value: 7,
                    suffix: Some(IntSuffix::U32)
                },
                TokenKind::IntLit {
                    value: 7,
                    suffix: Some(IntSuffix::U64)
                },
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn integer_overflow_is_an_error() {
        let err = lex("999999999999999999999999").unwrap_err();
        assert!(err.message.contains("out of range"));
    }

    #[test]
    fn float_literals_dot_and_exponent() {
        let ks = kinds("2.5 0.0 1e5 1.5e-2");
        let floats: Vec<f64> = ks
            .into_iter()
            .filter_map(|k| match k {
                TokenKind::FloatLit(v) => Some(v),
                _ => None,
            })
            .collect();
        assert_eq!(floats.len(), 4);
        assert!((floats[0] - 2.5).abs() < 1e-12);
        assert!((floats[1] - 0.0).abs() < 1e-12);
        assert!((floats[2] - 1e5).abs() < 1e-12);
        assert!((floats[3] - 1.5e-2).abs() < 1e-12);
    }

    #[test]
    fn int_dot_ident_is_not_a_float() {
        // `3.foo` should be INT DOT IDENT so that method/field access on
        // numeric literals parses.
        assert_eq!(
            kinds("3.foo"),
            vec![
                TokenKind::IntLit {
                    value: 3,
                    suffix: None
                },
                TokenKind::Dot,
                TokenKind::Ident("foo".into()),
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn exponent_without_digits_is_an_error() {
        let err = lex("1e").unwrap_err();
        assert!(err.message.contains("exponent"));
    }

    #[test]
    fn string_literals_with_escapes() {
        assert_eq!(
            kinds(r#""hello""#),
            vec![TokenKind::StringLit("hello".into()), TokenKind::Eof]
        );
        assert_eq!(
            kinds(r#""a\nb\t\"c\\""#),
            vec![
                TokenKind::StringLit("a\nb\t\"c\\".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn string_literal_with_utf8() {
        let ks = kinds("\"café · 日本\"");
        assert_eq!(
            ks,
            vec![
                TokenKind::StringLit("café · 日本".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn unterminated_string_errors() {
        let err = lex("\"no closing").unwrap_err();
        assert!(err.message.contains("unterminated"));
    }

    #[test]
    fn unterminated_on_newline_errors() {
        let err = lex("\"start\nend\"").unwrap_err();
        assert!(err.message.contains("newline"));
    }

    #[test]
    fn unknown_escape_errors() {
        let err = lex(r#""bad \q escape""#).unwrap_err();
        assert!(err.message.contains("escape"));
    }

    #[test]
    fn punctuation_and_all_operators() {
        assert_eq!(
            kinds("{}()[],:;.?| -> => = == != < > <= >= && || ! + - * / %"),
            vec![
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBracket,
                TokenKind::RBracket,
                TokenKind::Comma,
                TokenKind::Colon,
                TokenKind::Semi,
                TokenKind::Dot,
                TokenKind::Question,
                TokenKind::Pipe,
                TokenKind::Arrow,
                TokenKind::FatArrow,
                TokenKind::Eq,
                TokenKind::EqEq,
                TokenKind::NotEq,
                TokenKind::Lt,
                TokenKind::Gt,
                TokenKind::LtEq,
                TokenKind::GtEq,
                TokenKind::AndAnd,
                TokenKind::OrOr,
                TokenKind::Bang,
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::Percent,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn line_comments_are_skipped() {
        let src = "let x = 1 // trailing comment\nlet y = 2";
        assert_eq!(
            kinds(src),
            vec![
                TokenKind::Let,
                TokenKind::Ident("x".into()),
                TokenKind::Eq,
                TokenKind::IntLit {
                    value: 1,
                    suffix: None
                },
                TokenKind::Let,
                TokenKind::Ident("y".into()),
                TokenKind::Eq,
                TokenKind::IntLit {
                    value: 2,
                    suffix: None
                },
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn whitespace_forms_are_all_trivia() {
        assert_eq!(
            kinds("  \t\r\n  fn  "),
            vec![TokenKind::Fn, TokenKind::Eof]
        );
    }

    #[test]
    fn illegal_character_errors_with_span() {
        let err = lex("let x @ 1").unwrap_err();
        assert!(err.message.contains("unexpected character"));
        assert_eq!(err.span.line, 1);
        // `@` is at column 7 (1-based): `let x ` = 6 chars then @ at col 7.
        assert_eq!(err.span.col, 7);
    }

    #[test]
    fn lone_ampersand_has_helpful_message() {
        let err = lex("a & b").unwrap_err();
        assert!(err.message.contains("&&"));
    }

    #[test]
    fn span_tracks_line_and_col_across_newlines() {
        let src = "fn\n  foo";
        let toks = lex_ok(src);
        // `fn` at (line=1, col=1)
        assert_eq!(toks[0].kind, TokenKind::Fn);
        assert_eq!(toks[0].span.line, 1);
        assert_eq!(toks[0].span.col, 1);
        // `foo` at (line=2, col=3) after two spaces of indent.
        assert_eq!(toks[1].kind, TokenKind::Ident("foo".into()));
        assert_eq!(toks[1].span.line, 2);
        assert_eq!(toks[1].span.col, 3);
        // Eof
        assert!(matches!(toks[2].kind, TokenKind::Eof));
    }

    #[test]
    fn byte_offsets_are_correct() {
        // 0 1 2 3 4 5 6 7 8
        // l e t   x   =   1
        let toks = lex_ok("let x = 1");
        // let @ [0, 3)
        assert_eq!((toks[0].span.start, toks[0].span.end), (0, 3));
        // x @ [4, 5)
        assert_eq!((toks[1].span.start, toks[1].span.end), (4, 5));
        // = @ [6, 7)
        assert_eq!((toks[2].span.start, toks[2].span.end), (6, 7));
        // 1 @ [8, 9)
        assert_eq!((toks[3].span.start, toks[3].span.end), (8, 9));
    }

    #[test]
    fn hello_world_program_lexes_cleanly() {
        let src = r#"import std/io

fn main(): Result<unit, unit> io {
    io.println("hello, world")
    Ok(unit)
}
"#;
        // Should not error; we don't enumerate every token, just the
        // structural highlights.
        let toks = lex_ok(src);
        let only_kinds: Vec<_> = toks.iter().map(|t| &t.kind).collect();

        assert!(only_kinds.contains(&&TokenKind::Import));
        assert!(only_kinds.contains(&&TokenKind::Fn));
        assert!(only_kinds.contains(&&TokenKind::Unit));
        // `io` should appear as an Ident, not a keyword.
        assert!(only_kinds.iter().any(|k| matches!(k, TokenKind::Ident(s) if s == "io")));
        assert!(only_kinds.iter().any(|k| matches!(
            k,
            TokenKind::StringLit(s) if s == "hello, world"
        )));
    }

    #[test]
    fn sum_of_squares_program_lexes_cleanly() {
        let src = r#"fn sum_of_squares(n: i32): i32 {
    var total: i32 = 0
    for i in range(1, n + 1) {
        total = total + i * i
    }
    total
}
"#;
        let toks = lex_ok(src);
        let kinds: Vec<_> = toks.into_iter().map(|t| t.kind).collect();
        assert!(kinds.contains(&TokenKind::Var));
        assert!(kinds.contains(&TokenKind::For));
        assert!(kinds.contains(&TokenKind::In));
        // `i32` is an ident, not a keyword.
        assert!(kinds.iter().any(|k| matches!(k, TokenKind::Ident(s) if s == "i32")));
    }
}
