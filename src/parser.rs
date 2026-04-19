//! Recursive-descent parser. See `lumen-2s7` and `docs/grammar.ebnf`.
//!
//! Consumes a token stream from [`crate::lexer`] and produces an
//! [`ast::Module`]. Every AST node carries a span for downstream error
//! reporting.
//!
//! ## Structure
//!
//! The parser is organized top-down following the EBNF, one method per
//! production. A single [`Parser`] struct tracks the token cursor. Lookahead
//! is bounded to two tokens (`peek` + `peek_at(1)`).
//!
//! ## Disambiguations
//!
//! - `IDENT = expr` at statement position is parsed as an assignment rather
//!   than an expression-statement. We check with two-token lookahead for
//!   `IDENT` followed by a bare `=` (not `==`, `=>`).
//! - Struct literals (`Foo { ... }`) are disabled in the "head" position of
//!   `if`, `match`, and `for` conditions so that the opening `{` is
//!   unambiguously the body. Users who need a struct literal there must
//!   parenthesize it, matching Rust's convention.
//! - The trailing statement of a block, if it is a bare expression, is
//!   lifted to the block's `tail` expression so the block has a value.

use crate::ast::*;
use crate::lexer::{Token, TokenKind};
use crate::span::Span;

/// A parse-time failure with source location and a human-readable message.
#[derive(Clone, Debug)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.span.line, self.span.col, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse a full token stream into a [`Module`].
pub fn parse(tokens: Vec<Token>) -> Result<Module, ParseError> {
    let mut p = Parser::new(tokens);
    p.parse_module()
}

/// Parse a single expression from a snippet of source — used to compile
/// the body of a `\{...}` interpolation. Errors carry the surrounding
/// string literal's span so the user sees a location they recognize.
fn parse_interp_expr(source: &str, _line: u32, _col: u32, str_span: Span) -> Result<Expr, ParseError> {
    let tokens = crate::lexer::lex(source).map_err(|e| ParseError {
        span: str_span,
        message: format!("in string interpolation: {}", e.message),
    })?;
    let mut p = Parser::new(tokens);
    let expr = p.parse_expr(ExprCtx::normal()).map_err(|e| ParseError {
        span: str_span,
        message: format!("in string interpolation: {}", e.message),
    })?;
    if !matches!(p.peek_kind(), TokenKind::Eof) {
        return Err(ParseError {
            span: str_span,
            message: "in string interpolation: unexpected trailing tokens after expression".into(),
        });
    }
    Ok(expr)
}

// ---------------------------------------------------------------------------
// Parser state
// ---------------------------------------------------------------------------

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

/// Two bits of parser state threaded through expression parsing:
/// whether struct literals are allowed in this position.
#[derive(Clone, Copy)]
struct ExprCtx {
    allow_struct_lit: bool,
}

impl ExprCtx {
    fn normal() -> Self {
        Self {
            allow_struct_lit: true,
        }
    }
    fn no_struct() -> Self {
        Self {
            allow_struct_lit: false,
        }
    }
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    // --- Token cursor -----------------------------------------------------

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn peek_at(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset)
    }

    fn bump(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        if !matches!(tok.kind, TokenKind::Eof) {
            self.pos += 1;
        }
        tok
    }

    fn eat(&mut self, kind: &TokenKind) -> Option<Token> {
        if std::mem::discriminant(self.peek_kind()) == std::mem::discriminant(kind) {
            Some(self.bump())
        } else {
            None
        }
    }

    fn expect(&mut self, kind: &TokenKind, what: &str) -> Result<Token, ParseError> {
        if std::mem::discriminant(self.peek_kind()) == std::mem::discriminant(kind) {
            Ok(self.bump())
        } else {
            Err(self.error_here(format!(
                "expected {what}, found {}",
                describe_token(self.peek_kind())
            )))
        }
    }

    fn error_here(&self, message: String) -> ParseError {
        ParseError {
            span: self.peek().span,
            message,
        }
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Eof)
    }

    // --- Top level --------------------------------------------------------

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        let mut imports = Vec::new();
        while matches!(self.peek_kind(), TokenKind::Import) {
            imports.push(self.parse_import()?);
        }

        let mut items = Vec::new();
        while !self.at_eof() {
            items.push(self.parse_item()?);
        }

        Ok(Module { imports, items })
    }

    fn parse_import(&mut self) -> Result<Import, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Import, "`import`")?;

        let mut path = Vec::new();
        let head = self.expect_ident("module name after `import`")?;
        path.push(head.0);

        while self.eat(&TokenKind::Slash).is_some() {
            let seg = self.expect_ident("module name segment after `/`")?;
            path.push(seg.0);
        }

        // Optional alias: `import std/raylib as rl`
        let alias = if matches!(self.peek_kind(), TokenKind::As) {
            self.bump();
            Some(self.expect_ident("alias name after `as`")?.0)
        } else {
            None
        };

        let end = self.tokens[self.pos.saturating_sub(1)].span;
        Ok(Import {
            path,
            alias,
            span: merge(start, end),
        })
    }

    fn parse_item(&mut self) -> Result<Item, ParseError> {
        match self.peek_kind() {
            TokenKind::Fn => self.parse_fn_decl().map(Item::Fn),
            TokenKind::Type => self.parse_type_decl().map(Item::Type),
            TokenKind::Extern => self.parse_extern_fn_decl().map(Item::ExternFn),
            TokenKind::Actor => self.parse_actor_decl().map(Item::Actor),
            TokenKind::Msg => self.parse_msg_handler().map(Item::MsgHandler),
            other => Err(self.error_here(format!(
                "expected `fn`, `type`, `extern`, `actor`, or `msg` at top level, found {}",
                describe_token(other)
            ))),
        }
    }

    fn parse_extern_fn_decl(&mut self) -> Result<ExternFnDecl, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Extern, "`extern`")?;
        self.expect(&TokenKind::Fn, "`fn` after `extern`")?;
        let (name, name_span) = self.expect_ident("extern function name")?;

        self.expect(&TokenKind::LParen, "`(` after function name")?;
        let mut params = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RParen) {
            loop {
                params.push(self.parse_param()?);
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen, "`)` to close parameter list")?;

        self.expect(&TokenKind::Colon, "`:` before return type")?;
        let return_type = self.parse_type()?;
        let mut end = return_type.span;

        // Optional link name: `link "c_symbol_name"`
        let link_name = if matches!(self.peek_kind(), TokenKind::Ident(ref s) if s == "link") {
            self.bump(); // consume `link`
            match self.peek_kind().clone() {
                TokenKind::StringLit(s) => {
                    let tok = self.bump();
                    end = tok.span;
                    Some(s)
                }
                _ => return Err(self.error_here("expected string literal after `link`".into())),
            }
        } else {
            None
        };

        Ok(ExternFnDecl {
            name,
            name_span,
            params,
            return_type,
            link_name,
            span: merge(start, end),
        })
    }

    fn parse_actor_decl(&mut self) -> Result<ActorDecl, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Actor, "`actor`")?;
        let (name, name_span) = self.expect_ident("actor name")?;
        let fields = self.parse_struct_body()?;
        let end = self.tokens[self.pos.saturating_sub(1)].span;
        Ok(ActorDecl {
            name,
            name_span,
            fields,
            span: merge(start, end),
        })
    }

    fn parse_msg_handler(&mut self) -> Result<MsgHandlerDecl, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Msg, "`msg`")?;
        let (actor_name, _) = self.expect_ident("actor name")?;
        self.expect(&TokenKind::Dot, "`.` after actor name")?;
        let (name, name_span) = self.expect_ident("message handler name")?;

        self.expect(&TokenKind::LParen, "`(`")?;
        // First param must be `self` (consumed but not stored).
        let self_tok = self.expect_ident("`self` parameter")?;
        if self_tok.0 != "self" {
            return Err(ParseError {
                span: self_tok.1,
                message: "first parameter of a msg handler must be `self`".into(),
            });
        }
        let mut params = Vec::new();
        while self.eat(&TokenKind::Comma).is_some() {
            if matches!(self.peek_kind(), TokenKind::RParen) {
                break;
            }
            params.push(self.parse_param()?);
        }
        self.expect(&TokenKind::RParen, "`)`")?;

        self.expect(&TokenKind::Colon, "`:` before return type")?;
        let return_type = self.parse_type()?;

        let body = self.parse_block()?;
        let end = body.span;
        Ok(MsgHandlerDecl {
            actor_name,
            name,
            name_span,
            params,
            return_type,
            body,
            span: merge(start, end),
        })
    }

    // --- Function declarations --------------------------------------------

    fn parse_fn_decl(&mut self) -> Result<FnDecl, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Fn, "`fn`")?;
        let (name, name_span) = self.expect_ident("function name")?;

        self.expect(&TokenKind::LParen, "`(` after function name")?;
        let mut params = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RParen) {
            loop {
                params.push(self.parse_param()?);
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen, "`)` to close parameter list")?;

        self.expect(&TokenKind::Colon, "`:` before return type")?;
        let return_type = self.parse_type()?;

        // Optional effect annotation. `io` and `pure` are contextual keywords
        // — they lex as plain identifiers so they can also be used as module
        // names (`std/io`) and stdlib values (`io.println(...)`). We only
        // recognize them here, between the return type and the opening brace
        // of the body.
        let effect = match self.peek_kind() {
            TokenKind::Ident(name) if name == "io" && self.peek_at(1).is_some_and(|t| matches!(t.kind, TokenKind::LBrace)) => {
                self.bump();
                Effect::Io
            }
            TokenKind::Ident(name) if name == "pure" && self.peek_at(1).is_some_and(|t| matches!(t.kind, TokenKind::LBrace)) => {
                self.bump();
                Effect::Pure
            }
            _ => Effect::Pure,
        };

        let body = self.parse_block()?;
        let end = body.span;

        Ok(FnDecl {
            name,
            name_span,
            params,
            return_type,
            effect,
            body,
            span: merge(start, end),
        })
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let (name, name_span) = self.expect_ident("parameter name")?;
        self.expect(&TokenKind::Colon, "`:` before parameter type")?;
        let ty = self.parse_type()?;
        let span = merge(name_span, ty.span);
        Ok(Param { name, ty, span })
    }

    // --- Type declarations ------------------------------------------------

    fn parse_type_decl(&mut self) -> Result<TypeDecl, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Type, "`type`")?;
        let (name, name_span) = self.expect_ident("type name")?;
        self.expect(&TokenKind::Eq, "`=` in type declaration")?;

        // Sum type body starts with `|` or an IDENT followed by another `|`
        // or EOF/next decl. Struct body starts with `{`. A sum with a single
        // variant and no `|` looks like a struct-less variant — but the
        // grammar makes this ambiguous; we commit to "if next is `{`, it's a
        // struct body; otherwise sum body." Users can always write a leading
        // `|` to disambiguate.
        let body = match self.peek_kind() {
            TokenKind::LBrace => TypeBody::Struct(self.parse_struct_body()?),
            _ => TypeBody::Sum(self.parse_sum_body()?),
        };

        let end = self.tokens[self.pos.saturating_sub(1)].span;
        Ok(TypeDecl {
            name,
            name_span,
            body,
            span: merge(start, end),
        })
    }

    fn parse_struct_body(&mut self) -> Result<Vec<Field>, ParseError> {
        self.expect(&TokenKind::LBrace, "`{` for struct body")?;
        let mut fields = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RBrace) {
            loop {
                fields.push(self.parse_field()?);
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
                // allow trailing comma
                if matches!(self.peek_kind(), TokenKind::RBrace) {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RBrace, "`}` to close struct body")?;
        Ok(fields)
    }

    fn parse_field(&mut self) -> Result<Field, ParseError> {
        let (name, name_span) = self.expect_ident("field name")?;
        self.expect(&TokenKind::Colon, "`:` before field type")?;
        let ty = self.parse_type()?;
        Ok(Field {
            name,
            span: merge(name_span, ty.span),
            ty,
        })
    }

    fn parse_sum_body(&mut self) -> Result<Vec<Variant>, ParseError> {
        // Optional leading `|` (Rust-style).
        self.eat(&TokenKind::Pipe);

        let mut variants = Vec::new();
        variants.push(self.parse_variant()?);
        while self.eat(&TokenKind::Pipe).is_some() {
            variants.push(self.parse_variant()?);
        }
        Ok(variants)
    }

    fn parse_variant(&mut self) -> Result<Variant, ParseError> {
        let (name, name_span) = self.expect_ident("variant name")?;
        let payload = match self.peek_kind() {
            TokenKind::LBrace => Some(VariantPayload::Named(self.parse_struct_body()?)),
            TokenKind::LParen => {
                self.bump();
                let mut tys = Vec::new();
                if !matches!(self.peek_kind(), TokenKind::RParen) {
                    loop {
                        tys.push(self.parse_type()?);
                        if self.eat(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                self.expect(&TokenKind::RParen, "`)` to close positional variant")?;
                Some(VariantPayload::Positional(tys))
            }
            _ => None,
        };
        let end = self.tokens[self.pos.saturating_sub(1)].span;
        Ok(Variant {
            name,
            payload,
            span: merge(name_span, end),
        })
    }

    // --- Types ------------------------------------------------------------

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        // Tuple type: (T1, T2, ...)
        if matches!(self.peek_kind(), TokenKind::LParen) {
            let start = self.bump().span;
            let mut elems = Vec::new();
            if !matches!(self.peek_kind(), TokenKind::RParen) {
                loop {
                    elems.push(self.parse_type()?);
                    if self.eat(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
            }
            let end = self.expect(&TokenKind::RParen, "`)` to close tuple type")?.span;
            return Ok(Type {
                kind: TypeKind::Tuple(elems),
                span: merge(start, end),
            });
        }

        // Function pointer type: fn(T1, T2): R
        if matches!(self.peek_kind(), TokenKind::Fn) {
            let start = self.bump().span;
            self.expect(&TokenKind::LParen, "`(` after `fn` in function pointer type")?;
            let mut params = Vec::new();
            if !matches!(self.peek_kind(), TokenKind::RParen) {
                loop {
                    params.push(self.parse_type()?);
                    if self.eat(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RParen, "`)` to close function pointer type params")?;
            self.expect(&TokenKind::Colon, "`:` before function pointer return type")?;
            let ret = self.parse_type()?;
            let end = ret.span;
            return Ok(Type {
                kind: TypeKind::FnPtr { params, ret: Box::new(ret) },
                span: merge(start, end),
            });
        }

        let (name, name_span) = match self.peek_kind() {
            TokenKind::Unit => {
                let tok = self.bump();
                return Ok(Type {
                    kind: TypeKind::Named {
                        name: "unit".into(),
                        args: Vec::new(),
                    },
                    span: tok.span,
                });
            }
            TokenKind::Ident(_) => self.expect_ident("type name")?,
            other => {
                return Err(self.error_here(format!(
                    "expected a type name, found {}",
                    describe_token(other)
                )));
            }
        };

        let mut args = Vec::new();
        let mut end = name_span;
        if matches!(self.peek_kind(), TokenKind::Lt) {
            self.bump();
            loop {
                args.push(self.parse_type()?);
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
            }
            let gt = self.expect(&TokenKind::Gt, "`>` to close type arguments")?;
            end = gt.span;
        }

        Ok(Type {
            kind: TypeKind::Named { name, args },
            span: merge(name_span, end),
        })
    }

    // --- Blocks and statements --------------------------------------------

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::LBrace, "`{` to open block")?;

        let mut stmts = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::RBrace | TokenKind::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        let end = self.expect(&TokenKind::RBrace, "`}` to close block")?.span;

        // Lift a trailing expression statement into `tail`, matching the
        // grammar's `{ stmt* tail_expr? }` shape.
        let tail = match stmts.last().map(|s| &s.kind) {
            Some(StmtKind::Expr(_)) => {
                let last = stmts.pop().unwrap();
                if let StmtKind::Expr(e) = last.kind {
                    Some(Box::new(e))
                } else {
                    unreachable!()
                }
            }
            _ => None,
        };

        Ok(Block {
            stmts,
            tail,
            span: merge(start, end),
        })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek_kind() {
            TokenKind::Let => self.parse_let_stmt(),
            TokenKind::Var => self.parse_var_stmt(),
            TokenKind::For => self.parse_for_stmt(),
            TokenKind::Return => self.parse_return_stmt(),
            TokenKind::Ident(_) if self.is_assign_ahead() => self.parse_assign_stmt(),
            _ => {
                let start = self.peek().span;
                let expr = self.parse_expr(ExprCtx::normal())?;
                let span = merge(start, expr.span);
                Ok(Stmt {
                    kind: StmtKind::Expr(expr),
                    span,
                })
            }
        }
    }

    /// True iff the current token is an IDENT followed by `=`, `+=`, `-=`,
    /// or `*=` (not `==`, `=>`).
    fn is_assign_ahead(&self) -> bool {
        let Some(next) = self.peek_at(1) else {
            return false;
        };
        matches!(next.kind, TokenKind::Eq | TokenKind::PlusEq | TokenKind::MinusEq | TokenKind::StarEq)
    }

    fn parse_let_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Let, "`let`")?;
        // Destructuring let: `let (a, b) = expr`
        if matches!(self.peek_kind(), TokenKind::LParen) {
            self.bump();
            let mut names = Vec::new();
            loop {
                let (name, _) = self.expect_ident("binding name in tuple destructure")?;
                names.push(name);
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
            }
            self.expect(&TokenKind::RParen, "`)` to close destructuring")?;
            self.expect(&TokenKind::Eq, "`=` in let binding")?;
            let value = self.parse_expr(ExprCtx::normal())?;
            let span = merge(start, value.span);
            return Ok(Stmt {
                kind: StmtKind::LetTuple { names, value },
                span,
            });
        }
        let (name, _) = self.expect_ident("binding name")?;
        let ty = if self.eat(&TokenKind::Colon).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Eq, "`=` in let binding")?;
        let value = self.parse_expr(ExprCtx::normal())?;
        let span = merge(start, value.span);
        Ok(Stmt {
            kind: StmtKind::Let { name, ty, value },
            span,
        })
    }

    fn parse_var_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Var, "`var`")?;
        let (name, _) = self.expect_ident("binding name")?;
        let ty = if self.eat(&TokenKind::Colon).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Eq, "`=` in var binding")?;
        let value = self.parse_expr(ExprCtx::normal())?;
        let span = merge(start, value.span);
        Ok(Stmt {
            kind: StmtKind::Var { name, ty, value },
            span,
        })
    }

    fn parse_assign_stmt(&mut self) -> Result<Stmt, ParseError> {
        let (name, name_span) = self.expect_ident("assignment target")?;
        // Check for compound assignment (+=, -=, *=) and desugar.
        let compound_op = match self.peek_kind() {
            TokenKind::PlusEq => { self.bump(); Some(BinOp::Add) }
            TokenKind::MinusEq => { self.bump(); Some(BinOp::Sub) }
            TokenKind::StarEq => { self.bump(); Some(BinOp::Mul) }
            _ => { self.expect(&TokenKind::Eq, "`=` in assignment")?; None }
        };
        let rhs = self.parse_expr(ExprCtx::normal())?;
        let value = if let Some(op) = compound_op {
            // Desugar: `x += e` → `x = x + e`
            let lhs = Expr { kind: ExprKind::Ident(name.clone()), span: name_span };
            let span = merge(name_span, rhs.span);
            Expr {
                kind: ExprKind::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) },
                span,
            }
        } else {
            rhs
        };
        let span = merge(name_span, value.span);
        Ok(Stmt {
            kind: StmtKind::Assign { name, value },
            span,
        })
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::For, "`for`")?;
        let (binder, _) = self.expect_ident("loop binding")?;
        self.expect(&TokenKind::In, "`in` after loop binding")?;
        // Struct literals are disabled in the iterator expression so the
        // opening brace of the loop body is unambiguous.
        let iter = self.parse_expr(ExprCtx::no_struct())?;
        let body = self.parse_block()?;
        let span = merge(start, body.span);
        Ok(Stmt {
            kind: StmtKind::For { binder, iter, body },
            span,
        })
    }

    fn parse_return_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Return, "`return`")?;
        let value = match self.peek_kind() {
            TokenKind::RBrace | TokenKind::Eof => None,
            _ => Some(self.parse_expr(ExprCtx::normal())?),
        };
        let end = value
            .as_ref()
            .map(|e| e.span)
            .unwrap_or(start);
        Ok(Stmt {
            kind: StmtKind::Return(value),
            span: merge(start, end),
        })
    }

    // --- Expressions ------------------------------------------------------

    fn parse_expr(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        match self.peek_kind() {
            TokenKind::If => self.parse_if_expr(),
            TokenKind::Match => self.parse_match_expr(),
            TokenKind::Spawn => self.parse_spawn_expr(),
            TokenKind::Send => self.parse_send_expr(),
            TokenKind::Ask => self.parse_ask_expr(),
            _ => self.parse_or_expr(ctx),
        }
    }

    fn parse_spawn_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Spawn, "`spawn`")?;
        let (name, _name_span) = self.expect_ident("actor type name")?;
        // Parse struct literal body.
        self.expect(&TokenKind::LBrace, "`{` for initial state")?;
        let mut fields = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RBrace) {
            loop {
                fields.push(self.parse_field_init()?);
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
                if matches!(self.peek_kind(), TokenKind::RBrace) {
                    break;
                }
            }
        }
        let end = self.expect(&TokenKind::RBrace, "`}`")?.span;
        Ok(Expr {
            kind: ExprKind::Spawn {
                actor_name: name,
                fields,
            },
            span: merge(start, end),
        })
    }

    fn parse_send_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Send, "`send`")?;
        let handle = self.parse_postfix_expr(ExprCtx::no_struct())?;
        // The postfix chain should end with a method call. Extract it.
        match handle.kind {
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => Ok(Expr {
                kind: ExprKind::Send {
                    handle: receiver,
                    method,
                    args,
                },
                span: merge(start, handle.span),
            }),
            _ => Err(ParseError {
                span: handle.span,
                message: "expected `send handle.method(args)`".into(),
            }),
        }
    }

    fn parse_ask_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Ask, "`ask`")?;
        let handle = self.parse_postfix_expr(ExprCtx::no_struct())?;
        match handle.kind {
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => Ok(Expr {
                kind: ExprKind::Ask {
                    handle: receiver,
                    method,
                    args,
                },
                span: merge(start, handle.span),
            }),
            _ => Err(ParseError {
                span: handle.span,
                message: "expected `ask handle.method(args)`".into(),
            }),
        }
    }

    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::If, "`if`")?;
        let cond = self.parse_expr(ExprCtx::no_struct())?;
        let then_block = self.parse_block()?;
        // else is optional: `if cond { ... }` without else has unit type.
        // `else if` is sugar: parsed as `else { if ... }`.
        let (else_block, end_span) = if self.eat(&TokenKind::Else).is_some() {
            if matches!(self.peek_kind(), TokenKind::If) {
                // else-if chain: parse nested if as the else block's tail.
                let nested_if = self.parse_if_expr()?;
                let sp = nested_if.span;
                let eb = Block { stmts: Vec::new(), tail: Some(Box::new(nested_if)), span: sp };
                (eb, sp)
            } else {
                let eb = self.parse_block()?;
                let sp = eb.span;
                (eb, sp)
            }
        } else {
            (Block { stmts: Vec::new(), tail: None, span: then_block.span }, then_block.span)
        };
        let span = merge(start, end_span);
        Ok(Expr {
            kind: ExprKind::If {
                cond: Box::new(cond),
                then_block,
                else_block,
            },
            span,
        })
    }

    fn parse_match_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek().span;
        self.expect(&TokenKind::Match, "`match`")?;
        let scrutinee = self.parse_expr(ExprCtx::no_struct())?;
        self.expect(&TokenKind::LBrace, "`{` to open match body")?;

        let mut arms = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RBrace) {
            loop {
                arms.push(self.parse_match_arm()?);
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
                if matches!(self.peek_kind(), TokenKind::RBrace) {
                    break;
                }
            }
        }
        let end = self.expect(&TokenKind::RBrace, "`}` to close match body")?.span;

        Ok(Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(scrutinee),
                arms,
            },
            span: merge(start, end),
        })
    }

    fn parse_match_arm(&mut self) -> Result<MatchArm, ParseError> {
        let pattern = self.parse_pattern()?;
        self.expect(&TokenKind::FatArrow, "`=>` between pattern and body")?;
        let body = self.parse_expr(ExprCtx::normal())?;
        let span = merge(pattern.span, body.span);
        Ok(MatchArm {
            pattern,
            body,
            span,
        })
    }

    fn parse_or_expr(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_and_expr(ctx)?;
        while matches!(self.peek_kind(), TokenKind::OrOr) {
            self.bump();
            let rhs = self.parse_and_expr(ctx)?;
            lhs = bin(lhs, BinOp::Or, rhs);
        }
        Ok(lhs)
    }

    fn parse_and_expr(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_eq_expr(ctx)?;
        while matches!(self.peek_kind(), TokenKind::AndAnd) {
            self.bump();
            let rhs = self.parse_eq_expr(ctx)?;
            lhs = bin(lhs, BinOp::And, rhs);
        }
        Ok(lhs)
    }

    fn parse_eq_expr(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_rel_expr(ctx)?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::EqEq => BinOp::Eq,
                TokenKind::NotEq => BinOp::NotEq,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_rel_expr(ctx)?;
            lhs = bin(lhs, op, rhs);
        }
        Ok(lhs)
    }

    fn parse_rel_expr(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_add_expr(ctx)?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Lt => BinOp::Lt,
                TokenKind::LtEq => BinOp::LtEq,
                TokenKind::Gt => BinOp::Gt,
                TokenKind::GtEq => BinOp::GtEq,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_add_expr(ctx)?;
            lhs = bin(lhs, op, rhs);
        }
        Ok(lhs)
    }

    fn parse_add_expr(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_mul_expr(ctx)?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_mul_expr(ctx)?;
            lhs = bin(lhs, op, rhs);
        }
        Ok(lhs)
    }

    fn parse_mul_expr(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_cast_expr(ctx)?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Rem,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_cast_expr(ctx)?;
            lhs = bin(lhs, op, rhs);
        }
        Ok(lhs)
    }

    fn parse_cast_expr(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        let expr = self.parse_unary_expr(ctx)?;
        if matches!(self.peek_kind(), TokenKind::As) {
            self.bump();
            let ty = self.parse_type()?;
            let span = merge(expr.span, ty.span);
            Ok(Expr {
                kind: ExprKind::Cast {
                    expr: Box::new(expr),
                    to: ty,
                },
                span,
            })
        } else {
            Ok(expr)
        }
    }

    fn parse_unary_expr(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        match self.peek_kind() {
            TokenKind::Minus => {
                let start = self.bump().span;
                let rhs = self.parse_unary_expr(ctx)?;
                let span = merge(start, rhs.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Neg,
                        rhs: Box::new(rhs),
                    },
                    span,
                })
            }
            TokenKind::Bang => {
                let start = self.bump().span;
                let rhs = self.parse_unary_expr(ctx)?;
                let span = merge(start, rhs.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Not,
                        rhs: Box::new(rhs),
                    },
                    span,
                })
            }
            _ => self.parse_postfix_expr(ctx),
        }
    }

    fn parse_postfix_expr(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary(ctx)?;
        loop {
            match self.peek_kind() {
                TokenKind::Dot => {
                    self.bump();
                    // Tuple field access: `.0`, `.1`, etc.
                    if let TokenKind::IntLit { value, .. } = self.peek_kind() {
                        let index = *value as u32;
                        let idx_span = self.bump().span;
                        let span = merge(expr.span, idx_span);
                        expr = Expr {
                            kind: ExprKind::TupleField {
                                receiver: Box::new(expr),
                                index,
                            },
                            span,
                        };
                        continue;
                    }
                    let (name, name_span) = self.expect_ident("field or method name")?;
                    if matches!(self.peek_kind(), TokenKind::LParen) {
                        let (args, args_end) = self.parse_call_args()?;
                        let span = merge(expr.span, args_end);
                        expr = Expr {
                            kind: ExprKind::MethodCall {
                                receiver: Box::new(expr),
                                method: name,
                                args,
                            },
                            span,
                        };
                    } else {
                        let span = merge(expr.span, name_span);
                        expr = Expr {
                            kind: ExprKind::Field {
                                receiver: Box::new(expr),
                                name,
                            },
                            span,
                        };
                    }
                }
                TokenKind::LParen => {
                    let (args, args_end) = self.parse_call_args()?;
                    let span = merge(expr.span, args_end);
                    expr = Expr {
                        kind: ExprKind::Call {
                            callee: Box::new(expr),
                            args,
                        },
                        span,
                    };
                }
                TokenKind::Question => {
                    let q = self.bump();
                    let span = merge(expr.span, q.span);
                    expr = Expr {
                        kind: ExprKind::Try(Box::new(expr)),
                        span,
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_call_args(&mut self) -> Result<(Vec<Arg>, Span), ParseError> {
        self.expect(&TokenKind::LParen, "`(`")?;
        let mut args = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RParen) {
            loop {
                args.push(self.parse_arg()?);
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
                if matches!(self.peek_kind(), TokenKind::RParen) {
                    break;
                }
            }
        }
        let end = self.expect(&TokenKind::RParen, "`)` to close arguments")?.span;
        Ok((args, end))
    }

    fn parse_arg(&mut self) -> Result<Arg, ParseError> {
        // Named arg: `IDENT : expr`. Only commit to this form if we see
        // exactly that; otherwise fall through to a positional expression.
        if let TokenKind::Ident(_) = self.peek_kind() {
            if let Some(next) = self.peek_at(1) {
                if matches!(next.kind, TokenKind::Colon) {
                    let (name, name_span) = self.expect_ident("argument name")?;
                    self.bump(); // :
                    let value = self.parse_expr(ExprCtx::normal())?;
                    let span = merge(name_span, value.span);
                    return Ok(Arg {
                        name: Some(name),
                        value,
                        span,
                    });
                }
            }
        }

        let value = self.parse_expr(ExprCtx::normal())?;
        let span = value.span;
        Ok(Arg {
            name: None,
            value,
            span,
        })
    }

    fn parse_primary(&mut self, ctx: ExprCtx) -> Result<Expr, ParseError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::IntLit { value, suffix } => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::IntLit { value, suffix },
                    span: tok.span,
                })
            }
            TokenKind::FloatLit(v) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::FloatLit(v),
                    span: tok.span,
                })
            }
            TokenKind::StringLit(s) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::StringLit(s),
                    span: tok.span,
                })
            }
            TokenKind::InterpolatedString(parts) => {
                self.bump();
                let mut pieces = Vec::with_capacity(parts.len());
                for part in parts {
                    match part {
                        crate::lexer::InterpPart::Lit(s) => {
                            pieces.push(crate::ast::InterpPiece::Lit(s));
                        }
                        crate::lexer::InterpPart::Expr { source, line, col } => {
                            let expr = parse_interp_expr(&source, line, col, tok.span)?;
                            pieces.push(crate::ast::InterpPiece::Expr(expr));
                        }
                    }
                }
                Ok(Expr {
                    kind: ExprKind::Interpolated(pieces),
                    span: tok.span,
                })
            }
            TokenKind::True => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::BoolLit(true),
                    span: tok.span,
                })
            }
            TokenKind::False => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::BoolLit(false),
                    span: tok.span,
                })
            }
            TokenKind::Unit => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::UnitLit,
                    span: tok.span,
                })
            }
            TokenKind::LParen => {
                self.bump();
                let first = self.parse_expr(ExprCtx::normal())?;
                if self.eat(&TokenKind::Comma).is_some() {
                    // Tuple literal: (e1, e2, ...)
                    let mut elems = vec![first];
                    if !matches!(self.peek_kind(), TokenKind::RParen) {
                        loop {
                            elems.push(self.parse_expr(ExprCtx::normal())?);
                            if self.eat(&TokenKind::Comma).is_none() {
                                break;
                            }
                            if matches!(self.peek_kind(), TokenKind::RParen) {
                                break;
                            }
                        }
                    }
                    let end = self.expect(&TokenKind::RParen, "`)` to close tuple")?.span;
                    Ok(Expr {
                        kind: ExprKind::TupleLit(elems),
                        span: merge(tok.span, end),
                    })
                } else {
                    // Paren expression: (e)
                    let end = self.expect(&TokenKind::RParen, "`)` to close parenthesized expression")?.span;
                    Ok(Expr {
                        kind: ExprKind::Paren(Box::new(first)),
                        span: merge(tok.span, end),
                    })
                }
            }
            TokenKind::LBrace => {
                let block = self.parse_block()?;
                let span = block.span;
                Ok(Expr {
                    kind: ExprKind::Block(Box::new(block)),
                    span,
                })
            }
            // Non-capturing lambda: fn(x: i32): i32 { return x + 1 }
            TokenKind::Fn => {
                self.bump();
                self.expect(&TokenKind::LParen, "`(` after `fn` in lambda")?;
                let mut params = Vec::new();
                if !matches!(self.peek_kind(), TokenKind::RParen) {
                    loop {
                        params.push(self.parse_param()?);
                        if self.eat(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                }
                self.expect(&TokenKind::RParen, "`)` to close lambda parameters")?;
                self.expect(&TokenKind::Colon, "`:` before lambda return type")?;
                let return_type = self.parse_type()?;
                let body = self.parse_block()?;
                let end = body.span;
                Ok(Expr {
                    kind: ExprKind::Lambda { params, return_type, body },
                    span: merge(tok.span, end),
                })
            }
            TokenKind::Ident(_) => {
                let (name, name_span) = self.expect_ident("identifier")?;

                // Struct literal? IDENT `{` ... `}`, but only when struct
                // literals are allowed in this context.
                if ctx.allow_struct_lit && matches!(self.peek_kind(), TokenKind::LBrace) {
                    return self.parse_struct_lit_tail(name, name_span);
                }

                Ok(Expr {
                    kind: ExprKind::Ident(name),
                    span: name_span,
                })
            }
            _ => Err(self.error_here(format!(
                "expected an expression, found {}",
                describe_token(&tok.kind)
            ))),
        }
    }

    fn parse_struct_lit_tail(&mut self, name: String, name_span: Span) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::LBrace, "`{`")?;
        let mut fields = Vec::new();
        let mut spread = None;
        if !matches!(self.peek_kind(), TokenKind::RBrace) {
            loop {
                // Check for `..expr` spread syntax.
                if matches!(self.peek_kind(), TokenKind::Dot) {
                    if let Some(next) = self.peek_at(1) {
                        if matches!(next.kind, TokenKind::Dot) {
                            self.bump(); // first .
                            self.bump(); // second .
                            spread = Some(Box::new(self.parse_expr(ExprCtx::normal())?));
                            // Trailing comma allowed after spread.
                            self.eat(&TokenKind::Comma);
                            break;
                        }
                    }
                }
                fields.push(self.parse_field_init()?);
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
                if matches!(self.peek_kind(), TokenKind::RBrace) {
                    break;
                }
            }
        }
        let end = self.expect(&TokenKind::RBrace, "`}` to close struct literal")?.span;
        Ok(Expr {
            kind: ExprKind::StructLit {
                name,
                name_span,
                fields,
                spread,
            },
            span: merge(name_span, end),
        })
    }

    fn parse_field_init(&mut self) -> Result<FieldInit, ParseError> {
        let (name, name_span) = self.expect_ident("field name")?;
        self.expect(&TokenKind::Colon, "`:` in field initializer")?;
        let value = self.parse_expr(ExprCtx::normal())?;
        Ok(FieldInit {
            name,
            span: merge(name_span, value.span),
            value,
        })
    }

    // --- Patterns ---------------------------------------------------------

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::Ident(ref s) if s == "_" => {
                self.bump();
                Ok(Pattern {
                    kind: PatternKind::Wildcard,
                    span: tok.span,
                })
            }
            TokenKind::IntLit { value, suffix } => {
                self.bump();
                Ok(Pattern {
                    kind: PatternKind::Literal(LiteralPattern::Int(value, suffix)),
                    span: tok.span,
                })
            }
            TokenKind::StringLit(s) => {
                self.bump();
                Ok(Pattern {
                    kind: PatternKind::Literal(LiteralPattern::String(s)),
                    span: tok.span,
                })
            }
            TokenKind::True => {
                self.bump();
                Ok(Pattern {
                    kind: PatternKind::Literal(LiteralPattern::Bool(true)),
                    span: tok.span,
                })
            }
            TokenKind::False => {
                self.bump();
                Ok(Pattern {
                    kind: PatternKind::Literal(LiteralPattern::Bool(false)),
                    span: tok.span,
                })
            }
            TokenKind::Unit => {
                self.bump();
                Ok(Pattern {
                    kind: PatternKind::Literal(LiteralPattern::Unit),
                    span: tok.span,
                })
            }
            TokenKind::Ident(_) => {
                let (name, name_span) = self.expect_ident("pattern identifier")?;
                let payload = match self.peek_kind() {
                    TokenKind::LBrace => {
                        self.bump();
                        let mut fields = Vec::new();
                        if !matches!(self.peek_kind(), TokenKind::RBrace) {
                            loop {
                                fields.push(self.parse_pat_field()?);
                                if self.eat(&TokenKind::Comma).is_none() {
                                    break;
                                }
                                if matches!(self.peek_kind(), TokenKind::RBrace) {
                                    break;
                                }
                            }
                        }
                        self.expect(&TokenKind::RBrace, "`}` to close variant pattern")?;
                        Some(VariantPatPayload::Named(fields))
                    }
                    TokenKind::LParen => {
                        self.bump();
                        let mut pats = Vec::new();
                        if !matches!(self.peek_kind(), TokenKind::RParen) {
                            loop {
                                pats.push(self.parse_pattern()?);
                                if self.eat(&TokenKind::Comma).is_none() {
                                    break;
                                }
                            }
                        }
                        self.expect(
                            &TokenKind::RParen,
                            "`)` to close positional variant pattern",
                        )?;
                        Some(VariantPatPayload::Positional(pats))
                    }
                    _ => None,
                };
                let end = self.tokens[self.pos.saturating_sub(1)].span;
                // If there's no payload and the name starts with a lowercase
                // letter, treat it as a bare binding; otherwise it's a
                // variant constructor pattern. This is only a parse-time
                // heuristic — the typechecker makes the final call once it
                // can see which names are constructors in scope.
                let kind = if payload.is_none()
                    && name.chars().next().is_some_and(|c| c.is_lowercase() || c == '_')
                {
                    PatternKind::Binding(name)
                } else {
                    PatternKind::Variant { name, payload }
                };
                Ok(Pattern {
                    kind,
                    span: merge(name_span, end),
                })
            }
            _ => Err(self.error_here(format!(
                "expected a pattern, found {}",
                describe_token(&tok.kind)
            ))),
        }
    }

    fn parse_pat_field(&mut self) -> Result<PatField, ParseError> {
        let (name, name_span) = self.expect_ident("field name in pattern")?;
        self.expect(&TokenKind::Colon, "`:` in field pattern")?;
        let pattern = self.parse_pattern()?;
        Ok(PatField {
            name,
            span: merge(name_span, pattern.span),
            pattern,
        })
    }

    // --- Helpers ----------------------------------------------------------

    fn expect_ident(&mut self, what: &str) -> Result<(String, Span), ParseError> {
        match self.peek_kind().clone() {
            TokenKind::Ident(name) => {
                let span = self.bump().span;
                Ok((name, span))
            }
            other => Err(self.error_here(format!(
                "expected {what}, found {}",
                describe_token(&other)
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn merge(a: Span, b: Span) -> Span {
    Span::new(a.start.min(b.start), a.end.max(b.end), a.line, a.col)
}

fn bin(lhs: Expr, op: BinOp, rhs: Expr) -> Expr {
    let span = merge(lhs.span, rhs.span);
    Expr {
        kind: ExprKind::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        span,
    }
}

fn describe_token(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Fn => "`fn`".into(),
        TokenKind::Let => "`let`".into(),
        TokenKind::Var => "`var`".into(),
        TokenKind::Type => "`type`".into(),
        TokenKind::Import => "`import`".into(),
        TokenKind::If => "`if`".into(),
        TokenKind::Else => "`else`".into(),
        TokenKind::Match => "`match`".into(),
        TokenKind::For => "`for`".into(),
        TokenKind::In => "`in`".into(),
        TokenKind::Return => "`return`".into(),
        TokenKind::Extern => "`extern`".into(),
        TokenKind::Actor => "`actor`".into(),
        TokenKind::Msg => "`msg`".into(),
        TokenKind::Spawn => "`spawn`".into(),
        TokenKind::Send => "`send`".into(),
        TokenKind::Ask => "`ask`".into(),
        TokenKind::As => "`as`".into(),
        TokenKind::True => "`true`".into(),
        TokenKind::False => "`false`".into(),
        TokenKind::Unit => "`unit`".into(),
        TokenKind::Ident(s) => format!("identifier `{s}`"),
        TokenKind::IntLit { .. } => "integer literal".into(),
        TokenKind::FloatLit(_) => "float literal".into(),
        TokenKind::StringLit(_) => "string literal".into(),
        TokenKind::InterpolatedString(_) => "interpolated string literal".into(),
        TokenKind::LBrace => "`{`".into(),
        TokenKind::RBrace => "`}`".into(),
        TokenKind::LParen => "`(`".into(),
        TokenKind::RParen => "`)`".into(),
        TokenKind::LBracket => "`[`".into(),
        TokenKind::RBracket => "`]`".into(),
        TokenKind::Comma => "`,`".into(),
        TokenKind::Colon => "`:`".into(),
        TokenKind::Semi => "`;` (semicolons are not used in Lumen)".into(),
        TokenKind::Dot => "`.`".into(),
        TokenKind::Question => "`?`".into(),
        TokenKind::Pipe => "`|`".into(),
        TokenKind::Arrow => "`->`".into(),
        TokenKind::FatArrow => "`=>`".into(),
        TokenKind::Eq => "`=`".into(),
        TokenKind::PlusEq => "`+=`".into(),
        TokenKind::MinusEq => "`-=`".into(),
        TokenKind::StarEq => "`*=`".into(),
        TokenKind::EqEq => "`==`".into(),
        TokenKind::NotEq => "`!=`".into(),
        TokenKind::Lt => "`<`".into(),
        TokenKind::Gt => "`>`".into(),
        TokenKind::LtEq => "`<=`".into(),
        TokenKind::GtEq => "`>=`".into(),
        TokenKind::AndAnd => "`&&`".into(),
        TokenKind::OrOr => "`||`".into(),
        TokenKind::Bang => "`!`".into(),
        TokenKind::Plus => "`+`".into(),
        TokenKind::Minus => "`-`".into(),
        TokenKind::Star => "`*`".into(),
        TokenKind::Slash => "`/`".into(),
        TokenKind::Percent => "`%`".into(),
        TokenKind::Eof => "end of file".into(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    fn parse_ok(src: &str) -> Module {
        let toks = lex(src).unwrap_or_else(|e| panic!("lex failed: {e}"));
        parse(toks).unwrap_or_else(|e| panic!("parse failed: {e}"))
    }

    fn parse_err(src: &str) -> ParseError {
        let toks = lex(src).unwrap_or_else(|e| panic!("lex failed: {e}"));
        parse(toks).expect_err("parse should have failed")
    }

    #[test]
    fn empty_module() {
        let m = parse_ok("");
        assert!(m.imports.is_empty());
        assert!(m.items.is_empty());
    }

    #[test]
    fn single_import() {
        let m = parse_ok("import std/io");
        assert_eq!(m.imports.len(), 1);
        assert_eq!(m.imports[0].path, vec!["std".to_string(), "io".to_string()]);
    }

    #[test]
    fn multiple_imports() {
        let m = parse_ok("import std/io\nimport std/list\nimport std/string");
        assert_eq!(m.imports.len(), 3);
    }

    #[test]
    fn fn_without_params() {
        let m = parse_ok("fn answer(): i32 { 42 }");
        assert_eq!(m.items.len(), 1);
        let Item::Fn(f) = &m.items[0] else { panic!() };
        assert_eq!(f.name, "answer");
        assert!(f.params.is_empty());
        assert!(f.body.stmts.is_empty());
        assert!(f.body.tail.is_some());
        assert_eq!(f.effect, Effect::Pure);
    }

    #[test]
    fn fn_with_params_and_io_effect() {
        let m = parse_ok("fn greet(name: string): unit io { io.println(name) }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        assert_eq!(f.params.len(), 1);
        assert_eq!(f.params[0].name, "name");
        assert_eq!(f.effect, Effect::Io);
    }

    #[test]
    fn struct_type_decl() {
        let m = parse_ok("type User = { name: string, age: i32 }");
        let Item::Type(t) = &m.items[0] else { panic!() };
        assert_eq!(t.name, "User");
        let TypeBody::Struct(fields) = &t.body else { panic!() };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "name");
        assert_eq!(fields[1].name, "age");
    }

    #[test]
    fn sum_type_decl_with_leading_pipe() {
        let m = parse_ok("type LoadError = | NotFound { path: string } | ParseError { line: i32, reason: string }");
        let Item::Type(t) = &m.items[0] else { panic!() };
        let TypeBody::Sum(variants) = &t.body else { panic!() };
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0].name, "NotFound");
        assert_eq!(variants[1].name, "ParseError");
        let Some(VariantPayload::Named(fs)) = &variants[0].payload else { panic!() };
        assert_eq!(fs[0].name, "path");
    }

    #[test]
    fn sum_type_with_positional_payload() {
        let m = parse_ok("type T = | A(i32) | B(string, bool)");
        let Item::Type(t) = &m.items[0] else { panic!() };
        let TypeBody::Sum(variants) = &t.body else { panic!() };
        assert_eq!(variants.len(), 2);
        let Some(VariantPayload::Positional(a)) = &variants[0].payload else { panic!() };
        assert_eq!(a.len(), 1);
        let Some(VariantPayload::Positional(b)) = &variants[1].payload else { panic!() };
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn let_and_var_and_assign() {
        let m = parse_ok("fn f(): i32 { let x: i32 = 1\nvar y: i32 = 2\ny = y + x\ny }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        assert_eq!(f.body.stmts.len(), 3);
        assert!(matches!(f.body.stmts[0].kind, StmtKind::Let { .. }));
        assert!(matches!(f.body.stmts[1].kind, StmtKind::Var { .. }));
        assert!(matches!(f.body.stmts[2].kind, StmtKind::Assign { .. }));
        assert!(f.body.tail.is_some());
    }

    #[test]
    fn for_loop_with_range() {
        let m = parse_ok("fn f(n: i32): i32 { var total: i32 = 0\nfor i in range(1, n) { total = total + i }\ntotal }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let for_stmt = f.body.stmts.iter().find(|s| matches!(s.kind, StmtKind::For { .. })).unwrap();
        let StmtKind::For { binder, body, .. } = &for_stmt.kind else { panic!() };
        assert_eq!(binder, "i");
        assert_eq!(body.stmts.len(), 1);
    }

    #[test]
    fn if_without_else_parses_as_unit() {
        // else is now optional — if without else produces unit.
        let m = parse_ok("fn f(): unit { if true { let x = 1 } }");
        assert_eq!(m.items.len(), 1);
    }

    #[test]
    fn match_with_named_variant_patterns() {
        let src = r#"
            fn area(s: Shape): f64 {
                match s {
                    Circle { radius: r } => r,
                    Rectangle { width: w, height: h } => w,
                    _ => 0.0,
                }
            }
        "#;
        let m = parse_ok(src);
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        let ExprKind::Match { arms, .. } = &tail.kind else { panic!() };
        assert_eq!(arms.len(), 3);
        assert!(matches!(arms[2].pattern.kind, PatternKind::Wildcard));
        let PatternKind::Variant { name, payload: Some(VariantPatPayload::Named(fs)) } =
            &arms[0].pattern.kind
        else { panic!() };
        assert_eq!(name, "Circle");
        assert_eq!(fs[0].name, "radius");
    }

    #[test]
    fn try_operator_postfix() {
        let m = parse_ok("fn f(): Result<i32, i32> io { let x = io.read()?\nOk(x) }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let StmtKind::Let { value, .. } = &f.body.stmts[0].kind else { panic!() };
        assert!(matches!(value.kind, ExprKind::Try(_)));
    }

    #[test]
    fn method_call_chain() {
        let m = parse_ok("fn f(): i32 { xs.iter().map(double).sum() }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        // The outermost node should be a method call `.sum()`
        let ExprKind::MethodCall { method, .. } = &tail.kind else { panic!() };
        assert_eq!(method, "sum");
    }

    #[test]
    fn cast_with_as() {
        let m = parse_ok("fn f(n: i32): i64 { n as i64 }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        assert!(matches!(tail.kind, ExprKind::Cast { .. }));
    }

    #[test]
    fn precedence_mul_binds_tighter_than_add() {
        let m = parse_ok("fn f(): i32 { 1 + 2 * 3 }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        let ExprKind::Binary { op: BinOp::Add, rhs, .. } = &tail.kind else { panic!() };
        // The right-hand side should be 2 * 3.
        let ExprKind::Binary { op: BinOp::Mul, .. } = &rhs.kind else { panic!() };
    }

    #[test]
    fn precedence_cmp_below_arith() {
        let m = parse_ok("fn f(): bool { 1 + 2 < 4 * 5 }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        let ExprKind::Binary { op: BinOp::Lt, .. } = &tail.kind else { panic!() };
    }

    #[test]
    fn unary_neg_and_not() {
        let m = parse_ok("fn f(): bool { !(-x == 0) }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        let ExprKind::Unary { op: UnaryOp::Not, .. } = &tail.kind else { panic!() };
    }

    #[test]
    fn struct_literal_in_expression() {
        let m = parse_ok("fn f(): User { User { name: \"a\", age: 1 } }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        let ExprKind::StructLit { name, fields, .. } = &tail.kind else { panic!() };
        assert_eq!(name, "User");
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn struct_literal_disallowed_in_if_head() {
        // `if c { 1 } else { 2 }` where c = `Foo{}` needs parens so the
        // opening brace of the then-block isn't eaten as a struct literal.
        // Without parens, `Foo` is a plain IDENT and `{` starts the then-body.
        let m = parse_ok("fn f(): i32 { if Foo { 1 } else { 2 } }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        let ExprKind::If { cond, .. } = &tail.kind else { panic!() };
        assert!(matches!(cond.kind, ExprKind::Ident(_)));
    }

    #[test]
    fn named_and_positional_args() {
        let m = parse_ok("fn f(): i32 { g(1, x: 2, 3, y: 4) }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        let ExprKind::Call { args, .. } = &tail.kind else { panic!() };
        assert_eq!(args.len(), 4);
        assert_eq!(args[0].name, None);
        assert_eq!(args[1].name.as_deref(), Some("x"));
        assert_eq!(args[2].name, None);
        assert_eq!(args[3].name.as_deref(), Some("y"));
    }

    #[test]
    fn return_statement() {
        let m = parse_ok("fn f(): i32 { return 1 }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        assert!(matches!(f.body.stmts[0].kind, StmtKind::Return(Some(_))));
    }

    #[test]
    fn semicolon_yields_friendly_error() {
        let err = parse_err("fn f(): i32 { let x = 1; x }");
        assert!(err.message.contains("semicolon"));
    }

    #[test]
    fn hello_world_parses() {
        let src = r#"
            import std/io

            fn main(): Result<unit, unit> io {
                io.println("hello, world")
                Ok(unit)
            }
        "#;
        let m = parse_ok(src);
        assert_eq!(m.imports.len(), 1);
        assert_eq!(m.items.len(), 1);
    }

    #[test]
    fn load_user_parses() {
        // The post-revision canonical example from the epic.
        let src = r#"
            import std/io
            import std/int

            type User = { name: string, age: i32 }

            type LoadError =
                | NotFound { path: string }
                | ParseError { line: i32, reason: string }

            fn load_user(path: string): Result<User, LoadError> io {
                let contents = io.read_file(path)?
                let lines = contents.split("\n")
                let name = lines.get(0).ok_or(ParseError { line: 0, reason: "no name" })?
                let age_str = lines.get(1).ok_or(ParseError { line: 1, reason: "no age" })?
                let age = int.parse_i32(age_str)?
                Ok(User { name: name, age: age })
            }
        "#;
        let m = parse_ok(src);
        assert_eq!(m.imports.len(), 2);
        assert_eq!(m.items.len(), 3);
    }

    #[test]
    fn sum_of_squares_parses() {
        let src = r#"
            fn sum_of_squares(n: i32): i32 {
                var total: i32 = 0
                for i in range(1, n + 1) {
                    total = total + i * i
                }
                total
            }
        "#;
        let m = parse_ok(src);
        assert_eq!(m.items.len(), 1);
    }

    #[test]
    fn shape_match_parses() {
        let src = r#"
            type Shape =
                | Circle { radius: f64 }
                | Rectangle { width: f64, height: f64 }
                | Triangle { base: f64, height: f64 }

            fn area(s: Shape): f64 {
                match s {
                    Circle { radius: r } => r,
                    Rectangle { width: w, height: h } => w,
                    Triangle { base: b, height: h } => b,
                }
            }
        "#;
        let m = parse_ok(src);
        assert_eq!(m.items.len(), 2);
    }

    #[test]
    fn shape_match_parses_error_on_missing_arrow() {
        let err = parse_err("fn f(s: i32): i32 { match s { 1 => 1, 2 2, _ => 0 } }");
        assert!(err.message.contains("=>"));
    }

    #[test]
    fn lambda_expression_parses() {
        let m = parse_ok("fn f(): i32 { let g = fn(x: i32): i32 { return x + 1 } \n g(5) }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let StmtKind::Let { value, .. } = &f.body.stmts[0].kind else { panic!() };
        assert!(matches!(value.kind, ExprKind::Lambda { .. }));
    }

    #[test]
    fn fn_ptr_type_parses() {
        let m = parse_ok("fn apply(f: fn(i32): i32, x: i32): i32 { f(x) }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        assert!(matches!(f.params[0].ty.kind, TypeKind::FnPtr { .. }));
    }

    #[test]
    fn extern_fn_parses() {
        let m = parse_ok("extern fn malloc(size: i64): i64");
        assert!(matches!(&m.items[0], Item::ExternFn(_)));
    }

    #[test]
    fn extern_fn_with_link_name() {
        let m = parse_ok(r#"extern fn sqrt(x: f64): f64 link "lumen_sqrt""#);
        let Item::ExternFn(ef) = &m.items[0] else { panic!() };
        assert_eq!(ef.name, "sqrt");
        assert_eq!(ef.link_name.as_deref(), Some("lumen_sqrt"));
    }

    #[test]
    fn actor_and_msg_handler_parse() {
        let src = r#"
            actor Counter { count: i32 }
            msg Counter.increment(self, n: i32): Counter {
                return Counter { count: self.count + n }
            }
        "#;
        let m = parse_ok(src);
        assert!(matches!(&m.items[0], Item::Actor(_)));
        assert!(matches!(&m.items[1], Item::MsgHandler(_)));
    }

    #[test]
    fn tuple_destructuring_parses() {
        let m = parse_ok("fn f(): i32 { let (a, b) = (1, 2) \n a }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        assert!(matches!(&f.body.stmts[0].kind, StmtKind::LetTuple { .. }));
    }

    #[test]
    fn block_expression_parses() {
        let m = parse_ok("fn f(): i32 { let x = { 42 } \n x }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let StmtKind::Let { value, .. } = &f.body.stmts[0].kind else { panic!() };
        assert!(matches!(value.kind, ExprKind::Block(_)));
    }

    #[test]
    fn for_loop_parses() {
        let m = parse_ok("fn f(): i32 { for i in range(0, 10) { i } \n 0 }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        assert!(matches!(&f.body.stmts[0].kind, StmtKind::For { .. }));
    }

    #[test]
    #[test]
    fn try_operator_parses() {
        let m = parse_ok("fn f(): Result<i32, i32> { let x = Ok(1)? \n Ok(x) }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let StmtKind::Let { value, .. } = &f.body.stmts[0].kind else { panic!() };
        assert!(matches!(value.kind, ExprKind::Try(_)));
    }

    #[test]
    fn else_if_chain_parses() {
        let m = parse_ok("fn f(x: i32): i32 { if x == 1 { 10 } else if x == 2 { 20 } else { 30 } }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        let tail = f.body.tail.as_ref().unwrap();
        let ExprKind::If { else_block, .. } = &tail.kind else { panic!() };
        // else block's tail should be a nested if
        let nested = else_block.tail.as_ref().unwrap();
        assert!(matches!(nested.kind, ExprKind::If { .. }));
    }

    #[test]
    fn compound_assignment_parses() {
        let m = parse_ok("fn f(): i32 { var x = 0 \n x += 5 \n x -= 2 \n x *= 3 \n return x }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        // x += 5 desugars to x = x + 5
        let StmtKind::Assign { value, .. } = &f.body.stmts[1].kind else { panic!() };
        assert!(matches!(value.kind, ExprKind::Binary { op: BinOp::Add, .. }));
    }

    #[test]
    fn block_comment_skipped() {
        let m = parse_ok("fn f(): i32 { /* this is a comment */ 42 }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        assert!(f.body.tail.is_some());
    }

    #[test]
    fn nested_block_comment_skipped() {
        let m = parse_ok("fn f(): i32 { /* outer /* nested */ still comment */ 42 }");
        let Item::Fn(f) = &m.items[0] else { panic!() };
        assert!(f.body.tail.is_some());
    }
}
