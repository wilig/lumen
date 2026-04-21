//! Canonical source formatter (lumen-43k).
//!
//! `lumen fmt FILE.lm` rewrites a Lumen source file to a single
//! canonical style. No configuration knobs — one style, no options.
//! Ships AST-based: lex → parse → re-emit from the AST. Comments are
//! preserved via a source-scanning pass that interleaves comment
//! ranges at AST-node boundaries.
//!
//! Known MVP limitations:
//!   - Comments inside expressions (mid-line comments inside a
//!     struct literal or between args) may move to a nearby boundary.
//!   - Trailing comments on the same line as an item migrate to
//!     their own line.
//!
//! Both are clean follow-ups. See lumen-43k-followup tickets.

use crate::ast::*;
use crate::lexer;
use crate::parser;

/// Format `src` and return the canonical version. Errors bubble up
/// as human-readable strings so CLI can surface them.
pub fn format(src: &str) -> Result<String, String> {
    let tokens = lexer::lex(src).map_err(|e| format!("lex: {}", e.message))?;
    let module = parser::parse(tokens).map_err(|e| format!("parse: {}", e.message))?;
    let comments = collect_comments(src);
    let mut ctx = FmtCtx::new(src, comments);
    emit_module(&module, &mut ctx);
    Ok(ctx.out)
}

// --- FmtCtx -------------------------------------------------------------

struct FmtCtx<'a> {
    out: String,
    indent: u32,
    #[allow(dead_code)]
    src: &'a str,
    comments: Vec<CommentRange>,
    /// Byte offset in source up through which we've already emitted
    /// (or considered emitting) comments. Ratchets forward only.
    comments_emitted_through: u32,
}

#[derive(Debug, Clone)]
struct CommentRange {
    start: u32,
    end: u32,
    text: String,
    kind: CommentKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CommentKind {
    Line,
    Block,
}

impl<'a> FmtCtx<'a> {
    fn new(src: &'a str, comments: Vec<CommentRange>) -> Self {
        Self {
            out: String::new(),
            indent: 0,
            src,
            comments,
            comments_emitted_through: 0,
        }
    }

    fn write_indent(&mut self) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
    }

    fn newline(&mut self) {
        if !self.out.is_empty() && !self.out.ends_with('\n') {
            self.out.push('\n');
        }
    }

    fn blank_line(&mut self) {
        self.newline();
        if !self.out.ends_with("\n\n") {
            self.out.push('\n');
        }
    }

    /// Emit any comments with `start >= from && start < until` that
    /// haven't been emitted yet. Called at AST-node boundaries to
    /// preserve comments without a full lexer rewrite.
    fn flush_comments_before(&mut self, until: u32) {
        let from = self.comments_emitted_through;
        let mut any = false;
        let mut emitted: Vec<CommentRange> = Vec::new();
        for c in &self.comments {
            if c.start >= from && c.start < until {
                emitted.push(c.clone());
            }
        }
        for c in emitted {
            if !any && !self.out.is_empty() && !self.out.ends_with('\n') {
                self.out.push(' ');
            }
            if !any {
                self.newline();
            }
            self.write_indent();
            self.out.push_str(&c.text);
            self.newline();
            any = true;
            if c.end > self.comments_emitted_through {
                self.comments_emitted_through = c.end;
            }
        }
        if until > self.comments_emitted_through {
            self.comments_emitted_through = until;
        }
    }
}

// --- Module + items -----------------------------------------------------

fn emit_module(m: &Module, ctx: &mut FmtCtx) {
    // Emit leading comments (before first import/item).
    let first_pos = m.imports.first().map(|i| i.span.start)
        .or_else(|| m.items.first().map(item_span_start))
        .unwrap_or(u32::MAX);
    ctx.flush_comments_before(first_pos);
    let had_leading = !ctx.out.is_empty();

    // Sort imports alphabetically by path for canonical ordering.
    let mut sorted_imports: Vec<&Import> = m.imports.iter().collect();
    sorted_imports.sort_by(|a, b| a.path.cmp(&b.path));

    if had_leading && (!sorted_imports.is_empty() || !m.items.is_empty()) {
        // Blank line between the module-level comment block and
        // the first import/item — matches the repo convention.
        ctx.blank_line();
    }

    for imp in &sorted_imports {
        emit_import(imp, ctx);
        ctx.newline();
    }
    if !m.imports.is_empty() && !m.items.is_empty() {
        ctx.blank_line();
    }

    for (i, item) in m.items.iter().enumerate() {
        ctx.flush_comments_before(item_span_start(item));
        emit_item(item, ctx);
        if i + 1 < m.items.len() {
            // Consecutive extern fns stick together (matches the
            // repo style — groups of FFI declarations without blank
            // lines). Other item boundaries get a blank line.
            let next = &m.items[i + 1];
            let extern_run = matches!(item, Item::ExternFn(_))
                && matches!(next, Item::ExternFn(_));
            if extern_run {
                ctx.newline();
            } else {
                ctx.blank_line();
            }
        }
    }
    // Tail comments — comments after the last item.
    ctx.flush_comments_before(u32::MAX);
    ctx.newline();
}

fn item_span_start(item: &Item) -> u32 {
    match item {
        Item::Fn(f) => f.span.start,
        Item::Type(t) => t.span.start,
        Item::ExternFn(e) => e.span.start,
        Item::Actor(a) => a.span.start,
        Item::MsgHandler(m) => m.span.start,
        Item::GlobalLet(g) => g.span.start,
    }
}

fn emit_import(imp: &Import, ctx: &mut FmtCtx) {
    ctx.out.push_str("import ");
    ctx.out.push_str(&imp.path.join("/"));
    if let Some(alias) = &imp.alias {
        ctx.out.push_str(" as ");
        ctx.out.push_str(alias);
    }
}

fn emit_item(item: &Item, ctx: &mut FmtCtx) {
    match item {
        Item::Fn(f) => emit_fn_decl(f, ctx),
        Item::Type(t) => emit_type_decl(t, ctx),
        Item::ExternFn(e) => emit_extern_fn_decl(e, ctx),
        Item::Actor(a) => emit_actor_decl(a, ctx),
        Item::MsgHandler(m) => emit_msg_handler(m, ctx),
        Item::GlobalLet(g) => emit_global_let(g, ctx),
    }
}

// --- Fn decls ----------------------------------------------------------

fn emit_fn_decl(f: &FnDecl, ctx: &mut FmtCtx) {
    ctx.write_indent();
    ctx.out.push_str("fn ");
    ctx.out.push_str(&f.name);
    emit_type_params(&f.type_params, ctx);
    emit_params(&f.params, ctx);
    ctx.out.push_str(": ");
    emit_type(&f.return_type, ctx);
    if f.effect == Effect::Io {
        ctx.out.push_str(" io");
    }
    ctx.out.push(' ');
    emit_block(&f.body, ctx);
}

fn emit_extern_fn_decl(e: &ExternFnDecl, ctx: &mut FmtCtx) {
    ctx.write_indent();
    ctx.out.push_str("extern fn ");
    ctx.out.push_str(&e.name);
    emit_type_params(&e.type_params, ctx);
    emit_params(&e.params, ctx);
    ctx.out.push_str(": ");
    emit_type(&e.return_type, ctx);
    if let Some(link) = &e.link_name {
        ctx.out.push_str(" link \"");
        ctx.out.push_str(link);
        ctx.out.push('"');
    }
}

fn emit_msg_handler(m: &MsgHandlerDecl, ctx: &mut FmtCtx) {
    ctx.write_indent();
    ctx.out.push_str("msg ");
    ctx.out.push_str(&m.actor_name);
    ctx.out.push('.');
    ctx.out.push_str(&m.name);
    ctx.out.push('(');
    ctx.out.push_str("self");
    for p in &m.params {
        ctx.out.push_str(", ");
        ctx.out.push_str(&p.name);
        ctx.out.push_str(": ");
        emit_type(&p.ty, ctx);
    }
    ctx.out.push_str("): ");
    emit_type(&m.return_type, ctx);
    ctx.out.push(' ');
    emit_block(&m.body, ctx);
}

fn emit_type_params(tps: &[String], ctx: &mut FmtCtx) {
    if tps.is_empty() { return; }
    ctx.out.push('<');
    for (i, tp) in tps.iter().enumerate() {
        if i > 0 { ctx.out.push_str(", "); }
        ctx.out.push_str(tp);
    }
    ctx.out.push('>');
}

fn emit_params(params: &[Param], ctx: &mut FmtCtx) {
    ctx.out.push('(');
    for (i, p) in params.iter().enumerate() {
        if i > 0 { ctx.out.push_str(", "); }
        ctx.out.push_str(&p.name);
        ctx.out.push_str(": ");
        emit_type(&p.ty, ctx);
    }
    ctx.out.push(')');
}

// --- Type decls + types -----------------------------------------------

fn emit_type_decl(t: &TypeDecl, ctx: &mut FmtCtx) {
    ctx.write_indent();
    ctx.out.push_str("type ");
    ctx.out.push_str(&t.name);
    emit_type_params(&t.type_params, ctx);
    ctx.out.push_str(" = ");
    match &t.body {
        TypeBody::Struct(fields) => {
            ctx.out.push('{');
            ctx.newline();
            ctx.indent += 1;
            for (i, f) in fields.iter().enumerate() {
                ctx.write_indent();
                ctx.out.push_str(&f.name);
                ctx.out.push_str(": ");
                emit_type(&f.ty, ctx);
                if i + 1 < fields.len() {
                    ctx.out.push(',');
                }
                ctx.newline();
            }
            ctx.indent -= 1;
            ctx.write_indent();
            ctx.out.push('}');
        }
        TypeBody::Sum(variants) => {
            for (i, v) in variants.iter().enumerate() {
                if i > 0 { ctx.out.push_str(" | "); }
                emit_variant(v, ctx);
            }
        }
    }
}

fn emit_variant(v: &Variant, ctx: &mut FmtCtx) {
    ctx.out.push_str(&v.name);
    if let Some(payload) = &v.payload {
        match payload {
            VariantPayload::Named(fields) => {
                ctx.out.push_str(" {");
                for (i, f) in fields.iter().enumerate() {
                    if i > 0 { ctx.out.push(','); }
                    ctx.out.push(' ');
                    ctx.out.push_str(&f.name);
                    ctx.out.push_str(": ");
                    emit_type(&f.ty, ctx);
                }
                ctx.out.push_str(" }");
            }
            VariantPayload::Positional(types) => {
                ctx.out.push('(');
                for (i, t) in types.iter().enumerate() {
                    if i > 0 { ctx.out.push_str(", "); }
                    emit_type(t, ctx);
                }
                ctx.out.push(')');
            }
        }
    }
}

fn emit_actor_decl(a: &ActorDecl, ctx: &mut FmtCtx) {
    ctx.write_indent();
    ctx.out.push_str("actor ");
    ctx.out.push_str(&a.name);
    ctx.out.push_str(" {");
    if a.fields.is_empty() {
        ctx.out.push('}');
        return;
    }
    ctx.newline();
    ctx.indent += 1;
    for (i, f) in a.fields.iter().enumerate() {
        ctx.write_indent();
        ctx.out.push_str(&f.name);
        ctx.out.push_str(": ");
        emit_type(&f.ty, ctx);
        if i + 1 < a.fields.len() {
            ctx.out.push(',');
        }
        ctx.newline();
    }
    ctx.indent -= 1;
    ctx.write_indent();
    ctx.out.push('}');
}

fn emit_global_let(g: &GlobalLetDecl, ctx: &mut FmtCtx) {
    ctx.write_indent();
    ctx.out.push_str(if g.mutable { "var " } else { "let " });
    ctx.out.push_str(&g.name);
    if let Some(ty) = &g.ty {
        ctx.out.push_str(": ");
        emit_type(ty, ctx);
    }
    ctx.out.push_str(" = ");
    emit_expr(&g.value, ctx);
}

fn emit_type(t: &Type, ctx: &mut FmtCtx) {
    match &t.kind {
        TypeKind::Named { name, args } => {
            ctx.out.push_str(name);
            if !args.is_empty() {
                ctx.out.push('<');
                for (i, a) in args.iter().enumerate() {
                    if i > 0 { ctx.out.push_str(", "); }
                    emit_type(a, ctx);
                }
                ctx.out.push('>');
            }
        }
        TypeKind::Tuple(elems) => {
            ctx.out.push('(');
            for (i, e) in elems.iter().enumerate() {
                if i > 0 { ctx.out.push_str(", "); }
                emit_type(e, ctx);
            }
            ctx.out.push(')');
        }
        TypeKind::FnPtr { params, ret } => {
            ctx.out.push_str("fn(");
            for (i, p) in params.iter().enumerate() {
                if i > 0 { ctx.out.push_str(", "); }
                emit_type(p, ctx);
            }
            ctx.out.push_str("): ");
            emit_type(ret, ctx);
        }
    }
}

// --- Blocks + statements -----------------------------------------------

fn emit_block(block: &Block, ctx: &mut FmtCtx) {
    if block.stmts.is_empty() && block.tail.is_none() {
        ctx.out.push_str("{}");
        return;
    }
    // Single-statement / single-tail blocks with no intervening
    // comments collapse to `{ body }` inline — matches the repo's
    // one-liners like `fn WIN_WIDTH(): i32 { return 800 }`. Fall
    // back to multi-line if the inline form overflows 80 columns
    // on the current line.
    let one_liner = block_one_liner(block, ctx);
    if let Some(inline) = one_liner {
        let current_col = current_column(&ctx.out);
        if current_col + inline.len() + 4 /* "{ " + " }" */ <= 80 {
            ctx.out.push_str("{ ");
            ctx.out.push_str(&inline);
            ctx.out.push_str(" }");
            return;
        }
    }
    ctx.out.push('{');
    ctx.newline();
    ctx.indent += 1;
    for stmt in &block.stmts {
        ctx.flush_comments_before(stmt.span.start);
        ctx.write_indent();
        emit_stmt(stmt, ctx);
        ctx.newline();
    }
    if let Some(tail) = &block.tail {
        ctx.flush_comments_before(tail.span.start);
        ctx.write_indent();
        emit_expr(tail, ctx);
        ctx.newline();
    }
    ctx.indent -= 1;
    ctx.write_indent();
    ctx.out.push('}');
}

/// If the block is a single stmt or single tail with no embedded
/// comments, render it into a string; caller decides whether it fits
/// inline. Returns None if the block is multi-item.
fn block_one_liner(block: &Block, parent: &FmtCtx) -> Option<String> {
    let has_inner_comments = parent.comments.iter().any(|c|
        c.start >= block.span.start && c.end <= block.span.end
    );
    if has_inner_comments { return None; }
    let num = block.stmts.len() + if block.tail.is_some() { 1 } else { 0 };
    if num != 1 { return None; }
    let mut sub = FmtCtx::new(parent.src, Vec::new());
    sub.indent = 0;
    if let Some(stmt) = block.stmts.first() {
        emit_stmt(stmt, &mut sub);
    } else if let Some(tail) = &block.tail {
        emit_expr(tail, &mut sub);
    }
    // Only inline if the emitted body contains no newline.
    if sub.out.contains('\n') { return None; }
    Some(sub.out)
}

fn current_column(s: &str) -> usize {
    match s.rfind('\n') {
        Some(i) => s.len() - i - 1,
        None => s.len(),
    }
}

fn emit_stmt(s: &Stmt, ctx: &mut FmtCtx) {
    match &s.kind {
        StmtKind::Let { name, ty, value } => {
            ctx.out.push_str("let ");
            ctx.out.push_str(name);
            if let Some(t) = ty {
                ctx.out.push_str(": ");
                emit_type(t, ctx);
            }
            ctx.out.push_str(" = ");
            emit_expr(value, ctx);
        }
        StmtKind::Var { name, ty, value } => {
            ctx.out.push_str("var ");
            ctx.out.push_str(name);
            if let Some(t) = ty {
                ctx.out.push_str(": ");
                emit_type(t, ctx);
            }
            ctx.out.push_str(" = ");
            emit_expr(value, ctx);
        }
        StmtKind::LetTuple { names, value } => {
            ctx.out.push_str("let (");
            for (i, n) in names.iter().enumerate() {
                if i > 0 { ctx.out.push_str(", "); }
                ctx.out.push_str(n);
            }
            ctx.out.push_str(") = ");
            emit_expr(value, ctx);
        }
        StmtKind::Assign { name, value } => {
            ctx.out.push_str(name);
            ctx.out.push_str(" = ");
            emit_expr(value, ctx);
        }
        StmtKind::Expr(e) => emit_expr(e, ctx),
        StmtKind::For { binder, iter, body } => {
            ctx.out.push_str("for ");
            ctx.out.push_str(binder);
            ctx.out.push_str(" in ");
            emit_expr(iter, ctx);
            ctx.out.push(' ');
            emit_block(body, ctx);
        }
        StmtKind::Return(value) => {
            ctx.out.push_str("return");
            if let Some(e) = value {
                ctx.out.push(' ');
                emit_expr(e, ctx);
            }
        }
    }
}

// --- Expressions -------------------------------------------------------

fn emit_expr(e: &Expr, ctx: &mut FmtCtx) {
    match &e.kind {
        ExprKind::IntLit { value, suffix } => {
            use crate::lexer::IntSuffix;
            ctx.out.push_str(&value.to_string());
            if let Some(suf) = suffix {
                ctx.out.push_str(match suf {
                    IntSuffix::I32 => "i32",
                    IntSuffix::I64 => "i64",
                    IntSuffix::U32 => "u32",
                    IntSuffix::U64 => "u64",
                });
            }
        }
        ExprKind::FloatLit(v) => {
            // Keep the %g format; this loses the original representation
            // but is canonical. "3.0" stays "3.0".
            let s = format!("{v}");
            ctx.out.push_str(&s);
            if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                ctx.out.push_str(".0");
            }
        }
        ExprKind::StringLit(s) => {
            ctx.out.push('"');
            escape_string(s, &mut ctx.out);
            ctx.out.push('"');
        }
        ExprKind::CharLit(cp) => {
            ctx.out.push('\'');
            match char::from_u32(*cp) {
                Some('\\') => ctx.out.push_str("\\\\"),
                Some('\'') => ctx.out.push_str("\\'"),
                Some('\n') => ctx.out.push_str("\\n"),
                Some('\r') => ctx.out.push_str("\\r"),
                Some('\t') => ctx.out.push_str("\\t"),
                Some(c) if c.is_ascii() && !c.is_ascii_control() => ctx.out.push(c),
                Some(c) => ctx.out.push_str(&format!("\\u{{{:x}}}", c as u32)),
                None => ctx.out.push_str(&format!("\\u{{{:x}}}", cp)),
            }
            ctx.out.push('\'');
        }
        ExprKind::BoolLit(b) => ctx.out.push_str(if *b { "true" } else { "false" }),
        ExprKind::UnitLit => ctx.out.push_str("unit"),
        ExprKind::Ident(name) => ctx.out.push_str(name),
        ExprKind::Paren(inner) => {
            ctx.out.push('(');
            emit_expr(inner, ctx);
            ctx.out.push(')');
        }
        ExprKind::Block(b) => emit_block(b, ctx),
        ExprKind::Binary { op, lhs, rhs } => {
            emit_expr(lhs, ctx);
            ctx.out.push(' ');
            ctx.out.push_str(binop_str(*op));
            ctx.out.push(' ');
            emit_expr(rhs, ctx);
        }
        ExprKind::Unary { op, rhs } => {
            ctx.out.push_str(match op {
                UnaryOp::Neg => "-",
                UnaryOp::Not => "!",
            });
            emit_expr(rhs, ctx);
        }
        ExprKind::Call { callee, args } => {
            emit_expr(callee, ctx);
            emit_args(args, ctx);
        }
        ExprKind::MethodCall { receiver, method, args } => {
            emit_expr(receiver, ctx);
            ctx.out.push('.');
            ctx.out.push_str(method);
            emit_args(args, ctx);
        }
        ExprKind::Field { receiver, name } => {
            emit_expr(receiver, ctx);
            ctx.out.push('.');
            ctx.out.push_str(name);
        }
        ExprKind::TupleField { receiver, index } => {
            emit_expr(receiver, ctx);
            ctx.out.push('.');
            ctx.out.push_str(&index.to_string());
        }
        ExprKind::Try(inner) => {
            emit_expr(inner, ctx);
            ctx.out.push('?');
        }
        ExprKind::TupleLit(elems) => {
            ctx.out.push('(');
            for (i, el) in elems.iter().enumerate() {
                if i > 0 { ctx.out.push_str(", "); }
                emit_expr(el, ctx);
            }
            if elems.len() == 1 {
                ctx.out.push(',');
            }
            ctx.out.push(')');
        }
        ExprKind::StructLit { name, fields, spread, .. } => {
            ctx.out.push_str(name);
            ctx.out.push_str(" {");
            for (i, fi) in fields.iter().enumerate() {
                if i > 0 { ctx.out.push(','); }
                ctx.out.push(' ');
                ctx.out.push_str(&fi.name);
                ctx.out.push_str(": ");
                emit_expr(&fi.value, ctx);
            }
            if let Some(s) = spread {
                if !fields.is_empty() { ctx.out.push(','); }
                ctx.out.push_str(" ..");
                emit_expr(s, ctx);
            }
            ctx.out.push_str(" }");
        }
        ExprKind::Cast { expr, to } => {
            emit_expr(expr, ctx);
            ctx.out.push_str(" as ");
            emit_type(to, ctx);
        }
        ExprKind::If { cond, then_block, else_block } => {
            ctx.out.push_str("if ");
            emit_expr(cond, ctx);
            ctx.out.push(' ');
            emit_block(then_block, ctx);
            // Only emit else if it's non-empty.
            if !else_block.stmts.is_empty() || else_block.tail.is_some() {
                ctx.out.push_str(" else ");
                emit_block(else_block, ctx);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            ctx.out.push_str("match ");
            emit_expr(scrutinee, ctx);
            ctx.out.push_str(" {");
            ctx.newline();
            ctx.indent += 1;
            for arm in arms {
                ctx.write_indent();
                emit_pattern(&arm.pattern, ctx);
                ctx.out.push_str(" => ");
                emit_expr(&arm.body, ctx);
                ctx.out.push(',');
                ctx.newline();
            }
            ctx.indent -= 1;
            ctx.write_indent();
            ctx.out.push('}');
        }
        ExprKind::Lambda { params, return_type, body } => {
            ctx.out.push_str("fn");
            emit_params(params, ctx);
            ctx.out.push_str(": ");
            emit_type(return_type, ctx);
            ctx.out.push(' ');
            emit_block(body, ctx);
        }
        ExprKind::Spawn { actor_name, fields } => {
            ctx.out.push_str("spawn ");
            ctx.out.push_str(actor_name);
            ctx.out.push_str(" {");
            for (i, fi) in fields.iter().enumerate() {
                if i > 0 { ctx.out.push(','); }
                ctx.out.push(' ');
                ctx.out.push_str(&fi.name);
                ctx.out.push_str(": ");
                emit_expr(&fi.value, ctx);
            }
            ctx.out.push_str(" }");
        }
        ExprKind::Send { handle, method, args } => {
            ctx.out.push_str("send ");
            emit_expr(handle, ctx);
            ctx.out.push('.');
            ctx.out.push_str(method);
            emit_args(args, ctx);
        }
        ExprKind::Ask { handle, method, args } => {
            ctx.out.push_str("ask ");
            emit_expr(handle, ctx);
            ctx.out.push('.');
            ctx.out.push_str(method);
            emit_args(args, ctx);
        }
        ExprKind::Interpolated(parts) => {
            ctx.out.push('"');
            for part in parts {
                match part {
                    InterpPiece::Lit(s) => escape_string(s, &mut ctx.out),
                    InterpPiece::Expr(e) => {
                        ctx.out.push_str("${");
                        emit_expr(e, ctx);
                        ctx.out.push('}');
                    }
                }
            }
            ctx.out.push('"');
        }
        ExprKind::Arena(body) => {
            ctx.out.push_str("arena ");
            emit_block(body, ctx);
        }
    }
}

fn emit_args(args: &[Arg], ctx: &mut FmtCtx) {
    ctx.out.push('(');
    for (i, a) in args.iter().enumerate() {
        if i > 0 { ctx.out.push_str(", "); }
        if let Some(name) = &a.name {
            ctx.out.push_str(name);
            ctx.out.push_str(": ");
        }
        emit_expr(&a.value, ctx);
    }
    ctx.out.push(')');
}

fn emit_pattern(p: &Pattern, ctx: &mut FmtCtx) {
    match &p.kind {
        PatternKind::Wildcard => ctx.out.push('_'),
        PatternKind::Binding(n) => ctx.out.push_str(n),
        PatternKind::Literal(l) => emit_literal_pattern(l, ctx),
        PatternKind::Variant { name, payload } => {
            ctx.out.push_str(name);
            if let Some(payload) = payload {
                match payload {
                    VariantPatPayload::Named(fields) => {
                        ctx.out.push_str(" {");
                        for (i, f) in fields.iter().enumerate() {
                            if i > 0 { ctx.out.push(','); }
                            ctx.out.push(' ');
                            ctx.out.push_str(&f.name);
                            ctx.out.push_str(": ");
                            emit_pattern(&f.pattern, ctx);
                        }
                        ctx.out.push_str(" }");
                    }
                    VariantPatPayload::Positional(pats) => {
                        ctx.out.push('(');
                        for (i, pat) in pats.iter().enumerate() {
                            if i > 0 { ctx.out.push_str(", "); }
                            emit_pattern(pat, ctx);
                        }
                        ctx.out.push(')');
                    }
                }
            }
        }
    }
}

fn emit_literal_pattern(l: &LiteralPattern, ctx: &mut FmtCtx) {
    use crate::lexer::IntSuffix;
    match l {
        LiteralPattern::Int(v, suffix) => {
            ctx.out.push_str(&v.to_string());
            if let Some(s) = suffix {
                ctx.out.push_str(match s {
                    IntSuffix::I32 => "i32",
                    IntSuffix::I64 => "i64",
                    IntSuffix::U32 => "u32",
                    IntSuffix::U64 => "u64",
                });
            }
        }
        LiteralPattern::Bool(b) => ctx.out.push_str(if *b { "true" } else { "false" }),
        LiteralPattern::String(s) => {
            ctx.out.push('"');
            escape_string(s, &mut ctx.out);
            ctx.out.push('"');
        }
        LiteralPattern::Unit => ctx.out.push_str("unit"),
    }
}

fn binop_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*",
        BinOp::Div => "/", BinOp::Rem => "%",
        BinOp::Eq => "==", BinOp::NotEq => "!=",
        BinOp::Lt => "<", BinOp::Gt => ">",
        BinOp::LtEq => "<=", BinOp::GtEq => ">=",
        BinOp::And => "&&", BinOp::Or => "||",
    }
}

fn escape_string(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
}

// --- Comment extraction -------------------------------------------------

/// Walk `src` once, collecting line and block comment ranges. Mirrors
/// the skip_trivia logic in the lexer but records rather than
/// discards.
fn collect_comments(src: &str) -> Vec<CommentRange> {
    let bytes = src.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        // Skip whitespace.
        if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
            i += 1;
            continue;
        }
        // Skip string literals so we don't mistake their contents.
        if b == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() { i += 1; }
                i += 1;
            }
            if i < bytes.len() { i += 1; }
            continue;
        }
        if b == b'\'' {
            // Char literal — walk until the closing '.
            i += 1;
            while i < bytes.len() && bytes[i] != b'\'' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() { i += 1; }
                i += 1;
            }
            if i < bytes.len() { i += 1; }
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            let start = i as u32;
            while i < bytes.len() && bytes[i] != b'\n' { i += 1; }
            let end = i as u32;
            out.push(CommentRange {
                start, end,
                text: src[start as usize..end as usize].to_string(),
                kind: CommentKind::Line,
            });
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            let start = i as u32;
            i += 2;
            let mut depth: u32 = 1;
            while i + 1 < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1; i += 2;
                } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1; i += 2;
                } else {
                    i += 1;
                }
            }
            let end = i as u32;
            out.push(CommentRange {
                start, end,
                text: src[start as usize..end as usize].to_string(),
                kind: CommentKind::Block,
            });
            continue;
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(s: &str) -> String {
        format(s).unwrap()
    }

    #[test]
    fn idempotent_on_canonical_hello() {
        let src = "import std/io\n\nfn main(): i32 io {\n    io.println(\"hi\")\n    return 0\n}\n";
        assert_eq!(fmt(src), src);
    }

    #[test]
    fn normalizes_indent_and_spacing() {
        // Trivial single-statement bodies inline to `{ ... }`.
        let in_  = "fn  main(  ):i32{return  42}";
        let want = "fn main(): i32 { return 42 }\n";
        assert_eq!(fmt(in_), want);
    }

    #[test]
    fn multi_stmt_bodies_go_multiline() {
        let in_ = "fn f(): i32 { let x = 1  return x }";
        let want = "fn f(): i32 {\n    let x = 1\n    return x\n}\n";
        assert_eq!(fmt(in_), want);
    }

    #[test]
    fn preserves_top_level_line_comments() {
        let src = "// top comment\nimport std/io\n";
        let out = fmt(src);
        assert!(out.contains("// top comment"), "got:\n{out}");
    }

    #[test]
    fn sorts_imports() {
        let in_ = "import std/list\nimport std/io\n";
        let out = fmt(in_);
        let io_pos = out.find("std/io").unwrap();
        let list_pos = out.find("std/list").unwrap();
        assert!(io_pos < list_pos, "imports not sorted:\n{out}");
    }
}
