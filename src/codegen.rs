//! AST → Wasm codegen. See `lumen-awl`.
//!
//! Targets Wasmtime + WASI. Memory model: a single linear memory with a
//! bump allocator (no free). Strings are `(ptr: i32, len: i32)` pairs held
//! as two i32s on the Wasm stack. Structs and sum-type payloads live in
//! linear memory, pointed to by an i32.
//!
//! ## Scope of this module (phase 1)
//!
//! This file implements the core, memory-free slice of codegen:
//!
//! - All five numeric types (`i32`, `i64`, `u32`, `u64`, `f64`), with the
//!   arithmetic/comparison/logical operators routed to the correct
//!   signed/unsigned Wasm opcode.
//! - `bool` and `unit` (both lowered to `i32`).
//! - User-defined monomorphic functions, calls between them, and
//!   parameters + local variables from `let`/`var` with `var` re-assignment.
//! - `if`/`else` expressions and blocks.
//! - `return` statements.
//! - `as` casts between any two numeric types.
//! - `main` and all other top-level functions are exported.
//!
//! Deferred to follow-up codegen issues and not yet handled here:
//!
//! - Strings, structs, sum types, `match`, `Option`/`Result`, `?`, linear
//!   memory, and `for` loops.

use std::collections::HashMap;

use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
    TypeSection, ValType,
};

use crate::ast::{self, BinOp, Expr, ExprKind, FnDecl, Item, StmtKind, UnaryOp};
use crate::lexer::IntSuffix;
use crate::span::Span;
use crate::types::{ModuleInfo, Ty};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CodegenError {
    pub span: Span,
    pub message: String,
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.span.line, self.span.col, self.message)
    }
}

impl std::error::Error for CodegenError {}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Compile a type-checked module into a Wasm binary. The [`ModuleInfo`]
/// argument must come from [`crate::types::typecheck`] on the same module
/// — codegen trusts the types it sees and will panic if given a mismatched
/// pair.
pub fn compile(module: &ast::Module, info: &ModuleInfo) -> Result<Vec<u8>, CodegenError> {
    let mut cg = Codegen::new(info);
    cg.compile_module(module)?;
    Ok(cg.finish())
}

// ---------------------------------------------------------------------------
// Codegen state
// ---------------------------------------------------------------------------

struct Codegen<'a> {
    info: &'a ModuleInfo,
    /// Fn name → Wasm function index (== type index; we don't dedupe types
    /// in this pass).
    fn_indices: HashMap<String, u32>,
    types: TypeSection,
    functions: FunctionSection,
    exports: ExportSection,
    code: CodeSection,
    next_index: u32,
}

impl<'a> Codegen<'a> {
    fn new(info: &'a ModuleInfo) -> Self {
        Self {
            info,
            fn_indices: HashMap::new(),
            types: TypeSection::new(),
            functions: FunctionSection::new(),
            exports: ExportSection::new(),
            code: CodeSection::new(),
            next_index: 0,
        }
    }

    fn compile_module(&mut self, module: &ast::Module) -> Result<(), CodegenError> {
        // Pass 1: assign a function + type index to every `fn` item and add
        // its signature to the type section. This needs to finish before
        // body codegen so that forward calls resolve.
        for item in &module.items {
            if let Item::Fn(f) = item {
                let sig = self
                    .info
                    .fns
                    .get(&f.name)
                    .expect("type checker should have populated this fn");
                let idx = self.next_index;
                self.next_index += 1;
                self.fn_indices.insert(f.name.clone(), idx);

                let params_val: Vec<ValType> = sig
                    .params
                    .iter()
                    .map(|(_, t)| wasm_val_type(t, f.span))
                    .collect::<Result<_, _>>()?;
                let returns_val: Vec<ValType> = match &sig.ret {
                    Ty::Unit => vec![ValType::I32], // unit lowers to a zero i32
                    ret => vec![wasm_val_type(ret, f.span)?],
                };

                self.types.ty().function(params_val, returns_val);
                self.functions.function(idx);
                self.exports.export(f.name.as_str(), ExportKind::Func, idx);
            }
        }

        // Pass 2: emit each function body.
        for item in &module.items {
            if let Item::Fn(f) = item {
                let func = self.compile_fn(f)?;
                self.code.function(&func);
            }
        }

        Ok(())
    }

    fn finish(self) -> Vec<u8> {
        let mut module = Module::new();
        module.section(&self.types);
        module.section(&self.functions);
        module.section(&self.exports);
        module.section(&self.code);
        module.finish()
    }

    // --- Function body ---------------------------------------------------

    fn compile_fn(&self, f: &FnDecl) -> Result<Function, CodegenError> {
        let sig = self
            .info
            .fns
            .get(&f.name)
            .expect("type checker populated sig");

        // First, walk the body to collect every local (params + let/var
        // bindings) so the Wasm locals header is accurate.
        let mut fb = FnBuilder::new(self, sig);
        // Parameters are locals 0..N already — register them.
        for (pname, pty) in &sig.params {
            fb.register_param(pname, pty.clone());
        }
        fb.collect_locals(&f.body)?;

        let mut function = Function::new_with_locals_types(fb.extra_locals_types.clone());
        fb.compile_block(&f.body, &mut function)?;

        // Implicit `unit` return: if the declared return is unit, push a
        // zero i32 so the function signature is satisfied.
        if matches!(sig.ret, Ty::Unit) {
            function.instruction(&Instruction::I32Const(0));
        }

        function.instruction(&Instruction::End);
        Ok(function)
    }
}

// ---------------------------------------------------------------------------
// Per-function compile state
// ---------------------------------------------------------------------------

struct FnBuilder<'a, 'b> {
    cg: &'a Codegen<'b>,
    sig: &'a crate::types::FnSig,
    /// Name → local index.
    locals: HashMap<String, u32>,
    /// Parallel vec of (name, ty) for every local in declaration order.
    /// Helps when emitting the locals header and for type-directed dispatch.
    local_types: Vec<Ty>,
    /// Types of locals that come *after* the parameters — this is exactly
    /// what `Function::new_with_locals_types` wants.
    extra_locals_types: Vec<ValType>,
    num_params: u32,
}

impl<'a, 'b> FnBuilder<'a, 'b> {
    fn new(cg: &'a Codegen<'b>, sig: &'a crate::types::FnSig) -> Self {
        Self {
            cg,
            sig,
            locals: HashMap::new(),
            local_types: Vec::new(),
            extra_locals_types: Vec::new(),
            num_params: 0,
        }
    }

    fn register_param(&mut self, name: &str, ty: Ty) {
        let idx = self.local_types.len() as u32;
        self.locals.insert(name.to_string(), idx);
        self.local_types.push(ty);
        self.num_params += 1;
    }

    /// Walk the block and register every `let`/`var` binding that appears
    /// inside it (including nested blocks, `for` bodies, and match/if arms).
    fn collect_locals(&mut self, block: &ast::Block) -> Result<(), CodegenError> {
        for stmt in &block.stmts {
            match &stmt.kind {
                StmtKind::Let { name, value, .. } | StmtKind::Var { name, value, .. } => {
                    let ty = self.infer_value_ty(value)?;
                    self.register_local(name, ty, value.span)?;
                    self.walk_expr(value)?;
                }
                StmtKind::Assign { value, .. } => {
                    self.walk_expr(value)?;
                }
                StmtKind::Expr(e) => self.walk_expr(e)?,
                StmtKind::For { .. } => {
                    // For-loops are not emitted by phase-1 codegen; the
                    // typechecker still accepts them because range(...) is
                    // a built-in stub. Fail cleanly rather than silently
                    // producing wrong Wasm.
                    return Err(CodegenError {
                        span: stmt.span,
                        message: "for-loops are not supported by phase-1 codegen yet".into(),
                    });
                }
                StmtKind::Return(Some(e)) => self.walk_expr(e)?,
                StmtKind::Return(None) => {}
            }
        }
        if let Some(tail) = &block.tail {
            self.walk_expr(tail)?;
        }
        Ok(())
    }

    fn walk_expr(&mut self, expr: &Expr) -> Result<(), CodegenError> {
        match &expr.kind {
            ExprKind::If {
                cond,
                then_block,
                else_block,
            } => {
                self.walk_expr(cond)?;
                self.collect_locals(then_block)?;
                self.collect_locals(else_block)?;
            }
            ExprKind::Block(b) => self.collect_locals(b)?,
            ExprKind::Paren(e) => self.walk_expr(e)?,
            ExprKind::Unary { rhs, .. } => self.walk_expr(rhs)?,
            ExprKind::Binary { lhs, rhs, .. } => {
                self.walk_expr(lhs)?;
                self.walk_expr(rhs)?;
            }
            ExprKind::Cast { expr, .. } => self.walk_expr(expr)?,
            ExprKind::Call { callee, args } => {
                self.walk_expr(callee)?;
                for a in args {
                    self.walk_expr(&a.value)?;
                }
            }
            ExprKind::Field { receiver, .. } => self.walk_expr(receiver)?,
            ExprKind::MethodCall { receiver, args, .. } => {
                self.walk_expr(receiver)?;
                for a in args {
                    self.walk_expr(&a.value)?;
                }
            }
            ExprKind::Try(inner) => self.walk_expr(inner)?,
            ExprKind::Match { scrutinee, arms } => {
                self.walk_expr(scrutinee)?;
                for arm in arms {
                    self.walk_expr(&arm.body)?;
                }
            }
            ExprKind::StructLit { fields, .. } => {
                for fi in fields {
                    self.walk_expr(&fi.value)?;
                }
            }
            // Leaves
            ExprKind::IntLit { .. }
            | ExprKind::FloatLit(_)
            | ExprKind::StringLit(_)
            | ExprKind::BoolLit(_)
            | ExprKind::UnitLit
            | ExprKind::Ident(_) => {}
        }
        Ok(())
    }

    fn register_local(&mut self, name: &str, ty: Ty, span: Span) -> Result<(), CodegenError> {
        let idx = self.local_types.len() as u32;
        let val = wasm_val_type(&ty, span)?;
        self.locals.insert(name.to_string(), idx);
        self.local_types.push(ty);
        self.extra_locals_types.push(val);
        Ok(())
    }

    /// Approximate the type of a value expression at collect-locals time so
    /// we can reserve a Wasm local slot of the right width. Because the
    /// type checker has already validated everything, this only needs to
    /// get the *shape* right (i32 vs i64 vs f64).
    fn infer_value_ty(&self, expr: &Expr) -> Result<Ty, CodegenError> {
        Ok(match &expr.kind {
            ExprKind::IntLit { suffix, .. } => match suffix {
                Some(IntSuffix::I32) | None => Ty::I32,
                Some(IntSuffix::I64) => Ty::I64,
                Some(IntSuffix::U32) => Ty::U32,
                Some(IntSuffix::U64) => Ty::U64,
            },
            ExprKind::FloatLit(_) => Ty::F64,
            ExprKind::BoolLit(_) => Ty::Bool,
            ExprKind::UnitLit => Ty::Unit,
            ExprKind::StringLit(_) => {
                return Err(CodegenError {
                    span: expr.span,
                    message: "string literals are not supported by phase-1 codegen yet".into(),
                });
            }
            ExprKind::Ident(name) => {
                if let Some(&idx) = self.locals.get(name) {
                    self.local_types[idx as usize].clone()
                } else {
                    return Err(CodegenError {
                        span: expr.span,
                        message: format!("unknown identifier `{name}`"),
                    });
                }
            }
            ExprKind::Paren(inner) => self.infer_value_ty(inner)?,
            ExprKind::Unary { op, rhs } => match op {
                UnaryOp::Neg => self.infer_value_ty(rhs)?,
                UnaryOp::Not => Ty::Bool,
            },
            ExprKind::Binary { op, lhs, .. } => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                    self.infer_value_ty(lhs)?
                }
                BinOp::Eq
                | BinOp::NotEq
                | BinOp::Lt
                | BinOp::Gt
                | BinOp::LtEq
                | BinOp::GtEq
                | BinOp::And
                | BinOp::Or => Ty::Bool,
            },
            ExprKind::Cast { to, .. } => resolve_builtin_numeric(to)?,
            ExprKind::Call { callee, .. } => {
                if let ExprKind::Ident(name) = &callee.kind {
                    if let Some(sig) = self.cg.info.fns.get(name) {
                        sig.ret.clone()
                    } else {
                        return Err(CodegenError {
                            span: expr.span,
                            message: format!("unknown function `{name}`"),
                        });
                    }
                } else {
                    return Err(CodegenError {
                        span: expr.span,
                        message: "only direct function calls are supported".into(),
                    });
                }
            }
            ExprKind::If { then_block, .. } => {
                if let Some(tail) = &then_block.tail {
                    self.infer_value_ty(tail)?
                } else {
                    Ty::Unit
                }
            }
            _ => {
                return Err(CodegenError {
                    span: expr.span,
                    message: "expression kind not supported by phase-1 codegen".into(),
                });
            }
        })
    }

    // ----------------------------------------------------------------------
    // Code emission
    // ----------------------------------------------------------------------

    fn compile_block(&self, block: &ast::Block, f: &mut Function) -> Result<(), CodegenError> {
        for stmt in &block.stmts {
            self.compile_stmt(stmt, f)?;
        }
        if let Some(tail) = &block.tail {
            self.compile_expr(tail, f)?;
        }
        Ok(())
    }

    fn compile_stmt(&self, stmt: &ast::Stmt, f: &mut Function) -> Result<(), CodegenError> {
        match &stmt.kind {
            StmtKind::Let { name, value, .. } | StmtKind::Var { name, value, .. } => {
                self.compile_expr(value, f)?;
                let idx = *self
                    .locals
                    .get(name)
                    .expect("local registered in pass-0");
                f.instruction(&Instruction::LocalSet(idx));
            }
            StmtKind::Assign { name, value } => {
                self.compile_expr(value, f)?;
                let idx = *self.locals.get(name).ok_or_else(|| CodegenError {
                    span: stmt.span,
                    message: format!("assignment to unknown local `{name}`"),
                })?;
                f.instruction(&Instruction::LocalSet(idx));
            }
            StmtKind::Expr(e) => {
                self.compile_expr(e, f)?;
                // Statement-position expressions still push their value; in
                // phase 1 we treat the block like `(seq stmts*, tail)` and
                // just let those sit on the stack. That's incorrect for
                // non-tail non-unit statements — enforce by dropping.
                let ty = self.infer_value_ty(e)?;
                if !matches!(ty, Ty::Unit) {
                    f.instruction(&Instruction::Drop);
                }
            }
            StmtKind::For { .. } => {
                // collect_locals already rejected this; be explicit.
                return Err(CodegenError {
                    span: stmt.span,
                    message: "for-loops are not supported by phase-1 codegen yet".into(),
                });
            }
            StmtKind::Return(Some(e)) => {
                self.compile_expr(e, f)?;
                f.instruction(&Instruction::Return);
            }
            StmtKind::Return(None) => {
                if matches!(self.sig.ret, Ty::Unit) {
                    f.instruction(&Instruction::I32Const(0));
                }
                f.instruction(&Instruction::Return);
            }
        }
        Ok(())
    }

    fn compile_expr(&self, expr: &Expr, f: &mut Function) -> Result<(), CodegenError> {
        match &expr.kind {
            ExprKind::IntLit { value, suffix } => {
                let ty = match suffix {
                    Some(IntSuffix::I64) => Ty::I64,
                    Some(IntSuffix::U64) => Ty::U64,
                    Some(IntSuffix::U32) => Ty::U32,
                    Some(IntSuffix::I32) | None => Ty::I32,
                };
                match ty {
                    Ty::I64 | Ty::U64 => {
                        f.instruction(&Instruction::I64Const(*value as i64));
                    }
                    _ => {
                        f.instruction(&Instruction::I32Const(*value as i32));
                    }
                }
            }
            ExprKind::FloatLit(v) => {
                f.instruction(&Instruction::F64Const((*v).into()));
            }
            ExprKind::BoolLit(b) => {
                f.instruction(&Instruction::I32Const(if *b { 1 } else { 0 }));
            }
            ExprKind::UnitLit => {
                f.instruction(&Instruction::I32Const(0));
            }
            ExprKind::Ident(name) => {
                let idx = *self.locals.get(name).ok_or_else(|| CodegenError {
                    span: expr.span,
                    message: format!("unknown identifier `{name}`"),
                })?;
                f.instruction(&Instruction::LocalGet(idx));
            }
            ExprKind::Paren(inner) => self.compile_expr(inner, f)?,
            ExprKind::Block(block) => self.compile_block(block, f)?,
            ExprKind::Unary { op, rhs } => {
                let rhs_ty = self.infer_value_ty(rhs)?;
                match op {
                    UnaryOp::Neg => match rhs_ty {
                        Ty::I32 | Ty::U32 => {
                            f.instruction(&Instruction::I32Const(0));
                            self.compile_expr(rhs, f)?;
                            f.instruction(&Instruction::I32Sub);
                        }
                        Ty::I64 | Ty::U64 => {
                            f.instruction(&Instruction::I64Const(0));
                            self.compile_expr(rhs, f)?;
                            f.instruction(&Instruction::I64Sub);
                        }
                        Ty::F64 => {
                            self.compile_expr(rhs, f)?;
                            f.instruction(&Instruction::F64Neg);
                        }
                        _ => {
                            return Err(CodegenError {
                                span: expr.span,
                                message: format!("unary `-` on non-numeric {}", rhs_ty.display()),
                            });
                        }
                    },
                    UnaryOp::Not => {
                        // !b := b == 0
                        self.compile_expr(rhs, f)?;
                        f.instruction(&Instruction::I32Eqz);
                    }
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                self.compile_binary(*op, lhs, rhs, f)?;
            }
            ExprKind::Cast { expr: inner, to } => {
                let from = self.infer_value_ty(inner)?;
                let to_ty = resolve_builtin_numeric(to)?;
                self.compile_expr(inner, f)?;
                emit_cast(&from, &to_ty, f);
            }
            ExprKind::Call { callee, args } => {
                let name = match &callee.kind {
                    ExprKind::Ident(n) => n.clone(),
                    _ => {
                        return Err(CodegenError {
                            span: callee.span,
                            message: "only direct function calls are supported".into(),
                        });
                    }
                };
                for a in args {
                    self.compile_expr(&a.value, f)?;
                }
                let idx = *self
                    .cg
                    .fn_indices
                    .get(&name)
                    .ok_or_else(|| CodegenError {
                        span: callee.span,
                        message: format!("unknown function `{name}`"),
                    })?;
                f.instruction(&Instruction::Call(idx));
            }
            ExprKind::If {
                cond,
                then_block,
                else_block,
            } => {
                let result_ty = self.if_result_ty(then_block)?;
                let block_ty = match result_ty {
                    Ty::Unit => wasm_encoder::BlockType::Result(ValType::I32),
                    ref t => wasm_encoder::BlockType::Result(wasm_val_type(t, expr.span)?),
                };
                self.compile_expr(cond, f)?;
                f.instruction(&Instruction::If(block_ty));
                self.compile_block(then_block, f)?;
                if matches!(result_ty, Ty::Unit) && then_block.tail.is_none() {
                    f.instruction(&Instruction::I32Const(0));
                }
                f.instruction(&Instruction::Else);
                self.compile_block(else_block, f)?;
                if matches!(result_ty, Ty::Unit) && else_block.tail.is_none() {
                    f.instruction(&Instruction::I32Const(0));
                }
                f.instruction(&Instruction::End);
            }
            _ => {
                return Err(CodegenError {
                    span: expr.span,
                    message: "expression kind not supported by phase-1 codegen".into(),
                });
            }
        }
        Ok(())
    }

    fn if_result_ty(&self, then_block: &ast::Block) -> Result<Ty, CodegenError> {
        match &then_block.tail {
            Some(e) => self.infer_value_ty(e),
            None => Ok(Ty::Unit),
        }
    }

    fn compile_binary(
        &self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        // Short-circuit logical operators.
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.compile_logical(op, lhs, rhs, f);
        }

        let lt = self.infer_value_ty(lhs)?;
        self.compile_expr(lhs, f)?;
        self.compile_expr(rhs, f)?;

        let instr = match (&lt, op) {
            (Ty::I32, BinOp::Add) | (Ty::U32, BinOp::Add) => Instruction::I32Add,
            (Ty::I32, BinOp::Sub) | (Ty::U32, BinOp::Sub) => Instruction::I32Sub,
            (Ty::I32, BinOp::Mul) | (Ty::U32, BinOp::Mul) => Instruction::I32Mul,
            (Ty::I32, BinOp::Div) => Instruction::I32DivS,
            (Ty::U32, BinOp::Div) => Instruction::I32DivU,
            (Ty::I32, BinOp::Rem) => Instruction::I32RemS,
            (Ty::U32, BinOp::Rem) => Instruction::I32RemU,
            (Ty::I32, BinOp::Eq) | (Ty::U32, BinOp::Eq) | (Ty::Bool, BinOp::Eq) => {
                Instruction::I32Eq
            }
            (Ty::I32, BinOp::NotEq) | (Ty::U32, BinOp::NotEq) | (Ty::Bool, BinOp::NotEq) => {
                Instruction::I32Ne
            }
            (Ty::I32, BinOp::Lt) => Instruction::I32LtS,
            (Ty::U32, BinOp::Lt) => Instruction::I32LtU,
            (Ty::I32, BinOp::LtEq) => Instruction::I32LeS,
            (Ty::U32, BinOp::LtEq) => Instruction::I32LeU,
            (Ty::I32, BinOp::Gt) => Instruction::I32GtS,
            (Ty::U32, BinOp::Gt) => Instruction::I32GtU,
            (Ty::I32, BinOp::GtEq) => Instruction::I32GeS,
            (Ty::U32, BinOp::GtEq) => Instruction::I32GeU,

            (Ty::I64, BinOp::Add) | (Ty::U64, BinOp::Add) => Instruction::I64Add,
            (Ty::I64, BinOp::Sub) | (Ty::U64, BinOp::Sub) => Instruction::I64Sub,
            (Ty::I64, BinOp::Mul) | (Ty::U64, BinOp::Mul) => Instruction::I64Mul,
            (Ty::I64, BinOp::Div) => Instruction::I64DivS,
            (Ty::U64, BinOp::Div) => Instruction::I64DivU,
            (Ty::I64, BinOp::Rem) => Instruction::I64RemS,
            (Ty::U64, BinOp::Rem) => Instruction::I64RemU,
            (Ty::I64, BinOp::Eq) | (Ty::U64, BinOp::Eq) => Instruction::I64Eq,
            (Ty::I64, BinOp::NotEq) | (Ty::U64, BinOp::NotEq) => Instruction::I64Ne,
            (Ty::I64, BinOp::Lt) => Instruction::I64LtS,
            (Ty::U64, BinOp::Lt) => Instruction::I64LtU,
            (Ty::I64, BinOp::LtEq) => Instruction::I64LeS,
            (Ty::U64, BinOp::LtEq) => Instruction::I64LeU,
            (Ty::I64, BinOp::Gt) => Instruction::I64GtS,
            (Ty::U64, BinOp::Gt) => Instruction::I64GtU,
            (Ty::I64, BinOp::GtEq) => Instruction::I64GeS,
            (Ty::U64, BinOp::GtEq) => Instruction::I64GeU,

            (Ty::F64, BinOp::Add) => Instruction::F64Add,
            (Ty::F64, BinOp::Sub) => Instruction::F64Sub,
            (Ty::F64, BinOp::Mul) => Instruction::F64Mul,
            (Ty::F64, BinOp::Div) => Instruction::F64Div,
            (Ty::F64, BinOp::Eq) => Instruction::F64Eq,
            (Ty::F64, BinOp::NotEq) => Instruction::F64Ne,
            (Ty::F64, BinOp::Lt) => Instruction::F64Lt,
            (Ty::F64, BinOp::LtEq) => Instruction::F64Le,
            (Ty::F64, BinOp::Gt) => Instruction::F64Gt,
            (Ty::F64, BinOp::GtEq) => Instruction::F64Ge,

            (ty, op) => {
                return Err(CodegenError {
                    span: lhs.span,
                    message: format!(
                        "phase-1 codegen has no lowering for {:?} on {}",
                        op,
                        ty.display()
                    ),
                });
            }
        };
        f.instruction(&instr);
        Ok(())
    }

    fn compile_logical(
        &self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        // Short-circuit `&&` and `||` via `if` blocks that produce an i32.
        self.compile_expr(lhs, f)?;
        f.instruction(&Instruction::If(wasm_encoder::BlockType::Result(ValType::I32)));
        match op {
            BinOp::And => {
                self.compile_expr(rhs, f)?;
            }
            BinOp::Or => {
                f.instruction(&Instruction::I32Const(1));
            }
            _ => unreachable!(),
        }
        f.instruction(&Instruction::Else);
        match op {
            BinOp::And => {
                f.instruction(&Instruction::I32Const(0));
            }
            BinOp::Or => {
                self.compile_expr(rhs, f)?;
            }
            _ => unreachable!(),
        }
        f.instruction(&Instruction::End);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn wasm_val_type(ty: &Ty, span: Span) -> Result<ValType, CodegenError> {
    Ok(match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::Unit => ValType::I32,
        Ty::I64 | Ty::U64 => ValType::I64,
        Ty::F64 => ValType::F64,
        other => {
            return Err(CodegenError {
                span,
                message: format!(
                    "phase-1 codegen only supports numeric / bool / unit; got {}",
                    other.display()
                ),
            });
        }
    })
}

fn resolve_builtin_numeric(ty: &ast::Type) -> Result<Ty, CodegenError> {
    match &ty.kind {
        ast::TypeKind::Named { name, args } if args.is_empty() => match name.as_str() {
            "i32" => Ok(Ty::I32),
            "i64" => Ok(Ty::I64),
            "u32" => Ok(Ty::U32),
            "u64" => Ok(Ty::U64),
            "f64" => Ok(Ty::F64),
            _ => Err(CodegenError {
                span: ty.span,
                message: format!("`as` target must be numeric, got `{name}`"),
            }),
        },
        _ => Err(CodegenError {
            span: ty.span,
            message: "`as` target must be a bare numeric type".into(),
        }),
    }
}

fn emit_cast(from: &Ty, to: &Ty, f: &mut Function) {
    use Ty::*;
    match (from, to) {
        // Identity / width-preserving: u32<->i32 and u64<->i64 are the same
        // Wasm type, so no instruction needed.
        (I32, I32) | (U32, U32) | (I32, U32) | (U32, I32) => {}
        (I64, I64) | (U64, U64) | (I64, U64) | (U64, I64) => {}
        (F64, F64) => {}

        // Widening integers.
        (I32, I64) => { f.instruction(&Instruction::I64ExtendI32S); }
        (U32, I64) | (I32, U64) | (U32, U64) => {
            f.instruction(&Instruction::I64ExtendI32U);
        }

        // Narrowing.
        (I64, I32) | (I64, U32) | (U64, I32) | (U64, U32) => {
            f.instruction(&Instruction::I32WrapI64);
        }

        // Int ↔ float.
        (I32, F64) => { f.instruction(&Instruction::F64ConvertI32S); }
        (U32, F64) => { f.instruction(&Instruction::F64ConvertI32U); }
        (I64, F64) => { f.instruction(&Instruction::F64ConvertI64S); }
        (U64, F64) => { f.instruction(&Instruction::F64ConvertI64U); }
        (F64, I32) => { f.instruction(&Instruction::I32TruncF64S); }
        (F64, U32) => { f.instruction(&Instruction::I32TruncF64U); }
        (F64, I64) => { f.instruction(&Instruction::I64TruncF64S); }
        (F64, U64) => { f.instruction(&Instruction::I64TruncF64U); }

        _ => {
            // Unreachable for well-typed input.
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;
    use crate::types::typecheck;

    fn compile_src(src: &str) -> Vec<u8> {
        let toks = lex(src).unwrap();
        let module = parse(toks).unwrap();
        let info = typecheck(&module).unwrap_or_else(|errs| {
            for e in errs {
                eprintln!("{e}");
            }
            panic!("typecheck failed");
        });
        compile(&module, &info).unwrap_or_else(|e| panic!("codegen failed: {e}"))
    }

    #[test]
    fn emitted_module_validates() {
        let wasm = compile_src("fn answer(): i32 { 42 }");
        // Basic shape check: starts with the Wasm magic number.
        assert_eq!(&wasm[0..4], b"\0asm");
        // Run it through wasmparser::validate to catch structural bugs.
        wasmparser::validate(&wasm).expect("emitted module must validate");
    }

    #[test]
    fn arithmetic_and_user_call_validate() {
        let src = r#"
            fn square(n: i32): i32 { n * n }
            fn sum_of_squares_three(): i32 { square(1) + square(2) + square(3) }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
    }

    #[test]
    fn i64_and_f64_validate() {
        let src = r#"
            fn big(): i64 { 1000000i64 * 1000i64 }
            fn fp(): f64 { 1.5 + 2.5 }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
    }

    #[test]
    fn if_expression_validates() {
        let src = "fn sign(n: i32): i32 { if n < 0 { -1 } else { if n == 0 { 0 } else { 1 } } }";
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
    }

    #[test]
    fn var_and_assign_validate() {
        let src = r#"
            fn triangle(n: i32): i32 {
                var total: i32 = 0
                total = total + n
                total = total + (n - 1)
                total
            }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
    }

    #[test]
    fn cast_validates() {
        let src = r#"
            fn widen(n: i32): i64 { n as i64 }
            fn to_float(n: i32): f64 { n as f64 }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
    }

    // --- End-to-end execution via Wasmtime --------------------------------

    fn run_i32(wasm: &[u8], func: &str) -> i32 {
        use wasmtime::*;
        let engine = Engine::default();
        let module = wasmtime::Module::new(&engine, wasm).expect("module builds");
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f = instance
            .get_typed_func::<(), i32>(&mut store, func)
            .expect("typed fn");
        f.call(&mut store, ()).expect("call")
    }

    fn run_i32_i32(wasm: &[u8], func: &str, arg: i32) -> i32 {
        use wasmtime::*;
        let engine = Engine::default();
        let module = wasmtime::Module::new(&engine, wasm).expect("module builds");
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f = instance
            .get_typed_func::<i32, i32>(&mut store, func)
            .expect("typed fn");
        f.call(&mut store, arg).expect("call")
    }

    fn run_i64(wasm: &[u8], func: &str) -> i64 {
        use wasmtime::*;
        let engine = Engine::default();
        let module = wasmtime::Module::new(&engine, wasm).expect("module builds");
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).expect("instance");
        let f = instance
            .get_typed_func::<(), i64>(&mut store, func)
            .expect("typed fn");
        f.call(&mut store, ()).expect("call")
    }

    #[test]
    fn run_constant_i32() {
        let wasm = compile_src("fn answer(): i32 { 42 }");
        assert_eq!(run_i32(&wasm, "answer"), 42);
    }

    #[test]
    fn run_arithmetic_chain() {
        let wasm = compile_src("fn value(): i32 { (1 + 2) * (3 + 4) - 5 }");
        assert_eq!(run_i32(&wasm, "value"), 16);
    }

    #[test]
    fn run_user_function_calls() {
        let src = r#"
            fn square(n: i32): i32 { n * n }
            fn value(): i32 { square(3) + square(4) }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "value"), 25);
    }

    #[test]
    fn run_with_parameter() {
        let src = "fn cube(n: i32): i32 { n * n * n }";
        let wasm = compile_src(src);
        assert_eq!(run_i32_i32(&wasm, "cube", 3), 27);
        assert_eq!(run_i32_i32(&wasm, "cube", 5), 125);
    }

    #[test]
    fn run_if_expression() {
        let src = "fn abs(n: i32): i32 { if n < 0 { -n } else { n } }";
        let wasm = compile_src(src);
        assert_eq!(run_i32_i32(&wasm, "abs", -7), 7);
        assert_eq!(run_i32_i32(&wasm, "abs", 5), 5);
        assert_eq!(run_i32_i32(&wasm, "abs", 0), 0);
    }

    #[test]
    fn run_var_accumulation() {
        // A manual loop-unroll that uses `var` + assignment.
        let src = r#"
            fn four_plus_three(): i32 {
                var total: i32 = 0
                total = total + 4
                total = total + 3
                total
            }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "four_plus_three"), 7);
    }

    #[test]
    fn run_i64_arithmetic() {
        let src = "fn big(): i64 { 1000000i64 * 1000i64 }";
        let wasm = compile_src(src);
        assert_eq!(run_i64(&wasm, "big"), 1_000_000_000);
    }

    #[test]
    fn run_cast_widening() {
        let src = "fn widen(n: i32): i64 { n as i64 }";
        let wasm = compile_src(src);
        use wasmtime::*;
        let engine = Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).unwrap();
        let f = instance.get_typed_func::<i32, i64>(&mut store, "widen").unwrap();
        assert_eq!(f.call(&mut store, 42).unwrap(), 42i64);
    }

    #[test]
    fn run_recursive_function() {
        let src = r#"
            fn fact(n: i32): i32 {
                if n <= 1 { 1 } else { n * fact(n - 1) }
            }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32_i32(&wasm, "fact", 5), 120);
        assert_eq!(run_i32_i32(&wasm, "fact", 10), 3_628_800);
    }
}
