//! AST → Wasm codegen. See `lumen-awl` (phase 1) and `lumen-tne` (strings).
//!
//! Targets Wasmtime + WASI. Memory model: a single linear memory with a
//! bump allocator (no free).
//!
//! ## String representation
//!
//! A `string` value is a single `i32` that points to a block in linear
//! memory laid out as `[len: i32 | bytes...]`. String literals are
//! interned into a data section at compile time at deterministic offsets,
//! and the bump pointer starts just past the last literal. `string +
//! string` is lowered to a call into the auto-emitted `string_concat`
//! helper, which allocates a fresh block via the bump pointer and copies
//! the two payloads in with `memory.copy`. The built-in `string_len(s)`
//! function reads the length word at `offset=0`.
//!
//! ## Struct representation
//!
//! A struct value is a single `i32` pointer into linear memory. Each
//! struct type gets a deterministic layout: fields are laid out in
//! declaration order, with `i64`/`u64`/`f64` aligned to 8 bytes and
//! everything else aligned to 4 bytes. The layout is computed once per
//! module during the pre-pass and reused by struct literals (for
//! `store`s) and field access (for `load`s). Struct literals allocate
//! via the same bump pointer as strings.
//!
//! ## Scope
//!
//! This module currently covers numerics, bool, unit, control flow, user
//! function calls (phase 1, `lumen-awl`), strings + concat + `string_len`
//! (phase 2a, `lumen-tne`), and structs + field access + struct literals
//! (phase 2b, `lumen-ci2`). Still deferred: sum types + match
//! (`lumen-mmx`), Option/Result + `?` (`lumen-rvp`), for-loops
//! (`lumen-w0g`), error frames (`lumen-x4a`), WASI IO (`lumen-nwf`).

use std::collections::HashMap;

use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ExportKind, ExportSection, Function, FunctionSection,
    GlobalSection, GlobalType, Instruction, MemArg, MemorySection, MemoryType, Module,
    TypeSection, ValType,
};

use crate::ast::{self, BinOp, Expr, ExprKind, FnDecl, Item, StmtKind, UnaryOp};
use crate::lexer::IntSuffix;
use crate::span::Span;
use crate::types::{ModuleInfo, Ty, TypeInfo};

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

/// Fixed index assigned to the auto-emitted `string_concat` helper. The
/// helper is always emitted (even for programs that don't use strings) so
/// fn index assignment is stable and simple.
const STRING_CONCAT_FN_INDEX: u32 = 0;
/// Reserve the first 8 bytes of linear memory as an unused null zone so an
/// all-zeros pointer (commonly the uninitialized value of a memory slot)
/// never names a real block.
const HEAP_START: u32 = 8;
/// Memory argument used for all of our 4-byte (i32-sized) loads and stores.
const I32_MEM_ARG: MemArg = MemArg {
    offset: 0,
    align: 2,
    memory_index: 0,
};

struct Codegen<'a> {
    info: &'a ModuleInfo,
    /// Fn name → Wasm function index. Lumen fn `f` gets index
    /// `1 + position_in_module`, because helper index 0 is reserved for
    /// `string_concat`.
    fn_indices: HashMap<String, u32>,
    types: TypeSection,
    functions: FunctionSection,
    memory: MemorySection,
    globals: GlobalSection,
    data: DataSection,
    exports: ExportSection,
    code: CodeSection,
    /// Offset in linear memory → `[len: i32 | bytes...]` for every unique
    /// string literal observed in the module. Built during pre-pass so the
    /// data section is known before bodies emit.
    string_offsets: HashMap<String, u32>,
    /// Deterministic per-type layout for every user struct. Computed once
    /// during the pre-pass and read by struct literal codegen (for stores)
    /// and field access (for loads).
    struct_layouts: HashMap<String, StructLayout>,
    /// First free byte after all the literal data segments — the initial
    /// value of the bump-pointer global.
    heap_ptr: u32,
}

#[derive(Clone, Debug)]
struct StructLayout {
    fields: Vec<StructFieldLayout>,
    total_size: u32,
}

#[derive(Clone, Debug)]
struct StructFieldLayout {
    name: String,
    ty: Ty,
    offset: u32,
    /// Wasm `MemArg.align` is log2 of the natural byte alignment. We store
    /// the alignment here in that already-encoded form.
    align_log2: u32,
    wasm_ty: ValType,
}

impl<'a> Codegen<'a> {
    fn new(info: &'a ModuleInfo) -> Self {
        Self {
            info,
            fn_indices: HashMap::new(),
            types: TypeSection::new(),
            functions: FunctionSection::new(),
            memory: MemorySection::new(),
            globals: GlobalSection::new(),
            data: DataSection::new(),
            exports: ExportSection::new(),
            code: CodeSection::new(),
            string_offsets: HashMap::new(),
            struct_layouts: HashMap::new(),
            heap_ptr: HEAP_START,
        }
    }

    fn compile_module(&mut self, module: &ast::Module) -> Result<(), CodegenError> {
        // Pass A0: compute a layout for every user struct so that struct
        // literals and field accesses can emit stable offsets.
        for (name, info) in &self.info.types {
            if let TypeInfo::Struct { fields, .. } = info {
                let layout = compute_struct_layout(fields)?;
                self.struct_layouts.insert(name.clone(), layout);
            }
        }

        // Pass A: intern every string literal into a deterministic offset
        // and emit the data section payloads. Do this before signatures so
        // the heap pointer is known by the time we emit the bump global.
        self.intern_string_literals(module);

        // Pass B: emit helper signatures first so their indices are fixed.
        // string_concat has type index 0 and fn index 0.
        self.types
            .ty()
            .function(vec![ValType::I32, ValType::I32], vec![ValType::I32]);
        self.functions.function(0);

        // Pass C: assign a function + type index to every user `fn` item.
        // User fn `i` lives at fn index `1 + i` (helper = 0).
        let mut next_index = 1u32;
        for item in &module.items {
            if let Item::Fn(f) = item {
                let sig = self
                    .info
                    .fns
                    .get(&f.name)
                    .expect("type checker should have populated this fn");
                let idx = next_index;
                next_index += 1;
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

        // Pass D: emit the linear memory (1 page = 64 KiB to start) and the
        // bump pointer global, initialized just past the last string
        // literal.
        self.memory.memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        self.globals.global(
            GlobalType {
                val_type: ValType::I32,
                mutable: true,
                shared: false,
            },
            &ConstExpr::i32_const(self.heap_ptr as i32),
        );
        self.exports.export("memory", ExportKind::Memory, 0);

        // Pass E: emit the string_concat helper body.
        self.code.function(&self.emit_string_concat_helper());

        // Pass F: emit each user function body.
        for item in &module.items {
            if let Item::Fn(f) = item {
                let func = self.compile_fn(f)?;
                self.code.function(&func);
            }
        }

        Ok(())
    }

    /// Walk the AST and assign a linear-memory offset to every distinct
    /// string literal. The data section payload is `[len: i32 little-endian
    /// | bytes...]` so `string_len(s)` can just `i32.load` at `offset=0`.
    fn intern_string_literals(&mut self, module: &ast::Module) {
        let mut seen: Vec<String> = Vec::new();
        for item in &module.items {
            if let Item::Fn(f) = item {
                self.scan_string_lits_in_block(&f.body, &mut seen);
            }
        }
        for s in seen {
            if self.string_offsets.contains_key(&s) {
                continue;
            }
            let offset = self.heap_ptr;
            self.string_offsets.insert(s.clone(), offset);

            let bytes = s.as_bytes();
            let len = bytes.len() as u32;
            let mut payload: Vec<u8> = Vec::with_capacity(4 + bytes.len());
            payload.extend_from_slice(&len.to_le_bytes());
            payload.extend_from_slice(bytes);

            self.data
                .active(0, &ConstExpr::i32_const(offset as i32), payload);
            self.heap_ptr += 4 + len;
            // Keep the bump pointer i32-aligned so future allocations stay
            // aligned for i32 loads/stores.
            while !self.heap_ptr.is_multiple_of(4) {
                self.heap_ptr += 1;
            }
        }
    }

    fn scan_string_lits_in_block(&self, block: &ast::Block, acc: &mut Vec<String>) {
        for stmt in &block.stmts {
            self.scan_string_lits_in_stmt(stmt, acc);
        }
        if let Some(tail) = &block.tail {
            self.scan_string_lits_in_expr(tail, acc);
        }
    }

    fn scan_string_lits_in_stmt(&self, stmt: &ast::Stmt, acc: &mut Vec<String>) {
        match &stmt.kind {
            StmtKind::Let { value, .. }
            | StmtKind::Var { value, .. }
            | StmtKind::Assign { value, .. } => self.scan_string_lits_in_expr(value, acc),
            StmtKind::Expr(e) => self.scan_string_lits_in_expr(e, acc),
            StmtKind::For { iter, body, .. } => {
                self.scan_string_lits_in_expr(iter, acc);
                self.scan_string_lits_in_block(body, acc);
            }
            StmtKind::Return(Some(e)) => self.scan_string_lits_in_expr(e, acc),
            StmtKind::Return(None) => {}
        }
    }

    fn scan_string_lits_in_expr(&self, expr: &Expr, acc: &mut Vec<String>) {
        match &expr.kind {
            ExprKind::StringLit(s) => {
                if !acc.iter().any(|existing| existing == s) {
                    acc.push(s.clone());
                }
            }
            ExprKind::Paren(e) => self.scan_string_lits_in_expr(e, acc),
            ExprKind::Unary { rhs, .. } => self.scan_string_lits_in_expr(rhs, acc),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.scan_string_lits_in_expr(lhs, acc);
                self.scan_string_lits_in_expr(rhs, acc);
            }
            ExprKind::Cast { expr, .. } => self.scan_string_lits_in_expr(expr, acc),
            ExprKind::Call { callee, args } => {
                self.scan_string_lits_in_expr(callee, acc);
                for a in args {
                    self.scan_string_lits_in_expr(&a.value, acc);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.scan_string_lits_in_expr(receiver, acc);
                for a in args {
                    self.scan_string_lits_in_expr(&a.value, acc);
                }
            }
            ExprKind::Field { receiver, .. } => self.scan_string_lits_in_expr(receiver, acc),
            ExprKind::Try(inner) => self.scan_string_lits_in_expr(inner, acc),
            ExprKind::If {
                cond,
                then_block,
                else_block,
            } => {
                self.scan_string_lits_in_expr(cond, acc);
                self.scan_string_lits_in_block(then_block, acc);
                self.scan_string_lits_in_block(else_block, acc);
            }
            ExprKind::Match { scrutinee, arms } => {
                self.scan_string_lits_in_expr(scrutinee, acc);
                for arm in arms {
                    self.scan_string_lits_in_expr(&arm.body, acc);
                }
            }
            ExprKind::Block(b) => self.scan_string_lits_in_block(b, acc),
            ExprKind::StructLit { fields, .. } => {
                for fi in fields {
                    self.scan_string_lits_in_expr(&fi.value, acc);
                }
            }
            ExprKind::IntLit { .. }
            | ExprKind::FloatLit(_)
            | ExprKind::BoolLit(_)
            | ExprKind::UnitLit
            | ExprKind::Ident(_) => {}
        }
    }

    /// Emit the body of the `string_concat(a: i32, b: i32) -> i32` helper.
    /// Allocates a new `[len | bytes]` block at the bump pointer and
    /// `memory.copy`s the two source payloads into place.
    fn emit_string_concat_helper(&self) -> Function {
        // Local layout: 0=a, 1=b (params), 2=len_a, 3=len_b, 4=result.
        let mut f = Function::new_with_locals_types(vec![
            ValType::I32, // len_a
            ValType::I32, // len_b
            ValType::I32, // result
        ]);

        // len_a = i32.load(a)
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Load(I32_MEM_ARG));
        f.instruction(&Instruction::LocalSet(2));

        // len_b = i32.load(b)
        f.instruction(&Instruction::LocalGet(1));
        f.instruction(&Instruction::I32Load(I32_MEM_ARG));
        f.instruction(&Instruction::LocalSet(3));

        // result = bump_ptr
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::LocalSet(4));

        // bump_ptr = bump_ptr + 4 + len_a + len_b (keep i32 alignment by
        // rounding up to the next multiple of 4 inline).
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::I32Const(4));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(3));
        f.instruction(&Instruction::I32Add);
        // Align up to 4: x = (x + 3) & ~3
        f.instruction(&Instruction::I32Const(3));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Const(-4));
        f.instruction(&Instruction::I32And);
        f.instruction(&Instruction::GlobalSet(0));

        // *result = len_a + len_b
        f.instruction(&Instruction::LocalGet(4));
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::LocalGet(3));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Store(I32_MEM_ARG));

        // memory.copy(dst = result + 4, src = a + 4, len = len_a)
        f.instruction(&Instruction::LocalGet(4));
        f.instruction(&Instruction::I32Const(4));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Const(4));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::MemoryCopy {
            src_mem: 0,
            dst_mem: 0,
        });

        // memory.copy(dst = result + 4 + len_a, src = b + 4, len = len_b)
        f.instruction(&Instruction::LocalGet(4));
        f.instruction(&Instruction::I32Const(4));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(1));
        f.instruction(&Instruction::I32Const(4));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(3));
        f.instruction(&Instruction::MemoryCopy {
            src_mem: 0,
            dst_mem: 0,
        });

        // Return result
        f.instruction(&Instruction::LocalGet(4));
        f.instruction(&Instruction::End);

        f
    }

    fn finish(self) -> Vec<u8> {
        let mut module = Module::new();
        module.section(&self.types);
        module.section(&self.functions);
        module.section(&self.memory);
        module.section(&self.globals);
        module.section(&self.exports);
        module.section(&self.code);
        module.section(&self.data);
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
    /// Struct literals need a scratch i32 local to hold the freshly-
    /// allocated pointer while the fields are being written. Keyed by the
    /// struct-lit expression's `span.start` so emit-time lookups find the
    /// slot the pre-pass allocated.
    struct_lit_slots: HashMap<u32, u32>,
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
            struct_lit_slots: HashMap::new(),
        }
    }

    /// Allocate an anonymous i32 scratch local for internal codegen use
    /// (e.g., holding a struct-literal pointer while its fields are being
    /// written). Returns the local's index.
    fn alloc_scratch_i32(&mut self) -> u32 {
        let idx = self.local_types.len() as u32;
        self.local_types.push(Ty::I32);
        self.extra_locals_types.push(ValType::I32);
        idx
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
                // Reserve a scratch i32 for this literal up front so the
                // emit pass can hand the freshly-allocated pointer around.
                let slot = self.alloc_scratch_i32();
                self.struct_lit_slots.insert(expr.span.start, slot);
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
            ExprKind::StringLit(_) => Ty::String,
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
                    if name == "string_len" {
                        Ty::I32
                    } else if let Some(sig) = self.cg.info.fns.get(name) {
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
            ExprKind::StructLit { name, .. } => Ty::User(name.clone()),
            ExprKind::Field { receiver, name } => {
                let recv_ty = self.infer_value_ty(receiver)?;
                let type_name = match recv_ty {
                    Ty::User(n) => n,
                    _ => {
                        return Err(CodegenError {
                            span: expr.span,
                            message: "field access on a non-struct".into(),
                        });
                    }
                };
                let layout = self
                    .cg
                    .struct_layouts
                    .get(&type_name)
                    .ok_or_else(|| CodegenError {
                        span: expr.span,
                        message: format!("no layout for struct `{type_name}`"),
                    })?;
                layout
                    .fields
                    .iter()
                    .find(|f| f.name == *name)
                    .ok_or_else(|| CodegenError {
                        span: expr.span,
                        message: format!("struct `{type_name}` has no field `{name}`"),
                    })?
                    .ty
                    .clone()
            }
            _ => {
                return Err(CodegenError {
                    span: expr.span,
                    message: "expression kind not supported by current codegen".into(),
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
            ExprKind::StringLit(s) => {
                let offset = *self
                    .cg
                    .string_offsets
                    .get(s)
                    .expect("string literal should have been interned in pass A");
                f.instruction(&Instruction::I32Const(offset as i32));
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

                // Built-in `string_len(s)` = `i32.load` of the length word
                // at offset 0 of the string payload.
                if name == "string_len" {
                    if args.len() != 1 {
                        return Err(CodegenError {
                            span: expr.span,
                            message: format!(
                                "string_len expects 1 argument, found {}",
                                args.len()
                            ),
                        });
                    }
                    self.compile_expr(&args[0].value, f)?;
                    f.instruction(&Instruction::I32Load(I32_MEM_ARG));
                    return Ok(());
                }

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
            ExprKind::StructLit { name, fields, .. } => {
                self.compile_struct_lit(name, fields, expr.span, f)?;
            }
            ExprKind::Field { receiver, name } => {
                self.compile_field_access(receiver, name, f)?;
            }
            _ => {
                return Err(CodegenError {
                    span: expr.span,
                    message: "expression kind not supported by current codegen".into(),
                });
            }
        }
        Ok(())
    }

    fn compile_struct_lit(
        &self,
        name: &str,
        fields: &[ast::FieldInit],
        span: Span,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        let layout = self
            .cg
            .struct_layouts
            .get(name)
            .ok_or_else(|| CodegenError {
                span,
                message: format!("no layout for struct `{name}`"),
            })?;
        let slot = *self
            .struct_lit_slots
            .get(&span.start)
            .expect("walk_expr should have reserved a scratch slot");

        // Allocate a fresh block: slot = bump_ptr; bump_ptr += total (align 4).
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::LocalSet(slot));

        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::I32Const(layout.total_size as i32));
        f.instruction(&Instruction::I32Add);
        // Align up to 4: `(x + 3) & ~3`.
        f.instruction(&Instruction::I32Const(3));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Const(-4));
        f.instruction(&Instruction::I32And);
        f.instruction(&Instruction::GlobalSet(0));

        // Write each field in declaration order — the typechecker has
        // already verified every required field is present and well-typed.
        for layout_field in &layout.fields {
            let init = fields
                .iter()
                .find(|fi| fi.name == layout_field.name)
                .ok_or_else(|| CodegenError {
                    span,
                    message: format!(
                        "internal: struct literal for `{name}` is missing field `{}` \
                         (should have been caught by the typechecker)",
                        layout_field.name
                    ),
                })?;

            // Push the base pointer, then the field value, then store.
            f.instruction(&Instruction::LocalGet(slot));
            self.compile_expr(&init.value, f)?;
            let mem_arg = MemArg {
                offset: layout_field.offset as u64,
                align: layout_field.align_log2,
                memory_index: 0,
            };
            let instr = match layout_field.wasm_ty {
                ValType::I32 => Instruction::I32Store(mem_arg),
                ValType::I64 => Instruction::I64Store(mem_arg),
                ValType::F64 => Instruction::F64Store(mem_arg),
                _ => unreachable!("struct fields lower to i32/i64/f64"),
            };
            f.instruction(&instr);
        }

        // Leave the struct pointer on the stack as the expression value.
        f.instruction(&Instruction::LocalGet(slot));
        Ok(())
    }

    fn compile_field_access(
        &self,
        receiver: &Expr,
        field: &str,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        let recv_ty = self.infer_value_ty(receiver)?;
        let type_name = match recv_ty {
            Ty::User(n) => n,
            _ => {
                return Err(CodegenError {
                    span: receiver.span,
                    message: "field access on a non-struct".into(),
                });
            }
        };
        let layout = self
            .cg
            .struct_layouts
            .get(&type_name)
            .ok_or_else(|| CodegenError {
                span: receiver.span,
                message: format!("no layout for struct `{type_name}`"),
            })?;
        let field_layout = layout
            .fields
            .iter()
            .find(|f| f.name == field)
            .ok_or_else(|| CodegenError {
                span: receiver.span,
                message: format!("struct `{type_name}` has no field `{field}`"),
            })?;

        self.compile_expr(receiver, f)?;
        let mem_arg = MemArg {
            offset: field_layout.offset as u64,
            align: field_layout.align_log2,
            memory_index: 0,
        };
        let instr = match field_layout.wasm_ty {
            ValType::I32 => Instruction::I32Load(mem_arg),
            ValType::I64 => Instruction::I64Load(mem_arg),
            ValType::F64 => Instruction::F64Load(mem_arg),
            _ => unreachable!("struct fields lower to i32/i64/f64"),
        };
        f.instruction(&instr);
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

        // String + string → call into the auto-emitted string_concat
        // helper. Matches the typechecker's special case.
        if matches!(op, BinOp::Add) && matches!(lt, Ty::String) {
            self.compile_expr(lhs, f)?;
            self.compile_expr(rhs, f)?;
            f.instruction(&Instruction::Call(STRING_CONCAT_FN_INDEX));
            return Ok(());
        }

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
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::Unit | Ty::String | Ty::User(_) => ValType::I32,
        Ty::I64 | Ty::U64 => ValType::I64,
        Ty::F64 => ValType::F64,
        other => {
            return Err(CodegenError {
                span,
                message: format!(
                    "codegen does not yet support type {} (sum types land with lumen-mmx, Option/Result with lumen-rvp)",
                    other.display()
                ),
            });
        }
    })
}

/// Byte size of a value of this type in linear memory. Scalars are 4 or 8;
/// composite types that live behind a pointer (strings, user structs) take
/// 4 bytes for the i32 handle.
fn sizeof(ty: &Ty) -> u32 {
    match ty {
        Ty::I64 | Ty::U64 | Ty::F64 => 8,
        _ => 4,
    }
}

/// Natural byte alignment.
fn alignof(ty: &Ty) -> u32 {
    match ty {
        Ty::I64 | Ty::U64 | Ty::F64 => 8,
        _ => 4,
    }
}

fn compute_struct_layout(fields: &[(String, Ty)]) -> Result<StructLayout, CodegenError> {
    let mut offset: u32 = 0;
    let mut laid_out = Vec::with_capacity(fields.len());
    for (name, ty) in fields {
        let align = alignof(ty);
        // Round `offset` up to `align`.
        offset = (offset + align - 1) & !(align - 1);
        let wasm_ty = wasm_val_type(ty, Span::DUMMY)?;
        laid_out.push(StructFieldLayout {
            name: name.clone(),
            ty: ty.clone(),
            offset,
            align_log2: align.trailing_zeros(),
            wasm_ty,
        });
        offset += sizeof(ty);
    }
    // Round total size up to 4 bytes so allocations stay i32-aligned.
    let total_size = (offset + 3) & !3;
    Ok(StructLayout {
        fields: laid_out,
        total_size,
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
    fn run_string_literal_length() {
        let src = r#"fn hello_len(): i32 { string_len("hello") }"#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
        assert_eq!(run_i32(&wasm, "hello_len"), 5);
    }

    #[test]
    fn run_string_concat_length() {
        // The star criterion: string concat compiles and runs end-to-end.
        let src = r#"fn total(): i32 { string_len("hello, " + "world") }"#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
        assert_eq!(run_i32(&wasm, "total"), 12);
    }

    #[test]
    fn run_string_concat_three_way() {
        let src = r#"fn total(): i32 { string_len("a" + "bc" + "def") }"#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "total"), 6);
    }

    #[test]
    fn run_string_concat_with_variable() {
        let src = r#"
            fn greet(): i32 {
                let greeting = "hello, " + "world"
                string_len(greeting)
            }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "greet"), 12);
    }

    #[test]
    fn run_string_literal_deduplicated() {
        // Using the same literal twice should not crash — the scanner
        // de-dupes by content before assigning offsets.
        let src = r#"
            fn a(): i32 { string_len("hi") }
            fn b(): i32 { string_len("hi") }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "a"), 2);
        assert_eq!(run_i32(&wasm, "b"), 2);
    }

    // --- Structs -----------------------------------------------------------

    #[test]
    fn run_struct_literal_and_field_access() {
        let src = r#"
            type Point = { x: i32, y: i32 }

            fn make_and_read(): i32 {
                let p = Point { x: 3, y: 4 }
                p.x + p.y
            }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
        assert_eq!(run_i32(&wasm, "make_and_read"), 7);
    }

    #[test]
    fn run_struct_with_mixed_field_widths() {
        // i64 field forces 8-byte alignment; make sure the layout holds.
        let src = r#"
            type Mixed = { tag: i32, big: i64, small: i32 }

            fn sum(): i64 {
                let m = Mixed { tag: 1, big: 100i64, small: 2 }
                m.big
            }

            fn tag(): i32 {
                let m = Mixed { tag: 7, big: 0i64, small: 0 }
                m.tag + m.small
            }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
        assert_eq!(run_i64(&wasm, "sum"), 100);
        assert_eq!(run_i32(&wasm, "tag"), 7);
    }

    #[test]
    fn run_struct_as_argument_and_return() {
        let src = r#"
            type Point = { x: i32, y: i32 }

            fn origin(): Point {
                Point { x: 0, y: 0 }
            }

            fn translate(p: Point, dx: i32, dy: i32): Point {
                Point { x: p.x + dx, y: p.y + dy }
            }

            fn manhattan(): i32 {
                let p = translate(origin(), 3, 4)
                p.x + p.y
            }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
        assert_eq!(run_i32(&wasm, "manhattan"), 7);
    }

    #[test]
    fn run_struct_with_string_field() {
        // Make sure the string field's i32 pointer layout interacts with
        // the bump allocator without corrupting itself.
        let src = r#"
            type Greet = { prefix: string }

            fn greeting_len(): i32 {
                let g = Greet { prefix: "hello" }
                string_len(g.prefix)
            }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "greeting_len"), 5);
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
