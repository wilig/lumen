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
    CodeSection, ConstExpr, DataSection, EntityType, ExportKind, ExportSection, Function,
    FunctionSection, GlobalSection, GlobalType, ImportSection, Instruction, MemArg,
    MemorySection, MemoryType, Module, TypeSection, ValType,
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

// STRING_CONCAT_FN_INDEX is now `self.num_imports + 0` (determined at
// compile time based on whether WASI imports are present).
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
    /// `num_imports + 1 + position_in_module`, because WASI imports
    /// come first (if any), then helper index 0, then user fns.
    fn_indices: HashMap<String, u32>,
    /// Whether the source module uses `import std/io`.
    uses_wasi: bool,
    /// Number of imported WASI functions (0 or 1 for fd_write).
    num_imports: u32,
    /// Function index of the auto-emitted `io_println` helper, if WASI
    /// is in use.
    io_println_idx: Option<u32>,
    /// Function index of the auto-emitted `print_frames` helper.
    print_frames_idx: Option<u32>,
    /// Function index of the auto-emitted `int_to_string_i32` helper.
    int_to_string_idx: u32,
    types: TypeSection,
    imports: ImportSection,
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
            uses_wasi: false,
            num_imports: 0,
            io_println_idx: None,
            print_frames_idx: None,
            int_to_string_idx: 0, // set during compile_module
            types: TypeSection::new(),
            imports: ImportSection::new(),
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
        // Check if the source imports std/io.
        self.uses_wasi = module
            .imports
            .iter()
            .any(|im| im.path == ["std", "io"]);

        // If WASI is used, ensure "\n" is interned early for io_println.
        if self.uses_wasi {
            // We'll intern "\n" as a string literal so it's in the data
            // section. The helper uses it for the trailing newline.
        }

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
        // Also intern the formatted error-frame messages for every `?`
        // site in the source, so they're available in the data section.
        self.intern_frame_messages(module);
        // Ensure "\n" is interned for io_println even if the user didn't
        // use it as a literal.
        if self.uses_wasi && !self.string_offsets.contains_key("\n") {
            let offset = self.heap_ptr;
            self.string_offsets.insert("\n".to_string(), offset);
            let mut payload = Vec::with_capacity(8);
            payload.extend_from_slice(&1u32.to_le_bytes());
            payload.push(b'\n');
            self.data
                .active(0, &ConstExpr::i32_const(offset as i32), payload);
            self.heap_ptr += 8; // 5 bytes rounded up to 8 for alignment
        }

        // Determine function index offsets. When WASI is in use we import
        // fd_write at index 0, shifting all internal functions by 1.
        let mut type_idx = 0u32;

        if self.uses_wasi {
            // Import fd_write: (i32, i32, i32, i32) -> i32
            self.types
                .ty()
                .function(vec![ValType::I32; 4], vec![ValType::I32]);
            let fd_write_type_idx = type_idx;
            type_idx += 1;
            self.imports.import(
                "wasi_snapshot_preview1",
                "fd_write",
                EntityType::Function(fd_write_type_idx),
            );
            self.num_imports = 1;
        }

        // Pass B: emit internal helper signatures.
        // string_concat: (i32, i32) -> i32
        self.types
            .ty()
            .function(vec![ValType::I32, ValType::I32], vec![ValType::I32]);
        type_idx += 1;
        self.functions.function(type_idx - 1);

        if self.uses_wasi {
            // io_println: (i32) -> i32 (unit)
            self.types
                .ty()
                .function(vec![ValType::I32], vec![ValType::I32]);
            self.io_println_idx = Some(self.num_imports + 1);
            self.functions.function(type_idx);
            type_idx += 1;

            // print_frames: () -> i32 (unit)
            self.types.ty().function(vec![], vec![ValType::I32]);
            self.print_frames_idx = Some(self.num_imports + 2);
            self.functions.function(type_idx);
            #[allow(unused_assignments)]
            {
                type_idx += 1;
            }
        }

        // int_to_string_i32: (i32) -> i32 (string pointer). Always emitted.
        let base_helpers: u32 = if self.uses_wasi { 3 } else { 1 };
        self.types
            .ty()
            .function(vec![ValType::I32], vec![ValType::I32]);
        self.int_to_string_idx = self.num_imports + base_helpers;
        self.functions.function(type_idx);
        #[allow(unused_assignments)]
        {
            type_idx += 1;
        }

        // Pass C: assign a function + type index to every user `fn` item.
        let helpers = base_helpers + 1;
        let mut next_index = self.num_imports + helpers;
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
        // Global 1: frame_chain — head of the error-frame linked list.
        // 0 = no frames.
        self.globals.global(
            GlobalType {
                val_type: ValType::I32,
                mutable: true,
                shared: false,
            },
            &ConstExpr::i32_const(0),
        );
        self.exports.export("memory", ExportKind::Memory, 0);

        // Pass E: emit helper function bodies.
        self.code.function(&self.emit_string_concat_helper());
        if self.uses_wasi {
            self.code.function(&self.emit_io_println_helper());
            self.code.function(&self.emit_print_frames_helper());
        }
        self.code.function(&self.emit_int_to_string_helper());

        // Pass F: emit each user function body.
        for item in &module.items {
            if let Item::Fn(f) = item {
                let func = self.compile_fn(f)?;
                self.code.function(&func);
            }
        }

        Ok(())
    }

    /// Pre-scan every `?` site and intern a formatted frame message
    /// string for each one: `"  at {fn_name} (<source>:{line}:{col})"`.
    fn intern_frame_messages(&mut self, module: &ast::Module) {
        for item in &module.items {
            if let Item::Fn(f) = item {
                self.scan_try_sites(&f.name, &f.body);
            }
        }
    }

    fn scan_try_sites(&mut self, fn_name: &str, block: &ast::Block) {
        for stmt in &block.stmts {
            match &stmt.kind {
                StmtKind::Let { value, .. }
                | StmtKind::Var { value, .. }
                | StmtKind::Assign { value, .. } => self.scan_try_in_expr(fn_name, value),
                StmtKind::Expr(e) => self.scan_try_in_expr(fn_name, e),
                StmtKind::For { iter, body, .. } => {
                    self.scan_try_in_expr(fn_name, iter);
                    self.scan_try_sites(fn_name, body);
                }
                StmtKind::Return(Some(e)) => self.scan_try_in_expr(fn_name, e),
                StmtKind::Return(None) => {}
            }
        }
        if let Some(tail) = &block.tail {
            self.scan_try_in_expr(fn_name, tail);
        }
    }

    fn scan_try_in_expr(&mut self, fn_name: &str, expr: &Expr) {
        match &expr.kind {
            ExprKind::Try(inner) => {
                self.scan_try_in_expr(fn_name, inner);
                // Intern the frame message for this `?` site.
                let msg = format!(
                    "  at {} (<source>:{}:{})",
                    fn_name, expr.span.line, expr.span.col
                );
                self.intern_string(&msg);
            }
            ExprKind::Paren(e) => self.scan_try_in_expr(fn_name, e),
            ExprKind::Unary { rhs, .. } => self.scan_try_in_expr(fn_name, rhs),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.scan_try_in_expr(fn_name, lhs);
                self.scan_try_in_expr(fn_name, rhs);
            }
            ExprKind::Cast { expr, .. } => self.scan_try_in_expr(fn_name, expr),
            ExprKind::Call { callee, args } => {
                self.scan_try_in_expr(fn_name, callee);
                for a in args {
                    self.scan_try_in_expr(fn_name, &a.value);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.scan_try_in_expr(fn_name, receiver);
                for a in args {
                    self.scan_try_in_expr(fn_name, &a.value);
                }
            }
            ExprKind::Field { receiver, .. } => self.scan_try_in_expr(fn_name, receiver),
            ExprKind::If {
                cond,
                then_block,
                else_block,
            } => {
                self.scan_try_in_expr(fn_name, cond);
                self.scan_try_sites(fn_name, then_block);
                self.scan_try_sites(fn_name, else_block);
            }
            ExprKind::Match { scrutinee, arms } => {
                self.scan_try_in_expr(fn_name, scrutinee);
                for arm in arms {
                    self.scan_try_in_expr(fn_name, &arm.body);
                }
            }
            ExprKind::Block(b) => self.scan_try_sites(fn_name, b),
            ExprKind::StructLit { fields, .. } => {
                for fi in fields {
                    self.scan_try_in_expr(fn_name, &fi.value);
                }
            }
            _ => {}
        }
    }

    /// Intern a string into the data section if not already present.
    fn intern_string(&mut self, s: &str) {
        if self.string_offsets.contains_key(s) {
            return;
        }
        let offset = self.heap_ptr;
        self.string_offsets.insert(s.to_string(), offset);
        let bytes = s.as_bytes();
        let len = bytes.len() as u32;
        let mut payload = Vec::with_capacity(4 + bytes.len());
        payload.extend_from_slice(&len.to_le_bytes());
        payload.extend_from_slice(bytes);
        self.data
            .active(0, &ConstExpr::i32_const(offset as i32), payload);
        self.heap_ptr += 4 + len;
        while !self.heap_ptr.is_multiple_of(4) {
            self.heap_ptr += 1;
        }
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
            self.intern_string(&s);
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

    /// Emit the body of the `io_println(s: i32) -> i32` helper.
    /// Concatenates `\n` to the string, then writes via WASI fd_write.
    fn emit_io_println_helper(&self) -> Function {
        // Locals: 0=s (param), 1=with_nl, 2=iov_buf
        let mut f = Function::new_with_locals_types(vec![ValType::I32, ValType::I32]);

        let nl_offset = *self
            .string_offsets
            .get("\n")
            .expect("\\n should be interned");
        let concat_idx = self.num_imports; // string_concat fn index

        // with_nl = string_concat(s, "\n")
        f.instruction(&Instruction::LocalGet(0)); // s
        f.instruction(&Instruction::I32Const(nl_offset as i32));
        f.instruction(&Instruction::Call(concat_idx));
        f.instruction(&Instruction::LocalSet(1)); // with_nl

        // Allocate iov (8 bytes) + nwritten (4 bytes) = 12 bytes.
        // Emit inline bump alloc.
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::I32Const(12));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Const(3));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Const(-4));
        f.instruction(&Instruction::I32And);
        f.instruction(&Instruction::GlobalSet(0));
        f.instruction(&Instruction::LocalSet(2)); // iov_buf

        // iov[0].ptr = with_nl + 4  (skip length prefix)
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::LocalGet(1));
        f.instruction(&Instruction::I32Const(4));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Store(I32_MEM_ARG)); // *(iov_buf+0) = ptr

        // iov[0].len = i32.load(with_nl)
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::LocalGet(1));
        f.instruction(&Instruction::I32Load(I32_MEM_ARG));
        f.instruction(&Instruction::I32Store(MemArg {
            offset: 4,
            align: 2,
            memory_index: 0,
        })); // *(iov_buf+4) = len

        // fd_write(fd=1, iovs=iov_buf, iovs_count=1, nwritten=iov_buf+8)
        f.instruction(&Instruction::I32Const(1)); // fd = stdout
        f.instruction(&Instruction::LocalGet(2)); // iovs
        f.instruction(&Instruction::I32Const(1)); // iovs_count
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::I32Const(8));
        f.instruction(&Instruction::I32Add); // nwritten ptr
        f.instruction(&Instruction::Call(0)); // fd_write is import index 0
        f.instruction(&Instruction::Drop); // ignore errno

        // Return unit (0).
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::End);

        f
    }

    /// Emit the body of `int_to_string_i32(n: i32) -> i32` (string ptr).
    /// Converts an i32 to its decimal string representation in linear memory.
    fn emit_int_to_string_helper(&self) -> Function {
        // Locals: 0=n (param), 1=scratch, 2=pos, 3=is_neg, 4=abs_val, 5=len, 6=result
        let mut f = Function::new_with_locals_types(vec![
            ValType::I32, // scratch
            ValType::I32, // pos
            ValType::I32, // is_neg
            ValType::I32, // abs_val
            ValType::I32, // len
            ValType::I32, // result
        ]);

        let store8 = MemArg {
            offset: 0,
            align: 0,
            memory_index: 0,
        };

        // scratch = bump_alloc(16)
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::LocalSet(1));
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::I32Const(16));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::GlobalSet(0));

        // pos = scratch + 15
        f.instruction(&Instruction::LocalGet(1));
        f.instruction(&Instruction::I32Const(15));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalSet(2));

        // is_neg = n < 0
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::I32LtS);
        f.instruction(&Instruction::LocalSet(3));

        // abs_val = is_neg ? (0 - n) : n
        f.instruction(&Instruction::LocalGet(3));
        f.instruction(&Instruction::If(wasm_encoder::BlockType::Result(ValType::I32)));
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Sub);
        f.instruction(&Instruction::Else);
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::End);
        f.instruction(&Instruction::LocalSet(4));

        // if abs_val == 0: write '0'
        f.instruction(&Instruction::LocalGet(4));
        f.instruction(&Instruction::I32Eqz);
        f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::I32Const(48)); // '0'
        f.instruction(&Instruction::I32Store8(store8));
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::I32Const(1));
        f.instruction(&Instruction::I32Sub);
        f.instruction(&Instruction::LocalSet(2));
        f.instruction(&Instruction::Else);

        // while abs_val > 0: extract digits
        f.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty));
        f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));
        f.instruction(&Instruction::LocalGet(4));
        f.instruction(&Instruction::I32Eqz);
        f.instruction(&Instruction::BrIf(1));

        // *(pos) = abs_val % 10 + '0'
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::LocalGet(4));
        f.instruction(&Instruction::I32Const(10));
        f.instruction(&Instruction::I32RemS);
        f.instruction(&Instruction::I32Const(48));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Store8(store8));

        // pos--
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::I32Const(1));
        f.instruction(&Instruction::I32Sub);
        f.instruction(&Instruction::LocalSet(2));

        // abs_val /= 10
        f.instruction(&Instruction::LocalGet(4));
        f.instruction(&Instruction::I32Const(10));
        f.instruction(&Instruction::I32DivS);
        f.instruction(&Instruction::LocalSet(4));

        f.instruction(&Instruction::Br(0));
        f.instruction(&Instruction::End); // end loop
        f.instruction(&Instruction::End); // end block
        f.instruction(&Instruction::End); // end else

        // if is_neg: write '-'
        f.instruction(&Instruction::LocalGet(3));
        f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::I32Const(45)); // '-'
        f.instruction(&Instruction::I32Store8(store8));
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::I32Const(1));
        f.instruction(&Instruction::I32Sub);
        f.instruction(&Instruction::LocalSet(2));
        f.instruction(&Instruction::End);

        // len = scratch + 15 - pos
        f.instruction(&Instruction::LocalGet(1));
        f.instruction(&Instruction::I32Const(15));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::I32Sub);
        f.instruction(&Instruction::LocalSet(5));

        // result = bump_alloc(4 + len)
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::LocalSet(6));
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::LocalGet(5));
        f.instruction(&Instruction::I32Const(4));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Const(3));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Const(-4));
        f.instruction(&Instruction::I32And);
        f.instruction(&Instruction::GlobalSet(0));

        // store len at result+0
        f.instruction(&Instruction::LocalGet(6));
        f.instruction(&Instruction::LocalGet(5));
        f.instruction(&Instruction::I32Store(I32_MEM_ARG));

        // memory.copy(result+4, pos+1, len)
        f.instruction(&Instruction::LocalGet(6));
        f.instruction(&Instruction::I32Const(4));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::I32Const(1));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalGet(5));
        f.instruction(&Instruction::MemoryCopy {
            src_mem: 0,
            dst_mem: 0,
        });

        // return result
        f.instruction(&Instruction::LocalGet(6));
        f.instruction(&Instruction::End);

        f
    }

    /// Emit the body of `print_frames() -> i32`. Walks the global
    /// frame_chain (global 1) and for each frame calls io_println on
    /// the frame's message string.
    fn emit_print_frames_helper(&self) -> Function {
        let io_println_idx = self.io_println_idx.expect("WASI must be in use");
        // Locals: 0 = current frame pointer
        let mut f = Function::new_with_locals_types(vec![ValType::I32]);

        // current = global.get 1 (frame_chain)
        f.instruction(&Instruction::GlobalGet(1));
        f.instruction(&Instruction::LocalSet(0));

        // block $break
        //   loop $continue
        //     if current == 0: br $break
        //     io_println(*(current + 0))  // message string pointer
        //     current = *(current + 4)    // next pointer
        //     br $continue
        //   end
        // end
        f.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty));
        f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));

        // Break if current == 0
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Eqz);
        f.instruction(&Instruction::BrIf(1));

        // io_println(current.message)
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Load(I32_MEM_ARG)); // message ptr
        f.instruction(&Instruction::Call(io_println_idx));
        f.instruction(&Instruction::Drop); // discard unit return

        // current = current.next
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I32Load(MemArg {
            offset: 4,
            align: 2,
            memory_index: 0,
        }));
        f.instruction(&Instruction::LocalSet(0));

        f.instruction(&Instruction::Br(0)); // loop
        f.instruction(&Instruction::End); // end loop
        f.instruction(&Instruction::End); // end block

        // Return unit.
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::End);

        f
    }

    fn finish(self) -> Vec<u8> {
        let mut module = Module::new();
        module.section(&self.types);
        if self.uses_wasi {
            module.section(&self.imports);
        }
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

        let mut fb = FnBuilder::new(self, sig);
        fb.current_fn_name = f.name.clone();
        // Parameters occupy slots [0, num_params).
        for (pname, pty) in &sig.params {
            fb.register_param(pname, pty.clone());
        }

        // Walk pass: allocate every extra local we'll need and populate
        // `slot_for_span`. Maintains its own scope stack as it traverses.
        fb.push_scope();
        for (i, (pname, _)) in sig.params.iter().enumerate() {
            fb.declare(pname.clone(), i as u32);
        }
        fb.walk_block(&f.body)?;
        fb.pop_scope();
        fb.reset_scopes();

        // Emit pass: now that the locals header is fixed, emit the body.
        let mut function = Function::new_with_locals_types(fb.extra_locals_types.clone());
        fb.push_scope();
        for (i, (pname, _)) in sig.params.iter().enumerate() {
            fb.declare(pname.clone(), i as u32);
        }
        fb.compile_block(&f.body, &mut function)?;
        fb.pop_scope();

        // Implicit `unit` return: if the declared return is unit, push a
        // zero i32 so the function signature is satisfied.
        if matches!(sig.ret, Ty::Unit) {
            function.instruction(&Instruction::I32Const(0));
        }

        // If this is `main` and it returns a Result, call print_frames
        // before returning. If no error was propagated, the chain is
        // empty and nothing prints. If there was an error, each `?` site
        // pushed a frame, and they'll all appear on stdout.
        if f.name == "main" && matches!(sig.ret, Ty::Result(_, _)) {
            if let Some(print_frames_idx) = self.print_frames_idx {
                function.instruction(&Instruction::Call(print_frames_idx));
                function.instruction(&Instruction::Drop);
            }
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
    /// Scope stack for name resolution. Both the walk pass and the emit
    /// pass push/pop scopes as they enter and leave blocks and match arms,
    /// and both fully rebuild it from scratch each time. Match arms in
    /// particular need real scoping so two sibling arms can each bind a
    /// `radius` local without colliding.
    scopes: Vec<HashMap<String, u32>>,
    /// Per-slot Lumen type, indexed by slot index. Slots `[0, num_params)`
    /// are the parameters; slots `[num_params, ...)` are extras allocated
    /// during the walk pass. Stable across passes.
    slot_types: Vec<Ty>,
    /// Types of the locals beyond the parameters, in the format
    /// `Function::new_with_locals_types` wants. Stable across passes.
    extra_locals_types: Vec<ValType>,
    num_params: u32,
    /// Span-keyed slot lookups for synthesized bindings. Used for:
    /// - `let` / `var` stmts (key = stmt span.start)
    /// - `for` loop binders (key = stmt span.start)
    /// - struct literal scratch pointers (key = expr span.start)
    /// - match scratch pointer for the scrutinee (key = match expr span.start)
    /// - pattern binding slots (key = pattern span.start)
    /// - `?` scratch (key = try expr span.start)
    ///
    /// Populated during the walk pass, read during emit.
    slot_for_span: HashMap<u32, u32>,
    /// Extra scratch slots that certain emit paths need beyond the
    /// primary `slot_for_span` entry. Sum constructor calls, for
    /// instance, need one slot for the payload pointer *and* one for
    /// the sum pointer. Keyed the same way, ordered by emission need.
    aux_slots: HashMap<u32, Vec<u32>>,
    /// Cached expression types. Populated during the walk pass for
    /// expressions whose type inference requires pattern-binding scope
    /// context (e.g. match arm bodies).
    ///
    /// Read during emit to avoid re-inferring in a scope where the
    /// bindings don't exist yet.
    expr_type_cache: HashMap<u32, Ty>,
    /// Name of the function currently being compiled, used to format
    /// error-frame messages.
    current_fn_name: String,
}

impl<'a, 'b> FnBuilder<'a, 'b> {
    fn new(cg: &'a Codegen<'b>, sig: &'a crate::types::FnSig) -> Self {
        Self {
            cg,
            sig,
            scopes: Vec::new(),
            slot_types: Vec::new(),
            extra_locals_types: Vec::new(),
            num_params: 0,
            slot_for_span: HashMap::new(),
            aux_slots: HashMap::new(),
            expr_type_cache: HashMap::new(),
            current_fn_name: String::new(),
        }
    }

    // --- Scope stack ------------------------------------------------------

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare(&mut self, name: String, slot: u32) {
        self.scopes
            .last_mut()
            .expect("must be inside a scope")
            .insert(name, slot);
    }

    fn lookup(&self, name: &str) -> Option<u32> {
        for scope in self.scopes.iter().rev() {
            if let Some(&slot) = scope.get(name) {
                return Some(slot);
            }
        }
        None
    }

    fn lookup_ty(&self, name: &str) -> Option<Ty> {
        self.lookup(name).map(|s| self.slot_types[s as usize].clone())
    }

    /// Clear the scope stack so a fresh pass (walk → emit) can repopulate.
    fn reset_scopes(&mut self) {
        self.scopes.clear();
    }

    // --- Slot allocation --------------------------------------------------

    /// Allocate a local slot of the given Lumen type.
    fn alloc_local(&mut self, ty: Ty, span: Span) -> Result<u32, CodegenError> {
        let val_type = wasm_val_type(&ty, span)?;
        let idx = self.num_params + self.extra_locals_types.len() as u32;
        self.slot_types.push(ty);
        self.extra_locals_types.push(val_type);
        Ok(idx)
    }

    /// Allocate an anonymous i32 scratch local for internal codegen use
    /// (struct literal pointer, match scrutinee pointer, etc.).
    fn alloc_scratch_i32(&mut self) -> u32 {
        let idx = self.num_params + self.extra_locals_types.len() as u32;
        self.slot_types.push(Ty::I32);
        self.extra_locals_types.push(ValType::I32);
        idx
    }

    fn register_param(&mut self, _name: &str, ty: Ty) {
        assert_eq!(self.slot_types.len() as u32, self.num_params);
        self.slot_types.push(ty);
        self.num_params += 1;
    }

    // ----------------------------------------------------------------------
    // Walk pass: allocate slots and populate slot_for_span
    // ----------------------------------------------------------------------

    fn walk_block(&mut self, block: &ast::Block) -> Result<(), CodegenError> {
        self.push_scope();
        for stmt in &block.stmts {
            self.walk_stmt(stmt)?;
        }
        if let Some(tail) = &block.tail {
            self.walk_expr(tail)?;
        }
        self.pop_scope();
        Ok(())
    }

    fn walk_stmt(&mut self, stmt: &ast::Stmt) -> Result<(), CodegenError> {
        match &stmt.kind {
            StmtKind::Let { name, ty, value } | StmtKind::Var { name, ty, value } => {
                self.walk_expr(value)?;
                let bound_ty = match ty {
                    Some(annot) => self.resolve_ast_type(annot)?,
                    None => self.infer_expr_ty(value)?,
                };
                let slot = self.alloc_local(bound_ty, stmt.span)?;
                self.slot_for_span.insert(stmt.span.start, slot);
                self.declare(name.clone(), slot);
            }
            StmtKind::Assign { value, .. } => {
                self.walk_expr(value)?;
            }
            StmtKind::Expr(e) => self.walk_expr(e)?,
            StmtKind::For { binder, iter, body } => {
                self.walk_expr(iter)?;
                self.push_scope();
                // Allocate the loop binder local (the element variable).
                let binder_slot = self.alloc_local(Ty::I32, stmt.span)?;
                self.slot_for_span.insert(stmt.span.start, binder_slot);
                self.declare(binder.clone(), binder_slot);
                // Allocate a counter local for the range lower bound
                // and an end-bound local. These are aux slots for the
                // for-loop desugaring.
                let counter_slot = self.alloc_scratch_i32();
                let end_slot = self.alloc_scratch_i32();
                self.aux_slots
                    .insert(stmt.span.start, vec![counter_slot, end_slot]);
                self.walk_block(body)?;
                self.pop_scope();
            }
            StmtKind::Return(Some(e)) => self.walk_expr(e)?,
            StmtKind::Return(None) => {}
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
                self.walk_block(then_block)?;
                self.walk_block(else_block)?;
            }
            ExprKind::Block(b) => self.walk_block(b)?,
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
                // If this is a sum-type constructor (built-in Ok/Err/Some
                // or a user variant with a payload), reserve two scratch
                // i32 slots for the payload and sum block pointers.
                if let ExprKind::Ident(name) = &callee.kind {
                    let is_sum_ctor = matches!(name.as_str(), "Ok" | "Err" | "Some")
                        || self.find_sum_for_variant(name).is_some();
                    if is_sum_ctor {
                        let payload_slot = self.alloc_scratch_i32();
                        let sum_slot = self.alloc_scratch_i32();
                        self.aux_slots
                            .insert(expr.span.start, vec![payload_slot, sum_slot]);
                    }
                }
            }
            ExprKind::Field { receiver, .. } => self.walk_expr(receiver)?,
            ExprKind::MethodCall { receiver, args, .. } => {
                self.walk_expr(receiver)?;
                for a in args {
                    self.walk_expr(&a.value)?;
                }
            }
            ExprKind::Try(inner) => {
                self.walk_expr(inner)?;
                // `?` needs a scratch slot for the sum pointer, plus
                // a scratch for the frame allocation on the Err path.
                let slot = self.alloc_scratch_i32();
                self.slot_for_span.insert(expr.span.start, slot);
                let frame_slot = self.alloc_scratch_i32();
                self.aux_slots.insert(expr.span.start, vec![frame_slot]);
            }
            ExprKind::Match { scrutinee, arms } => {
                self.walk_expr(scrutinee)?;
                // Scratch slot for the evaluated scrutinee pointer, plus
                // a second shared i32 slot for the payload pointer that
                // each arm's pattern binds use while destructuring.
                let scrut_slot = self.alloc_scratch_i32();
                let payload_slot = self.alloc_scratch_i32();
                self.slot_for_span.insert(expr.span.start, scrut_slot);
                self.aux_slots.insert(expr.span.start, vec![payload_slot]);

                let scrut_ty = self.infer_expr_ty(scrutinee)?;
                let mut cached_result_ty = None;
                for arm in arms {
                    self.push_scope();
                    self.walk_pattern(&arm.pattern, &scrut_ty)?;
                    self.walk_expr(&arm.body)?;
                    // Cache the result type from the first arm while its
                    // bindings are still in scope. Emit can't re-infer this
                    // because it pushes arm scopes one at a time.
                    if cached_result_ty.is_none() {
                        cached_result_ty = Some(self.infer_expr_ty(&arm.body)?);
                    }
                    self.pop_scope();
                }
                if let Some(ty) = cached_result_ty {
                    self.expr_type_cache.insert(expr.span.start, ty);
                }
            }
            ExprKind::StructLit { name, fields, .. } => {
                // Primary scratch slot: holds the struct pointer (for
                // plain structs) or the payload pointer (for named-field
                // variant constructors).
                let slot = self.alloc_scratch_i32();
                self.slot_for_span.insert(expr.span.start, slot);
                // If this is a named-field variant constructor, allocate
                // a second scratch for the sum block pointer.
                if !self.cg.struct_layouts.contains_key(name)
                    && self.find_sum_for_variant(name).is_some()
                {
                    let sum_slot = self.alloc_scratch_i32();
                    self.aux_slots.insert(expr.span.start, vec![sum_slot]);
                }
                for fi in fields {
                    self.walk_expr(&fi.value)?;
                }
            }
            // Identifier: usually a leaf, but `None` and bare zero-
            // payload variant constructors need a sum-block scratch slot.
            ExprKind::Ident(name) => {
                let needs_slot = name == "None"
                    || (self.lookup(name).is_none()
                        && self.find_sum_for_variant(name).is_some());
                if needs_slot {
                    let sum_slot = self.alloc_scratch_i32();
                    self.aux_slots.insert(expr.span.start, vec![sum_slot]);
                }
            }
            // Leaves
            ExprKind::IntLit { .. }
            | ExprKind::FloatLit(_)
            | ExprKind::StringLit(_)
            | ExprKind::BoolLit(_)
            | ExprKind::UnitLit => {}
        }
        Ok(())
    }

    /// Walk a pattern, allocating slots and declaring names for any
    /// bindings it contains. The `expected` type is the type of the
    /// sub-scrutinee this pattern is matching against.
    fn walk_pattern(&mut self, pat: &ast::Pattern, expected: &Ty) -> Result<(), CodegenError> {
        use ast::{PatternKind, VariantPatPayload};
        match &pat.kind {
            PatternKind::Wildcard | PatternKind::Literal(_) => {}
            PatternKind::Binding(name) => {
                let slot = self.alloc_local(expected.clone(), pat.span)?;
                self.slot_for_span.insert(pat.span.start, slot);
                self.declare(name.clone(), slot);
            }
            PatternKind::Variant {
                name: variant_name,
                payload,
            } => {
                // Zero-payload bindings need no slot work; they match by
                // tag only.
                let Some(payload) = payload.as_ref() else {
                    return Ok(());
                };
                let Some(fields) = self.variant_field_types(expected, variant_name) else {
                    return Ok(());
                };
                match payload {
                    VariantPatPayload::Named(pf_list) => {
                        for pf in pf_list {
                            let field_ty = fields
                                .iter()
                                .find(|(n, _)| n == &pf.name)
                                .map(|(_, t)| t.clone())
                                .unwrap_or(Ty::Error);
                            self.walk_pattern(&pf.pattern, &field_ty)?;
                        }
                    }
                    VariantPatPayload::Positional(pats) => {
                        for (i, p) in pats.iter().enumerate() {
                            let field_ty = fields
                                .get(i)
                                .map(|(_, t)| t.clone())
                                .unwrap_or(Ty::Error);
                            self.walk_pattern(p, &field_ty)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Resolve an AST `Type` to a Lumen `Ty`. Mirrors the typechecker's
    /// resolution rules but does not report errors — anything the
    /// typechecker rejected has already been filtered out.
    fn resolve_ast_type(&self, ty: &ast::Type) -> Result<Ty, CodegenError> {
        match &ty.kind {
            ast::TypeKind::Named { name, args } => Ok(match name.as_str() {
                "i32" if args.is_empty() => Ty::I32,
                "i64" if args.is_empty() => Ty::I64,
                "u32" if args.is_empty() => Ty::U32,
                "u64" if args.is_empty() => Ty::U64,
                "f64" if args.is_empty() => Ty::F64,
                "bool" if args.is_empty() => Ty::Bool,
                "string" if args.is_empty() => Ty::String,
                "unit" if args.is_empty() => Ty::Unit,
                "Option" if args.len() == 1 => {
                    Ty::Option(Box::new(self.resolve_ast_type(&args[0])?))
                }
                "Result" if args.len() == 2 => Ty::Result(
                    Box::new(self.resolve_ast_type(&args[0])?),
                    Box::new(self.resolve_ast_type(&args[1])?),
                ),
                "List" if args.len() == 1 => {
                    Ty::List(Box::new(self.resolve_ast_type(&args[0])?))
                }
                _ if args.is_empty() && self.cg.info.types.contains_key(name) => {
                    Ty::User(name.clone())
                }
                _ => {
                    return Err(CodegenError {
                        span: ty.span,
                        message: format!("unresolved type `{name}`"),
                    });
                }
            }),
        }
    }

    /// Return the list of `(field_name, field_ty)` pairs for a given
    /// variant of a given scrutinee type. For positional payloads the
    /// names are synthesized as `_0`, `_1`, ... Callers use the returned
    /// order for positional matching and the name for named matching.
    fn variant_field_types(
        &self,
        scrut_ty: &Ty,
        variant_name: &str,
    ) -> Option<Vec<(String, Ty)>> {
        match scrut_ty {
            Ty::User(type_name) => {
                let Some(TypeInfo::Sum { variants, .. }) = self.cg.info.types.get(type_name)
                else {
                    return None;
                };
                let v = variants.iter().find(|v| v.name == variant_name)?;
                Some(match &v.payload {
                    None => Vec::new(),
                    Some(crate::types::VariantPayloadInfo::Named(fs)) => fs.clone(),
                    Some(crate::types::VariantPayloadInfo::Positional(tys)) => tys
                        .iter()
                        .enumerate()
                        .map(|(i, t)| (format!("_{i}"), t.clone()))
                        .collect(),
                })
            }
            Ty::Option(inner) => match variant_name {
                "None" => Some(Vec::new()),
                "Some" => Some(vec![("_0".into(), (**inner).clone())]),
                _ => None,
            },
            Ty::Result(ok, err) => match variant_name {
                "Ok" => Some(vec![("_0".into(), (**ok).clone())]),
                "Err" => Some(vec![("_0".into(), (**err).clone())]),
                _ => None,
            },
            _ => None,
        }
    }

    /// Return the runtime tag value for a given variant of a given
    /// scrutinee type. Tags are assigned sequentially starting at 0, and
    /// Option/Result use the conventional None=0/Some=1 and Ok=0/Err=1.
    fn variant_tag(&self, scrut_ty: &Ty, variant_name: &str) -> Option<u32> {
        match scrut_ty {
            Ty::User(type_name) => {
                let Some(TypeInfo::Sum { variants, .. }) = self.cg.info.types.get(type_name)
                else {
                    return None;
                };
                variants
                    .iter()
                    .position(|v| v.name == variant_name)
                    .map(|i| i as u32)
            }
            Ty::Option(_) => match variant_name {
                "None" => Some(0),
                "Some" => Some(1),
                _ => None,
            },
            Ty::Result(_, _) => match variant_name {
                "Ok" => Some(0),
                "Err" => Some(1),
                _ => None,
            },
            _ => None,
        }
    }

    /// Search all user sum types for a variant with this name. Returns
    /// the owning sum type name if found. Used by `compile_call` to
    /// recognize bare-identifier variant constructors.
    fn find_sum_for_variant(&self, variant_name: &str) -> Option<String> {
        for (type_name, info) in &self.cg.info.types {
            if let TypeInfo::Sum { variants, .. } = info {
                if variants.iter().any(|v| v.name == variant_name) {
                    return Some(type_name.clone());
                }
            }
        }
        None
    }

    // ----------------------------------------------------------------------
    // Walk-time type inference
    // ----------------------------------------------------------------------

    /// Get the shape-correct Lumen type for an expression at walk time.
    /// Relies on the walk-pass scope stack to resolve identifiers.
    fn infer_expr_ty(&self, expr: &Expr) -> Result<Ty, CodegenError> {
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
                if let Some(ty) = self.lookup_ty(name) {
                    ty
                } else if name == "None" {
                    // `None` without context — callers who need a concrete
                    // Option<T> will override this via check_expr.
                    Ty::Option(Box::new(Ty::Error))
                } else if let Some(sum_name) = self.find_sum_for_variant(name) {
                    Ty::User(sum_name)
                } else {
                    return Err(CodegenError {
                        span: expr.span,
                        message: format!("unknown identifier `{name}`"),
                    });
                }
            }
            ExprKind::Paren(inner) => self.infer_expr_ty(inner)?,
            ExprKind::Unary { op, rhs } => match op {
                UnaryOp::Neg => self.infer_expr_ty(rhs)?,
                UnaryOp::Not => Ty::Bool,
            },
            ExprKind::Binary { op, lhs, .. } => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                    self.infer_expr_ty(lhs)?
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
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(name) = &callee.kind {
                    match name.as_str() {
                        "string_len" => Ty::I32,
                        "Ok" => {
                            let ok_ty = args
                                .first()
                                .map(|a| self.infer_expr_ty(&a.value))
                                .transpose()?
                                .unwrap_or(Ty::Error);
                            Ty::Result(Box::new(ok_ty), Box::new(Ty::Error))
                        }
                        "Err" => {
                            let err_ty = args
                                .first()
                                .map(|a| self.infer_expr_ty(&a.value))
                                .transpose()?
                                .unwrap_or(Ty::Error);
                            Ty::Result(Box::new(Ty::Error), Box::new(err_ty))
                        }
                        "Some" => {
                            let inner = args
                                .first()
                                .map(|a| self.infer_expr_ty(&a.value))
                                .transpose()?
                                .unwrap_or(Ty::Error);
                            Ty::Option(Box::new(inner))
                        }
                        _ => {
                            if let Some(sig) = self.cg.info.fns.get(name) {
                                sig.ret.clone()
                            } else if let Some(sum_name) = self.find_sum_for_variant(name) {
                                Ty::User(sum_name)
                            } else {
                                return Err(CodegenError {
                                    span: expr.span,
                                    message: format!("unknown function `{name}`"),
                                });
                            }
                        }
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
                    self.infer_expr_ty(tail)?
                } else {
                    Ty::Unit
                }
            }
            ExprKind::Match { arms, .. } => arms
                .first()
                .map(|a| self.infer_expr_ty(&a.body))
                .transpose()?
                .unwrap_or(Ty::Error),
            ExprKind::StructLit { name, .. } => {
                if self.cg.struct_layouts.contains_key(name) {
                    Ty::User(name.clone())
                } else if let Some(sum_name) = self.find_sum_for_variant(name) {
                    Ty::User(sum_name)
                } else {
                    Ty::User(name.clone())
                }
            }
            ExprKind::Field { receiver, name } => {
                let recv_ty = self.infer_expr_ty(receiver)?;
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
            ExprKind::Try(inner) => {
                let inner_ty = self.infer_expr_ty(inner)?;
                match inner_ty {
                    Ty::Result(ok, _) => *ok,
                    Ty::Option(inner) => *inner,
                    _ => Ty::Error,
                }
            }
            ExprKind::MethodCall { receiver, method, .. } => {
                if let ExprKind::Ident(mod_name) = &receiver.kind {
                    if mod_name == "int" && method == "to_string_i32" {
                        return Ok(Ty::String);
                    }
                    if mod_name == "io" && method == "println" {
                        return Ok(Ty::Unit);
                    }
                }
                return Err(CodegenError {
                    span: expr.span,
                    message: format!("method `.{method}(...)` not supported"),
                });
            }
            ExprKind::Block(b) => match &b.tail {
                Some(tail) => self.infer_expr_ty(tail)?,
                None => Ty::Unit,
            },
        })
    }

    // ----------------------------------------------------------------------
    // Code emission
    // ----------------------------------------------------------------------

    fn compile_block(
        &mut self,
        block: &ast::Block,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        self.push_scope();
        for stmt in &block.stmts {
            self.compile_stmt(stmt, f)?;
        }
        if let Some(tail) = &block.tail {
            self.compile_expr(tail, f)?;
        }
        self.pop_scope();
        Ok(())
    }

    fn compile_stmt(
        &mut self,
        stmt: &ast::Stmt,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        match &stmt.kind {
            StmtKind::Let { name, value, .. } | StmtKind::Var { name, value, .. } => {
                self.compile_expr(value, f)?;
                let idx = *self
                    .slot_for_span
                    .get(&stmt.span.start)
                    .expect("walk pass should have allocated this local");
                f.instruction(&Instruction::LocalSet(idx));
                self.declare(name.clone(), idx);
            }
            StmtKind::Assign { name, value } => {
                self.compile_expr(value, f)?;
                let idx = self.lookup(name).ok_or_else(|| CodegenError {
                    span: stmt.span,
                    message: format!("assignment to unknown local `{name}`"),
                })?;
                f.instruction(&Instruction::LocalSet(idx));
            }
            StmtKind::Expr(e) => {
                self.compile_expr(e, f)?;
                // Statement-position expressions aren't consumed as a
                // value, so drop whatever they leave on the Wasm stack.
                // This applies even to "unit"-returning calls because we
                // encode unit as i32(0).
                f.instruction(&Instruction::Drop);
            }
            StmtKind::For { binder, iter, body } => {
                self.compile_for_range(binder, iter, body, stmt.span, f)?;
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

    fn compile_expr(
        &mut self,
        expr: &Expr,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
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
                if let Some(idx) = self.lookup(name) {
                    f.instruction(&Instruction::LocalGet(idx));
                } else if name == "None" {
                    self.compile_none_constructor(expr.span, f)?;
                } else if let Some(sum_name) = self.find_sum_for_variant(name) {
                    // Zero-payload variant constructor, e.g. `type T = | A | B`
                    self.compile_zero_variant_constructor(&sum_name, name, expr.span, f)?;
                } else {
                    let scope_dump: Vec<String> = self
                        .scopes
                        .iter()
                        .map(|s| {
                            let keys: Vec<&str> =
                                s.keys().map(String::as_str).collect();
                            format!("[{}]", keys.join(", "))
                        })
                        .collect();
                    return Err(CodegenError {
                        span: expr.span,
                        message: format!(
                            "unknown identifier `{name}` (scopes: {})",
                            scope_dump.join(" > ")
                        ),
                    });
                }
            }
            ExprKind::Paren(inner) => self.compile_expr(inner, f)?,
            ExprKind::Block(block) => self.compile_block(block, f)?,
            ExprKind::Unary { op, rhs } => {
                let rhs_ty = self.infer_expr_ty(rhs)?;
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
                let from = self.infer_expr_ty(inner)?;
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

                // Built-in Option / Result constructors. Each of these
                // allocates a one-field payload block and wraps it in a
                // sum block with the right tag.
                match name.as_str() {
                    "Ok" => {
                        return self.compile_single_field_variant(
                            0, &args[0].value, expr.span, f,
                        );
                    }
                    "Err" => {
                        return self.compile_single_field_variant(
                            1, &args[0].value, expr.span, f,
                        );
                    }
                    "Some" => {
                        return self.compile_single_field_variant(
                            1, &args[0].value, expr.span, f,
                        );
                    }
                    _ => {}
                }

                // User-defined sum type variant constructor?
                if let Some(sum_name) = self.find_sum_for_variant(&name) {
                    return self.compile_user_variant_constructor(
                        &sum_name, &name, args, expr.span, f,
                    );
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
            ExprKind::Try(inner) => {
                self.compile_try(inner, expr.span, f)?;
            }
            ExprKind::Match { scrutinee, arms } => {
                self.compile_match(scrutinee, arms, expr.span, f)?;
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
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Module-qualified stdlib calls.
                if let ExprKind::Ident(mod_name) = &receiver.kind {
                    if mod_name == "int" && method == "to_string_i32" {
                        if let Some(arg) = args.first() {
                            self.compile_expr(&arg.value, f)?;
                        }
                        f.instruction(&Instruction::Call(self.cg.int_to_string_idx));
                        return Ok(());
                    }
                    if mod_name == "io" && method == "println" {
                        let idx = self.cg.io_println_idx.ok_or_else(|| CodegenError {
                            span: expr.span,
                            message: "`io.println` requires `import std/io`".into(),
                        })?;
                        if let Some(arg) = args.first() {
                            self.compile_expr(&arg.value, f)?;
                        }
                        f.instruction(&Instruction::Call(idx));
                        return Ok(());
                    }
                }
                return Err(CodegenError {
                    span: expr.span,
                    message: format!(
                        "method `.{method}(...)` not supported by codegen"
                    ),
                });
            }
            ExprKind::StructLit { name, fields, .. } => {
                self.compile_struct_lit(name, fields, expr.span, f)?;
            }
            ExprKind::Field { receiver, name } => {
                self.compile_field_access(receiver, name, f)?;
            }
        }
        Ok(())
    }

    fn compile_struct_lit(
        &mut self,
        name: &str,
        fields: &[ast::FieldInit],
        span: Span,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        // If this name is actually a named-field sum variant, dispatch
        // to the variant-constructor path instead of the struct-literal
        // path.
        if !self.cg.struct_layouts.contains_key(name) {
            if let Some(sum_name) = self.find_sum_for_variant(name) {
                return self.compile_named_variant_constructor(
                    &sum_name, name, fields, span, f,
                );
            }
        }

        let layout = self
            .cg
            .struct_layouts
            .get(name)
            .ok_or_else(|| CodegenError {
                span,
                message: format!("no layout for struct `{name}`"),
            })?;
        let slot = *self
            .slot_for_span
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
        &mut self,
        receiver: &Expr,
        field: &str,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        let recv_ty = self.infer_expr_ty(receiver)?;
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
            Some(e) => self.infer_expr_ty(e),
            None => Ok(Ty::Unit),
        }
    }

    // ----------------------------------------------------------------------
    // For loops
    // ----------------------------------------------------------------------

    /// Compile `for binder in range(start, end) { body }` as a counted
    /// Wasm loop. Recognizes only `range(start, end)` as the iterator
    /// expression for now; general iterators will come later.
    fn compile_for_range(
        &mut self,
        binder: &str,
        iter: &Expr,
        body: &ast::Block,
        span: Span,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        // Verify the iterator is `range(start, end)`.
        let (start_expr, end_expr) = match &iter.kind {
            ExprKind::Call { callee, args }
                if matches!(&callee.kind, ExprKind::Ident(n) if n == "range")
                    && args.len() == 2 =>
            {
                (&args[0].value, &args[1].value)
            }
            _ => {
                return Err(CodegenError {
                    span: iter.span,
                    message: "for-loop codegen only supports `range(start, end)` as iterator"
                        .into(),
                });
            }
        };

        let binder_slot = *self
            .slot_for_span
            .get(&span.start)
            .expect("walk reserved binder slot");
        let aux = self.aux_slots.get(&span.start).cloned().unwrap();
        let counter_slot = aux[0];
        let end_slot = aux[1];

        // counter = start
        self.compile_expr(start_expr, f)?;
        f.instruction(&Instruction::LocalSet(counter_slot));

        // end = end
        self.compile_expr(end_expr, f)?;
        f.instruction(&Instruction::LocalSet(end_slot));

        // block $break
        //   loop $continue
        //     if counter >= end { br $break }
        //     binder = counter
        //     <body>
        //     counter = counter + 1
        //     br $continue
        //   end
        // end
        f.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty));
        f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));

        // Break test: counter >= end → br 1 (exit outer block)
        f.instruction(&Instruction::LocalGet(counter_slot));
        f.instruction(&Instruction::LocalGet(end_slot));
        f.instruction(&Instruction::I32GeS);
        f.instruction(&Instruction::BrIf(1));

        // Bind the element variable.
        self.push_scope();
        f.instruction(&Instruction::LocalGet(counter_slot));
        f.instruction(&Instruction::LocalSet(binder_slot));
        self.declare(binder.to_string(), binder_slot);

        // Body.
        self.compile_block(body, f)?;

        self.pop_scope();

        // Increment counter.
        f.instruction(&Instruction::LocalGet(counter_slot));
        f.instruction(&Instruction::I32Const(1));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalSet(counter_slot));

        // Jump back.
        f.instruction(&Instruction::Br(0));

        f.instruction(&Instruction::End); // end loop
        f.instruction(&Instruction::End); // end block

        Ok(())
    }

    // ----------------------------------------------------------------------
    // Sum-type construction
    // ----------------------------------------------------------------------

    /// Emit a sum-value construction for the shape `Ok(v)` / `Err(e)` /
    /// `Some(v)` — a one-field variant with the given tag. Allocates a
    /// payload block big enough for the field, stores the value, then
    /// allocates the 8-byte sum block (tag + payload pointer) and leaves
    /// the sum pointer on the stack.
    fn compile_single_field_variant(
        &mut self,
        tag: u32,
        value: &Expr,
        call_span: Span,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        let val_ty = self.infer_expr_ty(value)?;
        let payload_size = sizeof(&val_ty).max(4);
        let wasm_ty = wasm_val_type(&val_ty, value.span)?;
        let align_log2 = alignof(&val_ty).trailing_zeros();
        let aux = self.aux_slots.get(&call_span.start).cloned().unwrap();
        let payload_slot = aux[0];
        let sum_slot = aux[1];
        self.emit_sum_alloc_one(
            tag,
            value,
            wasm_ty,
            payload_size,
            align_log2,
            payload_slot,
            sum_slot,
            f,
        )
    }

    /// Emit a `None` constructor: 8-byte sum block with tag=0 and
    /// payload_ptr=0.
    fn compile_none_constructor(
        &mut self,
        span: Span,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        let sum_slot = self.aux_slots.get(&span.start).unwrap()[0];
        self.emit_sum_block_no_payload(0, sum_slot, f);
        Ok(())
    }

    /// Emit a zero-payload variant constructor (e.g. `type T = | A | B`,
    /// called as the bare identifier `A`). Just tag + null payload.
    fn compile_zero_variant_constructor(
        &mut self,
        sum_name: &str,
        variant_name: &str,
        span: Span,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        let tag = self
            .variant_tag(&Ty::User(sum_name.to_string()), variant_name)
            .ok_or_else(|| CodegenError {
                span,
                message: format!("unknown variant `{variant_name}` of `{sum_name}`"),
            })?;
        let sum_slot = self.aux_slots.get(&span.start).unwrap()[0];
        self.emit_sum_block_no_payload(tag, sum_slot, f);
        Ok(())
    }

    /// Emit a user variant constructor invoked as a call, e.g.
    /// `Circle(1.0)` (positional) or `NotFound("path.txt")`. Named-field
    /// variants go through the struct-literal path, not here.
    fn compile_named_variant_constructor(
        &mut self,
        sum_name: &str,
        variant_name: &str,
        fields: &[ast::FieldInit],
        span: Span,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        let scrut_ty = Ty::User(sum_name.to_string());
        let tag = self
            .variant_tag(&scrut_ty, variant_name)
            .expect("typechecker verified variant");
        let variant_fields = self
            .variant_field_types(&scrut_ty, variant_name)
            .unwrap_or_default();
        let layout = compute_struct_layout(&variant_fields)?;

        // Primary slot (slot_for_span) = payload pointer.
        // aux_slots[0] = sum pointer.
        let payload_slot = *self
            .slot_for_span
            .get(&span.start)
            .expect("walk reserved payload slot");
        let sum_slot = self
            .aux_slots
            .get(&span.start)
            .and_then(|v| v.first().copied())
            .expect("walk reserved sum slot");

        // Allocate payload block.
        self.emit_bump_alloc(layout.total_size, f);
        f.instruction(&Instruction::LocalSet(payload_slot));

        // Write each field (look up the user-provided initializer by name).
        for layout_field in &layout.fields {
            let init = fields
                .iter()
                .find(|fi| fi.name == layout_field.name)
                .ok_or_else(|| CodegenError {
                    span,
                    message: format!(
                        "internal: variant `{variant_name}` literal is missing field `{}`",
                        layout_field.name
                    ),
                })?;
            f.instruction(&Instruction::LocalGet(payload_slot));
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
                _ => unreachable!(),
            };
            f.instruction(&instr);
        }

        // Allocate sum block + write tag and payload pointer.
        self.emit_bump_alloc(8, f);
        f.instruction(&Instruction::LocalSet(sum_slot));

        f.instruction(&Instruction::LocalGet(sum_slot));
        f.instruction(&Instruction::I32Const(tag as i32));
        f.instruction(&Instruction::I32Store(I32_MEM_ARG));

        f.instruction(&Instruction::LocalGet(sum_slot));
        f.instruction(&Instruction::LocalGet(payload_slot));
        f.instruction(&Instruction::I32Store(MemArg {
            offset: 4,
            align: 2,
            memory_index: 0,
        }));

        f.instruction(&Instruction::LocalGet(sum_slot));
        Ok(())
    }

    fn compile_user_variant_constructor(
        &mut self,
        sum_name: &str,
        variant_name: &str,
        args: &[ast::Arg],
        span: Span,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        let scrut_ty = Ty::User(sum_name.to_string());
        let tag = self.variant_tag(&scrut_ty, variant_name).unwrap();
        let fields = self
            .variant_field_types(&scrut_ty, variant_name)
            .unwrap_or_default();
        let aux = self.aux_slots.get(&span.start).cloned().unwrap();
        let payload_slot = aux[0];
        let sum_slot = aux[1];

        if fields.is_empty() {
            self.emit_sum_block_no_payload(tag, sum_slot, f);
            return Ok(());
        }
        let layout = compute_struct_layout(&fields)?;

        // Allocate the payload block and write each positional arg in
        // the corresponding slot.
        self.emit_bump_alloc(layout.total_size, f);
        // Stack: [payload_ptr]. Stash it in the pre-allocated scratch.
        f.instruction(&Instruction::LocalSet(payload_slot));

        for (i, field) in layout.fields.iter().enumerate() {
            let arg = args.get(i).ok_or_else(|| CodegenError {
                span,
                message: format!(
                    "variant `{variant_name}` expects {} args, found {}",
                    layout.fields.len(),
                    args.len()
                ),
            })?;
            f.instruction(&Instruction::LocalGet(payload_slot));
            self.compile_expr(&arg.value, f)?;
            let mem_arg = MemArg {
                offset: field.offset as u64,
                align: field.align_log2,
                memory_index: 0,
            };
            let instr = match field.wasm_ty {
                ValType::I32 => Instruction::I32Store(mem_arg),
                ValType::I64 => Instruction::I64Store(mem_arg),
                ValType::F64 => Instruction::F64Store(mem_arg),
                _ => unreachable!(),
            };
            f.instruction(&instr);
        }

        // Allocate the 8-byte sum header into sum_slot.
        self.emit_bump_alloc(8, f);
        f.instruction(&Instruction::LocalSet(sum_slot));

        // *sum_ptr = tag
        f.instruction(&Instruction::LocalGet(sum_slot));
        f.instruction(&Instruction::I32Const(tag as i32));
        f.instruction(&Instruction::I32Store(I32_MEM_ARG));

        // *(sum_ptr + 4) = payload_ptr
        f.instruction(&Instruction::LocalGet(sum_slot));
        f.instruction(&Instruction::LocalGet(payload_slot));
        f.instruction(&Instruction::I32Store(MemArg {
            offset: 4,
            align: 2,
            memory_index: 0,
        }));

        // Leave sum_ptr on the stack.
        f.instruction(&Instruction::LocalGet(sum_slot));
        Ok(())
    }

    /// Emit an 8-byte sum block with the given tag and a null payload
    /// pointer, leaving the sum block pointer on the stack. Uses the
    /// pre-allocated `sum_slot` to stash the block pointer.
    fn emit_sum_block_no_payload(&mut self, tag: u32, sum_slot: u32, f: &mut Function) {
        self.emit_bump_alloc(8, f);
        f.instruction(&Instruction::LocalSet(sum_slot));

        // *sum_ptr = tag
        f.instruction(&Instruction::LocalGet(sum_slot));
        f.instruction(&Instruction::I32Const(tag as i32));
        f.instruction(&Instruction::I32Store(I32_MEM_ARG));

        // *(sum_ptr + 4) = 0
        f.instruction(&Instruction::LocalGet(sum_slot));
        f.instruction(&Instruction::I32Const(0));
        f.instruction(&Instruction::I32Store(MemArg {
            offset: 4,
            align: 2,
            memory_index: 0,
        }));

        f.instruction(&Instruction::LocalGet(sum_slot));
    }

    /// Emit a one-field-payload sum construction. Handles Ok/Err/Some
    /// (built-ins) and any future single-field user variants. Uses
    /// pre-allocated scratch slots.
    #[allow(clippy::too_many_arguments)]
    fn emit_sum_alloc_one(
        &mut self,
        tag: u32,
        value: &Expr,
        wasm_ty: ValType,
        payload_size: u32,
        align_log2: u32,
        payload_slot: u32,
        sum_slot: u32,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        // Allocate payload block (at least 4 bytes; round up to 4 if the
        // value is an i32-width type, or 8 for i64/f64).
        let padded = (payload_size + 3) & !3;
        self.emit_bump_alloc(padded, f);
        f.instruction(&Instruction::LocalSet(payload_slot));

        // *payload = value
        f.instruction(&Instruction::LocalGet(payload_slot));
        self.compile_expr(value, f)?;
        let mem_arg = MemArg {
            offset: 0,
            align: align_log2,
            memory_index: 0,
        };
        match wasm_ty {
            ValType::I32 => f.instruction(&Instruction::I32Store(mem_arg)),
            ValType::I64 => f.instruction(&Instruction::I64Store(mem_arg)),
            ValType::F64 => f.instruction(&Instruction::F64Store(mem_arg)),
            _ => unreachable!(),
        };

        // Allocate sum block.
        self.emit_bump_alloc(8, f);
        f.instruction(&Instruction::LocalSet(sum_slot));

        // *sum = tag
        f.instruction(&Instruction::LocalGet(sum_slot));
        f.instruction(&Instruction::I32Const(tag as i32));
        f.instruction(&Instruction::I32Store(I32_MEM_ARG));

        // *(sum + 4) = payload_ptr
        f.instruction(&Instruction::LocalGet(sum_slot));
        f.instruction(&Instruction::LocalGet(payload_slot));
        f.instruction(&Instruction::I32Store(MemArg {
            offset: 4,
            align: 2,
            memory_index: 0,
        }));

        // Leave sum on the stack.
        f.instruction(&Instruction::LocalGet(sum_slot));
        Ok(())
    }

    /// Bump-allocate `size` bytes and leave the original bump_ptr (the
    /// start of the new block) on the stack. Always rounds the bump
    /// pointer up to the next multiple of 4 afterwards so follow-on
    /// allocations stay i32-aligned.
    fn emit_bump_alloc(&mut self, size: u32, f: &mut Function) {
        f.instruction(&Instruction::GlobalGet(0)); // start pointer
        f.instruction(&Instruction::GlobalGet(0));
        f.instruction(&Instruction::I32Const(size as i32));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Const(3));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::I32Const(-4));
        f.instruction(&Instruction::I32And);
        f.instruction(&Instruction::GlobalSet(0));
    }

    // ----------------------------------------------------------------------
    // `?` operator
    // ----------------------------------------------------------------------

    fn compile_try(
        &mut self,
        inner: &Expr,
        span: Span,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        let inner_ty = self.infer_expr_ty(inner)?;
        let (err_tag, ok_payload_ty) = match inner_ty.clone() {
            Ty::Result(ok, _) => (1u32, *ok),
            Ty::Option(inner) => (0u32, *inner),
            _ => {
                return Err(CodegenError {
                    span,
                    message: format!(
                        "`?` requires a Result or Option, got {}",
                        inner_ty.display()
                    ),
                });
            }
        };

        // Evaluate the inner into a scratch slot we can re-inspect.
        let scratch = *self
            .slot_for_span
            .get(&span.start)
            .expect("walk pass reserves a scratch for `?`");
        self.compile_expr(inner, f)?;
        f.instruction(&Instruction::LocalSet(scratch));

        // if tag == err_tag { push error frame; return inner; }
        f.instruction(&Instruction::LocalGet(scratch));
        f.instruction(&Instruction::I32Load(I32_MEM_ARG));
        f.instruction(&Instruction::I32Const(err_tag as i32));
        f.instruction(&Instruction::I32Eq);
        f.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));

        // Push an error frame onto the global frame_chain (global 1).
        let frame_msg = format!(
            "  at {} (<source>:{}:{})",
            self.current_fn_name, span.line, span.col
        );
        if let Some(&msg_offset) = self.cg.string_offsets.get(&frame_msg) {
            let frame_slot = self.aux_slots.get(&span.start).unwrap()[0];

            // Allocate 8-byte frame { message: i32, next: i32 }.
            f.instruction(&Instruction::GlobalGet(0));
            f.instruction(&Instruction::LocalSet(frame_slot));
            f.instruction(&Instruction::GlobalGet(0));
            f.instruction(&Instruction::I32Const(8));
            f.instruction(&Instruction::I32Add);
            f.instruction(&Instruction::GlobalSet(0));

            // frame.message = msg_offset
            f.instruction(&Instruction::LocalGet(frame_slot));
            f.instruction(&Instruction::I32Const(msg_offset as i32));
            f.instruction(&Instruction::I32Store(I32_MEM_ARG));

            // frame.next = current frame_chain head
            f.instruction(&Instruction::LocalGet(frame_slot));
            f.instruction(&Instruction::GlobalGet(1));
            f.instruction(&Instruction::I32Store(MemArg {
                offset: 4,
                align: 2,
                memory_index: 0,
            }));

            // frame_chain = &frame
            f.instruction(&Instruction::LocalGet(frame_slot));
            f.instruction(&Instruction::GlobalSet(1));
        }

        // If we're in main, print the accumulated frames before
        // returning, because the `return` instruction below bypasses
        // the epilogue at the end of compile_fn.
        if self.current_fn_name == "main" {
            if let Some(pf_idx) = self.cg.print_frames_idx {
                f.instruction(&Instruction::Call(pf_idx));
                f.instruction(&Instruction::Drop);
            }
        }

        f.instruction(&Instruction::LocalGet(scratch));
        f.instruction(&Instruction::Return);
        f.instruction(&Instruction::End);

        // Happy path: load payload_ptr, load T, leave on stack.
        f.instruction(&Instruction::LocalGet(scratch));
        f.instruction(&Instruction::I32Load(MemArg {
            offset: 4,
            align: 2,
            memory_index: 0,
        }));
        let wasm_ty = wasm_val_type(&ok_payload_ty, span)?;
        let mem_arg = MemArg {
            offset: 0,
            align: alignof(&ok_payload_ty).trailing_zeros(),
            memory_index: 0,
        };
        match wasm_ty {
            ValType::I32 => f.instruction(&Instruction::I32Load(mem_arg)),
            ValType::I64 => f.instruction(&Instruction::I64Load(mem_arg)),
            ValType::F64 => f.instruction(&Instruction::F64Load(mem_arg)),
            _ => unreachable!(),
        };
        Ok(())
    }

    // ----------------------------------------------------------------------
    // match expressions (sum types only for now)
    // ----------------------------------------------------------------------

    fn compile_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[ast::MatchArm],
        span: Span,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        let scrut_ty = self.infer_expr_ty(scrutinee)?;
        if !matches!(scrut_ty, Ty::User(_) | Ty::Option(_) | Ty::Result(_, _)) {
            return Err(CodegenError {
                span: scrutinee.span,
                message: format!(
                    "codegen only supports match on sum types; got {}",
                    scrut_ty.display()
                ),
            });
        }

        let scrut_slot = *self
            .slot_for_span
            .get(&span.start)
            .expect("walk pass reserves a scrutinee slot");
        let payload_ptr_slot = self
            .aux_slots
            .get(&span.start)
            .and_then(|v| v.first().copied())
            .expect("walk pass reserves a payload pointer slot");

        // Evaluate the scrutinee once.
        self.compile_expr(scrutinee, f)?;
        f.instruction(&Instruction::LocalSet(scrut_slot));

        // Use the result type cached during the walk pass. The walk could
        // safely infer the first arm body's type because its pattern
        // bindings were in scope; emit can't do that here.
        let result_ty = self
            .expr_type_cache
            .get(&span.start)
            .cloned()
            .unwrap_or(Ty::Unit);
        let block_ty = wasm_encoder::BlockType::Result(wasm_val_type(
            &result_ty, span,
        )?);

        self.compile_match_arms(
            arms,
            0,
            scrut_slot,
            payload_ptr_slot,
            &scrut_ty,
            block_ty,
            f,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_match_arms(
        &mut self,
        arms: &[ast::MatchArm],
        idx: usize,
        scrut_slot: u32,
        payload_ptr_slot: u32,
        scrut_ty: &Ty,
        block_ty: wasm_encoder::BlockType,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        use ast::PatternKind;
        if idx >= arms.len() {
            // Exhaustive guarantee: unreachable.
            f.instruction(&Instruction::Unreachable);
            return Ok(());
        }
        let arm = &arms[idx];

        // Wildcard/binding arm: always matches; emit body directly.
        let always = matches!(
            arm.pattern.kind,
            PatternKind::Wildcard | PatternKind::Binding(_)
        );
        if always {
            self.push_scope();
            if let PatternKind::Binding(name) = &arm.pattern.kind {
                let slot = *self
                    .slot_for_span
                    .get(&arm.pattern.span.start)
                    .expect("walk reserves pattern slots");
                f.instruction(&Instruction::LocalGet(scrut_slot));
                f.instruction(&Instruction::LocalSet(slot));
                self.declare(name.clone(), slot);
            }
            self.compile_expr(&arm.body, f)?;
            self.pop_scope();
            return Ok(());
        }

        // Variant arm: compare tag, emit matched body on true, recurse on
        // false.
        let PatternKind::Variant {
            name: variant_name,
            payload,
        } = &arm.pattern.kind
        else {
            return Err(CodegenError {
                span: arm.pattern.span,
                message: "codegen only supports wildcard / binding / variant patterns in match"
                    .into(),
            });
        };

        let tag = self
            .variant_tag(scrut_ty, variant_name)
            .ok_or_else(|| CodegenError {
                span: arm.pattern.span,
                message: format!("unknown variant `{variant_name}`"),
            })?;

        // Test: scrut_slot.tag == tag
        f.instruction(&Instruction::LocalGet(scrut_slot));
        f.instruction(&Instruction::I32Load(I32_MEM_ARG));
        f.instruction(&Instruction::I32Const(tag as i32));
        f.instruction(&Instruction::I32Eq);
        f.instruction(&Instruction::If(block_ty));

        // Matched branch.
        self.push_scope();
        if payload.is_some() {
            let fields = self
                .variant_field_types(scrut_ty, variant_name)
                .unwrap_or_default();
            let layout = compute_struct_layout(&fields)?;
            self.emit_pattern_binds(
                payload.as_ref().unwrap(),
                &layout,
                scrut_slot,
                payload_ptr_slot,
                f,
            )?;
        }
        self.compile_expr(&arm.body, f)?;
        self.pop_scope();

        f.instruction(&Instruction::Else);
        self.compile_match_arms(
            arms,
            idx + 1,
            scrut_slot,
            payload_ptr_slot,
            scrut_ty,
            block_ty,
            f,
        )?;
        f.instruction(&Instruction::End);

        Ok(())
    }

    fn emit_pattern_binds(
        &mut self,
        payload: &ast::VariantPatPayload,
        layout: &StructLayout,
        scrut_slot: u32,
        payload_slot: u32,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        use ast::{PatternKind, VariantPatPayload};

        // Load payload_ptr into the pre-allocated scratch so we can
        // rebase each field load off of it.
        f.instruction(&Instruction::LocalGet(scrut_slot));
        f.instruction(&Instruction::I32Load(MemArg {
            offset: 4,
            align: 2,
            memory_index: 0,
        }));
        f.instruction(&Instruction::LocalSet(payload_slot));

        // Helper closure-ish: for a given layout field and a sub-pattern,
        // if the sub-pattern binds a name, load the field and store it.
        let sub_pats: Vec<(&StructFieldLayout, &ast::Pattern)> = match payload {
            VariantPatPayload::Named(pfs) => pfs
                .iter()
                .filter_map(|pf| {
                    layout
                        .fields
                        .iter()
                        .find(|lf| lf.name == pf.name)
                        .map(|lf| (lf, &pf.pattern))
                })
                .collect(),
            VariantPatPayload::Positional(pats) => pats
                .iter()
                .enumerate()
                .filter_map(|(i, pat)| layout.fields.get(i).map(|lf| (lf, pat)))
                .collect(),
        };

        for (lf, pat) in sub_pats {
            match &pat.kind {
                PatternKind::Binding(name) => {
                    let slot = *self
                        .slot_for_span
                        .get(&pat.span.start)
                        .expect("walk reserves pattern slots");
                    // local = *(payload_ptr + offset)
                    f.instruction(&Instruction::LocalGet(payload_slot));
                    let mem_arg = MemArg {
                        offset: lf.offset as u64,
                        align: lf.align_log2,
                        memory_index: 0,
                    };
                    match lf.wasm_ty {
                        ValType::I32 => f.instruction(&Instruction::I32Load(mem_arg)),
                        ValType::I64 => f.instruction(&Instruction::I64Load(mem_arg)),
                        ValType::F64 => f.instruction(&Instruction::F64Load(mem_arg)),
                        _ => unreachable!(),
                    };
                    f.instruction(&Instruction::LocalSet(slot));
                    self.declare(name.clone(), slot);
                }
                PatternKind::Wildcard | PatternKind::Literal(_) => {
                    // No binding to emit. Literal patterns on variant
                    // sub-fields aren't currently matched against; the
                    // typechecker guarantees shape.
                }
                PatternKind::Variant { .. } => {
                    // Nested variant patterns would require recursive
                    // destructuring and a deeper payload walk. Defer.
                    return Err(CodegenError {
                        span: pat.span,
                        message: "nested variant patterns are not supported by codegen yet"
                            .into(),
                    });
                }
            }
        }
        Ok(())
    }

    fn compile_binary(
        &mut self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        f: &mut Function,
    ) -> Result<(), CodegenError> {
        // Short-circuit logical operators.
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.compile_logical(op, lhs, rhs, f);
        }

        let lt = self.infer_expr_ty(lhs)?;

        // String + string → call into the auto-emitted string_concat
        // helper. Matches the typechecker's special case.
        if matches!(op, BinOp::Add) && matches!(lt, Ty::String) {
            self.compile_expr(lhs, f)?;
            self.compile_expr(rhs, f)?;
            f.instruction(&Instruction::Call(self.cg.num_imports));
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
        &mut self,
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
        Ty::I32
        | Ty::U32
        | Ty::Bool
        | Ty::Unit
        | Ty::String
        | Ty::User(_)
        | Ty::Option(_)
        | Ty::Result(_, _)
        | Ty::List(_) => ValType::I32,
        Ty::I64 | Ty::U64 => ValType::I64,
        Ty::F64 => ValType::F64,
        Ty::Error => {
            return Err(CodegenError {
                span,
                message: "internal: Ty::Error reached codegen".into(),
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

    // --- Sum types: Option / Result / match / ? -------------------------

    #[test]
    fn run_some_none_construction() {
        // Option<i32> constructors allocate; we can't read the value back
        // without match, so just verify the pointers aren't 0 (Some) and
        // that None returns a non-null header either.
        let src = r#"
            fn one(): Option<i32> { Some(1) }
            fn empty(): Option<i32> { None }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
    }

    #[test]
    fn run_match_option_extracts_value() {
        let src = r#"
            fn unwrap_or(o: Option<i32>, default: i32): i32 {
                match o {
                    Some(n) => n,
                    None => default,
                }
            }

            fn with_some(): i32 { unwrap_or(Some(42), 0) }
            fn with_none(): i32 { unwrap_or(None, 99) }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
        assert_eq!(run_i32(&wasm, "with_some"), 42);
        assert_eq!(run_i32(&wasm, "with_none"), 99);
    }

    #[test]
    fn run_match_result_extracts_value() {
        let src = r#"
            fn safe_div(a: i32, b: i32): Result<i32, i32> {
                if b == 0 { Err(1) } else { Ok(a / b) }
            }

            fn test_ok(): i32 {
                match safe_div(10, 2) {
                    Ok(v) => v,
                    Err(_) => -1,
                }
            }

            fn test_err(): i32 {
                match safe_div(10, 0) {
                    Ok(v) => v,
                    Err(code) => code * 100,
                }
            }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "test_ok"), 5);
        assert_eq!(run_i32(&wasm, "test_err"), 100);
    }

    #[test]
    fn run_try_operator_happy_path() {
        let src = r#"
            fn first(): Result<i32, i32> { Ok(10) }
            fn second(): Result<i32, i32> { Ok(20) }

            fn sum(): Result<i32, i32> {
                let a = first()?
                let b = second()?
                Ok(a + b)
            }

            fn run(): i32 {
                match sum() {
                    Ok(v) => v,
                    Err(_) => -1,
                }
            }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "run"), 30);
    }

    #[test]
    fn run_try_operator_short_circuits_on_err() {
        let src = r#"
            fn fail(): Result<i32, i32> { Err(42) }

            fn chain(): Result<i32, i32> {
                let x = fail()?
                Ok(x + 100)
            }

            fn run(): i32 {
                match chain() {
                    Ok(_) => 0,
                    Err(code) => code,
                }
            }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "run"), 42);
    }

    #[test]
    fn run_user_sum_type_with_positional_variant() {
        let src = r#"
            type Token =
                | Number(i32)
                | Word(string)

            fn number_val(): i32 {
                let t = Number(7)
                match t {
                    Number(n) => n,
                    _ => -1,
                }
            }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "number_val"), 7);
    }

    #[test]
    fn run_user_sum_type_with_named_fields() {
        let src = r#"
            type Shape =
                | Circle { radius: i32 }
                | Rectangle { width: i32, height: i32 }

            fn area(s: Shape): i32 {
                match s {
                    Circle { radius: r } => r * r * 3,
                    Rectangle { width: w, height: h } => w * h,
                }
            }

            fn circle_area(): i32 { area(Circle { radius: 2 }) }
            fn rect_area(): i32 { area(Rectangle { width: 3, height: 4 }) }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
        assert_eq!(run_i32(&wasm, "circle_area"), 12);
        assert_eq!(run_i32(&wasm, "rect_area"), 12);
    }

    #[test]
    fn run_user_sum_type_with_zero_payload_variants() {
        let src = r#"
            type Light = | Red | Yellow | Green

            fn code(l: Light): i32 {
                match l {
                    Red => 1,
                    Yellow => 2,
                    Green => 3,
                }
            }

            fn red_code(): i32 { code(Red) }
            fn green_code(): i32 { code(Green) }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "red_code"), 1);
        assert_eq!(run_i32(&wasm, "green_code"), 3);
    }

    #[test]
    fn run_sibling_arms_reuse_binding_name() {
        // Two arms bind `h`; the scope refactor needs to put them in
        // separate scopes so they don't collide.
        let src = r#"
            type Rect = | A { h: i32 } | B { h: i32 }

            fn height(r: Rect): i32 {
                match r {
                    A { h: h } => h + 1,
                    B { h: h } => h + 2,
                }
            }

            fn run(): i32 { height(A { h: 10 }) + height(B { h: 100 }) }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "run"), 11 + 102);
    }

    // --- For loops --------------------------------------------------------

    #[test]
    fn run_sum_of_squares_with_for_loop() {
        let src = r#"
            fn sum_of_squares(n: i32): i32 {
                var total: i32 = 0
                for i in range(1, n + 1) {
                    total = total + i * i
                }
                total
            }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");
        // 1² + 2² + 3² + 4² + 5² = 1 + 4 + 9 + 16 + 25 = 55
        assert_eq!(run_i32_i32(&wasm, "sum_of_squares", 5), 55);
        assert_eq!(run_i32_i32(&wasm, "sum_of_squares", 0), 0);
        assert_eq!(run_i32_i32(&wasm, "sum_of_squares", 1), 1);
    }

    #[test]
    fn run_triangle_number_for_loop() {
        let src = r#"
            fn triangle(n: i32): i32 {
                var total: i32 = 0
                for i in range(1, n + 1) {
                    total = total + i
                }
                total
            }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32_i32(&wasm, "triangle", 10), 55);
        assert_eq!(run_i32_i32(&wasm, "triangle", 100), 5050);
    }

    #[test]
    fn run_empty_range_does_nothing() {
        let src = r#"
            fn f(): i32 {
                var total: i32 = 42
                for i in range(5, 5) {
                    total = total + i
                }
                total
            }
        "#;
        let wasm = compile_src(src);
        assert_eq!(run_i32(&wasm, "f"), 42);
    }

    // --- WASI / io.println ------------------------------------------------

    #[test]
    fn run_hello_world_via_wasi() {
        let src = r#"
            import std/io

            fn main(): i32 {
                io.println("hello, world")
                0
            }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");

        // Run via wasmtime-wasi (preview 1) and capture stdout.
        use wasmtime::*;
        let engine = Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        let mut linker = Linker::new(&engine);

        let stdout_pipe = wasmtime_wasi::p2::pipe::MemoryOutputPipe::new(4096);
        let wasi = wasmtime_wasi::WasiCtxBuilder::new()
            .stdout(stdout_pipe.clone())
            .build_p1();
        let mut store = Store::new(&engine, wasi);

        wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |s| s).unwrap();
        let instance = linker.instantiate(&mut store, &module).unwrap();
        let f = instance.get_typed_func::<(), i32>(&mut store, "main").unwrap();
        let ret = f.call(&mut store, ()).unwrap();
        assert_eq!(ret, 0);

        drop(store);
        let output_bytes = stdout_pipe.try_into_inner().unwrap();
        let output = String::from_utf8(output_bytes.into()).unwrap();
        assert_eq!(output, "hello, world\n");
    }

    // --- Stdlib: int.to_string_i32 -----------------------------------------

    #[test]
    fn run_int_to_string_positive() {
        let src = r#"
            import std/io
            fn main(): i32 {
                io.println(int.to_string_i32(42))
                io.println(int.to_string_i32(0))
                io.println(int.to_string_i32(-7))
                io.println(int.to_string_i32(12345))
                0
            }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("validate");

        use wasmtime::*;
        let engine = Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        let mut linker = Linker::new(&engine);
        let stdout = wasmtime_wasi::p2::pipe::MemoryOutputPipe::new(4096);
        let wasi = wasmtime_wasi::WasiCtxBuilder::new()
            .stdout(stdout.clone())
            .build_p1();
        let mut store = Store::new(&engine, wasi);
        wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |s| s).unwrap();
        let inst = linker.instantiate(&mut store, &module).unwrap();
        let f = inst.get_typed_func::<(), i32>(&mut store, "main").unwrap();
        f.call(&mut store, ()).unwrap();
        drop(store);

        let out = String::from_utf8(stdout.try_into_inner().unwrap().into()).unwrap();
        assert_eq!(out, "42\n0\n-7\n12345\n");
    }

    // --- Error frames -----------------------------------------------------

    #[test]
    fn run_error_frames_print_on_err() {
        let src = r#"
            import std/io

            fn inner(): Result<i32, i32> {
                Err(42)
            }

            fn middle(): Result<i32, i32> {
                let x = inner()?
                Ok(x)
            }

            fn main(): Result<i32, i32> {
                let y = middle()?
                Ok(y)
            }
        "#;
        let wasm = compile_src(src);
        wasmparser::validate(&wasm).expect("module must validate");

        use wasmtime::*;
        let engine = Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        let mut linker = Linker::new(&engine);

        let stdout_pipe = wasmtime_wasi::p2::pipe::MemoryOutputPipe::new(4096);
        let wasi = wasmtime_wasi::WasiCtxBuilder::new()
            .stdout(stdout_pipe.clone())
            .build_p1();
        let mut store = Store::new(&engine, wasi);
        wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |s| s).unwrap();

        let instance = linker.instantiate(&mut store, &module).unwrap();
        let f = instance.get_typed_func::<(), i32>(&mut store, "main").unwrap();
        let _ret = f.call(&mut store, ()).unwrap();

        drop(store);
        let output_bytes = stdout_pipe.try_into_inner().unwrap();
        let output = String::from_utf8(output_bytes.into()).unwrap();

        // Should have two frames: one from middle's `?` and one from main's `?`.
        assert!(
            output.contains("at middle"),
            "expected frame from middle, got: {output}"
        );
        assert!(
            output.contains("at main"),
            "expected frame from main, got: {output}"
        );
    }

    #[test]
    fn run_error_frames_empty_on_ok() {
        // When everything succeeds, no frames should print.
        let src = r#"
            import std/io

            fn inner(): Result<i32, i32> { Ok(1) }
            fn main(): Result<i32, i32> {
                let x = inner()?
                Ok(x)
            }
        "#;
        let wasm = compile_src(src);

        use wasmtime::*;
        let engine = Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        let mut linker = Linker::new(&engine);

        let stdout_pipe = wasmtime_wasi::p2::pipe::MemoryOutputPipe::new(4096);
        let wasi = wasmtime_wasi::WasiCtxBuilder::new()
            .stdout(stdout_pipe.clone())
            .build_p1();
        let mut store = Store::new(&engine, wasi);
        wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |s| s).unwrap();

        let instance = linker.instantiate(&mut store, &module).unwrap();
        let f = instance.get_typed_func::<(), i32>(&mut store, "main").unwrap();
        f.call(&mut store, ()).unwrap();

        drop(store);
        let output_bytes = stdout_pipe.try_into_inner().unwrap();
        let output = String::from_utf8(output_bytes.into()).unwrap();
        assert!(output.is_empty(), "expected no output on Ok, got: {output}");
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
