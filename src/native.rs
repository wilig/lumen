//! Native backend via Cranelift. Produces a relocatable object file (.o)
//! that links with `cc` to produce a standalone executable.
//!
//! Same AST and type checker as the Wasm backend; only the codegen
//! differs. Uses `cranelift-frontend`'s `FunctionBuilder` for automatic
//! SSA construction.
//!
//! Memory model: same bump allocator as the Wasm backend, but in
//! process memory (a large `static mut` buffer). String representation
//! is identical: `[len: i32 | bytes...]` pointed to by an i64 address.
//! Structs and sum types use the same layouts. IO goes through libc
//! `write(2)` instead of WASI `fd_write`.

use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{AbiParam, BlockArg, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Type as CLType, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::ast::{self, BinOp, Expr, ExprKind, FnDecl, Item, StmtKind, UnaryOp};
use crate::lexer::IntSuffix;
use crate::span::Span;
use crate::types::{ModuleInfo, Ty, TypeInfo};

/// Compile a type-checked module to a native object file (bytes).
pub fn compile_native(
    module: &ast::Module,
    info: &mut ModuleInfo,
    imported_modules: &[(&str, &ast::Module)],
    module_paths: &HashMap<String, String>,
    debug: bool,
    source_path: &str,
) -> Result<Vec<u8>, NativeError> {
    let mut cg = NativeCodegen::new(info)?;
    cg.debug_mode = debug;
    cg.source_path = source_path.to_string();
    cg.dwarf = crate::dwarf::DwarfBuilder::new(source_path);
    for (mod_name, path) in module_paths {
        cg.dwarf.add_module_file(mod_name, path);
    }
    cg.compile_module(module, imported_modules)?;
    Ok(cg.finish())
}

#[derive(Debug)]
pub struct NativeError {
    pub span: Span,
    pub message: String,
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.span.line, self.span.col, self.message)
    }
}

impl std::error::Error for NativeError {}

// The pointer type for the target — i64 on 64-bit.
const PTR: CLType = cl_types::I64;

/// Where formatted output should go.
#[derive(Clone, Copy)]
enum PrintTarget {
    /// `debug.print` — writes to stderr, quotes strings (so `"hi"` is
    /// distinguishable from the bareword `hi`).
    Stderr,
    /// `io.println` — writes to stdout, prints strings as raw content.
    Stdout,
    /// String interpolation — appends to the given strbuf pointer.
    StrBuf(Value),
}

/// FuncIds for one of the formatting helper sets (stderr / stdout / strbuf).
struct FmtFuncs {
    i32: FuncId,
    i64: FuncId,
    f64: FuncId,
    bool: FuncId,
    str: FuncId,
    raw: FuncId,
    /// First argument prepended to every helper call (the strbuf ptr).
    /// `None` for stderr/stdout (those helpers take no extra arg).
    leading: Option<Value>,
}

// ---------------------------------------------------------------------------
// Native codegen state
// ---------------------------------------------------------------------------

struct NativeCodegen<'a> {
    info: &'a mut ModuleInfo,
    obj: ObjectModule,

    /// Lumen fn name → Cranelift FuncId.
    fn_ids: HashMap<String, FuncId>,
    /// Core infrastructure FuncIds (not module-managed).
    libc_malloc: FuncId,
    libc_free: FuncId,
    helper_concat: FuncId,
    helper_println: FuncId,
    helper_utf8_encode: FuncId,
    helper_assert: FuncId,
    helper_print_frames: FuncId,
    helper_rc_alloc: FuncId,
    helper_rc_incr: FuncId,
    helper_rc_decr: FuncId,
    rt_send: FuncId,
    rt_ask: FuncId,
    rt_drain: FuncId,
    // debug.print primitives
    debug_i32: FuncId,
    debug_i64: FuncId,
    debug_f64: FuncId,
    debug_bool: FuncId,
    debug_str: FuncId,
    debug_raw: FuncId,
    debug_newline: FuncId,
    io_i32: FuncId,
    io_i64: FuncId,
    io_f64: FuncId,
    io_bool: FuncId,
    io_str: FuncId,
    io_raw: FuncId,
    io_newline: FuncId,
    strbuf_new: FuncId,
    strbuf_finish: FuncId,
    strbuf_raw: FuncId,
    strbuf_str: FuncId,
    strbuf_i32: FuncId,
    strbuf_i64: FuncId,
    strbuf_f64: FuncId,
    strbuf_bool: FuncId,
    debug_init: FuncId,
    debug_push: FuncId,
    debug_pop: FuncId,
    string_eq: FuncId,
    debug_data_counter: usize,

    /// Per-actor dispatch function IDs (actor_name → FuncId).
    dispatch_fns: HashMap<String, FuncId>,

    /// DataId for the heap buffer (a large static byte array).
    heap_data: DataId,
    /// DataId for the bump pointer (an 8-byte mutable global).
    bump_ptr_data: DataId,
    /// DataId for the frame chain pointer.
    frame_chain_data: DataId,

    /// Interned string literals: content → DataId.
    string_data: HashMap<String, DataId>,
    /// Top-level `let`/`var` bindings materialized as static data.
    /// name → (data id, Ty). Keys include bindings from both the
    /// user's module and imported modules; the user's ones are
    /// defined (with initial value bytes), imported ones are Linkage::Import.
    global_data: HashMap<String, (DataId, Ty)>,

    /// Lambda FuncIds keyed by source span (line, col).
    lambda_ids: HashMap<(u32, u32), FuncId>,
    /// Lambda FnSigs (lambdas aren't in ModuleInfo, so we store sigs here).
    lambda_sigs: HashMap<String, crate::types::FnSig>,

    /// Imported module function FuncIds: C link name → FuncId.
    module_fn_ids: HashMap<String, FuncId>,
    /// Unqualified-name → FuncId scoped per imported module. When
    /// compiling a module's body the lookup checks here first
    /// (using FnEmitter::current_module) before falling back to the
    /// global fn_ids — that's how `len(xs)` inside std/list resolves
    /// to lumen_list_len even when std/map (which also exports `len`)
    /// is imported in the same program.
    local_fn_ids: HashMap<String, HashMap<String, FuncId>>,

    /// Uses WASI / io module.
    uses_io: bool,
    /// Debug mode: emit frame chain push/pop for stack traces.
    debug_mode: bool,
    /// Path of the source file being compiled (for assert messages).
    source_path: String,
    /// Generic fn ASTs, keyed by fn name. Generic fns aren't compiled
    /// directly; each call site requests a monomorphization with concrete
    /// type arguments, recorded in `monomorph_queue` and compiled later.
    generic_templates: HashMap<String, ast::FnDecl>,
    /// Pending monomorphizations to compile after the main pass:
    /// (mangled_name, func_id, substituted FnSig, substitution, original AST).
    monomorph_queue: Vec<MonomorphRequest>,
    /// Mangled names of monomorphizations already declared (so a second
    /// call site reusing the same instantiation just calls the existing
    /// FuncId rather than re-declaring).
    monomorph_done: HashSet<String>,
    /// DWARF debug info builder. Populated as user fns are defined and
    /// flushed to the object in `finish()`. See src/dwarf.rs (lumen-v3w).
    dwarf: crate::dwarf::DwarfBuilder,
}

struct MonomorphRequest {
    mangled_name: String,
    func_id: FuncId,
    sig: crate::types::FnSig,
    subs: HashMap<String, Ty>,
    decl: ast::FnDecl,
    /// When the original generic fn lives inside an imported module,
    /// the module's name. Threads through so the monomorphized body's
    /// unqualified calls resolve via the right local_fn_ids.
    current_module: Option<String>,
}

impl<'a> NativeCodegen<'a> {
    fn new(info: &'a mut ModuleInfo) -> Result<Self, NativeError> {
        let isa = cranelift_native::builder()
            .map_err(|e| NativeError {
                span: Span::DUMMY,
                message: format!("cranelift ISA: {e}"),
            })?
            .finish({
                let mut b = settings::builder();
                // Disable the verifier — it panics on valid but complex
                // IR patterns (nested if-without-else with return).
                b.set("enable_verifier", "false").ok();
                b.set("opt_level", "speed").ok();
                // Emit frame pointers and unwind tables so DWARF
                // debug info and `gdb bt` work out of the box.
                b.set("unwind_info", "true").ok();
                b.set("preserve_frame_pointers", "true").ok();
                // AArch64 (macOS) requires PIC; cranelift-object only handles
                // GOT-based aarch64 relocations, not direct ADRP ones.
                if cfg!(target_arch = "aarch64") {
                    b.set("is_pic", "true").ok();
                    b.set("use_colocated_libcalls", "false").ok();
                }
                settings::Flags::new(b)
            })
            .map_err(|e| NativeError {
                span: Span::DUMMY,
                message: format!("cranelift flags: {e}"),
            })?;

        let builder = ObjectBuilder::new(
            isa,
            "lumen_module",
            cranelift_module::default_libcall_names(),
        )
        .map_err(|e| NativeError {
            span: Span::DUMMY,
            message: format!("object builder: {e}"),
        })?;
        let mut obj = ObjectModule::new(builder);

        // malloc(size) -> ptr
        let mut malloc_sig = obj.make_signature();
        malloc_sig.params.push(AbiParam::new(PTR));
        malloc_sig.returns.push(AbiParam::new(PTR));
        let libc_malloc = obj
            .declare_function("malloc", Linkage::Import, &malloc_sig)
            .unwrap();

        // free(ptr)
        let mut free_sig = obj.make_signature();
        free_sig.params.push(AbiParam::new(PTR));
        let libc_free = obj
            .declare_function("free", Linkage::Import, &free_sig)
            .unwrap();

        // Heap: a 1MB static buffer. Enough for the prototype.
        let heap_data = obj.declare_data("lumen_heap", Linkage::Local, true, false).unwrap();
        let bump_ptr_data = obj.declare_data("lumen_bump", Linkage::Local, true, false).unwrap();
        let frame_chain_data =
            obj.declare_data("lumen_frame_chain", Linkage::Export, true, false).unwrap();

        // Define heap: 1MB of zeros.
        let mut desc = DataDescription::new();
        desc.define_zeroinit(1024 * 1024);
        obj.define_data(heap_data, &desc).unwrap();

        // Define bump_ptr: 8 bytes, initially 0 (will be patched after
        // string literals are placed).
        let mut desc = DataDescription::new();
        desc.define(vec![0u8; 8].into_boxed_slice());
        obj.define_data(bump_ptr_data, &desc).unwrap();

        // Define frame_chain: 8 bytes, initially 0.
        let mut desc = DataDescription::new();
        desc.define(vec![0u8; 8].into_boxed_slice());
        obj.define_data(frame_chain_data, &desc).unwrap();

        // String helpers (now in rt.c, linked externally).
        let mut concat_sig = obj.make_signature();
        concat_sig.params.push(AbiParam::new(PTR));
        concat_sig.params.push(AbiParam::new(PTR));
        concat_sig.returns.push(AbiParam::new(PTR));
        let helper_concat = obj
            .declare_function("lumen_concat", Linkage::Import, &concat_sig)
            .unwrap();

        let mut println_sig = obj.make_signature();
        println_sig.params.push(AbiParam::new(PTR));
        let helper_println = obj
            .declare_function("lumen_println", Linkage::Import, &println_sig)
            .unwrap();

        // lumen_utf8_encode(cp: i32, out: ptr) -> i32 bytes_written
        let mut utf8_enc_sig = obj.make_signature();
        utf8_enc_sig.params.push(AbiParam::new(cl_types::I32));
        utf8_enc_sig.params.push(AbiParam::new(PTR));
        utf8_enc_sig.returns.push(AbiParam::new(cl_types::I32));
        let helper_utf8_encode = obj
            .declare_function("lumen_utf8_encode", Linkage::Import, &utf8_enc_sig)
            .unwrap();

        // lumen_assert(cond: i32, msg: ptr, file: ptr, line: i32, col: i32, debug: i32)
        let mut assert_sig = obj.make_signature();
        assert_sig.params.push(AbiParam::new(cl_types::I32)); // cond
        assert_sig.params.push(AbiParam::new(PTR));           // msg (or 0)
        assert_sig.params.push(AbiParam::new(PTR));           // file
        assert_sig.params.push(AbiParam::new(cl_types::I32)); // line
        assert_sig.params.push(AbiParam::new(cl_types::I32)); // col
        assert_sig.params.push(AbiParam::new(cl_types::I32)); // debug_mode
        let helper_assert = obj
            .declare_function("lumen_assert", Linkage::Import, &assert_sig)
            .unwrap();

        // rc_alloc(size: i64) -> ptr: dispatches through the current
        // allocator (malloc + rc header by default, or a bump-allocated
        // region inside an `arena { ... }` block). Defined in rt.c so
        // the allocator vtable lives in one place.
        let mut rc_alloc_sig = obj.make_signature();
        rc_alloc_sig.params.push(AbiParam::new(PTR));
        rc_alloc_sig.returns.push(AbiParam::new(PTR));
        let helper_rc_alloc = obj
            .declare_function("lumen_rc_alloc", Linkage::Import, &rc_alloc_sig)
            .unwrap();

        // rc_incr(ptr): if ptr in heap, rc++
        let mut rc_incr_sig = obj.make_signature();
        rc_incr_sig.params.push(AbiParam::new(PTR));
        let helper_rc_incr = obj
            .declare_function("lumen_rc_incr", Linkage::Local, &rc_incr_sig)
            .unwrap();

        // rc_decr(ptr): if ptr in heap, rc--; if 0, free.
        // Exported so C runtime helpers (e.g. lumen_map_set when displacing
        // an old key/value) can decrement Lumen-owned references.
        let mut rc_decr_sig = obj.make_signature();
        rc_decr_sig.params.push(AbiParam::new(PTR));
        let helper_rc_decr = obj
            .declare_function("lumen_rc_decr", Linkage::Export, &rc_decr_sig)
            .unwrap();

        // --- Actor runtime functions (from runtime/rt.c) ---
        // lumen_rt_send(cell: ptr, dispatch: ptr, kind: i32, arg0: i64)
        let mut rt_send_sig = obj.make_signature();
        rt_send_sig.params.push(AbiParam::new(PTR)); // cell
        rt_send_sig.params.push(AbiParam::new(PTR)); // dispatch fn ptr
        rt_send_sig.params.push(AbiParam::new(cl_types::I32)); // msg_kind
        rt_send_sig.params.push(AbiParam::new(cl_types::I64)); // arg0
        let rt_send = obj
            .declare_function("lumen_rt_send", Linkage::Import, &rt_send_sig)
            .unwrap();

        // lumen_rt_ask(...) -> i64
        let mut rt_ask_sig = obj.make_signature();
        rt_ask_sig.params.push(AbiParam::new(PTR));
        rt_ask_sig.params.push(AbiParam::new(PTR));
        rt_ask_sig.params.push(AbiParam::new(cl_types::I32));
        rt_ask_sig.params.push(AbiParam::new(cl_types::I64));
        rt_ask_sig.returns.push(AbiParam::new(cl_types::I64));
        let rt_ask = obj
            .declare_function("lumen_rt_ask", Linkage::Import, &rt_ask_sig)
            .unwrap();

        // lumen_rt_drain()
        let rt_drain_sig = obj.make_signature();
        let rt_drain = obj
            .declare_function("lumen_rt_drain", Linkage::Import, &rt_drain_sig)
            .unwrap();

        // --- Arena allocator (lumen-z3e) ---
        // lumen_arena_new(initial: i64) -> i64
        let mut arena_new_sig = obj.make_signature();
        arena_new_sig.params.push(AbiParam::new(cl_types::I64));
        arena_new_sig.returns.push(AbiParam::new(cl_types::I64));
        let arena_new = obj
            .declare_function("lumen_arena_new", Linkage::Import, &arena_new_sig)
            .unwrap();
        // lumen_arena_free(arena: i64)
        let mut arena_free_sig = obj.make_signature();
        arena_free_sig.params.push(AbiParam::new(cl_types::I64));
        let arena_free = obj
            .declare_function("lumen_arena_free", Linkage::Import, &arena_free_sig)
            .unwrap();
        // lumen_allocator_push_arena(arena: i64) -> i64 (prev allocator)
        let mut push_sig = obj.make_signature();
        push_sig.params.push(AbiParam::new(cl_types::I64));
        push_sig.returns.push(AbiParam::new(cl_types::I64));
        let alloc_push = obj
            .declare_function("lumen_allocator_push_arena", Linkage::Import, &push_sig)
            .unwrap();
        // lumen_allocator_pop(prev: i64)
        let mut pop_sig = obj.make_signature();
        pop_sig.params.push(AbiParam::new(cl_types::I64));
        let alloc_pop = obj
            .declare_function("lumen_allocator_pop", Linkage::Import, &pop_sig)
            .unwrap();

        // --- debug.print primitives ---
        let mut di32_sig = obj.make_signature();
        di32_sig.params.push(AbiParam::new(cl_types::I32));
        let debug_i32 = obj.declare_function("lumen_debug_i32", Linkage::Import, &di32_sig).unwrap();

        let mut di64_sig = obj.make_signature();
        di64_sig.params.push(AbiParam::new(cl_types::I64));
        let debug_i64 = obj.declare_function("lumen_debug_i64", Linkage::Import, &di64_sig).unwrap();

        let mut df64_sig = obj.make_signature();
        df64_sig.params.push(AbiParam::new(cl_types::F64));
        let debug_f64 = obj.declare_function("lumen_debug_f64", Linkage::Import, &df64_sig).unwrap();

        let mut dbool_sig = obj.make_signature();
        dbool_sig.params.push(AbiParam::new(cl_types::I32));
        let debug_bool = obj.declare_function("lumen_debug_bool", Linkage::Import, &dbool_sig).unwrap();

        let mut dstr_sig = obj.make_signature();
        dstr_sig.params.push(AbiParam::new(PTR));
        let debug_str = obj.declare_function("lumen_debug_str", Linkage::Import, &dstr_sig).unwrap();

        let mut draw_sig = obj.make_signature();
        draw_sig.params.push(AbiParam::new(PTR));
        draw_sig.params.push(AbiParam::new(cl_types::I32));
        let debug_raw = obj.declare_function("lumen_debug_raw", Linkage::Import, &draw_sig).unwrap();

        let dnl_sig = obj.make_signature();
        let debug_newline = obj.declare_function("lumen_debug_newline", Linkage::Import, &dnl_sig).unwrap();

        // --- io.println primitives (mirror debug.* but write to stdout) ---
        let mut io_i32_sig = obj.make_signature();
        io_i32_sig.params.push(AbiParam::new(cl_types::I32));
        let io_i32 = obj.declare_function("lumen_io_i32", Linkage::Import, &io_i32_sig).unwrap();

        let mut io_i64_sig = obj.make_signature();
        io_i64_sig.params.push(AbiParam::new(cl_types::I64));
        let io_i64 = obj.declare_function("lumen_io_i64", Linkage::Import, &io_i64_sig).unwrap();

        let mut io_f64_sig = obj.make_signature();
        io_f64_sig.params.push(AbiParam::new(cl_types::F64));
        let io_f64 = obj.declare_function("lumen_io_f64", Linkage::Import, &io_f64_sig).unwrap();

        let mut io_bool_sig = obj.make_signature();
        io_bool_sig.params.push(AbiParam::new(cl_types::I32));
        let io_bool = obj.declare_function("lumen_io_bool", Linkage::Import, &io_bool_sig).unwrap();

        let mut io_str_sig = obj.make_signature();
        io_str_sig.params.push(AbiParam::new(PTR));
        let io_str = obj.declare_function("lumen_io_str", Linkage::Import, &io_str_sig).unwrap();

        let mut io_raw_sig = obj.make_signature();
        io_raw_sig.params.push(AbiParam::new(PTR));
        io_raw_sig.params.push(AbiParam::new(cl_types::I32));
        let io_raw = obj.declare_function("lumen_io_raw", Linkage::Import, &io_raw_sig).unwrap();

        let io_nl_sig = obj.make_signature();
        let io_newline = obj.declare_function("lumen_io_newline", Linkage::Import, &io_nl_sig).unwrap();

        // --- strbuf primitives (string interpolation) ---
        // All take the buffer ptr as the first arg.
        let mut sb_new_sig = obj.make_signature();
        sb_new_sig.returns.push(AbiParam::new(PTR));
        let strbuf_new = obj.declare_function("lumen_strbuf_new", Linkage::Import, &sb_new_sig).unwrap();

        let mut sb_finish_sig = obj.make_signature();
        sb_finish_sig.params.push(AbiParam::new(PTR));
        sb_finish_sig.returns.push(AbiParam::new(PTR));
        let strbuf_finish = obj.declare_function("lumen_strbuf_finish", Linkage::Import, &sb_finish_sig).unwrap();

        let mut sb_raw_sig = obj.make_signature();
        sb_raw_sig.params.push(AbiParam::new(PTR));           // buf
        sb_raw_sig.params.push(AbiParam::new(PTR));           // ptr
        sb_raw_sig.params.push(AbiParam::new(cl_types::I32)); // len
        let strbuf_raw = obj.declare_function("lumen_strbuf_raw", Linkage::Import, &sb_raw_sig).unwrap();

        let mut sb_str_sig = obj.make_signature();
        sb_str_sig.params.push(AbiParam::new(PTR));
        sb_str_sig.params.push(AbiParam::new(PTR));
        let strbuf_str = obj.declare_function("lumen_strbuf_str", Linkage::Import, &sb_str_sig).unwrap();

        let mut sb_i32_sig = obj.make_signature();
        sb_i32_sig.params.push(AbiParam::new(PTR));
        sb_i32_sig.params.push(AbiParam::new(cl_types::I32));
        let strbuf_i32 = obj.declare_function("lumen_strbuf_i32", Linkage::Import, &sb_i32_sig).unwrap();

        let mut sb_i64_sig = obj.make_signature();
        sb_i64_sig.params.push(AbiParam::new(PTR));
        sb_i64_sig.params.push(AbiParam::new(cl_types::I64));
        let strbuf_i64 = obj.declare_function("lumen_strbuf_i64", Linkage::Import, &sb_i64_sig).unwrap();

        let mut sb_f64_sig = obj.make_signature();
        sb_f64_sig.params.push(AbiParam::new(PTR));
        sb_f64_sig.params.push(AbiParam::new(cl_types::F64));
        let strbuf_f64 = obj.declare_function("lumen_strbuf_f64", Linkage::Import, &sb_f64_sig).unwrap();

        let mut sb_bool_sig = obj.make_signature();
        sb_bool_sig.params.push(AbiParam::new(PTR));
        sb_bool_sig.params.push(AbiParam::new(cl_types::I32));
        let strbuf_bool = obj.declare_function("lumen_strbuf_bool", Linkage::Import, &sb_bool_sig).unwrap();

        let dinit_sig = obj.make_signature();
        let debug_init = obj.declare_function("lumen_debug_init", Linkage::Import, &dinit_sig).unwrap();

        let mut dpush_sig = obj.make_signature();
        dpush_sig.params.push(AbiParam::new(PTR));
        let debug_push = obj.declare_function("lumen_debug_push", Linkage::Import, &dpush_sig).unwrap();

        let dpop_sig = obj.make_signature();
        let debug_pop = obj.declare_function("lumen_debug_pop", Linkage::Import, &dpop_sig).unwrap();

        let mut streq_sig = obj.make_signature();
        streq_sig.params.push(AbiParam::new(PTR));
        streq_sig.params.push(AbiParam::new(PTR));
        streq_sig.returns.push(AbiParam::new(cl_types::I32));
        let string_eq = obj.declare_function("lumen_string_eq", Linkage::Import, &streq_sig).unwrap();

        // --- Module functions are now declared from parsed std/*.lm files ---
        // (see compile_module → "Declare imported module extern fns")
        // print_frames: () -> void. Walks the frame_chain and prints each.
        let print_frames_sig = obj.make_signature();
        let helper_print_frames = obj
            .declare_function("lumen_print_frames", Linkage::Local, &print_frames_sig)
            .unwrap();

        Ok(Self {
            info,
            obj,
            fn_ids: HashMap::new(),
            libc_malloc,
            libc_free,
            helper_concat,
            helper_println,
            helper_utf8_encode,
            helper_assert,
            helper_print_frames,
            helper_rc_alloc,
            helper_rc_incr,
            helper_rc_decr,
            rt_send,
            rt_ask,
            rt_drain,
            debug_i32,
            debug_i64,
            debug_f64,
            debug_bool,
            debug_str,
            debug_raw,
            debug_newline,
            io_i32,
            io_i64,
            io_f64,
            io_bool,
            io_str,
            io_raw,
            io_newline,
            strbuf_new,
            strbuf_finish,
            strbuf_raw,
            strbuf_str,
            strbuf_i32,
            strbuf_i64,
            strbuf_f64,
            strbuf_bool,
            debug_init,
            debug_push,
            debug_pop,
            string_eq,
            debug_data_counter: 0,
            dispatch_fns: HashMap::new(),
            heap_data,
            bump_ptr_data,
            frame_chain_data,
            string_data: HashMap::new(),
            global_data: HashMap::new(),
            lambda_ids: HashMap::new(),
            lambda_sigs: HashMap::new(),
            module_fn_ids: {
                let mut m = HashMap::new();
                // Runtime allocator + arena functions (declared above
                // as Import; compile_arena_block looks them up via
                // module_func).
                m.insert("lumen_arena_new".to_string(), arena_new);
                m.insert("lumen_arena_free".to_string(), arena_free);
                m.insert("lumen_allocator_push_arena".to_string(), alloc_push);
                m.insert("lumen_allocator_pop".to_string(), alloc_pop);
                m
            },
            local_fn_ids: HashMap::new(),
            uses_io: false,
            debug_mode: false,
            source_path: String::new(),
            generic_templates: HashMap::new(),
            monomorph_queue: Vec::new(),
            monomorph_done: HashSet::new(),
            dwarf: crate::dwarf::DwarfBuilder::new("<unset>"),
        })
    }

    /// Declare a DataId for every module-level let/var binding. We
    /// have to walk both the user's module AND every imported
    /// module — not because any of them are visible across module
    /// boundaries (they aren't; globals are module-private), but
    /// because an imported module's own fn bodies may reference its
    /// own globals, and we compile all fn bodies in the same pass.
    /// Symbols are namespaced per module so two modules can each
    /// declare `var count: i32 = 0` without collision at the linker
    /// level. Only scalar initializers at MVP.
    fn declare_globals(
        &mut self,
        module: &ast::Module,
        imported_modules: &[(&str, &ast::Module)],
    ) -> Result<(), NativeError> {
        for &(mod_name, mod_ast) in imported_modules {
            self.declare_module_globals(mod_name, mod_ast)?;
        }
        self.declare_module_globals("user", module)?;
        Ok(())
    }

    fn declare_module_globals(
        &mut self,
        module_key: &str,
        module: &ast::Module,
    ) -> Result<(), NativeError> {
        for item in &module.items {
            let Item::GlobalLet(gl) = item else { continue };
            let ty = match self.info.globals.get(&gl.name) {
                Some((t, _)) => t.clone(),
                None => {
                    // For imported modules we don't populate info.globals
                    // (it holds only the user's module). Resolve the type
                    // from the annotation directly — required at MVP.
                    match &gl.ty {
                        Some(t) => crate::types::resolve_type(t, &self.info.types)
                            .map_err(|e| NativeError {
                                span: gl.span,
                                message: format!("resolve global `{}` type: {}", gl.name, e.message),
                            })?,
                        None => continue,
                    }
                }
            };
            let symbol = global_symbol(module_key, &gl.name);
            // Per-module key to keep same-named globals in different
            // modules distinct.
            let map_key = format!("{module_key}::{}", gl.name);
            if self.global_data.contains_key(&map_key) {
                continue;
            }
            let id = self.obj
                .declare_data(&symbol, Linkage::Local, gl.mutable, false)
                .map_err(|e| NativeError {
                    span: gl.span,
                    message: format!("declare global `{}`: {e}", gl.name),
                })?;
            let bytes = evaluate_const_initializer(&gl.value, &ty)
                .ok_or_else(|| NativeError {
                    span: gl.value.span,
                    message: "top-level let/var initializer must be a constant scalar (int/float/bool/char)".into(),
                })?;
            let mut desc = DataDescription::new();
            desc.define(bytes.into_boxed_slice());
            self.obj.define_data(id, &desc).map_err(|e| NativeError {
                span: gl.span,
                message: format!("define global `{}`: {e}", gl.name),
            })?;
            self.global_data.insert(map_key, (id, ty));
        }
        Ok(())
    }

    fn compile_module(&mut self, module: &ast::Module, imported_modules: &[(&str, &ast::Module)]) -> Result<(), NativeError> {
        self.uses_io = module.imports.iter().any(|im| im.path == ["std", "io"]);

        // Intern string literals + frame messages.
        self.intern_all_strings(module);

        // Materialize top-level let/var bindings as static data.
        // User's module: defined here with initial bytes.
        // Imported modules: declared as Import so they link to the
        // user's definition. We define the combined set below; each
        // binding's Lumen name is used as the symbol so Linkage::Import
        // from one compilation and Linkage::Export from another would
        // match if modules compiled separately — today we compile
        // everything in one pass, so every binding is defined in the
        // user module's object file and exported.
        self.declare_globals(module, imported_modules)?;

        // Declare all user functions. Generic ones are stashed as templates
        // — they get monomorphized on demand from each call site.
        for item in &module.items {
            if let Item::Fn(f) = item {
                if !f.type_params.is_empty() {
                    self.generic_templates.insert(f.name.clone(), f.clone());
                    continue;
                }
                let sig = self.build_sig(&f.name);
                let id = self
                    .obj
                    .declare_function(&f.name, Linkage::Export, &sig)
                    .unwrap();
                self.fn_ids.insert(f.name.clone(), id);
            }
        }

        // Declare msg handler functions (compiled as regular fns).
        for item in &module.items {
            if let Item::MsgHandler(mh) = item {
                let fn_name = format!("{}_{}", mh.actor_name, mh.name);
                let sig = self.build_sig(&fn_name);
                let id = self
                    .obj
                    .declare_function(&fn_name, Linkage::Local, &sig)
                    .unwrap();
                self.fn_ids.insert(fn_name, id);
            }
        }

        // Declare extern fns as imported symbols.
        for item in &module.items {
            if let Item::ExternFn(ef) = item {
                let sig = self.build_sig(&ef.name);
                // Use link_name for the linker symbol if specified,
                // but register under the Lumen-facing name in fn_ids.
                let symbol = ef.link_name.as_deref().unwrap_or(&ef.name);
                let id = self
                    .obj
                    .declare_function(symbol, Linkage::Import, &sig)
                    .unwrap();
                self.fn_ids.insert(ef.name.clone(), id);
            }
        }

        // Declare imported module extern fns from parsed std/*.lm files.
        for (_mod_name, links) in &self.info.module_link_names {
            let mod_sigs = self.info.modules.get(_mod_name);
            for (fn_name, link_name) in links {
                if self.module_fn_ids.contains_key(link_name.as_str()) {
                    continue;
                }
                if let Some(sig) = mod_sigs.and_then(|m| m.get(fn_name)) {
                    let cl_sig = self.build_sig_from(sig);
                    let id = self.obj.declare_function(link_name, Linkage::Import, &cl_sig).unwrap();
                    self.module_fn_ids.insert(link_name.clone(), id);
                }
            }
        }

        // Declare and compile fn items from imported modules. Two
        // passes so generic templates in one module are available
        // when another module's body compiles (e.g. std/string's
        // split calls list.new; if std/string is processed first,
        // std/list's templates must already be registered).
        for &(_mod_name, mod_ast) in imported_modules {
            self.intern_all_strings(mod_ast);
        }
        for &(mod_name, mod_ast) in imported_modules {
            for item in &mod_ast.items {
                if let Item::Fn(f) = item {
                    if !f.type_params.is_empty() {
                        let key = format!("{mod_name}.{}", f.name);
                        self.generic_templates.insert(key, f.clone());
                    }
                }
            }
        }
        for &(mod_name, mod_ast) in imported_modules {
            // Declare extern fns so the module's Lumen functions can call them.
            for item in &mod_ast.items {
                if let Item::ExternFn(ef) = item {
                    let symbol = ef.link_name.as_deref().unwrap_or(&ef.name);
                    if !self.module_fn_ids.contains_key(symbol) {
                        let sig = self.info.modules.get(mod_name)
                            .and_then(|m| m.get(&ef.name));
                        if let Some(sig) = sig {
                            let cl_sig = self.build_sig_from(sig);
                            if let Ok(id) = self.obj.declare_function(symbol, Linkage::Import, &cl_sig) {
                                self.module_fn_ids.insert(symbol.to_string(), id);
                            }
                        }
                    }
                    // Register under the Lumen-facing name in the
                    // module's LOCAL fn_ids so intra-module unqualified
                    // calls (e.g. `len(xs)` inside std/list) resolve to
                    // this module's extern, regardless of which other
                    // modules with same-named externs are imported.
                    if let Some(&id) = self.module_fn_ids.get(ef.link_name.as_deref().unwrap_or(&ef.name)) {
                        self.local_fn_ids
                            .entry(mod_name.to_string())
                            .or_insert_with(HashMap::new)
                            .insert(ef.name.clone(), id);
                    }
                }
            }
            // Declare each fn. Generic ones get stashed as templates and
            // monomorphized on demand from each call site.
            for item in &mod_ast.items {
                if let Item::Fn(f) = item {
                    if !f.type_params.is_empty() {
                        let key = format!("{mod_name}.{}", f.name);
                        self.generic_templates.insert(key, f.clone());
                        continue;
                    }
                    let mangled = format!("{mod_name}${}", f.name);
                    if let Some(mod_sigs) = self.info.modules.get(mod_name) {
                        if let Some(sig) = mod_sigs.get(&f.name) {
                            let cl_sig = self.build_sig_from(sig);
                            let id = self.obj.declare_function(&mangled, Linkage::Local, &cl_sig).unwrap();
                            self.fn_ids.insert(mangled.clone(), id);
                            self.lambda_sigs.insert(mangled, sig.clone());
                            // Also register in module_fn_ids so compile_method_call can find it.
                            self.module_fn_ids.insert(format!("{mod_name}:{}", f.name), id);
                        }
                    }
                }
            }
            // Register unmangled names for intra-module calls in the
            // module's local fn_ids (non-generic only — generics are
            // resolved via monomorphize_module_method_if_generic).
            for item in &mod_ast.items {
                if let Item::Fn(f) = item {
                    if !f.type_params.is_empty() { continue; }
                    let mangled = format!("{mod_name}${}", f.name);
                    if let Some(&id) = self.fn_ids.get(&mangled) {
                        self.local_fn_ids
                            .entry(mod_name.to_string())
                            .or_insert_with(HashMap::new)
                            .insert(f.name.clone(), id);
                    }
                }
            }
            // Compile each fn body with mangled name (skip generic templates —
            // they're compiled per monomorphization in the drain loop).
            let fn_ids = self.fn_ids.clone();
            for item in &mod_ast.items {
                if let Item::Fn(f) = item {
                    if !f.type_params.is_empty() { continue; }
                    let mangled = format!("{mod_name}${}", f.name);
                    if let Some(&func_id) = fn_ids.get(&mangled) {
                        let synthetic = FnDecl {
                            name: mangled,
                            name_span: f.name_span,
                            type_params: f.type_params.clone(),
                            params: f.params.clone(),
                            return_type: f.return_type.clone(),
                            effect: f.effect,
                            body: f.body.clone(),
                            span: f.span,
                        };
                        // Mark this as a module body so unqualified calls
                        // inside it resolve via local_fn_ids[mod_name].
                        self.define_module_function(&synthetic, func_id, mod_name)?;
                    }
                }
            }
        }

        // Define helper bodies (concat/println/itoa/rc_alloc are in rt.c).
        self.define_print_frames_helper()?;
        self.define_rc_incr_helper()?;
        self.define_rc_decr_helper()?;

        // Emit per-actor dispatch functions BEFORE user function bodies
        // (user fns reference dispatch via send/ask).
        let actors: Vec<String> = self.info.actors.keys().cloned().collect();
        for actor_name in &actors {
            self.emit_actor_dispatch(actor_name, module)?;
        }

        // Collect, declare, and define non-capturing lambdas.
        let mut lambdas = Vec::new();
        for item in &module.items {
            match item {
                Item::Fn(f) => collect_lambdas_block(&f.body, &mut lambdas),
                Item::MsgHandler(mh) => collect_lambdas_block(&mh.body, &mut lambdas),
                _ => {}
            }
        }
        for (i, lam) in lambdas.iter().enumerate() {
            let lam_name = format!("__lambda_{i}");
            let mut sig = self.obj.make_signature();
            for p in &lam.params {
                let pty = resolve_type_to_ty(&p.ty);
                sig.params.push(AbiParam::new(lumen_to_cl(&pty)));
            }
            let ret_ty = resolve_type_to_ty(&lam.return_type);
            if ret_ty != Ty::Unit {
                sig.returns.push(AbiParam::new(lumen_to_cl(&ret_ty)));
            } else {
                sig.returns.push(AbiParam::new(cl_types::I32));
            }
            let func_id = self.obj.declare_function(&lam_name, Linkage::Local, &sig).unwrap();
            self.lambda_ids.insert((lam.span.line, lam.span.col), func_id);
            let fn_sig = crate::types::FnSig {
                params: lam.params.iter().map(|p| {
                    (p.name.clone(), resolve_type_to_ty(&p.ty))
                }).collect(),
                ret: ret_ty,
                effect: ast::Effect::Pure,
                type_params: Vec::new(),
                is_extern: false,
            };
            let synthetic = FnDecl {
                name: lam_name.clone(),
                name_span: lam.span,
                type_params: Vec::new(),
                params: lam.params.clone(),
                return_type: lam.return_type.clone(),
                effect: ast::Effect::Pure,
                body: lam.body.clone(),
                span: lam.span,
            };
            self.lambda_sigs.insert(lam_name, fn_sig);
            self.define_function(&synthetic, func_id)?;
        }

        // Define user function bodies (skip generic templates — they're
        // compiled per monomorphization in the drain loop below).
        let fn_ids = self.fn_ids.clone();
        for item in &module.items {
            if let Item::Fn(f) = item {
                if !f.type_params.is_empty() { continue; }
                let func_id = fn_ids[&f.name];
                self.define_function(f, func_id)?;
            }
        }

        // Define msg handler bodies (compiled as regular fns with `self` param).
        for item in &module.items {
            if let Item::MsgHandler(mh) = item {
                let fn_name = format!("{}_{}", mh.actor_name, mh.name);
                let func_id = fn_ids[&fn_name];
                // Create a synthetic FnDecl with self prepended.
                let synthetic = FnDecl {
                    name: fn_name.clone(),
                    name_span: mh.name_span,
                    type_params: Vec::new(),
                    params: {
                        let mut ps = vec![ast::Param {
                            name: "self".to_string(),
                            ty: ast::Type {
                                kind: ast::TypeKind::Named {
                                    name: mh.actor_name.clone(),
                                    args: Vec::new(),
                                },
                                span: mh.name_span,
                            },
                            span: mh.name_span,
                        }];
                        ps.extend(mh.params.clone());
                        ps
                    },
                    return_type: mh.return_type.clone(),
                    effect: ast::Effect::Pure,
                    body: mh.body.clone(),
                    span: mh.span,
                };
                self.define_function(&synthetic, func_id)?;
            }
        }

        // Drain monomorphization queue. Each compiled body may queue more
        // monomorphizations (a generic fn calling another generic fn), so
        // loop until empty.
        while let Some(req) = self.monomorph_queue.pop() {
            self.define_monomorphization(req)?;
        }

        Ok(())
    }

    fn finish(self) -> Vec<u8> {
        let mut product = self.obj.finish();
        self.dwarf.emit(&mut product);
        product.emit().unwrap()
    }

    fn build_sig(&self, fn_name: &str) -> cranelift_codegen::ir::Signature {
        let sig = self.info.fns.get(fn_name)
            .or_else(|| self.lambda_sigs.get(fn_name))
            .unwrap_or_else(|| panic!("no sig for {fn_name}"));
        self.build_sig_from(sig)
    }

    fn build_sig_from(&self, sig: &crate::types::FnSig) -> cranelift_codegen::ir::Signature {
        let mut cl_sig = self.obj.make_signature();
        for (_, ty) in &sig.params {
            cl_sig.params.push(AbiParam::new(lumen_to_cl(ty)));
        }
        if sig.ret != Ty::Unit {
            cl_sig.returns.push(AbiParam::new(lumen_to_cl(&sig.ret)));
        }
        cl_sig
    }

    // --- String interning -----------------------------------------------

    fn intern_all_strings(&mut self, module: &ast::Module) {
        // Collect all string literals from the AST.
        let mut strings = Vec::new();
        for item in &module.items {
            if let Item::Fn(f) = item {
                collect_strings_block(&f.body, &mut strings);
                // Frame messages for each ? site.
                collect_try_frame_strings(&f.name, &f.body, &mut strings);
            }
        }
        strings.push("\n".to_string());
        strings.push(", ".to_string());
        strings.push("=".to_string());
        strings.push("true".to_string());
        strings.push("false".to_string());
        if !self.source_path.is_empty() {
            strings.push(self.source_path.clone());
        }
        // Intern param names for frame arg capture.
        for (fn_name, sig) in &self.info.fns {
            for (pname, _) in &sig.params {
                strings.push(pname.clone());
            }
            let _ = fn_name;
        }
        strings.sort();
        strings.dedup();

        for s in &strings {
            self.intern_string(s);
        }
    }

    fn intern_string(&mut self, s: &str) -> DataId {
        if let Some(&id) = self.string_data.get(s) {
            return id;
        }
        let name = format!("str_{}", self.string_data.len());
        let id = self
            .obj
            .declare_data(&name, Linkage::Local, false, false)
            .unwrap();
        let bytes = s.as_bytes();
        let len = bytes.len() as u32;
        let mut payload = Vec::with_capacity(4 + bytes.len());
        payload.extend_from_slice(&len.to_le_bytes());
        payload.extend_from_slice(bytes);
        // Pad to 8-byte alignment.
        while payload.len() % 8 != 0 {
            payload.push(0);
        }
        let mut desc = DataDescription::new();
        desc.define(payload.into_boxed_slice());
        self.obj.define_data(id, &desc).unwrap();
        self.string_data.insert(s.to_string(), id);
        id
    }

    // --- Function definition --------------------------------------------

    /// Emit `lumen_print_frames()`: walk the frame_chain linked list
    /// and call lumen_println on each frame's message string.
    /// Frame layout (64-bit): { message: ptr @+0, next: ptr @+8 } = 16 bytes.
    fn define_print_frames_helper(&mut self) -> Result<(), NativeError> {
        let sig = self.obj.make_signature(); // () -> void
        let mut ctx = self.obj.make_context();
        ctx.func.signature = sig;
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry = builder.create_block();
        builder.switch_to_block(entry);

        let flags = MemFlags::new();
        let chain_gv = self.obj.declare_data_in_func(self.frame_chain_data, builder.func);
        let chain_addr = builder.ins().global_value(PTR, chain_gv);
        let current_init = builder.ins().load(PTR, flags, chain_addr, 0);

        let cur_var = builder.declare_var(PTR);
        builder.def_var(cur_var, current_init);

        let header_bb = builder.create_block();
        let body_bb = builder.create_block();
        let exit_bb = builder.create_block();

        builder.ins().jump(header_bb, &[]);

        // Header: if current == 0, exit.
        builder.switch_to_block(header_bb);
        let cur = builder.use_var(cur_var);
        let zero = builder.ins().iconst(PTR, 0);
        let done = builder.ins().icmp(IntCC::Equal, cur, zero);
        builder.ins().brif(done, exit_bb, &[], body_bb, &[]);

        // Body: println(current.message), current = current.next.
        builder.switch_to_block(body_bb);
        let cur = builder.use_var(cur_var);
        let msg = builder.ins().load(PTR, flags, cur, 0); // message @+0
        let println_ref = self.obj.declare_func_in_func(self.helper_println, builder.func);
        builder.ins().call(println_ref, &[msg]);

        let cur = builder.use_var(cur_var);
        let next = builder.ins().load(PTR, flags, cur, 8); // next @+8
        builder.def_var(cur_var, next);
        builder.ins().jump(header_bb, &[]);

        // Exit.
        builder.switch_to_block(exit_bb);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();

        self.obj
            .define_function(self.helper_print_frames, &mut ctx)
            .unwrap();
        Ok(())
    }

    /// `lumen_rc_incr(ptr)`: if ptr != 0 and ptr is heap-allocated,
    /// increment the refcount at ptr-8.
    fn define_rc_incr_helper(&mut self) -> Result<(), NativeError> {
        let mut sig = self.obj.make_signature();
        sig.params.push(AbiParam::new(PTR));
        let mut ctx = self.obj.make_context();
        ctx.func.signature = sig;
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let ptr = builder.block_params(block)[0];
        let flags = MemFlags::new();

        // if ptr == 0, return
        let is_null = builder.ins().icmp_imm(IntCC::Equal, ptr, 0);
        let do_incr = builder.create_block();
        let exit = builder.create_block();
        builder.ins().brif(is_null, exit, &[], do_incr, &[]);

        builder.switch_to_block(do_incr);
        // Check the magic sentinel at ptr-4 to verify this is an
        // rc_alloc'd block. String literals and other static data in
        // .rodata don't have the rc header — writing to their memory
        // would segfault.
        let eight = builder.ins().iconst(PTR, 8);
        let header = builder.ins().isub(ptr, eight);
        let magic = builder.ins().load(cl_types::I32, flags, header, 4);
        let expected = builder.ins().iconst(cl_types::I32, 0x4C554D45u32 as i64);
        let is_rc = builder.ins().icmp(IntCC::Equal, magic, expected);
        let do_real_incr = builder.create_block();
        builder.ins().brif(is_rc, do_real_incr, &[], exit, &[]);

        builder.switch_to_block(do_real_incr);
        let rc = builder.ins().load(cl_types::I32, flags, header, 0);
        let new_rc = builder.ins().iadd_imm(rc, 1);
        builder.ins().store(flags, new_rc, header, 0);
        builder.ins().jump(exit, &[]);

        builder.switch_to_block(exit);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
        self.obj.define_function(self.helper_rc_incr, &mut ctx).unwrap();
        Ok(())
    }

    /// `lumen_rc_decr(ptr)`: decrement refcount; if it hits 0, free
    /// the block via libc free.
    fn define_rc_decr_helper(&mut self) -> Result<(), NativeError> {
        let mut sig = self.obj.make_signature();
        sig.params.push(AbiParam::new(PTR));
        let mut ctx = self.obj.make_context();
        ctx.func.signature = sig;
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let ptr = builder.block_params(block)[0];
        let flags = MemFlags::new();

        // if ptr == 0, return
        let is_null = builder.ins().icmp_imm(IntCC::Equal, ptr, 0);
        let do_decr = builder.create_block();
        let exit = builder.create_block();
        builder.ins().brif(is_null, exit, &[], do_decr, &[]);

        builder.switch_to_block(do_decr);
        let eight = builder.ins().iconst(PTR, 8);
        let header = builder.ins().isub(ptr, eight);

        // Check magic sentinel at header+4. If it doesn't match
        // 0x4C554D45, this isn't an rc_alloc'd block (e.g. a string
        // literal in .rodata). Skip.
        let magic = builder.ins().load(cl_types::I32, flags, header, 4);
        let expected = builder.ins().iconst(cl_types::I32, 0x4C554D45u32 as i64);
        let is_rc = builder.ins().icmp(IntCC::Equal, magic, expected);
        let do_real_decr = builder.create_block();
        builder.ins().brif(is_rc, do_real_decr, &[], exit, &[]);

        builder.switch_to_block(do_real_decr);
        let rc = builder.ins().load(cl_types::I32, flags, header, 0);
        let new_rc = builder.ins().iadd_imm(rc, -1);
        builder.ins().store(flags, new_rc, header, 0);

        // if new_rc == 0, free
        let is_zero = builder.ins().icmp_imm(IntCC::Equal, new_rc, 0);
        let do_free = builder.create_block();
        builder.ins().brif(is_zero, do_free, &[], exit, &[]);

        builder.switch_to_block(do_free);
        let free_ref = self.obj.declare_func_in_func(self.libc_free, builder.func);
        builder.ins().call(free_ref, &[header]);
        builder.ins().jump(exit, &[]);

        builder.switch_to_block(exit);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();
        self.obj.define_function(self.helper_rc_decr, &mut ctx).unwrap();
        Ok(())
    }

    /// Emit a dispatch function for an actor type:
    /// `dispatch(cell: ptr, kind: i32, arg0: i64, reply: ptr)`
    /// Loads state from cell, dispatches on kind to the right handler,
    /// stores new state back, writes reply if the handler is a query.
    fn emit_actor_dispatch(
        &mut self,
        actor_name: &str,
        _module: &ast::Module,
    ) -> Result<(), NativeError> {
        let dispatch_name = format!("{}_dispatch", actor_name);
        let mut sig = self.obj.make_signature();
        sig.params.push(AbiParam::new(PTR)); // cell
        sig.params.push(AbiParam::new(cl_types::I32)); // kind
        sig.params.push(AbiParam::new(cl_types::I64)); // arg0
        sig.params.push(AbiParam::new(PTR)); // reply ptr
        let func_id = self
            .obj
            .declare_function(&dispatch_name, Linkage::Export, &sig)
            .unwrap();
        self.dispatch_fns.insert(actor_name.to_string(), func_id);

        let mut ctx = self.obj.make_context();
        ctx.func.signature = sig;
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let cell = builder.block_params(entry)[0];
        let kind = builder.block_params(entry)[1];
        let arg0 = builder.block_params(entry)[2];
        let reply_ptr = builder.block_params(entry)[3];
        let flags = MemFlags::new();

        // Load current state from cell.
        let state = builder.ins().load(PTR, flags, cell, 0);

        // Dispatch: for each msg handler, if kind == N, call handler.
        let msgs = self.info.actors.get(actor_name).cloned().unwrap_or_default();
        let exit_bb = builder.create_block();

        for (i, msg) in msgs.iter().enumerate() {
            let fn_name = format!("{}_{}", actor_name, msg.name);
            let handler_id = *self.fn_ids.get(&fn_name).unwrap();
            let handler_ref = self.obj.declare_func_in_func(handler_id, builder.func);

            let match_bb = builder.create_block();
            let next_bb = builder.create_block();

            let kind_val = builder.ins().iconst(cl_types::I32, i as i64);
            let matches = builder.ins().icmp(IntCC::Equal, kind, kind_val);
            builder.ins().brif(matches, match_bb, &[], next_bb, &[]);

            builder.switch_to_block(match_bb);
            // Build args: (state, arg0 truncated to the right type).
            let is_mutation = matches!(msg.ret, Ty::User(ref n) if n == actor_name);
            // Tuple return (State, reply): first element is the actor
            // type (mutation), rest is the reply.
            let is_tuple_mutation = matches!(&msg.ret,
                Ty::Tuple(elems) if matches!(elems.first(), Some(Ty::User(ref n)) if n == actor_name)
            );
            let mut call_args = vec![state];
            if msg.params.len() == 1 {
                // Single arg: truncate arg0 to the param type.
                let (_, pty) = &msg.params[0];
                let arg = match lumen_to_cl(pty) {
                    cl_types::I32 => builder.ins().ireduce(cl_types::I32, arg0),
                    cl_types::F64 => builder.ins().bitcast(cl_types::F64, flags, arg0),
                    _ => arg0,
                };
                call_args.push(arg);
            } else if msg.params.len() > 1 {
                // Multi-arg: arg0 is a pointer to a packed struct.
                // Load each field from the blob.
                for (pname, pty) in &msg.params {
                    let (offset, _) = field_offset(
                        &msg.params.iter().map(|(n, t)| (n.clone(), t.clone())).collect::<Vec<_>>(),
                        pname,
                    );
                    let cl_ty = lumen_to_cl(pty);
                    let val = builder.ins().load(cl_ty, flags, arg0, offset);
                    call_args.push(val);
                }
            }
            let call = builder.ins().call(handler_ref, &call_args);
            let result = builder.inst_results(call)[0];

            if is_mutation {
                // Store new state in cell.
                builder.ins().store(flags, result, cell, 0);
            } else if is_tuple_mutation {
                // Tuple return: element 0 is new state, rest is the reply.
                // result is a tuple pointer. Extract and store new state.
                let new_state = builder.ins().load(PTR, flags, result, 0);
                builder.ins().store(flags, new_state, cell, 0);
                // Reply with the full tuple pointer (caller uses .1 etc.)
                builder.ins().store(flags, result, reply_ptr, 0);
            } else {
                // Write result as reply.
                let reply_val = match lumen_to_cl(&msg.ret) {
                    cl_types::I32 => builder.ins().sextend(cl_types::I64, result),
                    cl_types::F64 => builder.ins().bitcast(cl_types::I64, flags, result),
                    _ => result,
                };
                builder.ins().store(flags, reply_val, reply_ptr, 0);
            }
            builder.ins().jump(exit_bb, &[]);

            builder.switch_to_block(next_bb);
        }
        // Fallthrough (unknown kind): just jump to exit.
        builder.ins().jump(exit_bb, &[]);

        builder.switch_to_block(exit_bb);
        builder.ins().return_(&[]);

        builder.seal_all_blocks();
        builder.finalize();

        self.obj.define_function(func_id, &mut ctx)
            .unwrap_or_else(|e| panic!("dispatch define failed: {e}"));
        Ok(())
    }

    fn define_function(&mut self, f: &FnDecl, func_id: FuncId) -> Result<(), NativeError> {
        let sig = self.info.fns.get(&f.name).cloned()
            .unwrap_or_else(|| self.lambda_sigs[&f.name].clone());
        self.define_function_with_sig(f, func_id, &sig, HashMap::new(), None)
    }

    /// Define a non-generic fn body that lives inside an imported module
    /// — sets `current_module` so unqualified intra-module calls resolve
    /// via the module's local_fn_ids.
    fn define_module_function(
        &mut self,
        f: &FnDecl,
        func_id: FuncId,
        module: &str,
    ) -> Result<(), NativeError> {
        let sig = self.info.fns.get(&f.name).cloned()
            .unwrap_or_else(|| self.lambda_sigs[&f.name].clone());
        self.define_function_with_sig(f, func_id, &sig, HashMap::new(), Some(module.to_string()))
    }

    fn define_monomorphization(&mut self, req: MonomorphRequest) -> Result<(), NativeError> {
        // Use the mangled name for diagnostic spans / debug-frame messages,
        // but everything else mirrors define_function.
        let renamed = FnDecl {
            name: req.mangled_name.clone(),
            ..req.decl
        };
        let current_module = req.current_module.clone();
        self.define_function_with_sig(&renamed, req.func_id, &req.sig, req.subs, current_module)
    }

    fn define_function_with_sig(
        &mut self,
        f: &FnDecl,
        func_id: FuncId,
        sig: &crate::types::FnSig,
        active_subs: HashMap<String, Ty>,
        current_module: Option<String>,
    ) -> Result<(), NativeError> {
        let cl_sig = self.build_sig_from(sig);

        let mut ctx = self.obj.make_context();
        ctx.func.signature = cl_sig;
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        // (sealed later)

        // Capture before current_module is moved into the FnEmitter.
        let current_module_name: Option<String> = current_module.clone();

        {
            let mut fb = FnEmitter::new(self, &mut builder, sig, &f.name);
            fb.active_subs = active_subs;
            fb.current_module = current_module;

            // Declare params as variables. Pointer-typed params are
            // registered on the cleanup stack — the caller incr'd them
            // before the call, and our scope-exit decr balances it.
            for (i, (pname, pty)) in sig.params.iter().enumerate() {
                let var = fb.fresh_var(lumen_to_cl(pty));
                let val = fb.builder.block_params(entry)[i];
                fb.builder.def_var(var, val);
                fb.names.insert(pname.clone(), var);
                fb.name_types.insert(pname.clone(), pty.clone());
            }

            // Collect pointer-typed params for cleanup — the caller
            // rc_incr'd them before the call, and our scope-exit decr
            // balances it.
            let mut param_cleanup = Vec::new();
            for (pname, pty) in sig.params.iter() {
                if !is_scalar(pty) {
                    if let Some(&var) = fb.names.get(pname) {
                        param_cleanup.push((pname.clone(), var, pty.clone()));
                    }
                }
            }

            // Debug mode: install crash handler at program start.
            if fb.cg.debug_mode && f.name == "main" {
                let init_ref = fb.cg.obj.declare_func_in_func(
                    fb.cg.debug_init, fb.builder.func);
                fb.builder.ins().call(init_ref, &[]);
            }

            // Debug mode: push frame message for stack traces.
            let debug_enabled = fb.cg.debug_mode;
            if debug_enabled {
                let frame_msg = format!("  at {} (<source>:{}:{})", f.name, f.span.line, f.span.col);
                let msg_name = format!("__frame_msg_{}", fb.cg.debug_data_counter);
                fb.cg.debug_data_counter += 1;
                let msg_data_id = fb.cg.obj.declare_data(&msg_name, Linkage::Local, false, false).unwrap();
                let mut desc = DataDescription::new();
                let bytes = frame_msg.as_bytes();
                let mut payload = Vec::with_capacity(4 + bytes.len());
                payload.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
                payload.extend_from_slice(bytes);
                desc.define(payload.into_boxed_slice());
                fb.cg.obj.define_data(msg_data_id, &desc).unwrap();

                let msg_gv = fb.cg.obj.declare_data_in_func(msg_data_id, fb.builder.func);
                let msg_val = fb.builder.ins().global_value(PTR, msg_gv);
                let push_ref = fb.cg.obj.declare_func_in_func(fb.cg.debug_push, fb.builder.func);
                fb.builder.ins().call(push_ref, &[msg_val]);
            }

            // Compile body.
            fb.hit_return = false;
            let result = fb.compile_block_with_cleanup(&f.body, param_cleanup)?;

            // Debug mode: pop frame before returning.
            let emit_debug_pop = |fb: &mut FnEmitter| {
                if debug_enabled {
                    let pop_ref = fb.cg.obj.declare_func_in_func(fb.cg.debug_pop, fb.builder.func);
                    fb.builder.ins().call(pop_ref, &[]);
                }
            };

            if fb.hit_return {
                // The body ended with a return. We're on a dead block.
                emit_debug_pop(&mut fb);
                if sig.ret == Ty::Unit {
                    fb.builder.ins().return_(&[]);
                } else {
                    let ret_ty = lumen_to_cl(&sig.ret);
                    let dummy = fb.builder.ins().iconst(ret_ty, 0);
                    fb.builder.ins().return_(&[dummy]);
                }
            } else {
                if f.name == "main" {
                    let drain_ref = fb.cg.obj.declare_func_in_func(
                        fb.cg.rt_drain, fb.builder.func,
                    );
                    fb.builder.ins().call(drain_ref, &[]);
                }
                emit_debug_pop(&mut fb);
                if sig.ret == Ty::Unit {
                    fb.builder.ins().return_(&[]);
                } else {
                    fb.builder.ins().return_(&[result]);
                }
            }
        } // fb dropped here, releasing the mutable borrow on builder

        builder.seal_all_blocks();
        builder.finalize();

        self.obj
            .define_function(func_id, &mut ctx)
            .map_err(|e| NativeError {
                span: f.span,
                message: format!("define {}: {e}", f.name),
            })?;

        // Record every fn — imported-module fns get attributed to
        // their own source file via the module_index lookup so their
        // line numbers don't bleed into the user's main source.
        if let Some(compiled) = ctx.compiled_code() {
            let size = compiled.code_buffer().len() as u32;
            if f.span.line > 0 {
                let file_index = self.dwarf.module_file_index(current_module_name.as_deref());
                let mut rows: Vec<crate::dwarf::LineRow> = Vec::new();
                let mut last: Option<(u32, u32)> = None;
                for sl in compiled.buffer.get_srclocs_sorted() {
                    let packed = sl.loc.bits();
                    if packed == 0 || packed == !0 { continue; }
                    let line = (packed >> 12) & 0x000F_FFFF;
                    let col = packed & 0x0000_0FFF;
                    if last == Some((line, col)) { continue; }
                    last = Some((line, col));
                    rows.push(crate::dwarf::LineRow { offset: sl.start, line, col });
                }
                let params: Vec<crate::dwarf::Param> = sig.params.iter()
                    .map(|(n, ty)| crate::dwarf::Param {
                        name: n.clone(),
                        ty: ty_to_dwarf(ty),
                    })
                    .collect();
                let ret = ty_to_dwarf(&sig.ret);
                self.dwarf.record_function(&f.name, func_id, size, f.span.line, rows, file_index, params, ret);
            }
        }

        Ok(())
    }


    fn declare_data_in_func(
        &mut self,
        data_id: DataId,
        builder: &mut FunctionBuilder<'_>,
    ) -> cranelift_codegen::ir::GlobalValue {
        self.obj.declare_data_in_func(data_id, builder.func)
    }
}

// ---------------------------------------------------------------------------
// Per-function emitter
// ---------------------------------------------------------------------------

struct FnEmitter<'a, 'b, 'c> {
    cg: &'a mut NativeCodegen<'b>,
    builder: &'a mut FunctionBuilder<'c>,
    #[allow(dead_code)]
    sig: &'a crate::types::FnSig,
    fn_name: String,
    names: HashMap<String, Variable>,
    name_types: HashMap<String, Ty>,
    /// Stack of cleanup lists.
    cleanup_stack: Vec<Vec<(String, Variable, Ty)>>,
    /// Set to true when a `return` statement is compiled. Checked by
    /// compile_if to avoid emitting jumps after a terminated block.
    hit_return: bool,
    /// Active type-parameter substitutions when compiling a monomorphized
    /// generic fn body (T → I32, etc). Empty for non-generic fns.
    active_subs: HashMap<String, Ty>,
    /// When compiling a body that lives inside an imported stdlib module,
    /// the module name. Unqualified user-fn calls inside that body
    /// resolve via `cg.local_fn_ids[current_module]` first, falling back
    /// to the global `cg.fn_ids` only if not found there.
    current_module: Option<String>,
}

impl<'a, 'b, 'c> FnEmitter<'a, 'b, 'c> {
    fn new(
        cg: &'a mut NativeCodegen<'b>,
        builder: &'a mut FunctionBuilder<'c>,
        sig: &'a crate::types::FnSig,
        fn_name: &str,
    ) -> Self {
        Self {
            cg,
            builder,
            sig,
            fn_name: fn_name.to_string(),
            names: HashMap::new(),
            name_types: HashMap::new(),
            cleanup_stack: Vec::new(),
            hit_return: false,
            active_subs: HashMap::new(),
            current_module: None,
        }
    }

    fn fresh_var(&mut self, ty: CLType) -> Variable {
        self.builder.declare_var(ty)
    }

    /// Attach a Lumen source location to subsequent instructions. The
    /// line+col is packed into cranelift's opaque SourceLoc u32 (line
    /// in the high 20 bits, col in the low 12). `dwarf::unpack_srcloc`
    /// decodes on the way out.
    fn set_srcloc(&mut self, span: crate::span::Span) {
        let packed = crate::dwarf::pack_srcloc(span.line, span.col);
        self.builder.set_srcloc(cranelift_codegen::ir::SourceLoc::new(packed));
    }

    fn compile_block(&mut self, block: &ast::Block) -> Result<Value, NativeError> {
        self.compile_block_with_cleanup(block, Vec::new())
    }

    fn compile_block_with_cleanup(
        &mut self,
        block: &ast::Block,
        initial_cleanup: Vec<(String, Variable, Ty)>,
    ) -> Result<Value, NativeError> {
        self.cleanup_stack.push(initial_cleanup);
        let mut hit_return = false;
        for stmt in &block.stmts {
            self.compile_stmt(stmt)?;
            // If the statement was a `return`, the block is terminated.
            // Don't emit any more code — Cranelift rejects instructions
            // after a terminator.
            if matches!(stmt.kind, StmtKind::Return(_)) {
                hit_return = true;
                break;
            }
        }
        if hit_return {
            // Block terminated by return. Set the flag and switch to
            // a dead block. The CALLER (compile_if, define_function)
            // is responsible for using the right dummy type.
            self.cleanup_stack.pop();
            let dead_bb = self.builder.create_block();
            self.builder.switch_to_block(dead_bb);
            // Don't emit any instruction here — let the caller decide
            // what to emit based on what type it needs.
            // Return a *placeholder* — the caller should check hit_return
            // and emit the right-typed iconst before using this value.
            return Ok(self.builder.ins().iconst(cl_types::I32, 0));
        }
        let result = match &block.tail {
            Some(e) => self.compile_expr(e)?,
            None => self.builder.ins().iconst(cl_types::I32, 0),
        };
        // rc_decr all pointer-typed let bindings that were declared in
        // this block, EXCEPT skip any that happen to be the result value
        // (the tail expression might reference a local that we're about
        // to return from this block).
        // Protect the result from the cleanup pass: rc_incr it first,
        // then decr all locals (including the one that might hold the
        // same pointer), then the net effect is +0 for the result and
        // -1 for everything else.
        let result_is_ptr = block.tail.as_ref().map(|e| {
            self.infer_ty(e).map(|t| !is_scalar(&t)).unwrap_or(false)
        }).unwrap_or(false);
        if result_is_ptr {
            self.emit_rc_incr(result);
        }
        let cleanup = self.cleanup_stack.pop().unwrap_or_default();
        for (_name, var, ty) in &cleanup {
            let val = self.builder.use_var(*var);
            self.emit_rc_decr_typed(val, ty);
        }
        Ok(result)
    }

    fn compile_stmt(&mut self, stmt: &ast::Stmt) -> Result<(), NativeError> {
        // Attach the statement's source span to every instruction
        // emitted while lowering this stmt. Cranelift keeps this in
        // its MachSrcLoc table; we harvest it at function-end to
        // build the DWARF line program (lumen-ix6).
        self.set_srcloc(stmt.span);
        match &stmt.kind {
            StmtKind::Let { name, value, ty } | StmtKind::Var { name, value, ty } => {
                let inferred_ty = self.infer_ty(value)?;
                // Prefer the user's annotation when it's more specific than
                // what we inferred (e.g. `var m: Map<string, i32> = map.new()`
                // — map.new returns Map<_, Error>, but the annotation pins V).
                // Inside a generic body we also substitute type-parameter
                // names (e.g. `let r: List<B>` becomes `List<String>` when
                // monomorphizing with B=String).
                let lumen_ty = match ty {
                    Some(annot) => {
                        let raw = resolve_type_to_ty(annot);
                        let annot_ty = substitute_ty(raw, &self.active_subs);
                        // Concretize any deferred Ty::User in the annotation
                        // (e.g. `let p: Pair<T, i32>` inside a generic fn at
                        // T=I32 becomes Pair$I32_I32). Uses active_subs which
                        // is empty for non-generic fns, so this is a no-op
                        // outside generic contexts.
                        let active_subs = self.active_subs.clone();
                        let annot_ty = self.concretize_ty(&annot_ty, &active_subs);
                        // Prefer the annotation when the inferred type is a
                        // deferred user-type instantiation (e.g. swap(p1)
                        // infers Pair$GB_GA without call-site subs). The
                        // annotation — Pair<string, i32> → Pair$Str_I32 —
                        // carries the concrete instantiation.
                        if ty_more_specific(&annot_ty, &inferred_ty) || self.is_deferred_user_ty(&inferred_ty) {
                            annot_ty
                        } else {
                            inferred_ty
                        }
                    }
                    None => inferred_ty,
                };
                let val = self.compile_expr(value)?;
                // If the RHS is a borrowing expression (variable read, field
                // access), rc_incr because we're creating an additional
                // reference. Fresh values (calls, struct lits) already have
                // rc=1 from allocation.
                if !is_scalar(&lumen_ty) && is_borrowing_expr(&value.kind) {
                    self.emit_rc_incr(val);
                }
                let cl_ty = lumen_to_cl(&lumen_ty);
                let var = self.fresh_var(cl_ty);
                self.builder.def_var(var, val);
                self.names.insert(name.clone(), var);
                self.name_types.insert(name.clone(), lumen_ty.clone());
                // Register pointer-typed bindings for scope-exit cleanup.
                if !is_scalar(&lumen_ty) {
                    if let Some(list) = self.cleanup_stack.last_mut() {
                        list.push((name.clone(), var, lumen_ty));
                    }
                }
            }
            StmtKind::Assign { name, value } => {
                // Assignment to a module-level `var`: compile the RHS
                // and store into the static slot.
                if !self.names.contains_key(name) {
                    if let Some((id, ty)) = self.lookup_global(name).cloned() {
                        let val = self.compile_expr(value)?;
                        let gv = self.cg.obj.declare_data_in_func(id, self.builder.func);
                        let addr = self.builder.ins().global_value(PTR, gv);
                        // Pointer-typed globals would need rc decr on the
                        // old value and incr on the new — MVP is scalar
                        // only, so skip.
                        let _ = ty;
                        self.builder.ins().store(MemFlags::new(), val, addr, 0);
                        return Ok(());
                    }
                }
                // Compile the new value FIRST, before decrementing the old.
                // This prevents use-after-free when the RHS references the
                // variable being assigned to (e.g. `x = update(x)`).
                let val = self.compile_expr(value)?;
                // RC: if the RHS is a reference copy (variable read, field
                // access), rc_incr because we're creating an additional
                // reference. Fresh values (calls, struct lits) already have
                // rc=1 from allocation — no extra incr needed.
                if let Some(ty) = self.name_types.get(name).cloned() {
                    if !is_scalar(&ty) && is_borrowing_expr(&value.kind) {
                        self.emit_rc_incr(val);
                    }
                }
                // RC: decrement the old value now that the new one is ready.
                if let Some(ty) = self.name_types.get(name).cloned() {
                    if !is_scalar(&ty) {
                        if let Some(&var) = self.names.get(name) {
                            let old = self.builder.use_var(var);
                            self.emit_rc_decr_typed(old, &ty);
                        }
                    }
                }
                let var = *self.names.get(name).ok_or_else(|| NativeError {
                    span: stmt.span,
                    message: format!("unknown `{name}`"),
                })?;
                self.builder.def_var(var, val);
                // Refine name_types: if the RHS has a more specific list type
                // (e.g. List(User("Block")) vs List(I64)), update the tracked type
                // so list.get can extract the element type.
                if let Ok(rhs_ty) = self.infer_ty(value) {
                    if let Ty::List(ref inner) = rhs_ty {
                        if !matches!(**inner, Ty::I64 | Ty::I32 | Ty::Error) {
                            self.name_types.insert(name.clone(), rhs_ty);
                        }
                    }
                }
            }
            StmtKind::Expr(e) => {
                self.compile_expr(e)?;
            }
            StmtKind::For { binder, iter, body } => {
                self.compile_for(binder, iter, body, stmt.span)?;
            }
            StmtKind::Return(Some(e)) => {
                let val = self.compile_expr(e)?;
                // Protect the return value from cleanup, then decr all
                // locals so we don't leak them.
                let ret_is_ptr = self.infer_ty(e).map(|t| !is_scalar(&t)).unwrap_or(false);
                if ret_is_ptr {
                    self.emit_rc_incr(val);
                }
                // Collect cleanup targets first to avoid borrow conflict.
                let to_clean: Vec<(Variable, Ty)> = self.cleanup_stack.iter().rev()
                    .flat_map(|frame| frame.iter().map(|(_, var, ty)| (*var, ty.clone())))
                    .collect();
                for (var, ty) in &to_clean {
                    let v = self.builder.use_var(*var);
                    self.emit_rc_decr_typed(v, ty);
                }
                self.builder.ins().return_(&[val]);
                self.hit_return = true;
            }
            StmtKind::LetTuple { names, value } => {
                let ptr = self.compile_expr(value)?;
                let val_ty = self.infer_ty(value)?;
                let elems = match val_ty {
                    Ty::Tuple(ref e) => e.clone(),
                    _ => {
                        return Err(NativeError {
                            span: stmt.span,
                            message: "let tuple destructuring requires a tuple value".into(),
                        });
                    }
                };
                if names.len() != elems.len() {
                    return Err(NativeError {
                        span: stmt.span,
                        message: format!(
                            "tuple destructuring expects {} names, found {}",
                            elems.len(),
                            names.len()
                        ),
                    });
                }
                let fields = tuple_as_fields(&elems);
                for (i, name) in names.iter().enumerate() {
                    let field_name = format!("_{i}");
                    let (offset, fty) = field_offset(&fields, &field_name);
                    let cl_ty = lumen_to_cl(&fty);
                    let val = self.builder.ins().load(cl_ty, MemFlags::new(), ptr, offset);
                    let var = self.fresh_var(cl_ty);
                    self.builder.def_var(var, val);
                    self.names.insert(name.clone(), var);
                    self.name_types.insert(name.clone(), fty.clone());
                    if !is_scalar(&fty) {
                        if let Some(list) = self.cleanup_stack.last_mut() {
                            list.push((name.clone(), var, fty));
                        }
                    }
                }
            }
            StmtKind::Return(None) => {
                self.builder.ins().return_(&[]);
            }
        }
        Ok(())
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<Value, NativeError> {
        match &expr.kind {
            ExprKind::IntLit { value, suffix } => {
                let (ty, val) = match suffix {
                    Some(IntSuffix::I64) | Some(IntSuffix::U64) => {
                        (cl_types::I64, *value as i64)
                    }
                    _ => (cl_types::I32, *value as i64),
                };
                Ok(self.builder.ins().iconst(ty, val))
            }
            ExprKind::FloatLit(v) => Ok(self.builder.ins().f64const(*v)),
            ExprKind::CharLit(v) => Ok(self.builder.ins().iconst(cl_types::I32, *v as i64)),
            ExprKind::BoolLit(b) => {
                Ok(self.builder.ins().iconst(cl_types::I32, if *b { 1 } else { 0 }))
            }
            ExprKind::UnitLit => Ok(self.builder.ins().iconst(cl_types::I32, 0)),
            ExprKind::StringLit(s) => {
                let data_id = *self.cg.string_data.get(s).unwrap();
                let gv = self.cg.obj.declare_data_in_func(data_id, self.builder.func);
                Ok(self.builder.ins().global_value(PTR, gv))
            }
            ExprKind::Ident(name) => {
                if let Some(&var) = self.names.get(name) {
                    Ok(self.builder.use_var(var))
                } else if name == "None" {
                    self.build_sum_block(0, None)
                } else if let Some(sum_name) = self.resolved_variant_sum(expr.span)
                    .or_else(|| self.find_sum_for_variant(name))
                {
                    let tag = self.variant_tag(&Ty::User(sum_name), name).unwrap_or(0);
                    self.build_sum_block(tag, None)
                } else if let Some(&func_id) = self.cg.fn_ids.get(name) {
                    // A function name used as a value — emit its address as a PTR.
                    let func_ref = self.cg.obj.declare_func_in_func(func_id, self.builder.func);
                    Ok(self.builder.ins().func_addr(PTR, func_ref))
                } else if let Some((id, ty)) = self.lookup_global(name).cloned() {
                    // Module-level let/var binding: load from the
                    // static slot. Scoped to the current module.
                    let gv = self.cg.obj.declare_data_in_func(id, self.builder.func);
                    let addr = self.builder.ins().global_value(PTR, gv);
                    let cl_ty = lumen_to_cl(&ty);
                    Ok(self.builder.ins().load(cl_ty, MemFlags::new(), addr, 0))
                } else {
                    Err(NativeError {
                        span: expr.span,
                        message: format!("unknown identifier `{name}`"),
                    })
                }
            }
            ExprKind::Paren(inner) => self.compile_expr(inner),
            ExprKind::Binary { op, lhs, rhs } => self.compile_binary(*op, lhs, rhs, expr.span),
            ExprKind::Unary { op, rhs } => {
                let v = self.compile_expr(rhs)?;
                match op {
                    UnaryOp::Neg => {
                        let ty = lumen_to_cl(&self.infer_ty(rhs)?);
                        if ty == cl_types::F64 {
                            Ok(self.builder.ins().fneg(v))
                        } else {
                            Ok(self.builder.ins().ineg(v))
                        }
                    }
                    UnaryOp::Not => {
                        let zero = self.builder.ins().iconst(cl_types::I32, 0);
                        Ok(self.builder.ins().icmp(IntCC::Equal, v, zero))
                    }
                }
            }
            ExprKind::Cast { expr: inner, to } => {
                let v = self.compile_expr(inner)?;
                let from = self.infer_ty(inner)?;
                let to_ty = resolve_cast_target(to)?;
                Ok(emit_cast_cl(self.builder, v, &from, &to_ty))
            }
            ExprKind::Call { callee, args } => self.compile_call(callee, args, expr.span),
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => self.compile_method_call(receiver, method, args, expr.span),
            ExprKind::If {
                cond,
                then_block,
                else_block,
            } => self.compile_if(cond, then_block, else_block, expr.span),
            ExprKind::Field { receiver, name } => {
                let ptr = self.compile_expr(receiver)?;
                let recv_ty = self.infer_ty(receiver)?;
                let type_name = match recv_ty {
                    Ty::User(n) => n,
                    _ => {
                        // Receiver isn't a known struct type (e.g. from
                        // list.get which returns i64). Search all struct
                        // types for one that has this field.
                        // Collect all candidates sorted alphabetically
                        // for deterministic resolution.
                        let mut candidates: Vec<&str> = Vec::new();
                        for (tname, tinfo) in &self.cg.info.types {
                            if let TypeInfo::Struct { fields, .. } = tinfo {
                                if fields.iter().any(|(f, _)| f == name) {
                                    candidates.push(tname.as_str());
                                }
                            }
                        }
                        if candidates.is_empty() {
                            return Err(NativeError {
                                span: expr.span,
                                message: format!("no struct has field `{name}`"),
                            });
                        }
                        candidates.sort();
                        candidates[0].to_string()
                    }
                };
                let fields = get_struct_fields(&self.cg.info.types, &type_name);
                let (offset, fty) = field_offset(&fields, name);
                let cl_ty = lumen_to_cl(&fty);
                Ok(self.builder.ins().load(cl_ty, MemFlags::new(), ptr, offset))
            }
            ExprKind::StructLit { name, fields, spread, .. } => {
                self.compile_struct_lit(name, fields, spread.as_deref(), expr.span)
            }
            ExprKind::Block(b) => self.compile_block(b),
            ExprKind::Match { scrutinee, arms } => {
                self.compile_match(scrutinee, arms, expr.span)
            }
            ExprKind::Try(inner) => self.compile_try(inner, expr.span),
            ExprKind::Spawn {
                actor_name, fields,
            } => self.compile_spawn(actor_name, fields, expr.span),
            ExprKind::Send {
                handle, method, args,
            } => self.compile_send(handle, method, args, expr.span),
            ExprKind::Ask {
                handle, method, args,
            } => self.compile_ask(handle, method, args, expr.span),
            ExprKind::TupleLit(elems) => {
                // Build field list from element types.
                let elem_tys: Result<Vec<Ty>, NativeError> =
                    elems.iter().map(|e| self.infer_ty(e)).collect();
                let elem_tys = elem_tys?;
                let fields = tuple_as_fields(&elem_tys);
                let size = struct_size(&fields);
                let ptr = self.rc_alloc(size as i64)?;
                for (i, elem_expr) in elems.iter().enumerate() {
                    let val = self.compile_expr(elem_expr)?;
                    let (offset, _) = field_offset(&fields, &format!("_{i}"));
                    self.builder.ins().store(MemFlags::new(), val, ptr, offset);
                    // rc_incr pointer-typed elements: the tuple now holds
                    // a reference alongside the original let binding.
                    if !is_scalar(&elem_tys[i]) {
                        self.emit_rc_incr(val);
                    }
                }
                Ok(ptr)
            }
            ExprKind::TupleField { receiver, index } => {
                let ptr = self.compile_expr(receiver)?;
                let recv_ty = self.infer_ty(receiver)?;
                match recv_ty {
                    Ty::Tuple(ref elems) => {
                        let fields = tuple_as_fields(elems);
                        let field_name = format!("_{index}");
                        let (offset, fty) = field_offset(&fields, &field_name);
                        let cl_ty = lumen_to_cl(&fty);
                        Ok(self.builder.ins().load(cl_ty, MemFlags::new(), ptr, offset))
                    }
                    _ => Err(NativeError {
                        span: expr.span,
                        message: "tuple field access on non-tuple".into(),
                    }),
                }
            }
            ExprKind::Lambda { .. } => {
                // Look up the pre-compiled lambda by its source span.
                let key = (expr.span.line, expr.span.col);
                let func_id = *self.cg.lambda_ids.get(&key).ok_or_else(|| NativeError {
                    span: expr.span,
                    message: "lambda not found (internal error)".into(),
                })?;
                let func_ref = self.cg.obj.declare_func_in_func(func_id, self.builder.func);
                Ok(self.builder.ins().func_addr(PTR, func_ref))
            }
            ExprKind::Interpolated(parts) => self.compile_interpolated(parts),
            ExprKind::Arena(block) => self.compile_arena_block(block),
            _ => Err(NativeError {
                span: expr.span,
                message: "expression not supported in native backend".into(),
            }),
        }
    }

    /// `arena { ... }` — swap the allocator for the extent of the
    /// block, emit the body, then restore and free. Every rc_alloc
    /// inside the block routes through the arena; rc_incr/decr skip
    /// sentinel-less arena memory so per-value reference counting
    /// silently becomes a no-op, and the whole region goes away at
    /// the arena's close.
    fn compile_arena_block(&mut self, block: &ast::Block) -> Result<Value, NativeError> {
        let initial = self.builder.ins().iconst(cl_types::I64, 4096);
        let new_fid = self.module_func("lumen_arena_new");
        let new_ref = self.cg.obj.declare_func_in_func(new_fid, self.builder.func);
        let new_call = self.builder.ins().call(new_ref, &[initial]);
        let arena = self.builder.inst_results(new_call)[0];

        let push_fid = self.module_func("lumen_allocator_push_arena");
        let push_ref = self.cg.obj.declare_func_in_func(push_fid, self.builder.func);
        let push_call = self.builder.ins().call(push_ref, &[arena]);
        let prev = self.builder.inst_results(push_call)[0];

        // Body — value is discarded (arena yields unit).
        let _ = self.compile_block(block)?;

        let pop_fid = self.module_func("lumen_allocator_pop");
        let pop_ref = self.cg.obj.declare_func_in_func(pop_fid, self.builder.func);
        self.builder.ins().call(pop_ref, &[prev]);

        let free_fid = self.module_func("lumen_arena_free");
        let free_ref = self.cg.obj.declare_func_in_func(free_fid, self.builder.func);
        self.builder.ins().call(free_ref, &[arena]);

        Ok(self.builder.ins().iconst(cl_types::I32, 0))
    }

    fn compile_binary(
        &mut self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        _span: Span,
    ) -> Result<Value, NativeError> {
        let lt = self.infer_ty(lhs)?;

        // String == and != : content comparison via lumen_string_eq.
        if matches!(op, BinOp::Eq | BinOp::NotEq) && matches!(lt, Ty::String) {
            let a = self.compile_expr(lhs)?;
            let b = self.compile_expr(rhs)?;
            let func_ref = self.cg.obj.declare_func_in_func(self.cg.string_eq, self.builder.func);
            let call = self.builder.ins().call(func_ref, &[a, b]);
            let eq = self.builder.inst_results(call)[0];
            if matches!(op, BinOp::NotEq) {
                let one = self.builder.ins().iconst(cl_types::I32, 1);
                return Ok(self.builder.ins().isub(one, eq)); // flip 0↔1
            }
            return Ok(eq);
        }

        // String + string.
        if matches!(op, BinOp::Add) && matches!(lt, Ty::String) {
            let a = self.compile_expr(lhs)?;
            let b = self.compile_expr(rhs)?;
            let func_ref = self
                .cg
                .obj
                .declare_func_in_func(self.cg.helper_concat, self.builder.func);
            let call = self.builder.ins().call(func_ref, &[a, b]);
            return Ok(self.builder.inst_results(call)[0]);
        }

        let a = self.compile_expr(lhs)?;
        let b = self.compile_expr(rhs)?;
        let is_f64 = matches!(lt, Ty::F64);
        let is_signed = matches!(lt, Ty::I32 | Ty::I64);

        Ok(match op {
            BinOp::Add if is_f64 => self.builder.ins().fadd(a, b),
            BinOp::Sub if is_f64 => self.builder.ins().fsub(a, b),
            BinOp::Mul if is_f64 => self.builder.ins().fmul(a, b),
            BinOp::Div if is_f64 => self.builder.ins().fdiv(a, b),
            BinOp::Add => self.builder.ins().iadd(a, b),
            BinOp::Sub => self.builder.ins().isub(a, b),
            BinOp::Mul => self.builder.ins().imul(a, b),
            BinOp::Div if is_signed => self.builder.ins().sdiv(a, b),
            BinOp::Div => self.builder.ins().udiv(a, b),
            BinOp::Rem if is_signed => self.builder.ins().srem(a, b),
            BinOp::Rem => self.builder.ins().urem(a, b),
            BinOp::Eq if is_f64 => self.builder.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::Equal, a, b),
            BinOp::NotEq if is_f64 => self.builder.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::NotEqual, a, b),
            BinOp::Lt if is_f64 => self.builder.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::LessThan, a, b),
            BinOp::LtEq if is_f64 => self.builder.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::LessThanOrEqual, a, b),
            BinOp::Gt if is_f64 => self.builder.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::GreaterThan, a, b),
            BinOp::GtEq if is_f64 => self.builder.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::GreaterThanOrEqual, a, b),
            BinOp::Eq => self.builder.ins().icmp(IntCC::Equal, a, b),
            BinOp::NotEq => self.builder.ins().icmp(IntCC::NotEqual, a, b),
            BinOp::Lt if is_signed => self.builder.ins().icmp(IntCC::SignedLessThan, a, b),
            BinOp::Lt => self.builder.ins().icmp(IntCC::UnsignedLessThan, a, b),
            BinOp::LtEq if is_signed => self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, a, b),
            BinOp::LtEq => self.builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, a, b),
            BinOp::Gt if is_signed => self.builder.ins().icmp(IntCC::SignedGreaterThan, a, b),
            BinOp::Gt => self.builder.ins().icmp(IntCC::UnsignedGreaterThan, a, b),
            BinOp::GtEq if is_signed => self.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, b),
            BinOp::GtEq => self.builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, a, b),
            BinOp::And => self.builder.ins().band(a, b),
            BinOp::Or => self.builder.ins().bor(a, b),
        })
    }

    /// If `name` is a generic user fn, ensure a monomorphization exists for the
    /// given argument types and return (mangled_name, substituted_sig).
    /// Returns None for non-generic callees so the caller can use `name`
    /// as-is. `type_args_override` lets the caller supply explicit type
    /// arguments (e.g. from the type checker's call_resolutions side
    /// table) instead of inferring solely from arg types — needed for
    /// fns like `nothing<T>(): Maybe<T>` where T can't come from args.
    fn monomorphize_if_generic(
        &mut self,
        name: &str,
        arg_tys: &[Ty],
        type_args_override: Option<Vec<Ty>>,
    ) -> Option<(String, crate::types::FnSig)> {
        let sig = self.cg.info.fns.get(name)?.clone();
        // No module context for user-module fns.
        self.do_monomorphize(name, name, &sig, arg_tys, None, type_args_override)
    }

    /// Like monomorphize_if_generic, but for module-qualified calls
    /// (e.g. list.map). The template is stored under the qualified name
    /// "module.fn" so the user-fn and module-fn paths don't collide.
    fn monomorphize_module_method_if_generic(
        &mut self,
        module: &str,
        method: &str,
        arg_tys: &[Ty],
        type_args_override: Option<Vec<Ty>>,
    ) -> Option<(String, crate::types::FnSig)> {
        let sig = self.cg.info.modules.get(module)?.get(method)?.clone();
        let qualified = format!("{module}.{method}");
        self.do_monomorphize(&qualified, &qualified, &sig, arg_tys, Some(module.to_string()), type_args_override)
    }

    /// Walk `ty`, finding deferred `Ty::User` names (registered by the
    /// type-checker pre-pass with Generic args — e.g. `Pair$GT_I32`
    /// from `fn make<T>(): Pair<T, i32>`'s sig). For each, substitute
    /// its stored generic args via `subs` to get concrete args, register
    /// the concrete instantiation if not already present, and replace
    /// the Ty::User name with the concrete mangled name.
    /// True if `ty` is (or contains) a deferred generic-type instantiation —
    /// i.e. a Ty::User registered in `generic_type_args` whose args still
    /// mention Ty::Generic. These arise when infer_ty walks a generic
    /// call site without access to the type-checker's call_resolutions
    /// (e.g. swap(p1) infers Pair$GB_GA until the annotation refines it).
    fn is_deferred_user_ty(&self, ty: &Ty) -> bool {
        match ty {
            Ty::Generic(_) => true,
            Ty::User(name) => {
                self.cg.info.generic_type_args.get(name)
                    .map(|(_, args)| args.iter().any(|a| self.is_deferred_user_ty(a)))
                    .unwrap_or(false)
            }
            Ty::List(i) | Ty::Option(i) | Ty::Handle(i) => self.is_deferred_user_ty(i),
            Ty::Map(k, v) => self.is_deferred_user_ty(k) || self.is_deferred_user_ty(v),
            Ty::Result(o, e) => self.is_deferred_user_ty(o) || self.is_deferred_user_ty(e),
            Ty::Tuple(elems) => elems.iter().any(|t| self.is_deferred_user_ty(t)),
            Ty::FnPtr { params, ret } => {
                params.iter().any(|p| self.is_deferred_user_ty(p))
                    || self.is_deferred_user_ty(ret)
            }
            _ => false,
        }
    }

    fn concretize_ty(&mut self, ty: &Ty, subs: &HashMap<String, Ty>) -> Ty {
        match ty {
            Ty::User(name) => {
                let entry = self.cg.info.generic_type_args.get(name).cloned();
                if let Some((tmpl, args)) = entry {
                    let concrete_args: Vec<Ty> = args.into_iter()
                        .map(|a| substitute_ty(a, subs))
                        .map(|a| self.concretize_ty(&a, subs))
                        .collect();
                    let new_mangled = crate::types::mangle_type_instantiation(&tmpl, &concrete_args);
                    if !self.cg.info.types.contains_key(&new_mangled) {
                        let mut errors = Vec::new();
                        crate::types::register_one_generic_instantiation(
                            &tmpl, &concrete_args, self.cg.info, &mut errors,
                        );
                        // Errors here would mean a malformed template arity —
                        // already caught by the type checker. Ignore.
                    }
                    Ty::User(new_mangled)
                } else {
                    ty.clone()
                }
            }
            Ty::List(inner) => Ty::List(Box::new(self.concretize_ty(inner, subs))),
            Ty::Map(k, v) => Ty::Map(Box::new(self.concretize_ty(k, subs)), Box::new(self.concretize_ty(v, subs))),
            Ty::Option(inner) => Ty::Option(Box::new(self.concretize_ty(inner, subs))),
            Ty::Result(o, e) => Ty::Result(Box::new(self.concretize_ty(o, subs)), Box::new(self.concretize_ty(e, subs))),
            Ty::Handle(inner) => Ty::Handle(Box::new(self.concretize_ty(inner, subs))),
            Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(|t| self.concretize_ty(t, subs)).collect()),
            Ty::FnPtr { params, ret } => Ty::FnPtr {
                params: params.iter().map(|p| self.concretize_ty(p, subs)).collect(),
                ret: Box::new(self.concretize_ty(ret, subs)),
            },
            _ => ty.clone(),
        }
    }

    /// Shared monomorphization logic. `qualified_name` is the base for
    /// name mangling; `template_key` is the key into `generic_templates`.
    /// `module` records which imported module owns the template (so the
    /// monomorphized body's unqualified calls resolve via the right
    /// local_fn_ids); None for templates from the user's module.
    fn do_monomorphize(
        &mut self,
        qualified_name: &str,
        template_key: &str,
        sig: &crate::types::FnSig,
        arg_tys: &[Ty],
        module: Option<String>,
        type_args_override: Option<Vec<Ty>>,
    ) -> Option<(String, crate::types::FnSig)> {
        if sig.type_params.is_empty() {
            return None;
        }
        // Generic externs share one C symbol across T instantiations
        // — there's no body to specialize. The compile_call extern
        // branch handles the widen/narrow at the boundary.
        if sig.is_extern {
            return None;
        }

        // If the type checker resolved the type args from context (args
        // + expected return), use those directly. Otherwise infer from
        // arg types alone — works for the common case where T flows
        // from a fn parameter.
        //
        // The override may contain Ty::Generic when the call appears
        // inside a generic fn body (e.g. id(id(x)) inside id_twice<T>'s
        // body, where the type checker recorded [Generic("T")]). Apply
        // the current monomorphization's active_subs so we resolve to
        // the concrete instantiation (id$I32) rather than a deferred
        // one (id$GT) that nobody compiles.
        let active_subs = self.active_subs.clone();
        let type_args: Vec<Ty> = if let Some(args) = type_args_override {
            args.into_iter()
                .map(|a| substitute_ty(a, &active_subs))
                .map(|a| self.concretize_ty(&a, &active_subs))
                .collect()
        } else {
            let mut subs: HashMap<String, Ty> = HashMap::new();
            for (i, (_, pty)) in sig.params.iter().enumerate() {
                if let Some(at) = arg_tys.get(i) {
                    unify_into(pty, at, &mut subs);
                }
            }
            sig.type_params.iter()
                .map(|p| subs.get(p).cloned().unwrap_or(Ty::Error))
                .collect()
        };
        // Build subs map from type_args for the substitution + concretize
        // step below.
        let subs: HashMap<String, Ty> = sig.type_params.iter()
            .zip(type_args.iter())
            .map(|(p, a)| (p.clone(), a.clone()))
            .collect();
        let mangled = mangle_monomorph_name(qualified_name, &type_args);

        let subbed_params: Vec<(String, Ty)> = sig.params.iter()
            .map(|(n, t)| (n.clone(), substitute_ty(t.clone(), &subs)))
            .collect();
        let subbed_ret = substitute_ty(sig.ret.clone(), &subs);
        // Concretize any deferred Ty::User in the substituted sig — e.g.
        // `Pair$GT_I32` from `fn make<T>(): Pair<T, i32>` becomes
        // `Pair$I32_I32` here when monomorphizing make<I32>.
        let subbed_params: Vec<(String, Ty)> = subbed_params.into_iter()
            .map(|(n, t)| (n, self.concretize_ty(&t, &subs)))
            .collect();
        let subbed_ret = self.concretize_ty(&subbed_ret, &subs);
        let subbed_sig = crate::types::FnSig {
            params: subbed_params,
            ret: subbed_ret,
            effect: sig.effect,
            type_params: Vec::new(),
            is_extern: sig.is_extern,
        };

        if !self.cg.monomorph_done.contains(&mangled) {
            let cl_sig = self.cg.build_sig_from(&subbed_sig);
            let func_id = self.cg.obj
                .declare_function(&mangled, Linkage::Local, &cl_sig)
                .unwrap();
            self.cg.fn_ids.insert(mangled.clone(), func_id);
            self.cg.monomorph_done.insert(mangled.clone());
            if let Some(decl) = self.cg.generic_templates.get(template_key).cloned() {
                self.cg.monomorph_queue.push(MonomorphRequest {
                    mangled_name: mangled.clone(),
                    func_id,
                    sig: subbed_sig.clone(),
                    subs,
                    decl,
                    current_module: module,
                });
            }
        }

        Some((mangled, subbed_sig))
    }

    fn compile_call(
        &mut self,
        callee: &Expr,
        args: &[ast::Arg],
        span: Span,
    ) -> Result<Value, NativeError> {
        let name = match &callee.kind {
            ExprKind::Ident(n) => n.clone(),
            _ => {
                return Err(NativeError {
                    span,
                    message: "only direct calls".into(),
                })
            }
        };

        // Built-in string_len.
        if name == "string_len" {
            let s = self.compile_expr(&args[0].value)?;
            let len = self.builder.ins().load(cl_types::I32, MemFlags::new(), s, 0);
            return Ok(len);
        }

        // Built-in assert(cond) / assert(cond, msg).
        if name == "assert" {
            let cond = self.compile_expr(&args[0].value)?;
            let msg = if args.len() >= 2 {
                self.compile_expr(&args[1].value)?
            } else {
                self.builder.ins().iconst(PTR, 0)
            };
            let file_ptr = if !self.cg.source_path.is_empty() {
                let data_id = *self.cg.string_data.get(&self.cg.source_path).unwrap();
                let gv = self.cg.obj.declare_data_in_func(data_id, self.builder.func);
                self.builder.ins().global_value(PTR, gv)
            } else {
                self.builder.ins().iconst(PTR, 0)
            };
            let line = self.builder.ins().iconst(cl_types::I32, span.line as i64);
            let col = self.builder.ins().iconst(cl_types::I32, span.col as i64);
            let dbg = self.builder.ins().iconst(cl_types::I32, if self.cg.debug_mode { 1 } else { 0 });
            let func_ref = self.cg.obj.declare_func_in_func(self.cg.helper_assert, self.builder.func);
            self.builder.ins().call(func_ref, &[cond, msg, file_ptr, line, col, dbg]);
            return Ok(self.builder.ins().iconst(cl_types::I32, 0));
        }

        // __rc_incr<T>(v) / __rc_decr<T>(v): compile-time-conditional
        // rc adjustment. Inside a generic fn body, T can be Ty::Generic;
        // active_subs resolves it at monomorphization. If T is scalar
        // (or resolves to scalar), the call compiles to nothing.
        // Needed so pure-Lumen stdlib wrappers over raw externs can
        // manage ref counts without a runtime "is it a ptr" probe.
        if name == "__rc_incr" || name == "__rc_decr" {
            if let Some(arg) = args.first() {
                let arg_ty = self.infer_ty(&arg.value)?;
                let resolved = substitute_ty(arg_ty, &self.active_subs);
                if !is_scalar(&resolved) {
                    let val = self.compile_expr(&arg.value)?;
                    if name == "__rc_incr" {
                        self.emit_rc_incr(val);
                    } else {
                        self.emit_rc_decr_typed(val, &resolved);
                    }
                } else {
                    // Still evaluate the arg for any side effects,
                    // then discard. Scalars are free.
                    let _ = self.compile_expr(&arg.value)?;
                }
            }
            return Ok(self.builder.ins().iconst(cl_types::I32, 0));
        }

        // __is_ptr<T>(v): compile-time 1 or 0 based on whether T
        // resolves to a pointer-shaped type. The arg is evaluated for
        // side effects, then its value is discarded.
        if name == "__is_ptr" {
            if let Some(arg) = args.first() {
                let arg_ty = self.infer_ty(&arg.value)?;
                let resolved = substitute_ty(arg_ty, &self.active_subs);
                let _ = self.compile_expr(&arg.value)?;
                let flag = if is_scalar(&resolved) { 0 } else { 1 };
                return Ok(self.builder.ins().iconst(cl_types::I32, flag));
            }
            return Ok(self.builder.ins().iconst(cl_types::I32, 0));
        }

        // Built-in Option/Result constructors.
        match name.as_str() {
            "Ok" => return self.compile_single_field_constructor(0, &args[0].value),
            "Err" => return self.compile_single_field_constructor(1, &args[0].value),
            "Some" => return self.compile_single_field_constructor(1, &args[0].value),
            "None" => return self.build_sum_block(0, None),
            _ => {}
        }

        // User variant constructor (positional payload)?
        // Prefer the type-checker's recorded resolution (if the call was
        // type-checked against an expected generic-sum instantiation),
        // otherwise fall back to first-match across all sums.
        if let Some(sum_name) = self.resolved_variant_sum(span)
            .or_else(|| self.find_sum_for_variant(&name))
        {
            let tag = self.variant_tag(&Ty::User(sum_name.clone()), &name).unwrap_or(0);
            if args.is_empty() {
                return self.build_sum_block(tag, None);
            }
            // Positional variant: allocate payload with fields.
            let scrut_ty = Ty::User(sum_name);
            let fields = self.variant_field_types(&scrut_ty, &name).unwrap_or_default();
            let layout = fields.clone();
            let payload = self.build_payload_block(&layout, args, span)?;
            return self.build_sum_block(tag, Some(payload));
        }

        // Indirect call through a local FnPtr or I64 variable.
        if let Some(&var) = self.names.get(&name) {
            let local_ty = self.name_types.get(&name).cloned();
            let func_ptr_val = self.builder.use_var(var);
            let mut sig = self.cg.obj.make_signature();
            let indirect = match &local_ty {
                Some(Ty::FnPtr { params, ret }) => {
                    for p in params {
                        sig.params.push(AbiParam::new(lumen_to_cl(p)));
                    }
                    if **ret != Ty::Unit {
                        sig.returns.push(AbiParam::new(lumen_to_cl(ret)));
                    } else {
                        sig.returns.push(AbiParam::new(cl_types::I32));
                    }
                    true
                }
                // Opaque i64 fn address: infer signature from call-site args.
                // Return type assumed i32 (the most common case for MVPs).
                Some(Ty::I64) => {
                    for a in args {
                        let arg_ty = self.infer_ty(&a.value)?;
                        sig.params.push(AbiParam::new(lumen_to_cl(&arg_ty)));
                    }
                    sig.returns.push(AbiParam::new(cl_types::I32));
                    true
                }
                _ => false,
            };
            if indirect {
                let mut arg_vals = Vec::new();
                for a in args {
                    let val = self.compile_expr(&a.value)?;
                    arg_vals.push(val);
                }
                let sig_ref = self.builder.import_signature(sig);
                let call = self.builder.ins().call_indirect(sig_ref, func_ptr_val, &arg_vals);
                return Ok(self.builder.inst_results(call)[0]);
            }
        }

        // User function call. Compile args first — we need their types to
        // monomorphize generic callees.
        let mut arg_vals = Vec::new();
        let mut arg_tys = Vec::new();
        for a in args {
            arg_vals.push(self.compile_expr(&a.value)?);
            arg_tys.push(self.infer_ty(&a.value)?);
        }

        // If the callee is generic, dispatch to a monomorphization (creating
        // it on first sight). Otherwise call the original name. Pass any
        // type-args the type checker resolved from context (for arg-less
        // generic calls like `nothing<T>(): Maybe<T>`).
        let type_args_override = self.cg.info.call_resolutions.get(&span.start).cloned();
        // Intra-module unqualified call to a generic Lumen fn (not
        // an extern) declared in the current imported module —
        // monomorphize_if_generic only looks in info.fns; these
        // live in info.modules[current_module]. Generic externs
        // don't need monomorphization (one C symbol shared across T).
        let module_generic_non_extern = self.current_module.as_ref().and_then(|m| {
            self.cg.info.modules.get(m).and_then(|sigs| {
                sigs.get(&name).filter(|s| !s.type_params.is_empty() && !s.is_extern).cloned()
            }).map(|_| m.clone())
        });
        let (effective_name, effective_sig) = if let Some(m) = module_generic_non_extern {
            match self.monomorphize_module_method_if_generic(
                &m, &name, &arg_tys, type_args_override.clone(),
            ) {
                Some(pair) => (pair.0, Some(pair.1)),
                None => (name.clone(), None),
            }
        } else {
            match self.monomorphize_if_generic(&name, &arg_tys, type_args_override) {
                Some(pair) => (pair.0, Some(pair.1)),
                None => {
                    // Look up the sig in the current module first (so that
                    // intra-module unqualified calls inside an imported
                    // module — e.g. `substring(s, ...)` inside std/string's
                    // own `split` — find their sig and trigger rc_incr on
                    // pointer args). Fall back to user-module fns.
                    let sig = self.current_module
                        .as_ref()
                        .and_then(|m| self.cg.info.modules.get(m))
                        .and_then(|m| m.get(&name))
                        .cloned()
                        .or_else(|| self.cg.info.fns.get(&name).cloned());
                    (name.clone(), sig)
                }
            }
        };

        // Inside a module body, unqualified calls resolve via the
        // module's local_fn_ids first (avoids global-fn_ids collisions
        // when multiple stdlib modules export the same short name —
        // e.g. std/list.new vs std/map.new).
        let func_id = self.current_module
            .as_ref()
            .and_then(|m| self.cg.local_fn_ids.get(m))
            .and_then(|m| m.get(&effective_name))
            .copied()
            .or_else(|| self.cg.fn_ids.get(&effective_name).copied())
            .ok_or_else(|| NativeError {
                span,
                message: format!("unknown function `{name}`"),
            })?;
        let func_id = &func_id;
        let func_ref = self
            .cg
            .obj
            .declare_func_in_func(*func_id, self.builder.func);

        // Extern callee: widen narrow scalars to the I64-shaped ABI
        // slots (especially for generic externs where T-typed params
        // are registered as I64 but the call site may pass I32/Char).
        // Narrow the return similarly.
        if let Some(sig) = effective_sig.as_ref() {
            if sig.is_extern {
                let subs = self.current_call_subs(&name, span);
                let mut widened = Vec::with_capacity(arg_vals.len());
                for (i, val) in arg_vals.iter().enumerate() {
                    if let Some((_, param_ty)) = sig.params.get(i) {
                        let declared = substitute_ty(param_ty.clone(), &subs);
                        let arg_ty = &arg_tys[i];
                        let w = self.convert_for_call(*val, arg_ty, &declared);
                        let w = self.convert_for_call(w, &declared, param_ty);
                        widened.push(w);
                    } else {
                        widened.push(*val);
                    }
                }
                let call = self.builder.ins().call(func_ref, &widened);
                let results = self.builder.inst_results(call);
                if results.is_empty() {
                    return Ok(self.builder.ins().iconst(cl_types::I32, 0));
                }
                let result = results[0];
                let declared_ret = substitute_ty(sig.ret.clone(), &subs);
                return Ok(self.convert_for_call(result, &sig.ret, &declared_ret));
            }
        }

        // rc_incr each pointer argument so the callee's scope-exit
        // decr doesn't free values the caller still holds.
        if let Some(sig) = &effective_sig {
            for (i, (_, pty)) in sig.params.iter().enumerate() {
                if !is_scalar(pty) {
                    if let Some(&val) = arg_vals.get(i) {
                        self.emit_rc_incr(val);
                    }
                }
            }
        }
        let call = self.builder.ins().call(func_ref, &arg_vals);
        let results = self.builder.inst_results(call);
        if results.is_empty() {
            // Void function — return a unit placeholder.
            Ok(self.builder.ins().iconst(cl_types::I32, 0))
        } else {
            Ok(results[0])
        }
    }

    /// Build the T-substitution map for a call expression: from the
    /// typechecker's recorded call_resolutions[span] plus the current
    /// monomorphization's active_subs. Empty when the call isn't
    /// generic or has no resolution.
    fn current_call_subs(&self, name: &str, span: Span) -> HashMap<String, Ty> {
        let mut subs = HashMap::new();
        if let Some(type_args) = self.cg.info.call_resolutions.get(&span.start) {
            // Resolve type params from whichever source the name
            // lives under. For user-module fns it's info.fns; for
            // imported modules (when this is called via method-call
            // dispatch it's already handled separately).
            let sig = self.cg.info.fns.get(name).cloned().or_else(|| {
                self.current_module
                    .as_ref()
                    .and_then(|m| self.cg.info.modules.get(m))
                    .and_then(|m| m.get(name))
                    .cloned()
            });
            if let Some(sig) = sig {
                for (tp, ty) in sig.type_params.iter().zip(type_args.iter()) {
                    let resolved = substitute_ty(ty.clone(), &self.active_subs);
                    subs.insert(tp.clone(), resolved);
                }
            }
        }
        subs
    }

    // --- Method call helpers ------------------------------------------------

    /// Compile args, call a void builtin FuncId, return unit (i32 0).
    fn call_builtin_void(&mut self, func_id: FuncId, args: &[ast::Arg]) -> Result<Value, NativeError> {
        let mut vals = Vec::new();
        for a in args { vals.push(self.compile_expr(&a.value)?); }
        let func_ref = self.cg.obj.declare_func_in_func(func_id, self.builder.func);
        self.builder.ins().call(func_ref, &vals);
        Ok(self.builder.ins().iconst(cl_types::I32, 0))
    }

    /// Compile args, call a FuncId with Lumen-sig-aware argument
    /// widening (generic externs live in an i64 slot at the C ABI but
    /// callers pass their natural-width Lumen types — sextend / ireduce
    /// / bitcast bridges the gap). Return value is narrowed back down
    /// if the monomorphized return type is narrower than the registered
    /// slot.
    fn call_builtin_typed_sub(
        &mut self,
        func_id: FuncId,
        args: &[ast::Arg],
        sig: Option<&crate::types::FnSig>,
        subs: &HashMap<String, Ty>,
    ) -> Result<Value, NativeError> {
        let mut vals = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let val = self.compile_expr(&a.value)?;
            let val = if let Some(s) = sig {
                if let Some((_, param_ty)) = s.params.get(i) {
                    let declared = substitute_ty(param_ty.clone(), subs);
                    // The registered Cranelift sig used the ORIGINAL
                    // param type via lumen_to_cl (so Generic → I64).
                    // We need to widen the arg from the declared-post-sub
                    // Cranelift type to that original-registered type.
                    let arg_ty = self.infer_ty(&a.value)?;
                    let widened = self.convert_for_call(val, &arg_ty, &declared);
                    // Further widen from declared-post-sub to registered-pre-sub.
                    self.convert_for_call(widened, &declared, param_ty)
                } else { val }
            } else { val };
            vals.push(val);
        }
        let func_ref = self.cg.obj.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &vals);
        let result = self.builder.inst_results(call)[0];
        if let Some(s) = sig {
            let declared_ret = substitute_ty(s.ret.clone(), subs);
            // Narrow return from registered-pre-sub CL type down to
            // declared-post-sub CL type if narrower (generic T
            // monomorphized to i32/char/etc.).
            Ok(self.convert_for_call(result, &s.ret, &declared_ret))
        } else {
            Ok(result)
        }
    }

    /// Convert a compiled value between two Lumen types that share a
    /// Cranelift representation mismatch (scalar widening / narrowing
    /// at FFI boundary). Returns the original value untouched when no
    /// conversion is needed.
    fn convert_for_call(&mut self, val: Value, from: &Ty, to: &Ty) -> Value {
        let from_cl = lumen_to_cl(from);
        let to_cl = lumen_to_cl(to);
        if from_cl == to_cl {
            return val;
        }
        match (from_cl, to_cl) {
            (a, b) if a == cl_types::I32 && b == cl_types::I64 => {
                self.builder.ins().sextend(cl_types::I64, val)
            }
            (a, b) if a == cl_types::I64 && b == cl_types::I32 => {
                self.builder.ins().ireduce(cl_types::I32, val)
            }
            (a, b) if a == cl_types::F64 && b == cl_types::I64 => {
                self.builder.ins().bitcast(cl_types::I64, MemFlags::new(), val)
            }
            (a, b) if a == cl_types::I64 && b == cl_types::F64 => {
                self.builder.ins().bitcast(cl_types::F64, MemFlags::new(), val)
            }
            _ => val,
        }
    }

    /// Look up a module function's FuncId by its C link name.
    fn module_func(&mut self, link_name: &str) -> FuncId {
        *self.cg.module_fn_ids.get(link_name)
            .expect(&format!("module function not declared: {link_name}"))
    }

    fn compile_method_call(
        &mut self,
        receiver: &Expr,
        method: &str,
        args: &[ast::Arg],
        span: Span,
    ) -> Result<Value, NativeError> {
        if let ExprKind::Ident(mod_name) = &receiver.kind {
            // --- Special cases that need custom codegen ---

            // debug.print: compile-time specialized formatting (stderr)
            if mod_name == "debug" && method == "print" {
                let val = self.compile_expr(&args[0].value)?;
                let ty = self.infer_ty(&args[0].value)?;
                self.emit_fmt_value(PrintTarget::Stderr, val, &ty)?;
                let nl_ref = self.cg.obj.declare_func_in_func(self.cg.debug_newline, self.builder.func);
                self.builder.ins().call(nl_ref, &[]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }

            // io.println: compile-time specialized formatting (stdout).
            // For string args, fall through to the existing lumen_println
            // extern (faster — single FFI call).
            if mod_name == "io" && method == "println" && args.len() == 1 {
                let ty = self.infer_ty(&args[0].value)?;
                if !matches!(ty, Ty::String) {
                    let val = self.compile_expr(&args[0].value)?;
                    self.emit_fmt_value(PrintTarget::Stdout, val, &ty)?;
                    let nl_ref = self.cg.obj.declare_func_in_func(self.cg.io_newline, self.builder.func);
                    self.builder.ins().call(nl_ref, &[]);
                    return Ok(self.builder.ins().iconst(cl_types::I32, 0));
                }
            }

            // net.tcp_write: call + ireduce i64→i32
            if mod_name == "net" && method == "tcp_write" {
                let fd = self.compile_expr(&args[0].value)?;
                let data = self.compile_expr(&args[1].value)?;
                let fid = self.module_func("lumen_tcp_write");
                let func_ref = self.cg.obj.declare_func_in_func(fid, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[fd, data]);
                let result = self.builder.inst_results(call)[0];
                return Ok(self.builder.ins().ireduce(cl_types::I32, result));
            }
            // net.serve: second arg is func_addr
            if mod_name == "net" && method == "serve" {
                let port = self.compile_expr(&args[0].value)?;
                let handler_name = match &args[1].value.kind {
                    ExprKind::Ident(n) => n.clone(),
                    _ => return Err(NativeError { span, message: "net.serve: second arg must be a function name".into() }),
                };
                let handler_id = *self.cg.fn_ids.get(&handler_name).ok_or_else(|| {
                    NativeError { span, message: format!("unknown function `{handler_name}`") }
                })?;
                let handler_ref = self.cg.obj.declare_func_in_func(handler_id, self.builder.func);
                let handler_addr = self.builder.ins().func_addr(PTR, handler_ref);
                let serve_fid = self.module_func("lumen_net_serve");
                let serve_ref = self.cg.obj.declare_func_in_func(serve_fid, self.builder.func);
                self.builder.ins().call(serve_ref, &[port, handler_addr]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            // list.new / list.push / list.get / list.set used to live
            // here as codegen specials. They're now pure-Lumen generic
            // wrappers in std/list.lm over the _new_raw / _push_raw /
            // _get_raw / _set_raw externs — lumen-d86.
            // map.new(): uses the typechecker's recorded [K, V] in
            // call_resolutions (populated by check_expr when there's
            // a let/var annotation on the binding). Defaults to
            // string-keyed when no context — matches previous behavior.
            if mod_name == "map" && method == "new" {
                let resolved = self.cg.info.call_resolutions.get(&span.start).cloned();
                let key_is_ptr = match resolved.as_ref().and_then(|v| v.first()) {
                    Some(k) => !is_scalar(k),
                    None => true,
                };
                let flag = self.builder.ins().iconst(cl_types::I32, if key_is_ptr { 1 } else { 0 });
                let fid = self.module_func("lumen_map_new");
                let func_ref = self.cg.obj.declare_func_in_func(fid, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[flag]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            // map.remove: widen key to i64; pass value_is_ptr so C
            // can rc_decr the removed value when appropriate.
            if mod_name == "map" && method == "remove" {
                let m = self.compile_expr(&args[0].value)?;
                let k_raw = self.compile_expr(&args[1].value)?;
                let map_ty = self.infer_ty(&args[0].value)?;
                let (key_ty, val_ty) = match map_ty {
                    Ty::Map(k, v) => (*k, *v),
                    _ => (Ty::String, Ty::Error),
                };
                let value_is_ptr = !is_scalar(&val_ty);
                let k64 = self.widen_to_i64(k_raw, &key_ty);
                let flag = self.builder.ins().iconst(cl_types::I32, if value_is_ptr { 1 } else { 0 });
                let fid = self.module_func("lumen_map_remove");
                let func_ref = self.cg.obj.declare_func_in_func(fid, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[m, k64, flag]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "map" && method == "merge" {
                let a = self.compile_expr(&args[0].value)?;
                let b = self.compile_expr(&args[1].value)?;
                let map_ty = self.infer_ty(&args[0].value)?;
                let val_ty = match map_ty { Ty::Map(_, v) => *v, _ => Ty::Error };
                let value_is_ptr = !is_scalar(&val_ty);
                let flag = self.builder.ins().iconst(cl_types::I32, if value_is_ptr { 1 } else { 0 });
                let fid = self.module_func("lumen_map_merge");
                let func_ref = self.cg.obj.declare_func_in_func(fid, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[a, b, flag]);
                return Ok(self.builder.inst_results(call)[0]);
            }
        }
        // Fall through to imported module lookup.
        if let ExprKind::Ident(mod_name) = &receiver.kind {
            let is_void = self.cg.info.modules
                .get(mod_name.as_str())
                .and_then(|m| m.get(method))
                .map(|sig| sig.ret == Ty::Unit)
                .unwrap_or(false);

            // Check extern fn by link name.
            if let Some(link_name) = self.cg.info.module_link_names
                .get(mod_name.as_str())
                .and_then(|m| m.get(method))
            {
                if let Some(&func_id) = self.cg.module_fn_ids.get(link_name.as_str()) {
                    let sig = self.cg.info.modules
                        .get(mod_name.as_str())
                        .and_then(|m| m.get(method))
                        .cloned();
                    // Generic externs: look up the typechecker's
                    // resolved T-args for this call site (if any) so
                    // the return-side narrowing knows its target.
                    let mut subs = HashMap::new();
                    if let Some(sig_ref) = &sig {
                        if !sig_ref.type_params.is_empty() {
                            if let Some(type_args) = self.cg.info.call_resolutions.get(&span.start) {
                                for (tp, ty) in sig_ref.type_params.iter().zip(type_args.iter()) {
                                    let resolved = substitute_ty(ty.clone(), &self.active_subs);
                                    subs.insert(tp.clone(), resolved);
                                }
                            }
                        }
                    }
                    return if is_void {
                        self.call_builtin_void(func_id, args)
                    } else {
                        self.call_builtin_typed_sub(func_id, args, sig.as_ref(), &subs)
                    };
                }
            }
            // Generic Lumen fn in an imported module: monomorphize per
            // call-site arg types, then call the specialized FuncId.
            // Has to come before the non-generic Lumen-fn lookup because
            // generic fns aren't registered there.
            let is_generic = self.cg.info.modules
                .get(mod_name.as_str())
                .and_then(|m| m.get(method))
                .map(|sig| !sig.type_params.is_empty())
                .unwrap_or(false);
            if is_generic {
                // Compile args first to know their types.
                let mut arg_vals = Vec::with_capacity(args.len());
                let mut arg_tys = Vec::with_capacity(args.len());
                for a in args {
                    arg_vals.push(self.compile_expr(&a.value)?);
                    arg_tys.push(self.infer_ty(&a.value)?);
                }
                let type_args_override = self.cg.info.call_resolutions.get(&span.start).cloned();
                if let Some((mangled, subbed_sig)) =
                    self.monomorphize_module_method_if_generic(mod_name, method, &arg_tys, type_args_override)
                {
                    let func_id = *self.cg.fn_ids.get(&mangled).unwrap();
                    let func_ref = self.cg.obj.declare_func_in_func(func_id, self.builder.func);
                    // rc_incr each pointer-typed argument (caller-side ref count).
                    for (i, (_, pty)) in subbed_sig.params.iter().enumerate() {
                        if !is_scalar(pty) {
                            if let Some(&val) = arg_vals.get(i) {
                                self.emit_rc_incr(val);
                            }
                        }
                    }
                    let call = self.builder.ins().call(func_ref, &arg_vals);
                    let results = self.builder.inst_results(call);
                    if results.is_empty() {
                        return Ok(self.builder.ins().iconst(cl_types::I32, 0));
                    }
                    return Ok(results[0]);
                }
            }

            // Check Lumen fn by mod_name:method key.
            let fn_key = format!("{mod_name}:{method}");
            if let Some(&func_id) = self.cg.module_fn_ids.get(&fn_key) {
                // Non-generic Lumen module fn (e.g. string.char_at).
                // Compile args, rc_incr non-scalar ones (the callee's
                // scope-exit will rc_decr its params; without the incr
                // here the caller's pointer would get freed mid-use —
                // which is exactly what was breaking string.char_at(text)
                // and friends, see lumen-358).
                let mut arg_vals = Vec::with_capacity(args.len());
                for a in args {
                    arg_vals.push(self.compile_expr(&a.value)?);
                }
                let sig = self.cg.info.modules.get(mod_name)
                    .and_then(|m| m.get(method))
                    .cloned();
                if let Some(sig) = &sig {
                    for (i, (_, pty)) in sig.params.iter().enumerate() {
                        if !is_scalar(pty) {
                            if let Some(&val) = arg_vals.get(i) {
                                self.emit_rc_incr(val);
                            }
                        }
                    }
                }
                let func_ref = self.cg.obj.declare_func_in_func(func_id, self.builder.func);
                let call = self.builder.ins().call(func_ref, &arg_vals);
                let results = self.builder.inst_results(call);
                if is_void || results.is_empty() {
                    return Ok(self.builder.ins().iconst(cl_types::I32, 0));
                }
                return Ok(results[0]);
            }
        }
        Err(NativeError {
            span,
            message: format!("method `.{method}()` not supported in native backend"),
        })
    }

    fn compile_if(
        &mut self,
        cond: &Expr,
        then_block: &ast::Block,
        else_block: &ast::Block,
        _span: Span,
    ) -> Result<Value, NativeError> {
        let cond_val = self.compile_expr(cond)?;
        let then_bb = self.builder.create_block();
        let else_bb = self.builder.create_block();
        let merge_bb = self.builder.create_block();

        let result_ty = self
            .infer_block_ty(then_block)
            .map(|t| lumen_to_cl(&t))
            .unwrap_or(cl_types::I32);
        self.builder.append_block_param(merge_bb, result_ty);

        self.builder.ins().brif(cond_val, then_bb, &[], else_bb, &[]);

        self.builder.switch_to_block(then_bb);
        self.hit_return = false;
        let then_val = self.compile_block(then_block)?;
        let then_returned = self.hit_return;
        // If the block returned, the current block is a dead block.
        // Emit a correctly-typed dummy for the merge param.
        let then_merge_val = if then_returned {
            self.builder.ins().iconst(result_ty, 0)
        } else {
            then_val
        };
        self.builder.ins().jump(merge_bb, &[BlockArg::Value(then_merge_val)]);

        self.builder.switch_to_block(else_bb);
        self.hit_return = false;
        let else_val = self.compile_block(else_block)?;
        let else_returned = self.hit_return;
        let else_merge_val = if else_returned {
            self.builder.ins().iconst(result_ty, 0)
        } else {
            else_val
        };
        self.builder.ins().jump(merge_bb, &[BlockArg::Value(else_merge_val)]);

        self.builder.switch_to_block(merge_bb);
        // (sealed later)
        Ok(self.builder.block_params(merge_bb)[0])
    }

    fn compile_for(
        &mut self,
        binder: &str,
        iter: &Expr,
        body: &ast::Block,
        _span: Span,
    ) -> Result<(), NativeError> {
        let (start_expr, end_expr) = match &iter.kind {
            ExprKind::Call { callee, args }
                if matches!(&callee.kind, ExprKind::Ident(n) if n == "range")
                    && args.len() == 2 =>
            {
                (&args[0].value, &args[1].value)
            }
            _ => {
                return Err(NativeError {
                    span: iter.span,
                    message: "only range(start, end) supported".into(),
                })
            }
        };

        let start = self.compile_expr(start_expr)?;
        let end = self.compile_expr(end_expr)?;

        let counter_var = self.fresh_var(cl_types::I32);
        self.builder.def_var(counter_var, start);

        let binder_var = self.fresh_var(cl_types::I32);
        self.builder.def_var(binder_var, start);
        self.names.insert(binder.to_string(), binder_var);

        let header_bb = self.builder.create_block();
        let body_bb = self.builder.create_block();
        let exit_bb = self.builder.create_block();

        self.builder.ins().jump(header_bb, &[]);

        self.builder.switch_to_block(header_bb);
        // Yield point disabled for now — interferes with raylib games.
        // TODO: only emit yield when the program imports std/net or uses actors.

        let counter = self.builder.use_var(counter_var);
        let done = self.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, counter, end);
        self.builder.ins().brif(done, exit_bb, &[], body_bb, &[]);

        self.builder.switch_to_block(body_bb);
        // (sealed later)
        let counter = self.builder.use_var(counter_var);
        self.builder.def_var(binder_var, counter);

        self.compile_block(body)?;

        let counter = self.builder.use_var(counter_var);
        let one = self.builder.ins().iconst(cl_types::I32, 1);
        let next = self.builder.ins().iadd(counter, one);
        self.builder.def_var(counter_var, next);
        self.builder.ins().jump(header_bb, &[]);

        // (sealed later)
        self.builder.switch_to_block(exit_bb);
        // (sealed later)

        Ok(())
    }

    /// If `name` is a generic struct template, infer its concrete
    /// instantiation by walking the struct literal's field values to
    /// derive the type-param substitution, then return the mangled
    /// name (already registered by the type-checker pre-pass). For a
    /// non-generic struct, returns `name` unchanged.
    fn resolve_struct_lit_name(
        &mut self,
        name: &str,
        fields: &[ast::FieldInit],
    ) -> Result<String, NativeError> {
        let template = match self.cg.info.generic_type_templates.get(name).cloned() {
            Some(t) => t,
            None => return Ok(name.to_string()),
        };
        let template_fields = match &template.body {
            ast::TypeBody::Struct(fs) => fs.clone(),
            _ => return Ok(name.to_string()),
        };
        let mut subs: HashMap<String, Ty> = HashMap::new();
        for tf in &template_fields {
            if let Some(init) = fields.iter().find(|fi| fi.name == tf.name) {
                let val_ty = self.infer_ty(&init.value)?;
                let template_field_ty = crate::types::resolve_type_with_params(
                    &tf.ty, &self.cg.info.types, &template.type_params,
                ).unwrap_or(Ty::Error);
                crate::types::unify_for_subs(&template_field_ty, &val_ty, &mut subs);
            }
        }
        let type_args: Vec<Ty> = template.type_params.iter()
            .map(|p| subs.get(p).cloned().unwrap_or(Ty::Error))
            .collect();
        Ok(crate::types::mangle_type_instantiation(name, &type_args))
    }

    fn compile_struct_lit(
        &mut self,
        name: &str,
        fields: &[ast::FieldInit],
        spread: Option<&ast::Expr>,
        span: Span,
    ) -> Result<Value, NativeError> {
        // Generic struct template? Infer the mangled instantiation name
        // from the field-value types, then dispatch to that.
        let resolved_name = self.resolve_struct_lit_name(name, fields)?;
        let name = resolved_name.as_str();

        // Named-field variant constructor? (e.g. Circle { radius: 2 })
        if get_struct_fields(&self.cg.info.types, name).is_empty() {
            if let Some(sum_name) = self.resolved_variant_sum(span)
                .or_else(|| self.find_sum_for_variant(name))
            {
                let scrut_ty = Ty::User(sum_name);
                let tag = self.variant_tag(&scrut_ty, name).unwrap_or(0);
                let var_fields = self.variant_field_types(&scrut_ty, name).unwrap_or_default();
                if var_fields.is_empty() {
                    return self.build_sum_block(tag, None);
                }
                let total = struct_size(&var_fields);
                let payload = self.rc_alloc(total as i64)?;
                for (fname, _fty) in &var_fields {
                    let init = fields.iter().find(|fi| &fi.name == fname).unwrap();
                    let val = self.compile_expr(&init.value)?;
                    let (offset, _) = field_offset(&var_fields, fname);
                    self.builder.ins().store(MemFlags::new(), val, payload, offset);
                }
                return self.build_sum_block(tag, Some(payload));
            }
        }

        // Compile the spread base pointer (if any) before allocating the new
        // struct so that any side-effects happen in source order.
        let spread_val: Option<Value> = spread.map(|e| self.compile_expr(e)).transpose()?;

        let def_fields = get_struct_fields(&self.cg.info.types, name);
        let total_size = struct_size(&def_fields);
        let ptr = self.rc_alloc(total_size as i64)?;

        for (fname, fty) in &def_fields {
            let (offset, _) = field_offset(&def_fields, fname);
            if let Some(init) = fields.iter().find(|fi| &fi.name == fname) {
                // User explicitly provided this field.
                let val = self.compile_expr(&init.value)?;
                self.builder.ins().store(MemFlags::new(), val, ptr, offset);
                // rc_incr pointer-typed fields: the struct now holds a
                // reference alongside the original binding.
                if !is_scalar(fty) {
                    self.emit_rc_incr(val);
                }
            } else if let Some(base) = spread_val {
                // Field not provided — load from the spread base struct.
                let cl_ty = lumen_to_cl(fty);
                let val = self.builder.ins().load(cl_ty, MemFlags::new(), base, offset);
                self.builder.ins().store(MemFlags::new(), val, ptr, offset);
                if !is_scalar(fty) {
                    self.emit_rc_incr(val);
                }
            } else {
                return Err(NativeError {
                    span,
                    message: format!("missing field `{fname}`"),
                });
            }
        }
        Ok(ptr)
    }

    fn bump_alloc(&mut self, size: i64) -> Result<Value, NativeError> {
        let flags = MemFlags::new();
        let bump_gv = self
            .cg
            .obj
            .declare_data_in_func(self.cg.bump_ptr_data, self.builder.func);
        let heap_gv = self
            .cg
            .obj
            .declare_data_in_func(self.cg.heap_data, self.builder.func);

        let bump_addr = self.builder.ins().global_value(PTR, bump_gv);
        let old_off = self.builder.ins().load(PTR, flags, bump_addr, 0);
        let heap_base = self.builder.ins().global_value(PTR, heap_gv);
        let result = self.builder.ins().iadd(heap_base, old_off);

        let size_val = self.builder.ins().iconst(PTR, size);
        let new_off = self.builder.ins().iadd(old_off, size_val);
        let seven = self.builder.ins().iconst(PTR, 7);
        let new_off = self.builder.ins().iadd(new_off, seven);
        let mask = self.builder.ins().iconst(PTR, -8i64);
        let new_off = self.builder.ins().band(new_off, mask);
        self.builder.ins().store(flags, new_off, bump_addr, 0);

        Ok(result)
    }

    // --- RC helpers -------------------------------------------------------

    /// Allocate via RC: calls lumen_rc_alloc(size), returns a pointer
    /// with rc=1 already set.
    fn rc_alloc(&mut self, size: i64) -> Result<Value, NativeError> {
        let size_val = self.builder.ins().iconst(PTR, size);
        let func_ref = self
            .cg
            .obj
            .declare_func_in_func(self.cg.helper_rc_alloc, self.builder.func);
        let call = self.builder.ins().call(func_ref, &[size_val]);
        Ok(self.builder.inst_results(call)[0])
    }

    // --- value-formatting codegen (shared by debug.print and io.println) ---

    fn fmt_funcs(&self, target: PrintTarget) -> FmtFuncs {
        match target {
            PrintTarget::Stderr => FmtFuncs {
                i32: self.cg.debug_i32,
                i64: self.cg.debug_i64,
                f64: self.cg.debug_f64,
                bool: self.cg.debug_bool,
                str: self.cg.debug_str,
                raw: self.cg.debug_raw,
                leading: None,
            },
            PrintTarget::Stdout => FmtFuncs {
                i32: self.cg.io_i32,
                i64: self.cg.io_i64,
                f64: self.cg.io_f64,
                bool: self.cg.io_bool,
                str: self.cg.io_str,
                raw: self.cg.io_raw,
                leading: None,
            },
            PrintTarget::StrBuf(buf) => FmtFuncs {
                i32: self.cg.strbuf_i32,
                i64: self.cg.strbuf_i64,
                f64: self.cg.strbuf_f64,
                bool: self.cg.strbuf_bool,
                str: self.cg.strbuf_str,
                raw: self.cg.strbuf_raw,
                leading: Some(buf),
            },
        }
    }

    fn fmt_call(&mut self, fid: FuncId, leading: Option<Value>, args: &[Value]) {
        let func_ref = self.cg.obj.declare_func_in_func(fid, self.builder.func);
        if let Some(buf) = leading {
            let mut all = Vec::with_capacity(args.len() + 1);
            all.push(buf);
            all.extend_from_slice(args);
            self.builder.ins().call(func_ref, &all);
        } else {
            self.builder.ins().call(func_ref, args);
        }
    }

    /// Emit a raw string literal to the given stream/buffer.
    fn emit_fmt_raw(&mut self, target: PrintTarget, s: &str) {
        let fns = self.fmt_funcs(target);
        let name = format!("__dbg_{}", self.cg.string_data.len() + self.cg.debug_data_counter);
        self.cg.debug_data_counter += 1;
        let data_id = self.cg.obj.declare_data(&name, Linkage::Local, false, false).unwrap();
        let mut desc = DataDescription::new();
        desc.define(s.as_bytes().to_vec().into_boxed_slice());
        self.cg.obj.define_data(data_id, &desc).unwrap();
        let gv = self.cg.obj.declare_data_in_func(data_id, self.builder.func);
        let ptr = self.builder.ins().global_value(PTR, gv);
        let len = self.builder.ins().iconst(cl_types::I32, s.len() as i64);
        self.fmt_call(fns.raw, fns.leading, &[ptr, len]);
    }

    /// Emit code to print a value of the given type to the given stream/buffer.
    fn emit_fmt_value(&mut self, target: PrintTarget, val: Value, ty: &Ty) -> Result<(), NativeError> {
        let fns = self.fmt_funcs(target);
        match ty {
            Ty::I32 | Ty::U32 => self.fmt_call(fns.i32, fns.leading, &[val]),
            Ty::I64 | Ty::U64 => self.fmt_call(fns.i64, fns.leading, &[val]),
            Ty::F64 => self.fmt_call(fns.f64, fns.leading, &[val]),
            Ty::Bool => self.fmt_call(fns.bool, fns.leading, &[val]),
            Ty::String | Ty::Bytes => self.fmt_call(fns.str, fns.leading, &[val]),
            Ty::Char => {
                // Encode the scalar into UTF-8 bytes on the stack, then
                // emit as a raw byte sequence (no len prefix — char fmt
                // is 1..=4 bytes inline).
                let slot = self.builder.create_sized_stack_slot(
                    StackSlotData::new(StackSlotKind::ExplicitSlot, 4, 0),
                );
                let ptr = self.builder.ins().stack_addr(PTR, slot, 0);
                let enc_ref = self.cg.obj
                    .declare_func_in_func(self.cg.helper_utf8_encode, self.builder.func);
                let enc_call = self.builder.ins().call(enc_ref, &[val, ptr]);
                let n = self.builder.inst_results(enc_call)[0];
                self.fmt_call(fns.raw, fns.leading, &[ptr, n]);
            }
            Ty::Unit => {
                self.emit_fmt_raw(target, "unit");
            }
            Ty::User(name) => {
                let fields = get_struct_fields(&self.cg.info.types, name);
                if fields.is_empty() {
                    self.emit_fmt_raw(target, name);
                } else {
                    self.emit_fmt_raw(target, &format!("{name} {{ "));
                    for (i, (fname, fty)) in fields.iter().enumerate() {
                        if i > 0 { self.emit_fmt_raw(target, ", "); }
                        self.emit_fmt_raw(target, &format!("{fname}: "));
                        let (offset, _) = field_offset(&fields, fname);
                        let cl_ty = lumen_to_cl(fty);
                        let fval = self.builder.ins().load(cl_ty, MemFlags::new(), val, offset);
                        self.emit_struct_field_fmt(target, fval, fty)?;
                    }
                    self.emit_fmt_raw(target, " }");
                }
            }
            Ty::Tuple(elems) => {
                self.emit_fmt_raw(target, "(");
                let fields = tuple_as_fields(elems);
                for (i, (fname, fty)) in fields.iter().enumerate() {
                    if i > 0 { self.emit_fmt_raw(target, ", "); }
                    let (offset, _) = field_offset(&fields, fname);
                    let cl_ty = lumen_to_cl(fty);
                    let fval = self.builder.ins().load(cl_ty, MemFlags::new(), val, offset);
                    self.emit_struct_field_fmt(target, fval, fty)?;
                }
                self.emit_fmt_raw(target, ")");
            }
            Ty::List(inner) => {
                self.emit_fmt_raw(target, "[");
                let len_fid = self.module_func("lumen_list_len");
                let len_ref = self.cg.obj.declare_func_in_func(len_fid, self.builder.func);
                let len_call = self.builder.ins().call(len_ref, &[val]);
                let len = self.builder.inst_results(len_call)[0];
                let get_fid = self.module_func("lumen_list_get");
                let get_ref = self.cg.obj.declare_func_in_func(get_fid, self.builder.func);
                let counter = self.fresh_var(cl_types::I32);
                let zero = self.builder.ins().iconst(cl_types::I32, 0);
                self.builder.def_var(counter, zero);
                let header = self.builder.create_block();
                let body = self.builder.create_block();
                let exit = self.builder.create_block();
                self.builder.ins().jump(header, &[]);
                self.builder.switch_to_block(header);
                let i = self.builder.use_var(counter);
                let done = self.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, i, len);
                self.builder.ins().brif(done, exit, &[], body, &[]);
                self.builder.switch_to_block(body);
                let i = self.builder.use_var(counter);
                let is_first = self.builder.ins().icmp_imm(IntCC::Equal, i, 0);
                let sep_bb = self.builder.create_block();
                let after_sep = self.builder.create_block();
                self.builder.ins().brif(is_first, after_sep, &[], sep_bb, &[]);
                self.builder.switch_to_block(sep_bb);
                self.emit_fmt_raw(target, ", ");
                self.builder.ins().jump(after_sep, &[]);
                self.builder.switch_to_block(after_sep);
                let i = self.builder.use_var(counter);
                let elem_call = self.builder.ins().call(get_ref, &[val, i]);
                let elem = self.builder.inst_results(elem_call)[0];
                let elem_val = if lumen_to_cl(inner) == cl_types::I32 {
                    self.builder.ins().ireduce(cl_types::I32, elem)
                } else { elem };
                self.emit_struct_field_fmt(target, elem_val, inner)?;
                let one = self.builder.ins().iconst(cl_types::I32, 1);
                let i = self.builder.use_var(counter);
                let next = self.builder.ins().iadd(i, one);
                self.builder.def_var(counter, next);
                self.builder.ins().jump(header, &[]);
                self.builder.switch_to_block(exit);
                self.emit_fmt_raw(target, "]");
            }
            _ => {
                self.emit_fmt_raw(target, &format!("<{}>", ty.display()));
            }
        }
        Ok(())
    }

    /// Lower a `"hello \{x}"` interpolation: allocate a strbuf, append each
    /// piece (literal text or formatted expression value) into it, then call
    /// strbuf_finish to produce a fresh Lumen string (rc=1).
    fn compile_interpolated(&mut self, parts: &[ast::InterpPiece]) -> Result<Value, NativeError> {
        let new_ref = self.cg.obj.declare_func_in_func(self.cg.strbuf_new, self.builder.func);
        let new_call = self.builder.ins().call(new_ref, &[]);
        let buf = self.builder.inst_results(new_call)[0];
        let target = PrintTarget::StrBuf(buf);
        for piece in parts {
            match piece {
                ast::InterpPiece::Lit(s) => self.emit_fmt_raw(target, s),
                ast::InterpPiece::Expr(e) => {
                    let val = self.compile_expr(e)?;
                    let ty = self.infer_ty(e)?;
                    self.emit_fmt_value(target, val, &ty)?;
                }
            }
        }
        let finish_ref = self.cg.obj.declare_func_in_func(self.cg.strbuf_finish, self.builder.func);
        let finish_call = self.builder.ins().call(finish_ref, &[buf]);
        Ok(self.builder.inst_results(finish_call)[0])
    }

    /// Print a value that is nested inside a struct/list/tuple.
    /// For non-stderr targets (which print strings as raw content), nested
    /// strings still get quoted so `[\"a\", \"b\"]` stays unambiguous.
    fn emit_struct_field_fmt(&mut self, target: PrintTarget, val: Value, ty: &Ty) -> Result<(), NativeError> {
        let needs_quotes = !matches!(target, PrintTarget::Stderr)
            && matches!(ty, Ty::String | Ty::Bytes);
        if needs_quotes {
            let fns = self.fmt_funcs(target);
            self.emit_fmt_raw(target, "\"");
            self.fmt_call(fns.str, fns.leading, &[val]);
            self.emit_fmt_raw(target, "\"");
            return Ok(());
        }
        self.emit_fmt_value(target, val, ty)
    }


    /// Sign-extend a Cranelift I32 value to I64 when the source type
    /// lives in a 32-bit slot (i32, char, bool, etc.). Pointer-shaped
    /// and already-64-bit types pass through unchanged. Used when
    /// narrowing Lumen values into the runtime's uniform i64 map key
    /// / value slots.
    fn widen_to_i64(&mut self, v: Value, ty: &Ty) -> Value {
        if lumen_to_cl(ty) == cl_types::I32 {
            self.builder.ins().sextend(cl_types::I64, v)
        } else if lumen_to_cl(ty) == cl_types::F64 {
            self.builder.ins().bitcast(cl_types::I64, MemFlags::new(), v)
        } else {
            v
        }
    }

    /// Look up a module-level binding by its Lumen name, scoped to the
    /// current module being compiled. Globals are private — no
    /// cross-module access.
    fn lookup_global(&self, name: &str) -> Option<&(DataId, Ty)> {
        let module_key = self.current_module.as_deref().unwrap_or("user");
        self.cg.global_data.get(&format!("{module_key}::{name}"))
    }

    fn emit_rc_incr(&mut self, ptr: Value) {
        let func_ref = self
            .cg
            .obj
            .declare_func_in_func(self.cg.helper_rc_incr, self.builder.func);
        self.builder.ins().call(func_ref, &[ptr]);
    }

    fn emit_rc_decr(&mut self, ptr: Value) {
        let func_ref = self
            .cg
            .obj
            .declare_func_in_func(self.cg.helper_rc_decr, self.builder.func);
        self.builder.ins().call(func_ref, &[ptr]);
    }

    /// Type-aware rc_decr: if the type has pointer children and the
    /// refcount is about to hit 0, recursively decr the children BEFORE
    /// freeing the block. Falls back to generic rc_decr for leaf types.
    fn emit_rc_decr_typed(&mut self, ptr: Value, ty: &Ty) {
        if is_scalar(ty) {
            return;
        }
        if !self.type_has_ptr_children(ty) {
            // Strings, simple scalars wrapped in Option/Result with
            // scalar payloads — no recursive work needed.
            self.emit_rc_decr(ptr);
            return;
        }

        // Inline check: if rc == 1, decr children first, then generic decr.
        // If rc > 1, just generic decr (children stay alive).
        let flags = MemFlags::new();
        let is_null = self.builder.ins().icmp_imm(IntCC::Equal, ptr, 0);
        let check_bb = self.builder.create_block();
        let decr_children_bb = self.builder.create_block();
        let do_decr_bb = self.builder.create_block();
        let exit_bb = self.builder.create_block();

        self.builder.ins().brif(is_null, exit_bb, &[], check_bb, &[]);

        self.builder.switch_to_block(check_bb);
        let eight = self.builder.ins().iconst(PTR, 8);
        let header = self.builder.ins().isub(ptr, eight);
        // Check magic.
        let magic = self.builder.ins().load(cl_types::I32, flags, header, 4);
        let expected = self.builder.ins().iconst(cl_types::I32, 0x4C554D45u32 as i64);
        let is_rc = self.builder.ins().icmp(IntCC::Equal, magic, expected);
        self.builder.ins().brif(is_rc, do_decr_bb, &[], exit_bb, &[]);

        self.builder.switch_to_block(do_decr_bb);
        let rc = self.builder.ins().load(cl_types::I32, flags, header, 0);
        let will_free = self.builder.ins().icmp_imm(IntCC::Equal, rc, 1);
        self.builder.ins().brif(will_free, decr_children_bb, &[], exit_bb, &[]);

        // Decr children before the generic decr frees the block.
        self.builder.switch_to_block(decr_children_bb);
        self.emit_child_decrs(ptr, ty);
        self.builder.ins().jump(exit_bb, &[]);

        self.builder.switch_to_block(exit_bb);
        // Always call generic decr (handles the actual rc-- and free).
        self.emit_rc_decr(ptr);
    }

    /// Emit rc_decr calls for each pointer-typed child field of a value.
    fn emit_child_decrs(&mut self, ptr: Value, ty: &Ty) {
        let flags = MemFlags::new();
        match ty {
            Ty::User(name) => {
                let fields = get_struct_fields(&self.cg.info.types, name);
                for (fname, fty) in &fields {
                    if !is_scalar(fty) {
                        let (offset, _) = field_offset(&fields, fname);
                        let child = self.builder.ins().load(PTR, flags, ptr, offset);
                        // Recursive: children might have their own children.
                        self.emit_rc_decr_typed(child, fty);
                    }
                }
            }
            Ty::Option(_) | Ty::Result(_, _) => {
                // Sum type: payload_ptr is at offset +8.
                let payload = self.builder.ins().load(PTR, flags, ptr, 8);
                // Generic decr on the payload block (we don't know the
                // variant's field layout at this point without checking
                // the tag, so we just release the payload's own refcount).
                self.emit_rc_decr(payload);
            }
            Ty::Tuple(elems) => {
                let fields = tuple_as_fields(elems);
                for (fname, fty) in &fields {
                    if !is_scalar(fty) {
                        let (offset, _) = field_offset(&fields, fname);
                        let child = self.builder.ins().load(PTR, flags, ptr, offset);
                        self.emit_rc_decr_typed(child, fty);
                    }
                }
            }
            Ty::String | Ty::Bytes => {
                // Strings/bytes have no pointer children.
            }
            _ => {}
        }
    }

    /// Does this type contain pointer-typed fields that need recursive
    /// decrement on free?
    fn type_has_ptr_children(&self, ty: &Ty) -> bool {
        match ty {
            Ty::User(name) => {
                let fields = get_struct_fields(&self.cg.info.types, name);
                fields.iter().any(|(_, fty)| !is_scalar(fty))
            }
            Ty::Option(_) | Ty::Result(_, _) => true, // payload ptr
            Ty::Tuple(_) => true, // tuple fields might be pointers
            _ => false,
        }
    }


    // --- Sum-type helpers ------------------------------------------------

    /// Allocate a 16-byte sum block { tag: i32 @+0, payload_ptr: i64 @+8 }.
    fn build_sum_block(
        &mut self,
        tag: u32,
        payload_ptr: Option<Value>,
    ) -> Result<Value, NativeError> {
        let ptr = self.rc_alloc(16)?;
        let tag_val = self.builder.ins().iconst(cl_types::I32, tag as i64);
        self.builder.ins().store(MemFlags::new(), tag_val, ptr, 0);
        let payload = payload_ptr.unwrap_or_else(|| self.builder.ins().iconst(PTR, 0));
        self.builder.ins().store(MemFlags::new(), payload, ptr, 8);
        Ok(ptr)
    }

    /// Ok(v), Err(e), Some(v): allocate a field block for the single value,
    /// then wrap in a sum block.
    fn compile_single_field_constructor(
        &mut self,
        tag: u32,
        value: &Expr,
    ) -> Result<Value, NativeError> {
        let val = self.compile_expr(value)?;
        let ty = self.infer_ty(value)?;
        // Pointer-typed payloads gain a new owner (the field block), so
        // rc_incr to balance the eventual scope-exit decrement of `val`'s
        // local binding. Without this the wrapped value is freed before
        // the caller can match on it.
        if !is_scalar(&ty) {
            self.emit_rc_incr(val);
        }
        let size = native_sizeof(&ty);
        let field_ptr = self.rc_alloc(size as i64)?;
        self.builder.ins().store(MemFlags::new(), val, field_ptr, 0);
        self.build_sum_block(tag, Some(field_ptr))
    }

    /// Allocate a payload block for a positional variant and store each
    /// arg value at the correct offset.
    fn build_payload_block(
        &mut self,
        fields: &[(String, Ty)],
        args: &[ast::Arg],
        _span: Span,
    ) -> Result<Value, NativeError> {
        let total = struct_size(fields);
        let ptr = self.rc_alloc(total as i64)?;
        for (i, (fname, _fty)) in fields.iter().enumerate() {
            if let Some(arg) = args.get(i) {
                let val = self.compile_expr(&arg.value)?;
                let (offset, _) = field_offset(fields, fname);
                self.builder.ins().store(MemFlags::new(), val, ptr, offset);
            }
        }
        Ok(ptr)
    }

    /// Match on a sum type (Option, Result, or user sum). Compiled as a
    /// chain of if-else on the tag discriminant, same structure as the
    /// Wasm backend.
    fn compile_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[ast::MatchArm],
        _span: Span,
    ) -> Result<Value, NativeError> {
        let scrut_ty = self.infer_ty(scrutinee)?;
        let scrut_val = self.compile_expr(scrutinee)?;

        // Store scrutinee in a variable so we can re-read tag + payload.
        let scrut_var = self.fresh_var(PTR);
        self.builder.def_var(scrut_var, scrut_val);

        let result_ty = arms
            .first()
            .map(|a| self.infer_ty(&a.body))
            .transpose()?
            .unwrap_or(Ty::Unit);
        let cl_result = lumen_to_cl(&result_ty);

        let merge_bb = self.builder.create_block();
        self.builder.append_block_param(merge_bb, cl_result);

        self.compile_match_arms(arms, 0, scrut_var, &scrut_ty, cl_result, merge_bb)?;

        self.builder.switch_to_block(merge_bb);
        Ok(self.builder.block_params(merge_bb)[0])
    }

    fn compile_match_arms(
        &mut self,
        arms: &[ast::MatchArm],
        idx: usize,
        scrut_var: Variable,
        scrut_ty: &Ty,
        cl_result: CLType,
        merge_bb: cranelift_codegen::ir::Block,
    ) -> Result<(), NativeError> {
        use ast::PatternKind;

        if idx >= arms.len() {
            self.builder.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
            return Ok(());
        }
        let arm = &arms[idx];

        let always = matches!(
            arm.pattern.kind,
            PatternKind::Wildcard | PatternKind::Binding(_)
        );

        if always {
            if let PatternKind::Binding(name) = &arm.pattern.kind {
                let var = self.fresh_var(PTR);
                let v = self.builder.use_var(scrut_var);
                self.builder.def_var(var, v);
                self.names.insert(name.clone(), var);
            }
            let body_val = self.compile_expr(&arm.body)?;
            self.builder
                .ins()
                .jump(merge_bb, &[BlockArg::Value(body_val)]);
            return Ok(());
        }

        let PatternKind::Variant {
            name: variant_name,
            payload,
        } = &arm.pattern.kind
        else {
            // Unsupported pattern kind — treat as wildcard.
            let body_val = self.compile_expr(&arm.body)?;
            self.builder
                .ins()
                .jump(merge_bb, &[BlockArg::Value(body_val)]);
            return Ok(());
        };

        let tag = self.variant_tag(scrut_ty, variant_name).unwrap_or(0);
        let scrut_ptr = self.builder.use_var(scrut_var);
        let actual_tag = self.builder.ins().load(cl_types::I32, MemFlags::new(), scrut_ptr, 0);
        let tag_val = self.builder.ins().iconst(cl_types::I32, tag as i64);
        let matches = self.builder.ins().icmp(IntCC::Equal, actual_tag, tag_val);

        let matched_bb = self.builder.create_block();
        let next_bb = self.builder.create_block();
        self.builder.ins().brif(matches, matched_bb, &[], next_bb, &[]);

        // Matched branch.
        self.builder.switch_to_block(matched_bb);
        if let Some(payload_pat) = payload {
            let scrut_ptr = self.builder.use_var(scrut_var);
            let payload_ptr = self.builder.ins().load(PTR, MemFlags::new(), scrut_ptr, 8);
            self.bind_pattern_payload(payload_pat, &scrut_ty.clone(), variant_name, payload_ptr)?;
        }
        let body_val = self.compile_expr(&arm.body)?;
        self.builder
            .ins()
            .jump(merge_bb, &[BlockArg::Value(body_val)]);

        // Next branch.
        self.builder.switch_to_block(next_bb);
        self.compile_match_arms(arms, idx + 1, scrut_var, scrut_ty, cl_result, merge_bb)?;

        Ok(())
    }

    fn bind_pattern_payload(
        &mut self,
        payload: &ast::VariantPatPayload,
        scrut_ty: &Ty,
        variant_name: &str,
        payload_ptr: Value,
    ) -> Result<(), NativeError> {
        use ast::{PatternKind, VariantPatPayload};

        let fields = self.variant_field_types(scrut_ty, variant_name).unwrap_or_default();
        let sub_pats: Vec<(&str, &ast::Pattern, i32, Ty)> = match payload {
            VariantPatPayload::Named(pfs) => pfs
                .iter()
                .filter_map(|pf| {
                    let (off, fty) = field_offset(&fields, &pf.name);
                    Some((pf.name.as_str(), &pf.pattern, off, fty))
                })
                .collect(),
            VariantPatPayload::Positional(pats) => pats
                .iter()
                .enumerate()
                .filter_map(|(i, pat)| {
                    let (fname, fty) = fields.get(i)?;
                    let (off, _) = field_offset(&fields, fname);
                    Some(("", pat, off, fty.clone()))
                })
                .collect(),
        };

        for (_name, pat, offset, fty) in sub_pats {
            if let PatternKind::Binding(bind_name) = &pat.kind {
                let cl_ty = lumen_to_cl(&fty);
                let val = self.builder.ins().load(cl_ty, MemFlags::new(), payload_ptr, offset);
                let var = self.fresh_var(cl_ty);
                self.builder.def_var(var, val);
                self.names.insert(bind_name.clone(), var);
                self.name_types.insert(bind_name.clone(), fty);
            }
        }
        Ok(())
    }

    /// `expr?` — check tag, return on error, extract value on success.
    fn compile_try(
        &mut self,
        inner: &Expr,
        _span: Span,
    ) -> Result<Value, NativeError> {
        let inner_ty = self.infer_ty(inner)?;
        let (err_tag, ok_payload_ty) = match inner_ty {
            Ty::Result(ok, _) => (1u32, *ok),
            Ty::Option(inner) => (0u32, *inner),
            _ => {
                return Err(NativeError {
                    span: _span,
                    message: "`?` requires Result or Option".into(),
                })
            }
        };

        let sum_ptr = self.compile_expr(inner)?;
        let scrut_var = self.fresh_var(PTR);
        self.builder.def_var(scrut_var, sum_ptr);

        let tag = self.builder.ins().load(cl_types::I32, MemFlags::new(), sum_ptr, 0);
        let err_tag_val = self.builder.ins().iconst(cl_types::I32, err_tag as i64);
        let is_err = self.builder.ins().icmp(IntCC::Equal, tag, err_tag_val);

        let err_bb = self.builder.create_block();
        let ok_bb = self.builder.create_block();
        self.builder.ins().brif(is_err, err_bb, &[], ok_bb, &[]);

        // Err path: push an error frame, optionally print, return.
        self.builder.switch_to_block(err_bb);

        // Allocate 16-byte frame { message: ptr @+0, next: ptr @+8 }.
        let frame_msg = format!(
            "  at {} (<source>:{}:{})",
            self.fn_name, _span.line, _span.col
        );
        if let Some(&msg_data_id) = self.cg.string_data.get(&frame_msg) {
            let frame_ptr = self.bump_alloc(16)?;
            // frame.message = interned string
            let msg_gv = self.cg.obj.declare_data_in_func(msg_data_id, self.builder.func);
            let msg_val = self.builder.ins().global_value(PTR, msg_gv);
            self.builder.ins().store(MemFlags::new(), msg_val, frame_ptr, 0);
            // frame.next = current frame_chain head
            let chain_gv = self.cg.obj.declare_data_in_func(
                self.cg.frame_chain_data, self.builder.func,
            );
            let chain_addr = self.builder.ins().global_value(PTR, chain_gv);
            let old_head = self.builder.ins().load(PTR, MemFlags::new(), chain_addr, 0);
            self.builder.ins().store(MemFlags::new(), old_head, frame_ptr, 8);
            // frame_chain = &frame
            self.builder.ins().store(MemFlags::new(), frame_ptr, chain_addr, 0);
        }

        // If we're in main, print all accumulated frames before returning.
        if self.fn_name == "main" {
            let pf_ref = self.cg.obj.declare_func_in_func(
                self.cg.helper_print_frames, self.builder.func,
            );
            self.builder.ins().call(pf_ref, &[]);
        }

        let err_ptr = self.builder.use_var(scrut_var);
        self.builder.ins().return_(&[err_ptr]);

        // Ok path: load payload, extract value.
        self.builder.switch_to_block(ok_bb);
        let ok_ptr = self.builder.use_var(scrut_var);
        let payload_ptr = self.builder.ins().load(PTR, MemFlags::new(), ok_ptr, 8);
        let cl_ty = lumen_to_cl(&ok_payload_ty);
        let val = self.builder.ins().load(cl_ty, MemFlags::new(), payload_ptr, 0);
        Ok(val)
    }

    // --- Actor operations -------------------------------------------------

    /// `spawn Counter { count: 0 }` — allocate a mutable state cell (8
    /// bytes holding a ptr to the current actor state), build the initial
    /// state as a struct, store the state ptr in the cell, return cell ptr.
    fn compile_spawn(
        &mut self,
        actor_name: &str,
        fields: &[ast::FieldInit],
        span: Span,
    ) -> Result<Value, NativeError> {
        // Build the initial state (same as a struct literal).
        let state = self.compile_struct_lit(actor_name, fields, None, span)?;
        // Allocate an 8-byte mutable cell to hold the state pointer.
        let cell = self.rc_alloc(8)?;
        self.builder.ins().store(MemFlags::new(), state, cell, 0);
        Ok(cell)
    }

    /// `send handle.method(args)` — load state from cell, call handler,
    /// store new state if the handler returns the actor type.
    fn compile_send(
        &mut self,
        handle: &Expr,
        method: &str,
        args: &[ast::Arg],
        span: Span,
    ) -> Result<Value, NativeError> {
        self.compile_msg_dispatch(handle, method, args, span, false)
    }

    /// `ask handle.method(args)` — same as send but return the handler's result.
    fn compile_ask(
        &mut self,
        handle: &Expr,
        method: &str,
        args: &[ast::Arg],
        span: Span,
    ) -> Result<Value, NativeError> {
        self.compile_msg_dispatch(handle, method, args, span, true)
    }

    fn compile_msg_dispatch(
        &mut self,
        handle: &Expr,
        method: &str,
        args: &[ast::Arg],
        span: Span,
        return_result: bool,
    ) -> Result<Value, NativeError> {
        let handle_ty = self.infer_ty(handle)?;
        let actor_name = match &handle_ty {
            Ty::Handle(inner) => match inner.as_ref() {
                Ty::User(n) => n.clone(),
                _ => return Err(NativeError { span, message: "expected actor handle".into() }),
            },
            _ => return Err(NativeError { span, message: "expected actor handle".into() }),
        };

        let cell = self.compile_expr(handle)?;

        // Find the msg_kind index for this method.
        let msgs = self.cg.info.actors.get(&actor_name).cloned().unwrap_or_default();
        let msg_kind = msgs.iter().position(|m| m.name == method).unwrap_or(0) as i64;
        let kind_val = self.builder.ins().iconst(cl_types::I32, msg_kind);

        // Encode args into arg0 (i64). For 0 args: 0. For 1 arg: the
        // value directly. For 2+ args: pack into a malloc'd struct and
        // pass the pointer.
        let msg_sig = msgs.iter().find(|m| m.name == method);
        let arg0 = if args.is_empty() {
            self.builder.ins().iconst(cl_types::I64, 0)
        } else if args.len() == 1 {
            let v = self.compile_expr(&args[0].value)?;
            let ty = self.infer_ty(&args[0].value)?;
            match lumen_to_cl(&ty) {
                cl_types::I32 => self.builder.ins().sextend(cl_types::I64, v),
                cl_types::F64 => self.builder.ins().bitcast(cl_types::I64, MemFlags::new(), v),
                _ => v,
            }
        } else {
            // Multi-arg: allocate a struct, store each arg, pass ptr.
            let param_types: Vec<(String, Ty)> = msg_sig
                .map(|m| m.params.clone())
                .unwrap_or_default();
            let total = struct_size(&param_types);
            let blob = self.rc_alloc(total as i64)?;
            for (i, arg) in args.iter().enumerate() {
                let v = self.compile_expr(&arg.value)?;
                if let Some((pname, _)) = param_types.get(i) {
                    let (offset, _) = field_offset(&param_types, pname);
                    self.builder.ins().store(MemFlags::new(), v, blob, offset);
                }
            }
            blob // ptr is i64, used as arg0
        };

        // Get the dispatch function pointer.
        let dispatch_id = self.cg.dispatch_fns.get(&actor_name).ok_or_else(|| {
            NativeError { span, message: format!("no dispatch for actor `{actor_name}`") }
        })?;
        let dispatch_ref = self.cg.obj.declare_func_in_func(*dispatch_id, self.builder.func);
        let dispatch_addr = self.builder.ins().func_addr(PTR, dispatch_ref);

        if return_result {
            // ask: call lumen_rt_ask(cell, dispatch, kind, arg0) -> i64
            let rt_ref = self.cg.obj.declare_func_in_func(self.cg.rt_ask, self.builder.func);
            let call = self.builder.ins().call(rt_ref, &[cell, dispatch_addr, kind_val, arg0]);
            let result_i64 = self.builder.inst_results(call)[0];
            // Truncate to the handler's return type.
            let fn_name = format!("{}_{}", actor_name, method);
            let ret_ty = self.cg.info.fns.get(&fn_name).map(|s| &s.ret);
            Ok(match ret_ty.map(|t| lumen_to_cl(t)) {
                Some(cl_types::I32) => self.builder.ins().ireduce(cl_types::I32, result_i64),
                _ => result_i64,
            })
        } else {
            // send: call lumen_rt_send(cell, dispatch, kind, arg0)
            let rt_ref = self.cg.obj.declare_func_in_func(self.cg.rt_send, self.builder.func);
            self.builder.ins().call(rt_ref, &[cell, dispatch_addr, kind_val, arg0]);
            Ok(self.builder.ins().iconst(cl_types::I32, 0))
        }
    }

    fn variant_tag(&self, scrut_ty: &Ty, variant_name: &str) -> Option<u32> {
        match scrut_ty {
            Ty::User(type_name) => {
                if let Some(TypeInfo::Sum { variants, .. }) = self.cg.info.types.get(type_name) {
                    variants
                        .iter()
                        .position(|v| v.name == variant_name)
                        .map(|i| i as u32)
                } else {
                    None
                }
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

    fn variant_field_types(&self, scrut_ty: &Ty, variant_name: &str) -> Option<Vec<(String, Ty)>> {
        match scrut_ty {
            Ty::User(type_name) => {
                if let Some(TypeInfo::Sum { variants, .. }) = self.cg.info.types.get(type_name) {
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
                } else {
                    None
                }
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

    fn infer_ty(&mut self, expr: &Expr) -> Result<Ty, NativeError> {
        // Simplified type inference for codegen dispatch.
        Ok(match &expr.kind {
            ExprKind::IntLit { suffix, .. } => match suffix {
                Some(IntSuffix::I64) => Ty::I64,
                Some(IntSuffix::U64) => Ty::U64,
                Some(IntSuffix::U32) => Ty::U32,
                _ => Ty::I32,
            },
            ExprKind::FloatLit(_) => Ty::F64,
            ExprKind::CharLit(_) => Ty::Char,
            ExprKind::BoolLit(_) => Ty::Bool,
            ExprKind::UnitLit => Ty::Unit,
            ExprKind::StringLit(_) => Ty::String,
            ExprKind::Interpolated(_) => Ty::String,
            ExprKind::Ident(name) => {
                if let Some(ty) = self.name_types.get(name) {
                    return Ok(ty.clone());
                }
                // Variant constructor recorded by the type checker against
                // an expected generic-sum instantiation (e.g. bare `Empty`
                // resolved against expected = Node$I32).
                if let Some(sum_name) = self.resolved_variant_sum(expr.span) {
                    return Ok(Ty::User(sum_name));
                }
                // Bare zero-payload variant of any registered sum type.
                if let Some(sum_name) = self.find_sum_for_variant(name) {
                    return Ok(Ty::User(sum_name));
                }
                // If the ident names a known function, return its FnPtr type.
                if let Some(sig) = self.cg.info.fns.get(name) {
                    let params: Vec<Ty> = sig.params.iter().map(|(_, t)| t.clone()).collect();
                    return Ok(Ty::FnPtr { params, ret: Box::new(sig.ret.clone()) });
                }
                // Module-level let/var binding.
                if let Some((_, ty)) = self.lookup_global(name) {
                    return Ok(ty.clone());
                }
                Ty::I32 // fallback
            }
            ExprKind::Paren(e) => self.infer_ty(e)?,
            ExprKind::Binary { op, lhs, .. } => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                    self.infer_ty(lhs)?
                }
                _ => Ty::Bool,
            },
            ExprKind::Unary { op, rhs } => match op {
                UnaryOp::Neg => self.infer_ty(rhs)?,
                UnaryOp::Not => Ty::Bool,
            },
            ExprKind::Cast { to, .. } => resolve_cast_target(to)?,
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(name) = &callee.kind {
                    if name == "string_len" {
                        return Ok(Ty::I32);
                    }
                    if name == "assert" {
                        return Ok(Ty::Unit);
                    }
                    match name.as_str() {
                        "Ok" => {
                            let ok_ty = args.first().map(|a| self.infer_ty(&a.value)).transpose()?.unwrap_or(Ty::Error);
                            return Ok(Ty::Result(Box::new(ok_ty), Box::new(Ty::Error)));
                        }
                        "Err" => {
                            let err_ty = args.first().map(|a| self.infer_ty(&a.value)).transpose()?.unwrap_or(Ty::Error);
                            return Ok(Ty::Result(Box::new(Ty::Error), Box::new(err_ty)));
                        }
                        "Some" => {
                            let inner = args.first().map(|a| self.infer_ty(&a.value)).transpose()?.unwrap_or(Ty::Error);
                            return Ok(Ty::Option(Box::new(inner)));
                        }
                        "None" => return Ok(Ty::Option(Box::new(Ty::Error))),
                        _ => {}
                    }
                    if let Some(sig) = self.cg.info.fns.get(name).cloned() {
                        // Generic callee: prefer the type-checker's
                        // call_resolutions if available — it runs
                        // template-aware unification (Pair$GA_GB vs
                        // Pair$I32_Str). Fall back to our own unify,
                        // which only bridges built-in type ctors.
                        if !sig.type_params.is_empty() {
                            let mut subs: HashMap<String, Ty> = HashMap::new();
                            if let Some(type_args) = self.cg.info.call_resolutions.get(&expr.span.start).cloned() {
                                // The recorded type args reference the
                                // enclosing fn's type params (e.g. id(x)
                                // inside id_twice<T> records [Generic("T")]).
                                // Thread active_subs through so T → I32
                                // when monomorphizing id_twice<I32>.
                                let active = self.active_subs.clone();
                                for (p, a) in sig.type_params.iter().zip(type_args.iter()) {
                                    let resolved = substitute_ty(a.clone(), &active);
                                    subs.insert(p.clone(), resolved);
                                }
                            } else {
                                for (i, (_, pty)) in sig.params.iter().enumerate() {
                                    if let Some(arg) = args.get(i) {
                                        let at = self.infer_ty(&arg.value)?;
                                        unify_into(pty, &at, &mut subs);
                                    }
                                }
                            }
                            let subbed = substitute_ty(sig.ret.clone(), &subs);
                            return Ok(self.concretize_ty(&subbed, &subs));
                        }
                        return Ok(sig.ret.clone());
                    }
                    // Unqualified call inside an imported module wrapper:
                    // prefer the current module's sig (if set) — otherwise
                    // HashMap iteration order would pick whichever module
                    // exported the name first, which is nondeterministic.
                    if let Some(m) = self.current_module.as_ref() {
                        if let Some(sig) = self.cg.info.modules.get(m).and_then(|mf| mf.get(name)) {
                            return Ok(sig.ret.clone());
                        }
                    }
                    // Fall back to the first match across all modules
                    // (rare — only matters for non-generic, non-current-module
                    // unqualified calls). Iterate sorted by module name so
                    // the choice is at least deterministic across runs.
                    let mut mod_names: Vec<&String> = self.cg.info.modules.keys().collect();
                    mod_names.sort();
                    for m in mod_names {
                        if let Some(sig) = self.cg.info.modules.get(m).and_then(|mf| mf.get(name)) {
                            return Ok(sig.ret.clone());
                        }
                    }
                    if let Some(sum_name) = self.find_sum_for_variant(name) {
                        return Ok(Ty::User(sum_name));
                    }
                    // Calling a local FnPtr variable — return its ret type.
                    if let Some(Ty::FnPtr { ret, .. }) = self.name_types.get(name) {
                        return Ok(*ret.clone());
                    }
                }
                Ty::I32
            }
            ExprKind::MethodCall { receiver, method, args, .. } => {
                if let ExprKind::Ident(m) = &receiver.kind {
                    if m == "int" && method == "to_string_i32" {
                        return Ok(Ty::String);
                    }
                    if m == "io" && method == "println" {
                        return Ok(Ty::Unit);
                    }
                    if m == "bytes" && method == "len" {
                        return Ok(Ty::I32);
                    }
                    if m == "bytes" && method == "new" {
                        return Ok(Ty::Bytes);
                    }
                    if m == "bytes" && method == "get" {
                        return Ok(Ty::I32);
                    }
                    if m == "bytes" && method == "concat" {
                        return Ok(Ty::Bytes);
                    }
                    if m == "bytes" && method == "from_string" {
                        return Ok(Ty::Bytes);
                    }
                    if m == "string" && method == "from_bytes" {
                        return Ok(Ty::String);
                    }
                    // TCP socket operations
                    if m == "net" && method == "tcp_listen" {
                        return Ok(Ty::I32);
                    }
                    if m == "net" && method == "tcp_accept" {
                        return Ok(Ty::I32);
                    }
                    if m == "net" && method == "tcp_read" {
                        return Ok(Ty::Bytes);
                    }
                    if m == "net" && method == "tcp_write" {
                        return Ok(Ty::I32);
                    }
                    if m == "net" && method == "tcp_close" {
                        return Ok(Ty::Unit);
                    }
                    if m == "net" && method == "serve" {
                        return Ok(Ty::Unit);
                    }
                    if m == "net" && method == "gt_read" {
                        return Ok(Ty::Bytes);
                    }
                    if m == "net" && method == "gt_write" {
                        return Ok(Ty::I32);
                    }
                    // HTTP parsing/formatting
                    if m == "http" && (method == "parse_method" || method == "parse_path" || method == "parse_body") {
                        return Ok(Ty::String);
                    }
                    if m == "http" && method == "format_response" {
                        return Ok(Ty::Bytes);
                    }
                    // List<T> operations
                    if m == "list" && method == "new" {
                        // Element type starts as Error so a `var xs: List<T> = list.new()`
                        // annotation refines via ty_more_specific. Without annotation, the
                        // first list.push (which refines via its arg type) pins it.
                        return Ok(Ty::List(Box::new(Ty::Error)));
                    }
                    if m == "list" && method == "len" {
                        return Ok(Ty::I32);
                    }
                    if m == "list" && method == "push" {
                        // Refine List<I64> → List<T> from the second arg.
                        let list_ty = args.first().map(|a| self.infer_ty(&a.value)).transpose()?.unwrap_or(Ty::List(Box::new(Ty::I64)));
                        let elem_ty = args.get(1).map(|a| self.infer_ty(&a.value)).transpose()?;
                        if let (Ty::List(inner), Some(et)) = (&list_ty, elem_ty) {
                            if matches!(**inner, Ty::I64 | Ty::I32 | Ty::Error) && !matches!(et, Ty::I32 | Ty::I64 | Ty::F64) {
                                return Ok(Ty::List(Box::new(et)));
                            }
                        }
                        return Ok(list_ty);
                    }
                    if m == "list" && method == "get" {
                        // Extract element type from the list argument.
                        if let Some(first_arg) = args.first() {
                            let list_ty = self.infer_ty(&first_arg.value)?;
                            if let Ty::List(inner) = list_ty {
                                return Ok(*inner);
                            }
                        }
                        return Ok(Ty::I64);
                    }
                    if m == "list" && method == "set" {
                        // Returns the list pointer (same type as first arg).
                        if let Some(first_arg) = args.first() {
                            return self.infer_ty(&first_arg.value);
                        }
                        return Ok(Ty::List(Box::new(Ty::I32)));
                    }
                    if m == "list" && method == "remove" {
                        // Returns the list pointer (same type as first arg).
                        if let Some(first_arg) = args.first() {
                            return self.infer_ty(&first_arg.value);
                        }
                        return Ok(Ty::List(Box::new(Ty::I32)));
                    }
                    // --- map ---
                    if m == "map" && method == "new" {
                        return Ok(Ty::Map(Box::new(Ty::Error), Box::new(Ty::Error)));
                    }
                    if m == "map" && method == "set" {
                        let map_ty = args.first().map(|a| self.infer_ty(&a.value)).transpose()?
                            .unwrap_or(Ty::Map(Box::new(Ty::String), Box::new(Ty::Error)));
                        let val_ty = args.get(2).map(|a| self.infer_ty(&a.value)).transpose()?;
                        if let (Ty::Map(k, v), Some(vt)) = (&map_ty, val_ty) {
                            if matches!(**v, Ty::Error) {
                                return Ok(Ty::Map(k.clone(), Box::new(vt)));
                            }
                        }
                        return Ok(map_ty);
                    }
                    if m == "map" && method == "get" {
                        let map_ty = args.first().map(|a| self.infer_ty(&a.value)).transpose()?
                            .unwrap_or(Ty::Map(Box::new(Ty::String), Box::new(Ty::Error)));
                        let elem_ty = match map_ty { Ty::Map(_, v) => *v, _ => Ty::Error };
                        return Ok(Ty::Option(Box::new(elem_ty)));
                    }
                    if m == "map" && method == "contains" {
                        return Ok(Ty::Bool);
                    }
                    if m == "map" && method == "remove" {
                        if let Some(first_arg) = args.first() {
                            return self.infer_ty(&first_arg.value);
                        }
                        return Ok(Ty::Map(Box::new(Ty::String), Box::new(Ty::Error)));
                    }
                    if m == "map" && method == "len" {
                        return Ok(Ty::I32);
                    }
                    if m == "map" && method == "keys" {
                        let map_ty = args.first().map(|a| self.infer_ty(&a.value)).transpose()?
                            .unwrap_or(Ty::Map(Box::new(Ty::Error), Box::new(Ty::Error)));
                        let key_ty = match map_ty { Ty::Map(k, _) => *k, _ => Ty::Error };
                        return Ok(Ty::List(Box::new(key_ty)));
                    }
                    if m == "map" && method == "values" {
                        let map_ty = args.first().map(|a| self.infer_ty(&a.value)).transpose()?
                            .unwrap_or(Ty::Map(Box::new(Ty::String), Box::new(Ty::Error)));
                        let elem_ty = match map_ty { Ty::Map(_, v) => *v, _ => Ty::Error };
                        return Ok(Ty::List(Box::new(elem_ty)));
                    }
                    if m == "map" && method == "entries" {
                        let map_ty = args.first().map(|a| self.infer_ty(&a.value)).transpose()?
                            .unwrap_or(Ty::Map(Box::new(Ty::Error), Box::new(Ty::Error)));
                        let (key_ty, val_ty) = match map_ty {
                            Ty::Map(k, v) => (*k, *v),
                            _ => (Ty::Error, Ty::Error),
                        };
                        return Ok(Ty::List(Box::new(Ty::Tuple(vec![key_ty, val_ty]))));
                    }
                    if m == "map" && method == "merge" {
                        if let Some(first_arg) = args.first() {
                            return self.infer_ty(&first_arg.value);
                        }
                        return Ok(Ty::Map(Box::new(Ty::String), Box::new(Ty::Error)));
                    }
                    if m == "map" && method == "get_or" {
                        let map_ty = args.first().map(|a| self.infer_ty(&a.value)).transpose()?
                            .unwrap_or(Ty::Map(Box::new(Ty::String), Box::new(Ty::Error)));
                        let elem_ty = match map_ty { Ty::Map(_, v) => *v, _ => Ty::Error };
                        return Ok(elem_ty);
                    }
                    // --- Raylib ---
                    if m == "rl" {
                        return Ok(match method.as_ref() {
                            "init_window" | "close_window" | "set_target_fps"
                            | "begin_drawing" | "end_drawing" | "clear_background"
                            | "draw_text" | "draw_rect" | "draw_rect_i" | "draw_rect_pro"
                            | "draw_circle" | "draw_line"
                            | "set_camera" | "begin_mode_2d" | "end_mode_2d"
                            | "init_audio" => Ty::Unit,
                            "window_should_close" | "is_key_pressed" | "is_key_down"
                            | "is_gesture_detected" | "measure_text"
                            | "black" | "white" | "red" | "green" | "blue"
                            | "yellow" | "purple" | "darkblue" | "darkgray" | "gray"
                            | "color_alpha" => Ty::I32,
                            "get_frame_time" => Ty::F64,
                            _ => Ty::I32,
                        });
                    }
                    // --- Math ---
                    if m == "math" {
                        return Ok(Ty::F64);
                    }
                    // Fall through to imported module lookup. Clone the
                    // sig so we can re-borrow self mutably for infer_ty
                    // and concretize_ty calls below.
                    let sig = self.cg.info.modules.get(m.as_str())
                        .and_then(|mf| mf.get(method.as_str()))
                        .cloned();
                    if let Some(sig) = sig {
                        // Generic module fn: infer the substitution
                        // from arg types and apply to the return,
                        // then concretize any deferred Ty::User names.
                        if !sig.type_params.is_empty() {
                            let mut subs: HashMap<String, Ty> = HashMap::new();
                            for (i, (_, pty)) in sig.params.iter().enumerate() {
                                if let Some(arg) = args.get(i) {
                                    let at = self.infer_ty(&arg.value)?;
                                    unify_into(pty, &at, &mut subs);
                                }
                            }
                            let subbed = substitute_ty(sig.ret.clone(), &subs);
                            return Ok(self.concretize_ty(&subbed, &subs));
                        }
                        return Ok(sig.ret.clone());
                    }
                }
                Ty::I32
            }
            ExprKind::StructLit { name, fields, .. } => {
                // Variant constructor recorded by the type checker against
                // an expected generic-sum instantiation (e.g. `Just { value: 42 }`
                // resolved against expected = Maybe$I32) — that wins because
                // the bare name is a variant, not a registered type.
                if let Some(sum_name) = self.resolved_variant_sum(expr.span) {
                    return Ok(Ty::User(sum_name));
                }
                // Bare variant of a non-generic sum (find_sum_for_variant
                // covers it because the sum itself is registered in info.types).
                if let Some(sum_name) = self.find_sum_for_variant(name) {
                    return Ok(Ty::User(sum_name));
                }
                // For a generic-template struct, the actual concrete type
                // is the mangled instantiation derived from field types.
                let resolved = self.resolve_struct_lit_name(name, fields)?;
                Ty::User(resolved)
            }
            ExprKind::Field { receiver, name } => {
                let recv_ty = self.infer_ty(receiver)?;
                let type_name = if let Ty::User(tn) = recv_ty {
                    Some(tn)
                } else {
                    // Same alphabetical struct resolution as compile_expr.
                    let mut candidates: Vec<&str> = Vec::new();
                    for (tname, tinfo) in &self.cg.info.types {
                        if let TypeInfo::Struct { fields, .. } = tinfo {
                            if fields.iter().any(|(f, _)| f == name) {
                                candidates.push(tname.as_str());
                            }
                        }
                    }
                    candidates.sort();
                    candidates.first().map(|s| s.to_string())
                };
                if let Some(tn) = type_name {
                    let fields = get_struct_fields(&self.cg.info.types, &tn);
                    fields
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, t)| t.clone())
                        .unwrap_or(Ty::I32)
                } else {
                    Ty::I32
                }
            }
            ExprKind::Spawn { actor_name, .. } => {
                Ty::Handle(Box::new(Ty::User(actor_name.clone())))
            }
            ExprKind::Send { .. } => Ty::Unit,
            ExprKind::Ask { handle, method, .. } => {
                let handle_ty = self.infer_ty(handle)?;
                if let Ty::Handle(inner) = &handle_ty {
                    if let Ty::User(actor_name) = inner.as_ref() {
                        let fn_name = format!("{}_{}", actor_name, method);
                        if let Some(sig) = self.cg.info.fns.get(&fn_name) {
                            return Ok(sig.ret.clone());
                        }
                    }
                }
                Ty::I32
            }
            ExprKind::If { then_block, .. } => self.infer_block_ty(then_block).unwrap_or(Ty::Unit),
            ExprKind::Match { arms, .. } => arms
                .first()
                .map(|a| self.infer_ty(&a.body))
                .transpose()?
                .unwrap_or(Ty::I32),
            ExprKind::Try(inner) => {
                let inner_ty = self.infer_ty(inner)?;
                match inner_ty {
                    Ty::Result(ok, _) => *ok,
                    Ty::Option(inner) => *inner,
                    _ => Ty::Error,
                }
            }
            ExprKind::TupleLit(elems) => {
                let types: Result<Vec<Ty>, NativeError> =
                    elems.iter().map(|e| self.infer_ty(e)).collect();
                Ty::Tuple(types?)
            }
            ExprKind::TupleField { receiver, index } => {
                let recv_ty = self.infer_ty(receiver)?;
                match recv_ty {
                    Ty::Tuple(elems) => {
                        let idx = *index as usize;
                        elems.get(idx).cloned().unwrap_or(Ty::I32)
                    }
                    _ => Ty::I32,
                }
            }
            ExprKind::Lambda { params, return_type, .. } => {
                let param_tys: Vec<Ty> = params.iter()
                    .map(|p| resolve_type_to_ty(&p.ty))
                    .collect();
                let ret = resolve_type_to_ty(return_type);
                Ty::FnPtr { params: param_tys, ret: Box::new(ret) }
            }
            _ => Ty::I32,
        })
    }

    fn infer_block_ty(&mut self, block: &ast::Block) -> Option<Ty> {
        block.tail.as_ref().map(|e| self.infer_ty(e).unwrap_or(Ty::I32))
    }

    fn find_sum_for_variant(&self, name: &str) -> Option<String> {
        for (type_name, info) in &self.cg.info.types {
            if let TypeInfo::Sum { variants, .. } = info {
                if variants.iter().any(|v| v.name == name) {
                    return Some(type_name.clone());
                }
            }
        }
        None
    }

    /// If the type checker resolved this variant-constructor expression
    /// against an expected generic-sum instantiation (recorded in
    /// ModuleInfo.variant_resolutions), return that mangled sum name.
    fn resolved_variant_sum(&self, span: Span) -> Option<String> {
        self.cg.info.variant_resolutions.get(&span.start).cloned()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns true when the expression reads an existing reference rather than
/// producing a fresh allocation. Used to decide whether `var = expr` needs
/// rc_incr (copies need it, fresh values already have rc=1).
/// Compute initial bytes for a top-level binding from its initializer
/// expression. MVP: only scalar literals (optionally unary-negated).
/// Returns None if the initializer isn't reducible to a constant at
/// compile time, which the caller reports as an error.
fn evaluate_const_initializer(e: &Expr, ty: &Ty) -> Option<Vec<u8>> {
    let (magnitude_u64, negative) = extract_int_literal(e);
    if let Some(v) = magnitude_u64 {
        return Some(encode_int(v, negative, ty));
    }
    match &e.kind {
        ExprKind::FloatLit(v) => {
            if matches!(ty, Ty::F64) {
                Some(v.to_le_bytes().to_vec())
            } else { None }
        }
        ExprKind::BoolLit(b) => {
            if matches!(ty, Ty::Bool) {
                Some(vec![if *b { 1 } else { 0 }, 0, 0, 0])
            } else { None }
        }
        ExprKind::CharLit(c) => {
            if matches!(ty, Ty::Char) {
                Some(c.to_le_bytes().to_vec())
            } else { None }
        }
        ExprKind::UnitLit => {
            if matches!(ty, Ty::Unit) {
                Some(vec![0, 0, 0, 0])
            } else { None }
        }
        ExprKind::Paren(inner) => evaluate_const_initializer(inner, ty),
        _ => None,
    }
}

/// Unwrap `-literal` / `literal` into a (magnitude, negative) pair so
/// the caller can encode against the declared type's width.
fn extract_int_literal(e: &Expr) -> (Option<u64>, bool) {
    match &e.kind {
        ExprKind::IntLit { value, .. } => (Some(*value), false),
        ExprKind::Unary { op: UnaryOp::Neg, rhs } => {
            if let ExprKind::IntLit { value, .. } = &rhs.kind {
                return (Some(*value), true);
            }
            (None, false)
        }
        ExprKind::Paren(inner) => extract_int_literal(inner),
        _ => (None, false),
    }
}

fn encode_int(magnitude: u64, negative: bool, ty: &Ty) -> Vec<u8> {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::Unit | Ty::Char => {
            let v: i32 = if negative { -(magnitude as i64) as i32 } else { magnitude as i32 };
            v.to_le_bytes().to_vec()
        }
        Ty::I64 | Ty::U64 => {
            let v: i64 = if negative { -(magnitude as i64) } else { magnitude as i64 };
            v.to_le_bytes().to_vec()
        }
        _ => vec![0, 0, 0, 0, 0, 0, 0, 0],
    }
}

/// Symbol name for a top-level binding. Namespaced by module so two
/// different modules can have the same Lumen-level binding name.
fn global_symbol(module: &str, name: &str) -> String {
    format!("lumen_global_{}_{}", module.replace('/', "_"), name)
}

fn is_borrowing_expr(kind: &ExprKind) -> bool {
    matches!(
        kind,
        ExprKind::Ident(_) | ExprKind::Field { .. } | ExprKind::TupleField { .. }
    ) || matches!(kind, ExprKind::Paren(inner) if is_borrowing_expr(&inner.kind))
}

fn tuple_as_fields(elems: &[Ty]) -> Vec<(String, Ty)> {
    elems.iter().enumerate().map(|(i, t)| (format!("_{i}"), t.clone())).collect()
}

fn lumen_to_cl(ty: &Ty) -> CLType {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::Unit | Ty::Char => cl_types::I32,
        Ty::I64 | Ty::U64 => cl_types::I64,
        Ty::F64 => cl_types::F64,
        Ty::String | Ty::Bytes | Ty::User(_) | Ty::Option(_) | Ty::Result(_, _) | Ty::List(_) | Ty::Map(_, _) | Ty::Handle(_) | Ty::Tuple(_) => PTR,
        // Ty::Generic should only appear in pre-monomorphized signatures.
        // If it leaks into codegen, treat as pointer-sized — safest default.
        Ty::Generic(_) => PTR,
        Ty::FnPtr { .. } => PTR,
        Ty::Error => cl_types::I32,
    }
}

/// Map Lumen's internal Ty into the smaller DWARF-facing enum. Every
/// heap-allocated shape (strings, bytes, lists, structs, sums, tuples,
/// handles, fn ptrs) collapses to `Pointer` — gdb sees them as opaque
/// addresses until ax3's follow-up wires full struct layouts.
fn ty_to_dwarf(ty: &Ty) -> crate::dwarf::DwarfTy {
    use crate::dwarf::DwarfTy;
    match ty {
        Ty::I32 => DwarfTy::I32,
        Ty::I64 => DwarfTy::I64,
        Ty::U32 => DwarfTy::U32,
        Ty::U64 => DwarfTy::U64,
        Ty::F64 => DwarfTy::F64,
        Ty::Bool => DwarfTy::Bool,
        Ty::Char => DwarfTy::Char,
        Ty::Unit => DwarfTy::Unit,
        _ => DwarfTy::Pointer,
    }
}

fn is_scalar(ty: &Ty) -> bool {
    matches!(
        ty,
        Ty::I32 | Ty::U32 | Ty::I64 | Ty::U64 | Ty::F64 | Ty::Bool | Ty::Unit | Ty::Char
        // Lists and maps are treated as scalar for RC purposes: they
        // manage their own memory via realloc inside set/push/remove. RC
        // decrementing one after realloc moved it would double-free.
        | Ty::List(_)
        | Ty::Map(_, _)
        // Function pointers are just integer addresses — no RC needed.
        | Ty::FnPtr { .. }
    )
}

/// Recursively replace Ty::Generic(name) with the substitution if name
/// is bound. Also replaces Ty::User(name) when bound (because the codegen
/// AST resolver returns User for unknown idents, including type params
/// that live only in the generic body's annotations).
fn substitute_ty(ty: Ty, subs: &HashMap<String, Ty>) -> Ty {
    if subs.is_empty() { return ty; }
    match ty {
        Ty::Generic(name) => subs.get(&name).cloned().unwrap_or(Ty::Generic(name)),
        Ty::User(name) if subs.contains_key(&name) => subs[&name].clone(),
        Ty::List(inner) => Ty::List(Box::new(substitute_ty(*inner, subs))),
        Ty::Map(k, v) => Ty::Map(Box::new(substitute_ty(*k, subs)), Box::new(substitute_ty(*v, subs))),
        Ty::Option(inner) => Ty::Option(Box::new(substitute_ty(*inner, subs))),
        Ty::Result(o, e) => Ty::Result(Box::new(substitute_ty(*o, subs)), Box::new(substitute_ty(*e, subs))),
        Ty::Handle(inner) => Ty::Handle(Box::new(substitute_ty(*inner, subs))),
        Ty::Tuple(elems) => Ty::Tuple(elems.into_iter().map(|t| substitute_ty(t, subs)).collect()),
        Ty::FnPtr { params, ret } => Ty::FnPtr {
            params: params.into_iter().map(|t| substitute_ty(t, subs)).collect(),
            ret: Box::new(substitute_ty(*ret, subs)),
        },
        other => other,
    }
}

/// Unify a generic param type against a concrete arg type, recording any
/// new Ty::Generic bindings into `subs`. First binding wins (no
/// constraint solving — practical for unbounded MVP generics).
fn unify_into(generic: &Ty, concrete: &Ty, subs: &mut HashMap<String, Ty>) {
    match (generic, concrete) {
        (Ty::Generic(name), c) => {
            subs.entry(name.clone()).or_insert_with(|| c.clone());
        }
        (Ty::List(g), Ty::List(c)) => unify_into(g, c, subs),
        (Ty::Map(gk, gv), Ty::Map(ck, cv)) => { unify_into(gk, ck, subs); unify_into(gv, cv, subs); }
        (Ty::Option(g), Ty::Option(c)) => unify_into(g, c, subs),
        (Ty::Result(go, ge), Ty::Result(co, ce)) => { unify_into(go, co, subs); unify_into(ge, ce, subs); }
        (Ty::Handle(g), Ty::Handle(c)) => unify_into(g, c, subs),
        (Ty::Tuple(gs), Ty::Tuple(cs)) if gs.len() == cs.len() => {
            for (g, c) in gs.iter().zip(cs.iter()) { unify_into(g, c, subs); }
        }
        (Ty::FnPtr { params: gp, ret: gr }, Ty::FnPtr { params: cp, ret: cr }) => {
            for (g, c) in gp.iter().zip(cp.iter()) { unify_into(g, c, subs); }
            unify_into(gr, cr, subs);
        }
        _ => {}
    }
}

/// Build a deterministic mangled name for a monomorphization. Order of
/// type args follows the fn's declared `type_params` so two call sites
/// with the same instantiation produce the same name.
fn mangle_monomorph_name(base: &str, type_args: &[Ty]) -> String {
    let mut s = base.to_string();
    s.push('$');
    for (i, t) in type_args.iter().enumerate() {
        if i > 0 { s.push('_'); }
        s.push_str(&mangle_ty(t));
    }
    s
}

fn mangle_ty(t: &Ty) -> String {
    match t {
        Ty::I32 => "I32".into(),
        Ty::I64 => "I64".into(),
        Ty::U32 => "U32".into(),
        Ty::U64 => "U64".into(),
        Ty::F64 => "F64".into(),
        Ty::Bool => "Bool".into(),
        Ty::String => "Str".into(),
        Ty::Char => "Char".into(),
        Ty::Bytes => "Bytes".into(),
        Ty::Unit => "Unit".into(),
        Ty::List(inner) => format!("L{}", mangle_ty(inner)),
        Ty::Map(k, v) => format!("M{}V{}", mangle_ty(k), mangle_ty(v)),
        Ty::Option(inner) => format!("O{}", mangle_ty(inner)),
        Ty::Result(o, e) => format!("R{}E{}", mangle_ty(o), mangle_ty(e)),
        Ty::Handle(inner) => format!("H{}", mangle_ty(inner)),
        Ty::Tuple(elems) => format!("T{}", elems.iter().map(mangle_ty).collect::<Vec<_>>().join("_")),
        Ty::User(n) => format!("U{}", n),
        Ty::FnPtr { .. } => "Fn".into(),
        Ty::Generic(n) => format!("G{}", n),
        Ty::Error => "Err".into(),
    }
}

/// True when `annot` carries strictly more information than `inferred` —
/// i.e. the inferred type has Ty::Error in a generic slot that the
/// annotation pins to a real type. Used so `var m: Map<string, i32> = map.new()`
/// honors the annotation (map.new returns Map<_, Error>).
fn ty_more_specific(annot: &Ty, inferred: &Ty) -> bool {
    match (annot, inferred) {
        (Ty::List(a), Ty::List(b)) => !matches!(**a, Ty::Error) && matches!(**b, Ty::Error),
        (Ty::Map(ak, av), Ty::Map(bk, bv)) => {
            (!matches!(**ak, Ty::Error) && matches!(**bk, Ty::Error))
                || (!matches!(**av, Ty::Error) && matches!(**bv, Ty::Error))
        }
        (Ty::Option(a), Ty::Option(b)) => !matches!(**a, Ty::Error) && matches!(**b, Ty::Error),
        (Ty::Result(ao, ae), Ty::Result(bo, be)) => {
            (!matches!(**ao, Ty::Error) && matches!(**bo, Ty::Error))
                || (!matches!(**ae, Ty::Error) && matches!(**be, Ty::Error))
        }
        _ => false,
    }
}

fn native_sizeof(ty: &Ty) -> i32 {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::Unit | Ty::Char => 4,
        Ty::I64 | Ty::U64 | Ty::F64 => 8,
        Ty::String | Ty::Bytes | Ty::User(_) | Ty::Option(_) | Ty::Result(_, _) | Ty::List(_) | Ty::Map(_, _) | Ty::Handle(_) | Ty::Tuple(_) => 8, // pointer
        Ty::Generic(_) => 8, // see comment in lumen_to_cl
        Ty::FnPtr { .. } => 8,
        Ty::Error => 4,
    }
}

/// Convert an AST type to a Ty (simplified, for lambda signatures).
fn resolve_type_to_ty(ty: &ast::Type) -> Ty {
    match &ty.kind {
        ast::TypeKind::Named { name, args } => match (name.as_str(), args.len()) {
            ("i32", 0) => Ty::I32,
            ("i64", 0) => Ty::I64,
            ("u32", 0) => Ty::U32,
            ("u64", 0) => Ty::U64,
            ("f64", 0) => Ty::F64,
            ("bool", 0) => Ty::Bool,
            ("unit", 0) => Ty::Unit,
            ("char", 0) => Ty::Char,
            ("string", 0) | ("String", 0) => Ty::String,
            ("bytes", 0) | ("Bytes", 0) => Ty::Bytes,
            ("List", 1) => Ty::List(Box::new(resolve_type_to_ty(&args[0]))),
            ("Map", 2) => Ty::Map(
                Box::new(resolve_type_to_ty(&args[0])),
                Box::new(resolve_type_to_ty(&args[1])),
            ),
            ("Option", 1) => Ty::Option(Box::new(resolve_type_to_ty(&args[0]))),
            ("Result", 2) => Ty::Result(
                Box::new(resolve_type_to_ty(&args[0])),
                Box::new(resolve_type_to_ty(&args[1])),
            ),
            ("Handle", 1) => Ty::Handle(Box::new(resolve_type_to_ty(&args[0]))),
            (other, 0) => Ty::User(other.to_string()),
            // Generic user-type instantiation: produce the mangled
            // name. The type-checker pre-pass should have registered
            // the corresponding TypeInfo in info.types under this name.
            (other, _) => {
                let resolved_args: Vec<Ty> = args.iter().map(resolve_type_to_ty).collect();
                Ty::User(crate::types::mangle_type_instantiation(other, &resolved_args))
            }
        },
        ast::TypeKind::FnPtr { params, ret } => {
            let param_tys = params.iter().map(resolve_type_to_ty).collect();
            Ty::FnPtr { params: param_tys, ret: Box::new(resolve_type_to_ty(ret)) }
        }
        ast::TypeKind::Tuple(elems) => {
            Ty::Tuple(elems.iter().map(resolve_type_to_ty).collect())
        }
    }
}

fn resolve_cast_target(ty: &ast::Type) -> Result<Ty, NativeError> {
    match &ty.kind {
        ast::TypeKind::Named { name, args } if args.is_empty() => match name.as_str() {
            "i32" => Ok(Ty::I32),
            "i64" => Ok(Ty::I64),
            "u32" => Ok(Ty::U32),
            "u64" => Ok(Ty::U64),
            "f64" => Ok(Ty::F64),
            // char is represented as i32 at runtime; allow the cast so
            // users can convert an i32 codepoint (e.g. from a decoder)
            // into a char. There's no validity check here — callers that
            // need one should use a checked conversion helper.
            "char" => Ok(Ty::Char),
            _ => Err(NativeError {
                span: ty.span,
                message: format!("`as` target must be numeric, got `{name}`"),
            }),
        },
        _ => Err(NativeError {
            span: ty.span,
            message: "`as` target must be bare numeric type".into(),
        }),
    }
}

fn emit_cast_cl(builder: &mut FunctionBuilder, v: Value, from: &Ty, to: &Ty) -> Value {
    match (from, to) {
        (Ty::I32, Ty::I64) => builder.ins().sextend(cl_types::I64, v),
        (Ty::U32, Ty::I64) | (Ty::I32, Ty::U64) | (Ty::U32, Ty::U64) => {
            builder.ins().uextend(cl_types::I64, v)
        }
        (Ty::I64, Ty::I32) | (Ty::I64, Ty::U32) | (Ty::U64, Ty::I32) | (Ty::U64, Ty::U32) => {
            builder.ins().ireduce(cl_types::I32, v)
        }
        (Ty::I32, Ty::F64) => builder.ins().fcvt_from_sint(cl_types::F64, v),
        (Ty::U32, Ty::F64) => builder.ins().fcvt_from_uint(cl_types::F64, v),
        (Ty::I64, Ty::F64) => builder.ins().fcvt_from_sint(cl_types::F64, v),
        (Ty::U64, Ty::F64) => builder.ins().fcvt_from_uint(cl_types::F64, v),
        (Ty::F64, Ty::I32) => builder.ins().fcvt_to_sint(cl_types::I32, v),
        (Ty::F64, Ty::U32) => builder.ins().fcvt_to_uint(cl_types::I32, v),
        (Ty::F64, Ty::I64) => builder.ins().fcvt_to_sint(cl_types::I64, v),
        (Ty::F64, Ty::U64) => builder.ins().fcvt_to_uint(cl_types::I64, v),
        _ => v,
    }
}

fn get_struct_fields(types: &HashMap<String, TypeInfo>, name: &str) -> Vec<(String, Ty)> {
    match types.get(name) {
        Some(TypeInfo::Struct { fields, .. }) => fields.clone(),
        _ => Vec::new(),
    }
}

fn field_offset(fields: &[(String, Ty)], target: &str) -> (i32, Ty) {
    let mut offset = 0i32;
    for (name, ty) in fields {
        let align = match ty {
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::String | Ty::Bytes | Ty::User(_) => 8,
            _ => 4,
        };
        offset = (offset + align - 1) & !(align - 1);
        if name == target {
            return (offset, ty.clone());
        }
        let size = match ty {
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::String | Ty::Bytes | Ty::User(_) => 8,
            _ => 4,
        };
        offset += size;
    }
    (0, Ty::I32)
}

fn struct_size(fields: &[(String, Ty)]) -> i32 {
    let mut offset = 0i32;
    for (_, ty) in fields {
        let align = match ty {
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::String | Ty::Bytes | Ty::User(_) => 8,
            _ => 4,
        };
        offset = (offset + align - 1) & !(align - 1);
        let size = match ty {
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::String | Ty::Bytes | Ty::User(_) => 8,
            _ => 4,
        };
        offset += size;
    }
    (offset + 7) & !7
}

// --- Lambda collection ---------------------------------------------------

struct LambdaInfo {
    params: Vec<ast::Param>,
    return_type: ast::Type,
    body: ast::Block,
    span: Span,
}

fn collect_lambdas_block(block: &ast::Block, acc: &mut Vec<LambdaInfo>) {
    for stmt in &block.stmts {
        collect_lambdas_stmt(stmt, acc);
    }
    if let Some(tail) = &block.tail {
        collect_lambdas_expr(tail, acc);
    }
}

fn collect_lambdas_stmt(stmt: &ast::Stmt, acc: &mut Vec<LambdaInfo>) {
    match &stmt.kind {
        StmtKind::Let { value, .. }
        | StmtKind::Var { value, .. }
        | StmtKind::Assign { value, .. } => collect_lambdas_expr(value, acc),
        StmtKind::LetTuple { value, .. } => collect_lambdas_expr(value, acc),
        StmtKind::Expr(e) => collect_lambdas_expr(e, acc),
        StmtKind::For { iter, body, .. } => {
            collect_lambdas_expr(iter, acc);
            collect_lambdas_block(body, acc);
        }
        StmtKind::Return(Some(e)) => collect_lambdas_expr(e, acc),
        StmtKind::Return(None) => {}
    }
}

fn collect_lambdas_expr(expr: &Expr, acc: &mut Vec<LambdaInfo>) {
    match &expr.kind {
        ExprKind::Lambda { params, return_type, body } => {
            // Collect strings inside the lambda body too.
            collect_lambdas_block(body, acc);
            acc.push(LambdaInfo {
                params: params.clone(),
                return_type: return_type.clone(),
                body: body.clone(),
                span: expr.span,
            });
        }
        ExprKind::Paren(e) | ExprKind::Unary { rhs: e, .. } | ExprKind::Cast { expr: e, .. } => {
            collect_lambdas_expr(e, acc);
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            collect_lambdas_expr(lhs, acc);
            collect_lambdas_expr(rhs, acc);
        }
        ExprKind::Call { callee, args } => {
            collect_lambdas_expr(callee, acc);
            for a in args { collect_lambdas_expr(&a.value, acc); }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            collect_lambdas_expr(receiver, acc);
            for a in args { collect_lambdas_expr(&a.value, acc); }
        }
        ExprKind::If { cond, then_block, else_block } => {
            collect_lambdas_expr(cond, acc);
            collect_lambdas_block(then_block, acc);
            collect_lambdas_block(else_block, acc);
        }
        ExprKind::Block(b) => collect_lambdas_block(b, acc),
        ExprKind::StructLit { fields, spread, .. } => {
            for fi in fields { collect_lambdas_expr(&fi.value, acc); }
            if let Some(s) = spread { collect_lambdas_expr(s, acc); }
        }
        ExprKind::TupleLit(elems) => {
            for e in elems { collect_lambdas_expr(e, acc); }
        }
        ExprKind::Match { scrutinee, arms } => {
            collect_lambdas_expr(scrutinee, acc);
            for arm in arms { collect_lambdas_expr(&arm.body, acc); }
        }
        ExprKind::Interpolated(parts) => {
            for p in parts {
                if let ast::InterpPiece::Expr(e) = p {
                    collect_lambdas_expr(e, acc);
                }
            }
        }
        _ => {}
    }
}

// --- String collection ---------------------------------------------------

fn collect_strings_block(block: &ast::Block, acc: &mut Vec<String>) {
    for stmt in &block.stmts {
        collect_strings_stmt(stmt, acc);
    }
    if let Some(tail) = &block.tail {
        collect_strings_expr(tail, acc);
    }
}

fn collect_strings_stmt(stmt: &ast::Stmt, acc: &mut Vec<String>) {
    match &stmt.kind {
        StmtKind::Let { value, .. }
        | StmtKind::Var { value, .. }
        | StmtKind::Assign { value, .. } => collect_strings_expr(value, acc),
        StmtKind::LetTuple { value, .. } => collect_strings_expr(value, acc),
        StmtKind::Expr(e) => collect_strings_expr(e, acc),
        StmtKind::For { iter, body, .. } => {
            collect_strings_expr(iter, acc);
            collect_strings_block(body, acc);
        }
        StmtKind::Return(Some(e)) => collect_strings_expr(e, acc),
        StmtKind::Return(None) => {}
    }
}

fn collect_strings_expr(expr: &Expr, acc: &mut Vec<String>) {
    match &expr.kind {
        ExprKind::StringLit(s) => acc.push(s.clone()),
        ExprKind::Paren(e) | ExprKind::Unary { rhs: e, .. } | ExprKind::Cast { expr: e, .. } => {
            collect_strings_expr(e, acc)
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            collect_strings_expr(lhs, acc);
            collect_strings_expr(rhs, acc);
        }
        ExprKind::Call { callee, args } => {
            collect_strings_expr(callee, acc);
            for a in args {
                collect_strings_expr(&a.value, acc);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            collect_strings_expr(receiver, acc);
            for a in args {
                collect_strings_expr(&a.value, acc);
            }
        }
        ExprKind::Field { receiver, .. } => collect_strings_expr(receiver, acc),
        ExprKind::Try(e) => collect_strings_expr(e, acc),
        ExprKind::If {
            cond,
            then_block,
            else_block,
        } => {
            collect_strings_expr(cond, acc);
            collect_strings_block(then_block, acc);
            collect_strings_block(else_block, acc);
        }
        ExprKind::Match { scrutinee, arms } => {
            collect_strings_expr(scrutinee, acc);
            for arm in arms {
                collect_strings_expr(&arm.body, acc);
            }
        }
        ExprKind::Block(b) => collect_strings_block(b, acc),
        ExprKind::StructLit { fields, spread, .. } => {
            for fi in fields {
                collect_strings_expr(&fi.value, acc);
            }
            if let Some(s) = spread {
                collect_strings_expr(s, acc);
            }
        }
        ExprKind::TupleLit(elems) => {
            for e in elems {
                collect_strings_expr(e, acc);
            }
        }
        ExprKind::TupleField { receiver, .. } => collect_strings_expr(receiver, acc),
        ExprKind::Spawn { fields, .. } => {
            for fi in fields {
                collect_strings_expr(&fi.value, acc);
            }
        }
        ExprKind::Send { handle, args, .. } | ExprKind::Ask { handle, args, .. } => {
            collect_strings_expr(handle, acc);
            for a in args {
                collect_strings_expr(&a.value, acc);
            }
        }
        ExprKind::Lambda { body, .. } => collect_strings_block(body, acc),
        ExprKind::Interpolated(parts) => {
            for p in parts {
                if let ast::InterpPiece::Expr(e) = p {
                    collect_strings_expr(e, acc);
                }
            }
        }
        ExprKind::Arena(body) => collect_strings_block(body, acc),
        _ => {}
    }
}

fn collect_try_frame_strings(fn_name: &str, block: &ast::Block, acc: &mut Vec<String>) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { value, .. }
            | StmtKind::Var { value, .. }
            | StmtKind::Assign { value, .. } => collect_try_frame_expr(fn_name, value, acc),
            StmtKind::LetTuple { value, .. } => collect_try_frame_expr(fn_name, value, acc),
            StmtKind::Expr(e) => collect_try_frame_expr(fn_name, e, acc),
            StmtKind::For { iter, body, .. } => {
                collect_try_frame_expr(fn_name, iter, acc);
                collect_try_frame_strings(fn_name, body, acc);
            }
            StmtKind::Return(Some(e)) => collect_try_frame_expr(fn_name, e, acc),
            StmtKind::Return(None) => {}
        }
    }
    if let Some(tail) = &block.tail {
        collect_try_frame_expr(fn_name, tail, acc);
    }
}

fn collect_try_frame_expr(fn_name: &str, expr: &Expr, acc: &mut Vec<String>) {
    if let ExprKind::Try(inner) = &expr.kind {
        collect_try_frame_expr(fn_name, inner, acc);
        acc.push(format!("  at {}(", fn_name));
        acc.push(format!(") (<source>:{}:{})", expr.span.line, expr.span.col));
        acc.push(format!(
            "  at {} (<source>:{}:{})",
            fn_name, expr.span.line, expr.span.col
        ));
    }
    // Recurse into sub-expressions.
    match &expr.kind {
        ExprKind::Paren(e) | ExprKind::Unary { rhs: e, .. } | ExprKind::Cast { expr: e, .. } => {
            collect_try_frame_expr(fn_name, e, acc)
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            collect_try_frame_expr(fn_name, lhs, acc);
            collect_try_frame_expr(fn_name, rhs, acc);
        }
        ExprKind::Call { callee, args } => {
            collect_try_frame_expr(fn_name, callee, acc);
            for a in args {
                collect_try_frame_expr(fn_name, &a.value, acc);
            }
        }
        ExprKind::If {
            cond,
            then_block,
            else_block,
        } => {
            collect_try_frame_expr(fn_name, cond, acc);
            collect_try_frame_strings(fn_name, then_block, acc);
            collect_try_frame_strings(fn_name, else_block, acc);
        }
        ExprKind::Block(b) => collect_try_frame_strings(fn_name, b, acc),
        _ => {}
    }
}
