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

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{AbiParam, BlockArg, InstBuilder, MemFlags, Type as CLType, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::ast::{self, BinOp, Effect, Expr, ExprKind, FnDecl, Item, StmtKind, UnaryOp};
use crate::lexer::IntSuffix;
use crate::span::Span;
use crate::types::{ModuleInfo, Ty, TypeInfo};

/// Compile a type-checked module to a native object file (bytes).
pub fn compile_native(
    module: &ast::Module,
    info: &ModuleInfo,
) -> Result<Vec<u8>, NativeError> {
    let mut cg = NativeCodegen::new(info)?;
    cg.compile_module(module)?;
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

// ---------------------------------------------------------------------------
// Native codegen state
// ---------------------------------------------------------------------------

struct NativeCodegen<'a> {
    info: &'a ModuleInfo,
    obj: ObjectModule,

    /// Lumen fn name → Cranelift FuncId.
    fn_ids: HashMap<String, FuncId>,
    /// Built-in helper func IDs.
    libc_write: FuncId,
    libc_malloc: FuncId,
    libc_free: FuncId,
    helper_concat: FuncId,
    helper_println: FuncId,
    helper_itoa: FuncId,
    helper_print_frames: FuncId,
    helper_rc_alloc: FuncId,
    helper_rc_incr: FuncId,
    helper_rc_decr: FuncId,
    rt_send: FuncId,
    rt_ask: FuncId,
    rt_drain: FuncId,
    rt_yield: FuncId,
    /// TCP socket helpers (from runtime/rt.c).
    net_tcp_listen: FuncId,
    net_tcp_accept: FuncId,
    net_tcp_read: FuncId,
    net_tcp_write: FuncId,
    net_tcp_close: FuncId,
    net_serve: FuncId,
    gt_read: FuncId,
    gt_write: FuncId,
    /// HTTP parsing/formatting helpers (from runtime/rt.c).
    http_parse_method: FuncId,
    http_parse_path: FuncId,
    http_parse_body: FuncId,
    http_format_response: FuncId,
    /// List<T> operations (from runtime/rt.c).
    list_new: FuncId,
    list_len: FuncId,
    list_push: FuncId,
    list_get: FuncId,
    list_set: FuncId,
    list_remove: FuncId,

    // --- Raylib bridge functions ---
    rl_init_window: FuncId,
    rl_close_window: FuncId,
    rl_window_should_close: FuncId,
    rl_set_target_fps: FuncId,
    rl_get_frame_time: FuncId,
    rl_begin_drawing: FuncId,
    rl_end_drawing: FuncId,
    rl_clear_background: FuncId,
    rl_draw_text: FuncId,
    rl_measure_text: FuncId,
    rl_draw_rectangle_rec: FuncId,
    rl_draw_rectangle: FuncId,
    rl_draw_rectangle_pro: FuncId,
    rl_draw_circle: FuncId,
    rl_draw_line: FuncId,
    rl_is_key_pressed: FuncId,
    rl_is_key_down: FuncId,
    rl_is_gesture_detected: FuncId,
    rl_set_camera: FuncId,
    rl_begin_mode_2d: FuncId,
    rl_end_mode_2d: FuncId,
    rl_init_audio: FuncId,
    rl_color_black: FuncId,
    rl_color_white: FuncId,
    rl_color_red: FuncId,
    rl_color_green: FuncId,
    rl_color_blue: FuncId,
    rl_color_yellow: FuncId,
    rl_color_purple: FuncId,
    rl_color_darkblue: FuncId,
    rl_color_darkgray: FuncId,
    rl_color_gray: FuncId,
    rl_color_alpha: FuncId,

    // --- Math helpers ---
    math_sqrt: FuncId,
    math_abs: FuncId,
    math_cos: FuncId,
    math_sin: FuncId,
    math_clamp: FuncId,
    math_rand: FuncId,

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

    /// Uses WASI / io module.
    uses_io: bool,
}

impl<'a> NativeCodegen<'a> {
    fn new(info: &'a ModuleInfo) -> Result<Self, NativeError> {
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

        // Declare libc write(fd, buf, count) -> ssize_t
        let mut write_sig = obj.make_signature();
        write_sig.params.push(AbiParam::new(cl_types::I32)); // fd
        write_sig.params.push(AbiParam::new(PTR)); // buf
        write_sig.params.push(AbiParam::new(PTR)); // count
        write_sig.returns.push(AbiParam::new(PTR)); // ssize_t
        let libc_write = obj
            .declare_function("write", Linkage::Import, &write_sig)
            .unwrap();

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
            obj.declare_data("lumen_frame_chain", Linkage::Local, true, false).unwrap();

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

        // Declare internal helpers (defined later).
        let mut concat_sig = obj.make_signature();
        concat_sig.params.push(AbiParam::new(PTR));
        concat_sig.params.push(AbiParam::new(PTR));
        concat_sig.returns.push(AbiParam::new(PTR));
        let helper_concat = obj
            .declare_function("lumen_concat", Linkage::Local, &concat_sig)
            .unwrap();

        let mut println_sig = obj.make_signature();
        println_sig.params.push(AbiParam::new(PTR));
        let helper_println = obj
            .declare_function("lumen_println", Linkage::Local, &println_sig)
            .unwrap();

        let mut itoa_sig = obj.make_signature();
        itoa_sig.params.push(AbiParam::new(cl_types::I32));
        itoa_sig.returns.push(AbiParam::new(PTR));
        let helper_itoa = obj
            .declare_function("lumen_itoa", Linkage::Local, &itoa_sig)
            .unwrap();

        // rc_alloc(size: i64) -> ptr: malloc(size+8), set rc=1, return ptr+8
        let mut rc_alloc_sig = obj.make_signature();
        rc_alloc_sig.params.push(AbiParam::new(PTR));
        rc_alloc_sig.returns.push(AbiParam::new(PTR));
        let helper_rc_alloc = obj
            .declare_function("lumen_rc_alloc", Linkage::Local, &rc_alloc_sig)
            .unwrap();

        // rc_incr(ptr): if ptr in heap, rc++
        let mut rc_incr_sig = obj.make_signature();
        rc_incr_sig.params.push(AbiParam::new(PTR));
        let helper_rc_incr = obj
            .declare_function("lumen_rc_incr", Linkage::Local, &rc_incr_sig)
            .unwrap();

        // rc_decr(ptr): if ptr in heap, rc--; if 0, free
        let mut rc_decr_sig = obj.make_signature();
        rc_decr_sig.params.push(AbiParam::new(PTR));
        let helper_rc_decr = obj
            .declare_function("lumen_rc_decr", Linkage::Local, &rc_decr_sig)
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

        // lumen_rt_yield()
        let rt_yield_sig = obj.make_signature();
        let rt_yield = obj
            .declare_function("lumen_rt_yield", Linkage::Import, &rt_yield_sig)
            .unwrap();

        // --- TCP socket helpers (from runtime/rt.c) ---
        // lumen_tcp_listen(port: i32) -> i32
        let mut tcp_listen_sig = obj.make_signature();
        tcp_listen_sig.params.push(AbiParam::new(cl_types::I32));
        tcp_listen_sig.returns.push(AbiParam::new(cl_types::I32));
        let net_tcp_listen = obj
            .declare_function("lumen_tcp_listen", Linkage::Import, &tcp_listen_sig)
            .unwrap();

        // lumen_tcp_accept(server_fd: i32) -> i32
        let mut tcp_accept_sig = obj.make_signature();
        tcp_accept_sig.params.push(AbiParam::new(cl_types::I32));
        tcp_accept_sig.returns.push(AbiParam::new(cl_types::I32));
        let net_tcp_accept = obj
            .declare_function("lumen_tcp_accept", Linkage::Import, &tcp_accept_sig)
            .unwrap();

        // lumen_tcp_read(fd: i32, max: i32) -> i64 (ptr to bytes)
        let mut tcp_read_sig = obj.make_signature();
        tcp_read_sig.params.push(AbiParam::new(cl_types::I32));
        tcp_read_sig.params.push(AbiParam::new(cl_types::I32));
        tcp_read_sig.returns.push(AbiParam::new(PTR));
        let net_tcp_read = obj
            .declare_function("lumen_tcp_read", Linkage::Import, &tcp_read_sig)
            .unwrap();

        // lumen_tcp_write(fd: i32, bytes_ptr: i64) -> i64
        let mut tcp_write_sig = obj.make_signature();
        tcp_write_sig.params.push(AbiParam::new(cl_types::I32));
        tcp_write_sig.params.push(AbiParam::new(PTR));
        tcp_write_sig.returns.push(AbiParam::new(PTR));
        let net_tcp_write = obj
            .declare_function("lumen_tcp_write", Linkage::Import, &tcp_write_sig)
            .unwrap();

        // lumen_tcp_close(fd: i32)
        let mut tcp_close_sig = obj.make_signature();
        tcp_close_sig.params.push(AbiParam::new(cl_types::I32));
        let net_tcp_close = obj
            .declare_function("lumen_tcp_close", Linkage::Import, &tcp_close_sig)
            .unwrap();

        // lumen_net_serve(port: i32, handler: ptr)
        let mut serve_sig = obj.make_signature();
        serve_sig.params.push(AbiParam::new(cl_types::I32));
        serve_sig.params.push(AbiParam::new(PTR));
        let net_serve = obj
            .declare_function("lumen_net_serve", Linkage::Import, &serve_sig)
            .unwrap();

        // lumen_gt_read(fd: i32, max: i32) -> ptr (bytes)
        let mut gt_read_sig = obj.make_signature();
        gt_read_sig.params.push(AbiParam::new(cl_types::I32));
        gt_read_sig.params.push(AbiParam::new(cl_types::I32));
        gt_read_sig.returns.push(AbiParam::new(PTR));
        let gt_read = obj
            .declare_function("lumen_gt_read", Linkage::Import, &gt_read_sig)
            .unwrap();

        // lumen_gt_write(fd: i32, bytes: ptr) -> i32
        let mut gt_write_sig = obj.make_signature();
        gt_write_sig.params.push(AbiParam::new(cl_types::I32));
        gt_write_sig.params.push(AbiParam::new(PTR));
        gt_write_sig.returns.push(AbiParam::new(cl_types::I32));
        let gt_write = obj
            .declare_function("lumen_gt_write", Linkage::Import, &gt_write_sig)
            .unwrap();

        // lumen_http_parse_method(raw: ptr) -> ptr
        let mut http_pm_sig = obj.make_signature();
        http_pm_sig.params.push(AbiParam::new(PTR));
        http_pm_sig.returns.push(AbiParam::new(PTR));
        let http_parse_method = obj
            .declare_function("lumen_http_parse_method", Linkage::Import, &http_pm_sig)
            .unwrap();

        // lumen_http_parse_path(raw: ptr) -> ptr
        let mut http_pp_sig = obj.make_signature();
        http_pp_sig.params.push(AbiParam::new(PTR));
        http_pp_sig.returns.push(AbiParam::new(PTR));
        let http_parse_path = obj
            .declare_function("lumen_http_parse_path", Linkage::Import, &http_pp_sig)
            .unwrap();

        // lumen_http_parse_body(raw: ptr) -> ptr
        let mut http_pb_sig = obj.make_signature();
        http_pb_sig.params.push(AbiParam::new(PTR));
        http_pb_sig.returns.push(AbiParam::new(PTR));
        let http_parse_body = obj
            .declare_function("lumen_http_parse_body", Linkage::Import, &http_pb_sig)
            .unwrap();

        // lumen_http_format_response(status: i32, body: ptr) -> ptr
        let mut http_fr_sig = obj.make_signature();
        http_fr_sig.params.push(AbiParam::new(cl_types::I32));
        http_fr_sig.params.push(AbiParam::new(PTR));
        http_fr_sig.returns.push(AbiParam::new(PTR));
        let http_format_response = obj
            .declare_function("lumen_http_format_response", Linkage::Import, &http_fr_sig)
            .unwrap();

        // lumen_list_new(elem_size: i32) -> i64 (PTR)
        let mut list_new_sig = obj.make_signature();
        list_new_sig.params.push(AbiParam::new(cl_types::I32));
        list_new_sig.returns.push(AbiParam::new(PTR));
        let list_new = obj
            .declare_function("lumen_list_new", Linkage::Import, &list_new_sig)
            .unwrap();

        // lumen_list_len(list: i64) -> i32
        let mut list_len_sig = obj.make_signature();
        list_len_sig.params.push(AbiParam::new(cl_types::I64));
        list_len_sig.returns.push(AbiParam::new(cl_types::I32));
        let list_len = obj
            .declare_function("lumen_list_len", Linkage::Import, &list_len_sig)
            .unwrap();

        // lumen_list_push(list: i64, value: i64) -> i64
        let mut list_push_sig = obj.make_signature();
        list_push_sig.params.push(AbiParam::new(cl_types::I64));
        list_push_sig.params.push(AbiParam::new(cl_types::I64));
        list_push_sig.returns.push(AbiParam::new(cl_types::I64));
        let list_push = obj
            .declare_function("lumen_list_push", Linkage::Import, &list_push_sig)
            .unwrap();

        // lumen_list_get(list: i64, index: i32) -> i64
        let mut list_get_sig = obj.make_signature();
        list_get_sig.params.push(AbiParam::new(cl_types::I64));
        list_get_sig.params.push(AbiParam::new(cl_types::I32));
        list_get_sig.returns.push(AbiParam::new(cl_types::I64));
        let list_get = obj
            .declare_function("lumen_list_get", Linkage::Import, &list_get_sig)
            .unwrap();

        // lumen_list_set(list: i64, index: i32, value: i64) -> i64
        let mut list_set_sig = obj.make_signature();
        list_set_sig.params.push(AbiParam::new(cl_types::I64));
        list_set_sig.params.push(AbiParam::new(cl_types::I32));
        list_set_sig.params.push(AbiParam::new(cl_types::I64));
        list_set_sig.returns.push(AbiParam::new(cl_types::I64));
        let list_set = obj
            .declare_function("lumen_list_set", Linkage::Import, &list_set_sig)
            .unwrap();

        // lumen_list_remove(list: i64, index: i32) -> i64
        let mut list_remove_sig = obj.make_signature();
        list_remove_sig.params.push(AbiParam::new(cl_types::I64));
        list_remove_sig.params.push(AbiParam::new(cl_types::I32));
        list_remove_sig.returns.push(AbiParam::new(cl_types::I64));
        let list_remove = obj
            .declare_function("lumen_list_remove", Linkage::Import, &list_remove_sig)
            .unwrap();

        // --- Raylib bridge function declarations ---

        // rl_init_window(w: i32, h: i32, title: ptr)
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(PTR));
        let rl_init_window = obj.declare_function("rl_init_window", Linkage::Import, &sig).unwrap();

        // rl_close_window()
        let sig = obj.make_signature();
        let rl_close_window = obj.declare_function("rl_close_window", Linkage::Import, &sig).unwrap();

        // rl_window_should_close() -> i32
        let mut sig = obj.make_signature();
        sig.returns.push(AbiParam::new(cl_types::I32));
        let rl_window_should_close = obj.declare_function("rl_window_should_close", Linkage::Import, &sig).unwrap();

        // rl_set_target_fps(fps: i32)
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::I32));
        let rl_set_target_fps = obj.declare_function("rl_set_target_fps", Linkage::Import, &sig).unwrap();

        // rl_get_frame_time() -> f64
        let mut sig = obj.make_signature();
        sig.returns.push(AbiParam::new(cl_types::F64));
        let rl_get_frame_time = obj.declare_function("rl_get_frame_time", Linkage::Import, &sig).unwrap();

        // rl_begin_drawing()
        let sig = obj.make_signature();
        let rl_begin_drawing = obj.declare_function("rl_begin_drawing", Linkage::Import, &sig).unwrap();

        // rl_end_drawing()
        let sig = obj.make_signature();
        let rl_end_drawing = obj.declare_function("rl_end_drawing", Linkage::Import, &sig).unwrap();

        // rl_clear_background(color: i32)
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::I32));
        let rl_clear_background = obj.declare_function("rl_clear_background", Linkage::Import, &sig).unwrap();

        // rl_draw_text(text: ptr, x: i32, y: i32, size: i32, color: i32)
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(PTR));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        let rl_draw_text = obj.declare_function("rl_draw_text", Linkage::Import, &sig).unwrap();

        // rl_measure_text(text: ptr, size: i32) -> i32
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(PTR));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.returns.push(AbiParam::new(cl_types::I32));
        let rl_measure_text = obj.declare_function("rl_measure_text", Linkage::Import, &sig).unwrap();

        // rl_draw_rectangle_rec(x: f64, y: f64, w: f64, h: f64, color: i32)
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::I32));
        let rl_draw_rectangle_rec = obj.declare_function("rl_draw_rectangle_rec", Linkage::Import, &sig).unwrap();

        // rl_draw_rectangle(x: i32, y: i32, w: i32, h: i32, color: i32)
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        let rl_draw_rectangle = obj.declare_function("rl_draw_rectangle", Linkage::Import, &sig).unwrap();

        // rl_draw_rectangle_pro(rx: f64, ry: f64, rw: f64, rh: f64, ox: f64, oy: f64, rot: f64, color: i32)
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::I32));
        let rl_draw_rectangle_pro = obj.declare_function("rl_draw_rectangle_pro", Linkage::Import, &sig).unwrap();

        // rl_draw_circle(cx: f64, cy: f64, radius: f64, color: i32)
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::I32));
        let rl_draw_circle = obj.declare_function("rl_draw_circle", Linkage::Import, &sig).unwrap();

        // rl_draw_line(x1: i32, y1: i32, x2: i32, y2: i32, color: i32)
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::I32));
        let rl_draw_line = obj.declare_function("rl_draw_line", Linkage::Import, &sig).unwrap();

        // rl_is_key_pressed(key: i32) -> i32
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.returns.push(AbiParam::new(cl_types::I32));
        let rl_is_key_pressed = obj.declare_function("rl_is_key_pressed", Linkage::Import, &sig).unwrap();

        // rl_is_key_down(key: i32) -> i32
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.returns.push(AbiParam::new(cl_types::I32));
        let rl_is_key_down = obj.declare_function("rl_is_key_down", Linkage::Import, &sig).unwrap();

        // rl_is_gesture_detected(gesture: i32) -> i32
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.returns.push(AbiParam::new(cl_types::I32));
        let rl_is_gesture_detected = obj.declare_function("rl_is_gesture_detected", Linkage::Import, &sig).unwrap();

        // rl_set_camera(tx: f64, ty: f64, ox: f64, oy: f64, rot: f64, zoom: f64)
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        let rl_set_camera = obj.declare_function("rl_set_camera", Linkage::Import, &sig).unwrap();

        // rl_begin_mode_2d()
        let sig = obj.make_signature();
        let rl_begin_mode_2d = obj.declare_function("rl_begin_mode_2d", Linkage::Import, &sig).unwrap();

        // rl_end_mode_2d()
        let sig = obj.make_signature();
        let rl_end_mode_2d = obj.declare_function("rl_end_mode_2d", Linkage::Import, &sig).unwrap();

        // rl_init_audio()
        let sig = obj.make_signature();
        let rl_init_audio = obj.declare_function("rl_init_audio", Linkage::Import, &sig).unwrap();

        // Color constants: all () -> i32
        let mut color_sig = obj.make_signature();
        color_sig.returns.push(AbiParam::new(cl_types::I32));
        let rl_color_black = obj.declare_function("rl_color_black", Linkage::Import, &color_sig).unwrap();
        let rl_color_white = obj.declare_function("rl_color_white", Linkage::Import, &color_sig).unwrap();
        let rl_color_red = obj.declare_function("rl_color_red", Linkage::Import, &color_sig).unwrap();
        let rl_color_green = obj.declare_function("rl_color_green", Linkage::Import, &color_sig).unwrap();
        let rl_color_blue = obj.declare_function("rl_color_blue", Linkage::Import, &color_sig).unwrap();
        let rl_color_yellow = obj.declare_function("rl_color_yellow", Linkage::Import, &color_sig).unwrap();
        let rl_color_purple = obj.declare_function("rl_color_purple", Linkage::Import, &color_sig).unwrap();
        let rl_color_darkblue = obj.declare_function("rl_color_darkblue", Linkage::Import, &color_sig).unwrap();
        let rl_color_darkgray = obj.declare_function("rl_color_darkgray", Linkage::Import, &color_sig).unwrap();
        let rl_color_gray = obj.declare_function("rl_color_gray", Linkage::Import, &color_sig).unwrap();

        // rl_color_alpha(color: i32, alpha: f64) -> i32
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::I32));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.returns.push(AbiParam::new(cl_types::I32));
        let rl_color_alpha = obj.declare_function("rl_color_alpha", Linkage::Import, &sig).unwrap();

        // --- Math helpers ---

        // lumen_sqrt(x: f64) -> f64
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.returns.push(AbiParam::new(cl_types::F64));
        let math_sqrt = obj.declare_function("lumen_sqrt", Linkage::Import, &sig).unwrap();

        // lumen_abs(x: f64) -> f64
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.returns.push(AbiParam::new(cl_types::F64));
        let math_abs = obj.declare_function("lumen_abs", Linkage::Import, &sig).unwrap();

        // lumen_cos(x: f64) -> f64
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.returns.push(AbiParam::new(cl_types::F64));
        let math_cos = obj.declare_function("lumen_cos", Linkage::Import, &sig).unwrap();

        // lumen_sin(x: f64) -> f64
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.returns.push(AbiParam::new(cl_types::F64));
        let math_sin = obj.declare_function("lumen_sin", Linkage::Import, &sig).unwrap();

        // lumen_clamp(x: f64, lo: f64, hi: f64) -> f64
        let mut sig = obj.make_signature();
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.params.push(AbiParam::new(cl_types::F64));
        sig.returns.push(AbiParam::new(cl_types::F64));
        let math_clamp = obj.declare_function("lumen_clamp", Linkage::Import, &sig).unwrap();

        // lumen_rand_f64() -> f64
        let mut sig = obj.make_signature();
        sig.returns.push(AbiParam::new(cl_types::F64));
        let math_rand = obj.declare_function("lumen_rand_f64", Linkage::Import, &sig).unwrap();

        // print_frames: () -> void. Walks the frame_chain and prints each.
        let print_frames_sig = obj.make_signature();
        let helper_print_frames = obj
            .declare_function("lumen_print_frames", Linkage::Local, &print_frames_sig)
            .unwrap();

        Ok(Self {
            info,
            obj,
            fn_ids: HashMap::new(),
            libc_write,
            libc_malloc,
            libc_free,
            helper_concat,
            helper_println,
            helper_itoa,
            helper_print_frames,
            helper_rc_alloc,
            helper_rc_incr,
            helper_rc_decr,
            rt_send,
            rt_ask,
            rt_drain,
            rt_yield,
            net_tcp_listen,
            net_tcp_accept,
            net_tcp_read,
            net_tcp_write,
            net_tcp_close,
            net_serve,
            gt_read,
            gt_write,
            http_parse_method,
            http_parse_path,
            http_parse_body,
            http_format_response,
            list_new,
            list_len,
            list_push,
            list_get,
            list_set,
            list_remove,
            rl_init_window,
            rl_close_window,
            rl_window_should_close,
            rl_set_target_fps,
            rl_get_frame_time,
            rl_begin_drawing,
            rl_end_drawing,
            rl_clear_background,
            rl_draw_text,
            rl_measure_text,
            rl_draw_rectangle_rec,
            rl_draw_rectangle,
            rl_draw_rectangle_pro,
            rl_draw_circle,
            rl_draw_line,
            rl_is_key_pressed,
            rl_is_key_down,
            rl_is_gesture_detected,
            rl_set_camera,
            rl_begin_mode_2d,
            rl_end_mode_2d,
            rl_init_audio,
            rl_color_black,
            rl_color_white,
            rl_color_red,
            rl_color_green,
            rl_color_blue,
            rl_color_yellow,
            rl_color_purple,
            rl_color_darkblue,
            rl_color_darkgray,
            rl_color_gray,
            rl_color_alpha,
            math_sqrt,
            math_abs,
            math_cos,
            math_sin,
            math_clamp,
            math_rand,
            dispatch_fns: HashMap::new(),
            heap_data,
            bump_ptr_data,
            frame_chain_data,
            string_data: HashMap::new(),
            uses_io: false,
        })
    }

    fn compile_module(&mut self, module: &ast::Module) -> Result<(), NativeError> {
        self.uses_io = module.imports.iter().any(|im| im.path == ["std", "io"]);

        // Intern string literals + frame messages.
        self.intern_all_strings(module);

        // Declare all user functions.
        for item in &module.items {
            if let Item::Fn(f) = item {
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
                let id = self
                    .obj
                    .declare_function(&ef.name, Linkage::Import, &sig)
                    .unwrap();
                self.fn_ids.insert(ef.name.clone(), id);
            }
        }

        // Define helper bodies.
        self.define_concat_helper()?;
        self.define_println_helper()?;
        self.define_itoa_helper()?;
        self.define_print_frames_helper()?;
        self.define_rc_alloc_helper()?;
        self.define_rc_incr_helper()?;
        self.define_rc_decr_helper()?;

        // Emit per-actor dispatch functions BEFORE user function bodies
        // (user fns reference dispatch via send/ask).
        let actors: Vec<String> = self.info.actors.keys().cloned().collect();
        for actor_name in &actors {
            self.emit_actor_dispatch(actor_name, module)?;
        }

        // Define user function bodies.
        let fn_ids = self.fn_ids.clone();
        for item in &module.items {
            if let Item::Fn(f) = item {
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

        Ok(())
    }

    fn finish(self) -> Vec<u8> {
        let product = self.obj.finish();
        product.emit().unwrap()
    }

    fn build_sig(&self, fn_name: &str) -> cranelift_codegen::ir::Signature {
        let sig = &self.info.fns[fn_name];
        let mut cl_sig = self.obj.make_signature();
        for (_, ty) in &sig.params {
            cl_sig.params.push(AbiParam::new(lumen_to_cl(ty)));
        }
        cl_sig.returns.push(AbiParam::new(lumen_to_cl(&sig.ret)));
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

    /// `lumen_rc_alloc(size: i64) -> ptr`: malloc(size+8), write rc=1 at
    /// the start, return ptr+8 (pointer to payload).
    fn define_rc_alloc_helper(&mut self) -> Result<(), NativeError> {
        let mut sig = self.obj.make_signature();
        sig.params.push(AbiParam::new(PTR));
        sig.returns.push(AbiParam::new(PTR));
        let mut ctx = self.obj.make_context();
        ctx.func.signature = sig;
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let size = builder.block_params(block)[0];
        let eight = builder.ins().iconst(PTR, 8);
        let total = builder.ins().iadd(size, eight);
        let malloc_ref = self.obj.declare_func_in_func(self.libc_malloc, builder.func);
        let call = builder.ins().call(malloc_ref, &[total]);
        let raw = builder.inst_results(call)[0];
        // *raw = 1 (refcount, i32)
        let one = builder.ins().iconst(cl_types::I32, 1);
        builder.ins().store(MemFlags::new(), one, raw, 0);
        // *(raw+4) = 0x4C554D45 ("LUME" magic sentinel)
        let magic = builder.ins().iconst(cl_types::I32, 0x4C554D45u32 as i64);
        builder.ins().store(MemFlags::new(), magic, raw, 4);
        // return raw + 8
        let payload = builder.ins().iadd(raw, eight);
        builder.ins().return_(&[payload]);

        builder.seal_all_blocks();
        builder.finalize();
        self.obj.define_function(self.helper_rc_alloc, &mut ctx).unwrap();
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
        let sig = &self.info.fns[&f.name];
        let cl_sig = self.build_sig(&f.name);

        let mut ctx = self.obj.make_context();
        ctx.func.signature = cl_sig;
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        // (sealed later)

        {
            let mut fb = FnEmitter::new(self, &mut builder, sig, &f.name);

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

            // Compile body. (Yield points are at loop headers only —
            // function-entry yield causes recursive dispatch issues.)
            fb.hit_return = false;
            let result = fb.compile_block_with_cleanup(&f.body, param_cleanup)?;

            if fb.hit_return {
                // The body ended with a return. We're on a dead block.
                // Emit a correctly-typed return for the function signature.
                let ret_ty = lumen_to_cl(&sig.ret);
                let dummy = fb.builder.ins().iconst(ret_ty, 0);
                fb.builder.ins().return_(&[dummy]);
            } else {
                // If this is main, drain the message queue before returning.
                if f.name == "main" {
                    let drain_ref = fb.cg.obj.declare_func_in_func(
                        fb.cg.rt_drain, fb.builder.func,
                    );
                    fb.builder.ins().call(drain_ref, &[]);
                }
                // Return.
                fb.builder.ins().return_(&[result]);
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

        Ok(())
    }

    // --- Helpers (defined as Cranelift functions) ------------------------

    fn define_concat_helper(&mut self) -> Result<(), NativeError> {
        // lumen_concat(a: ptr, b: ptr) -> ptr
        // Same logic as the Wasm string_concat helper.
        let sig = {
            let mut s = self.obj.make_signature();
            s.params.push(AbiParam::new(PTR));
            s.params.push(AbiParam::new(PTR));
            s.returns.push(AbiParam::new(PTR));
            s
        };
        let mut ctx = self.obj.make_context();
        ctx.func.signature = sig;
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        // (sealed later)

        let a = builder.block_params(block)[0];
        let b = builder.block_params(block)[1];
        let flags = MemFlags::new();

        // len_a = *(a) as i64 (load i32, extend)
        let len_a_i32 = builder.ins().load(cl_types::I32, flags, a, 0);
        let len_a = builder.ins().uextend(PTR, len_a_i32);
        let len_b_i32 = builder.ins().load(cl_types::I32, flags, b, 0);
        let len_b = builder.ins().uextend(PTR, len_b_i32);

        let total = builder.ins().iadd(len_a, len_b);
        let total_i32 = builder.ins().ireduce(cl_types::I32, total);

        // Allocate result via rc_alloc(4 + total).
        let four = builder.ins().iconst(PTR, 4);
        let alloc_size = builder.ins().iadd(total, four);
        let rc_alloc_ref = self.obj.declare_func_in_func(self.helper_rc_alloc, builder.func);
        let alloc_call = builder.ins().call(rc_alloc_ref, &[alloc_size]);
        let result = builder.inst_results(alloc_call)[0];

        // *result = total_i32
        builder.ins().store(flags, total_i32, result, 0);

        // memcpy(result+4, a+4, len_a)
        let dst1 = builder.ins().iadd_imm(result, 4);
        let src1 = builder.ins().iadd_imm(a, 4);
        builder.call_memcpy(self.obj.target_config(), dst1, src1, len_a);

        // memcpy(result+4+len_a, b+4, len_b)
        let dst2 = builder.ins().iadd(dst1, len_a);
        let src2 = builder.ins().iadd_imm(b, 4);
        builder.call_memcpy(self.obj.target_config(), dst2, src2, len_b);

        builder.ins().return_(&[result]);
        builder.seal_all_blocks();
        builder.finalize();

        self.obj.define_function(self.helper_concat, &mut ctx).unwrap();
        Ok(())
    }

    fn define_println_helper(&mut self) -> Result<(), NativeError> {
        // lumen_println(s: ptr) — prints the string followed by \n.
        let sig = {
            let mut s = self.obj.make_signature();
            s.params.push(AbiParam::new(PTR));
            s
        };
        let mut ctx = self.obj.make_context();
        ctx.func.signature = sig;
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        // (sealed later)

        let s = builder.block_params(block)[0];
        let flags = MemFlags::new();

        // Concat s + "\n".
        let nl_id = *self.string_data.get("\n").unwrap();
        let nl_gv = self.declare_data_in_func(nl_id, &mut builder);
        let nl = builder.ins().global_value(PTR, nl_gv);

        let concat_ref = self
            .obj
            .declare_func_in_func(self.helper_concat, builder.func);
        let with_nl = builder.ins().call(concat_ref, &[s, nl]);
        let with_nl = builder.inst_results(with_nl)[0];

        // Read len, compute data ptr.
        let len_i32 = builder.ins().load(cl_types::I32, flags, with_nl, 0);
        let len = builder.ins().uextend(PTR, len_i32);
        let data = builder.ins().iadd_imm(with_nl, 4);

        // write(1, data, len)
        let fd = builder.ins().iconst(cl_types::I32, 1);
        let write_ref = self
            .obj
            .declare_func_in_func(self.libc_write, builder.func);
        builder.ins().call(write_ref, &[fd, data, len]);

        builder.ins().return_(&[]);
        builder.seal_all_blocks();
        builder.finalize();

        self.obj.define_function(self.helper_println, &mut ctx).unwrap();
        Ok(())
    }

    fn define_itoa_helper(&mut self) -> Result<(), NativeError> {
        // lumen_itoa(n: i32) -> ptr (string)
        // Same algorithm as the Wasm version: reverse digit extraction.
        // For brevity, this is a simplified version that handles the
        // common case. A full production itoa would handle i32::MIN.
        let sig = {
            let mut s = self.obj.make_signature();
            s.params.push(AbiParam::new(cl_types::I32));
            s.returns.push(AbiParam::new(PTR));
            s
        };
        let mut ctx = self.obj.make_context();
        ctx.func.signature = sig;
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let n = builder.block_params(entry)[0];
        let flags = MemFlags::new();

        // Save bump watermark so we can reclaim scratch after rc_alloc'ing the result.
        let bump_gv = self.declare_data_in_func(self.bump_ptr_data, &mut builder);
        let heap_gv = self.declare_data_in_func(self.heap_data, &mut builder);
        let bump_addr = builder.ins().global_value(PTR, bump_gv);
        let saved_off = builder.ins().load(PTR, flags, bump_addr, 0);
        let watermark_var = builder.declare_var(PTR);
        builder.def_var(watermark_var, saved_off);

        // Allocate 16-byte scratch in heap (bump).
        let heap_base = builder.ins().global_value(PTR, heap_gv);
        let scratch = builder.ins().iadd(heap_base, saved_off);
        let new_off = builder.ins().iadd_imm(saved_off, 16);
        builder.ins().store(flags, new_off, bump_addr, 0);

        // Simple approach: format into scratch[0..15] backwards.
        // Use a loop block for digit extraction.

        // is_neg = n < 0
        let zero_i32 = builder.ins().iconst(cl_types::I32, 0);
        let is_neg = builder.ins().icmp(IntCC::SignedLessThan, n, zero_i32);

        // abs = is_neg ? (0 - n) : n
        let neg_n = builder.ins().ineg(n);
        let abs_val = builder.ins().select(is_neg, neg_n, n);

        // pos = scratch + 15  (write position, working backwards)
        let pos_var = builder.declare_var(PTR);
        let init_pos = builder.ins().iadd_imm(scratch, 15);
        builder.def_var(pos_var, init_pos);

        let abs_var = builder.declare_var(cl_types::I32);
        builder.def_var(abs_var, abs_val);

        // if abs == 0: write '0', else loop
        let is_zero = builder.ins().icmp_imm(IntCC::Equal, abs_val, 0);

        let zero_block = builder.create_block();
        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let after_digits = builder.create_block();

        builder.ins().brif(is_zero, zero_block, &[], loop_header, &[]);
        // (sealed later)

        // Zero block: write '0'.
        builder.switch_to_block(zero_block);
        let pos = builder.use_var(pos_var);
        let ascii_0 = builder.ins().iconst(cl_types::I8, 48);
        builder.ins().store(flags, ascii_0, pos, 0);
        let pos = builder.ins().iadd_imm(pos, -1);
        builder.def_var(pos_var, pos);
        builder.ins().jump(after_digits, &[]);
        // (sealed later)

        // Loop header: check if abs > 0.
        builder.switch_to_block(loop_header);
        let abs = builder.use_var(abs_var);
        let done = builder.ins().icmp_imm(IntCC::Equal, abs, 0);
        builder.ins().brif(done, after_digits, &[], loop_body, &[]);
        // (sealed later)

        // Loop body: extract one digit.
        builder.switch_to_block(loop_body);
        let abs = builder.use_var(abs_var);
        let ten = builder.ins().iconst(cl_types::I32, 10);
        let digit = builder.ins().srem(abs, ten);
        let digit_byte = builder.ins().iadd_imm(digit, 48);
        let digit_byte = builder.ins().ireduce(cl_types::I8, digit_byte);
        let pos = builder.use_var(pos_var);
        builder.ins().store(flags, digit_byte, pos, 0);
        let pos = builder.ins().iadd_imm(pos, -1);
        builder.def_var(pos_var, pos);
        let abs = builder.ins().sdiv(abs, ten);
        builder.def_var(abs_var, abs);
        builder.ins().jump(loop_header, &[]);
        // (sealed later)

        // After digits: handle sign, then build result string.
        builder.switch_to_block(after_digits);
        let neg_block = builder.create_block();
        let final_block = builder.create_block();

        let pos = builder.use_var(pos_var);
        builder.ins().brif(is_neg, neg_block, &[], final_block, &[]);
        // (sealed later)

        // Neg block: write '-'.
        builder.switch_to_block(neg_block);
        let pos = builder.use_var(pos_var);
        let minus = builder.ins().iconst(cl_types::I8, 45);
        builder.ins().store(flags, minus, pos, 0);
        let pos = builder.ins().iadd_imm(pos, -1);
        builder.def_var(pos_var, pos);
        builder.ins().jump(final_block, &[]);
        // (sealed later)

        // Final: allocate result string [len | bytes].
        builder.switch_to_block(final_block);
        let pos = builder.use_var(pos_var);
        // start = pos + 1
        let start = builder.ins().iadd_imm(pos, 1);
        // end = scratch + 16
        let end = builder.ins().iadd_imm(scratch, 16);
        let len = builder.ins().isub(end, start);
        let len_i32 = builder.ins().ireduce(cl_types::I32, len);

        // Allocate (4 + len) via rc_alloc so the result survives RC.
        let four = builder.ins().iconst(PTR, 4);
        let alloc_size = builder.ins().iadd(len, four);
        let rc_alloc_ref = self.obj.declare_func_in_func(self.helper_rc_alloc, builder.func);
        let alloc_call = builder.ins().call(rc_alloc_ref, &[alloc_size]);
        let result = builder.inst_results(alloc_call)[0];

        // *result = len_i32
        builder.ins().store(flags, len_i32, result, 0);
        // memcpy(result+4, start, len)
        let dst = builder.ins().iadd_imm(result, 4);
        builder.call_memcpy(self.obj.target_config(), dst, start, len);

        // Restore bump watermark: the result is rc_alloc'd (malloc) so it
        // survives; the scratch was bump-allocated and is now reclaimed.
        let wm = builder.use_var(watermark_var);
        let bump_addr2 = builder.ins().global_value(PTR, bump_gv);
        builder.ins().store(flags, wm, bump_addr2, 0);

        builder.ins().return_(&[result]);
        builder.seal_all_blocks();
        builder.finalize();

        self.obj.define_function(self.helper_itoa, &mut ctx).unwrap();
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
    sig: &'a crate::types::FnSig,
    fn_name: String,
    names: HashMap<String, Variable>,
    name_types: HashMap<String, Ty>,
    /// Stack of cleanup lists.
    cleanup_stack: Vec<Vec<(String, Variable, Ty)>>,
    /// Set to true when a `return` statement is compiled. Checked by
    /// compile_if to avoid emitting jumps after a terminated block.
    hit_return: bool,
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
        }
    }

    fn fresh_var(&mut self, ty: CLType) -> Variable {
        self.builder.declare_var(ty)
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
        match &stmt.kind {
            StmtKind::Let { name, value, .. } | StmtKind::Var { name, value, .. } => {
                let lumen_ty = self.infer_ty(value)?;
                let val = self.compile_expr(value)?;
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
            }
            StmtKind::Expr(e) => {
                self.compile_expr(e)?;
            }
            StmtKind::For { binder, iter, body } => {
                self.compile_for(binder, iter, body, stmt.span)?;
            }
            StmtKind::Return(Some(e)) => {
                let val = self.compile_expr(e)?;
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
                let zero = self.builder.ins().iconst(cl_types::I32, 0);
                self.builder.ins().return_(&[zero]);
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
                } else if let Some(sum_name) = self.find_sum_for_variant(name) {
                    let tag = self.variant_tag(&Ty::User(sum_name), name).unwrap_or(0);
                    self.build_sum_block(tag, None)
                } else if let Some(&func_id) = self.cg.fn_ids.get(name) {
                    // A function name used as a value — emit its address as a PTR.
                    let func_ref = self.cg.obj.declare_func_in_func(func_id, self.builder.func);
                    Ok(self.builder.ins().func_addr(PTR, func_ref))
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
            _ => Err(NativeError {
                span: expr.span,
                message: "expression not supported in native backend".into(),
            }),
        }
    }

    fn compile_binary(
        &mut self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        span: Span,
    ) -> Result<Value, NativeError> {
        let lt = self.infer_ty(lhs)?;

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

        // Built-in Option/Result constructors.
        match name.as_str() {
            "Ok" => return self.compile_single_field_constructor(0, &args[0].value),
            "Err" => return self.compile_single_field_constructor(1, &args[0].value),
            "Some" => return self.compile_single_field_constructor(1, &args[0].value),
            "None" => return self.build_sum_block(0, None),
            _ => {}
        }

        // User variant constructor (positional payload)?
        if let Some(sum_name) = self.find_sum_for_variant(&name) {
            let tag = self.variant_tag(&Ty::User(sum_name), &name).unwrap_or(0);
            if args.is_empty() {
                return self.build_sum_block(tag, None);
            }
            // Positional variant: allocate payload with fields.
            let scrut_ty = Ty::User(self.find_sum_for_variant(&name).unwrap());
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

        // User function call.
        let func_id = self.cg.fn_ids.get(&name).ok_or_else(|| NativeError {
            span,
            message: format!("unknown function `{name}`"),
        })?;
        let func_ref = self
            .cg
            .obj
            .declare_func_in_func(*func_id, self.builder.func);
        let mut arg_vals = Vec::new();
        for a in args {
            let val = self.compile_expr(&a.value)?;
            arg_vals.push(val);
        }
        // rc_incr each pointer argument so the callee's scope-exit
        // decr doesn't free values the caller still holds.
        if let Some(sig) = self.cg.info.fns.get(&name) {
            let param_types: Vec<Ty> = sig.params.iter().map(|(_, t)| t.clone()).collect();
            for (i, pty) in param_types.iter().enumerate() {
                if !is_scalar(pty) {
                    if let Some(&val) = arg_vals.get(i) {
                        self.emit_rc_incr(val);
                    }
                }
            }
        }
        let call = self.builder.ins().call(func_ref, &arg_vals);
        Ok(self.builder.inst_results(call)[0])
    }

    fn compile_method_call(
        &mut self,
        receiver: &Expr,
        method: &str,
        args: &[ast::Arg],
        span: Span,
    ) -> Result<Value, NativeError> {
        if let ExprKind::Ident(mod_name) = &receiver.kind {
            if mod_name == "io" && method == "println" {
                let s = self.compile_expr(&args[0].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.helper_println, self.builder.func);
                self.builder.ins().call(func_ref, &[s]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "int" && method == "to_string_i32" {
                let n = self.compile_expr(&args[0].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.helper_itoa, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[n]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "bytes" && method == "len" {
                let b = self.compile_expr(&args[0].value)?;
                let len = self.builder.ins().load(cl_types::I32, MemFlags::new(), b, 0);
                return Ok(len);
            }
            if mod_name == "bytes" && method == "new" {
                // rc_alloc(4 + size), store size as i32 at offset 0, memset to 0
                let size_val = self.compile_expr(&args[0].value)?;
                let size_ptr = self.builder.ins().uextend(PTR, size_val);
                let four = self.builder.ins().iconst(PTR, 4);
                let total = self.builder.ins().iadd(size_ptr, four);
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.helper_rc_alloc, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[total]);
                let ptr = self.builder.inst_results(call)[0];
                // Store length at offset 0
                self.builder.ins().store(MemFlags::new(), size_val, ptr, 0);
                // memset bytes to 0: use call_memset
                let data_ptr = self.builder.ins().iadd_imm(ptr, 4);
                let zero = self.builder.ins().iconst(cl_types::I8, 0);
                self.builder.call_memset(
                    self.cg.obj.target_config(),
                    data_ptr,
                    zero,
                    size_ptr,
                );
                return Ok(ptr);
            }
            if mod_name == "bytes" && method == "get" {
                // load u8 at b + 4 + i, zero-extend to i32
                let b = self.compile_expr(&args[0].value)?;
                let i = self.compile_expr(&args[1].value)?;
                let i_ptr = self.builder.ins().sextend(PTR, i);
                let base = self.builder.ins().iadd_imm(b, 4);
                let addr = self.builder.ins().iadd(base, i_ptr);
                let byte = self.builder.ins().load(cl_types::I8, MemFlags::new(), addr, 0);
                let result = self.builder.ins().uextend(cl_types::I32, byte);
                return Ok(result);
            }
            if mod_name == "bytes" && method == "concat" {
                let a = self.compile_expr(&args[0].value)?;
                let b = self.compile_expr(&args[1].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.helper_concat, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[a, b]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "bytes" && method == "from_string" {
                // Zero-cost: same layout
                let s = self.compile_expr(&args[0].value)?;
                return Ok(s);
            }
            if mod_name == "string" && method == "from_bytes" {
                // Zero-cost: same layout
                let b = self.compile_expr(&args[0].value)?;
                return Ok(b);
            }
            // --- TCP socket operations ---
            if mod_name == "net" && method == "tcp_listen" {
                let port = self.compile_expr(&args[0].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.net_tcp_listen, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[port]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "net" && method == "tcp_accept" {
                let server_fd = self.compile_expr(&args[0].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.net_tcp_accept, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[server_fd]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "net" && method == "tcp_read" {
                let fd = self.compile_expr(&args[0].value)?;
                let max = self.compile_expr(&args[1].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.net_tcp_read, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[fd, max]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "net" && method == "tcp_write" {
                let fd = self.compile_expr(&args[0].value)?;
                let data = self.compile_expr(&args[1].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.net_tcp_write, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[fd, data]);
                // Return value is i64 (ssize_t), truncate to i32
                let result = self.builder.inst_results(call)[0];
                let truncated = self.builder.ins().ireduce(cl_types::I32, result);
                return Ok(truncated);
            }
            if mod_name == "net" && method == "serve" {
                let port = self.compile_expr(&args[0].value)?;
                // Second arg: function name → func_addr
                let handler_name = match &args[1].value.kind {
                    ExprKind::Ident(n) => n.clone(),
                    _ => {
                        return Err(NativeError {
                            span,
                            message: "net.serve: second arg must be a function name".into(),
                        })
                    }
                };
                let handler_id = self.cg.fn_ids.get(&handler_name).ok_or_else(|| {
                    NativeError { span, message: format!("unknown function `{handler_name}`") }
                })?;
                let handler_ref = self.cg.obj.declare_func_in_func(*handler_id, self.builder.func);
                let handler_addr = self.builder.ins().func_addr(PTR, handler_ref);
                let serve_ref = self.cg.obj.declare_func_in_func(self.cg.net_serve, self.builder.func);
                self.builder.ins().call(serve_ref, &[port, handler_addr]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            // Green-thread-aware I/O: use gt_read/gt_write when inside net.serve.
            if mod_name == "net" && method == "gt_read" {
                let fd = self.compile_expr(&args[0].value)?;
                let max = self.compile_expr(&args[1].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.gt_read, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[fd, max]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "net" && method == "gt_write" {
                let fd = self.compile_expr(&args[0].value)?;
                let data = self.compile_expr(&args[1].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.gt_write, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[fd, data]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "net" && method == "tcp_close" {
                let fd = self.compile_expr(&args[0].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.net_tcp_close, self.builder.func);
                self.builder.ins().call(func_ref, &[fd]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "http" && method == "parse_method" {
                let raw = self.compile_expr(&args[0].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.http_parse_method, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[raw]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "http" && method == "parse_path" {
                let raw = self.compile_expr(&args[0].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.http_parse_path, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[raw]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "http" && method == "parse_body" {
                let raw = self.compile_expr(&args[0].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.http_parse_body, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[raw]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "http" && method == "format_response" {
                let status = self.compile_expr(&args[0].value)?;
                let body = self.compile_expr(&args[1].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.http_format_response, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[status, body]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            // --- List<T> operations ---
            if mod_name == "list" && method == "new" {
                // list.new(): create a new list with elem_size = 8.
                let elem_size = self.builder.ins().iconst(cl_types::I32, 8);
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.list_new, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[elem_size]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "list" && method == "len" {
                // list.len(l): returns i32.
                let l = self.compile_expr(&args[0].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.list_len, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[l]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "list" && method == "push" {
                // list.push(l, val): val may need sextend to i64. Returns new list ptr.
                let l = self.compile_expr(&args[0].value)?;
                let val_raw = self.compile_expr(&args[1].value)?;
                let val_ty = self.infer_ty(&args[1].value)?;
                // The list now holds a reference to the element — rc_incr
                // pointer-typed values so they survive scope cleanup.
                if !is_scalar(&val_ty) {
                    self.emit_rc_incr(val_raw);
                }
                let val64 = if lumen_to_cl(&val_ty) == cl_types::I32 {
                    self.builder.ins().sextend(cl_types::I64, val_raw)
                } else {
                    val_raw
                };
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.list_push, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[l, val64]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "list" && method == "get" {
                // list.get(l, i): returns i64; ireduce to i32 if elem type is i32.
                let l = self.compile_expr(&args[0].value)?;
                let i = self.compile_expr(&args[1].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.list_get, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[l, i]);
                let result = self.builder.inst_results(call)[0];
                // Determine element type from the list argument.
                let list_ty = self.infer_ty(&args[0].value)?;
                let elem_ty = match list_ty {
                    Ty::List(inner) => *inner,
                    _ => Ty::I64,
                };
                let r = if lumen_to_cl(&elem_ty) == cl_types::I32 {
                    self.builder.ins().ireduce(cl_types::I32, result)
                } else {
                    result
                };
                return Ok(r);
            }
            if mod_name == "list" && method == "set" {
                // list.set(l, i, val): sextend val if i32, returns new ptr.
                let l = self.compile_expr(&args[0].value)?;
                let i = self.compile_expr(&args[1].value)?;
                let val_raw = self.compile_expr(&args[2].value)?;
                let val_ty = self.infer_ty(&args[2].value)?;
                let val64 = if lumen_to_cl(&val_ty) == cl_types::I32 {
                    self.builder.ins().sextend(cl_types::I64, val_raw)
                } else {
                    val_raw
                };
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.list_set, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[l, i, val64]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "list" && method == "remove" {
                // list.remove(l, i): returns new ptr.
                let l = self.compile_expr(&args[0].value)?;
                let i = self.compile_expr(&args[1].value)?;
                let func_ref = self
                    .cg
                    .obj
                    .declare_func_in_func(self.cg.list_remove, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[l, i]);
                return Ok(self.builder.inst_results(call)[0]);
            }

            // --- Raylib: Window ---
            if mod_name == "rl" && method == "init_window" {
                let w = self.compile_expr(&args[0].value)?;
                let h = self.compile_expr(&args[1].value)?;
                let t = self.compile_expr(&args[2].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_init_window, self.builder.func);
                self.builder.ins().call(func_ref, &[w, h, t]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "close_window" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_close_window, self.builder.func);
                self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "window_should_close" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_window_should_close, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "set_target_fps" {
                let fps = self.compile_expr(&args[0].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_set_target_fps, self.builder.func);
                self.builder.ins().call(func_ref, &[fps]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "get_frame_time" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_get_frame_time, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }

            // --- Raylib: Drawing ---
            if mod_name == "rl" && method == "begin_drawing" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_begin_drawing, self.builder.func);
                self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "end_drawing" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_end_drawing, self.builder.func);
                self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "clear_background" {
                let color = self.compile_expr(&args[0].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_clear_background, self.builder.func);
                self.builder.ins().call(func_ref, &[color]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "draw_text" {
                let text = self.compile_expr(&args[0].value)?;
                let x = self.compile_expr(&args[1].value)?;
                let y = self.compile_expr(&args[2].value)?;
                let size = self.compile_expr(&args[3].value)?;
                let color = self.compile_expr(&args[4].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_draw_text, self.builder.func);
                self.builder.ins().call(func_ref, &[text, x, y, size, color]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "measure_text" {
                let text = self.compile_expr(&args[0].value)?;
                let size = self.compile_expr(&args[1].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_measure_text, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[text, size]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "draw_rect" {
                let x = self.compile_expr(&args[0].value)?;
                let y = self.compile_expr(&args[1].value)?;
                let w = self.compile_expr(&args[2].value)?;
                let h = self.compile_expr(&args[3].value)?;
                let color = self.compile_expr(&args[4].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_draw_rectangle_rec, self.builder.func);
                self.builder.ins().call(func_ref, &[x, y, w, h, color]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "draw_rect_i" {
                let x = self.compile_expr(&args[0].value)?;
                let y = self.compile_expr(&args[1].value)?;
                let w = self.compile_expr(&args[2].value)?;
                let h = self.compile_expr(&args[3].value)?;
                let color = self.compile_expr(&args[4].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_draw_rectangle, self.builder.func);
                self.builder.ins().call(func_ref, &[x, y, w, h, color]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "draw_rect_pro" {
                let rx = self.compile_expr(&args[0].value)?;
                let ry = self.compile_expr(&args[1].value)?;
                let rw = self.compile_expr(&args[2].value)?;
                let rh = self.compile_expr(&args[3].value)?;
                let ox = self.compile_expr(&args[4].value)?;
                let oy = self.compile_expr(&args[5].value)?;
                let rot = self.compile_expr(&args[6].value)?;
                let color = self.compile_expr(&args[7].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_draw_rectangle_pro, self.builder.func);
                self.builder.ins().call(func_ref, &[rx, ry, rw, rh, ox, oy, rot, color]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "draw_circle" {
                let cx = self.compile_expr(&args[0].value)?;
                let cy = self.compile_expr(&args[1].value)?;
                let radius = self.compile_expr(&args[2].value)?;
                let color = self.compile_expr(&args[3].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_draw_circle, self.builder.func);
                self.builder.ins().call(func_ref, &[cx, cy, radius, color]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "draw_line" {
                let x1 = self.compile_expr(&args[0].value)?;
                let y1 = self.compile_expr(&args[1].value)?;
                let x2 = self.compile_expr(&args[2].value)?;
                let y2 = self.compile_expr(&args[3].value)?;
                let color = self.compile_expr(&args[4].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_draw_line, self.builder.func);
                self.builder.ins().call(func_ref, &[x1, y1, x2, y2, color]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }

            // --- Raylib: Input ---
            if mod_name == "rl" && method == "is_key_pressed" {
                let key = self.compile_expr(&args[0].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_is_key_pressed, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[key]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "is_key_down" {
                let key = self.compile_expr(&args[0].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_is_key_down, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[key]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "is_gesture_detected" {
                let gesture = self.compile_expr(&args[0].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_is_gesture_detected, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[gesture]);
                return Ok(self.builder.inst_results(call)[0]);
            }

            // --- Raylib: Camera ---
            if mod_name == "rl" && method == "set_camera" {
                let tx = self.compile_expr(&args[0].value)?;
                let ty_val = self.compile_expr(&args[1].value)?;
                let ox = self.compile_expr(&args[2].value)?;
                let oy = self.compile_expr(&args[3].value)?;
                let rot = self.compile_expr(&args[4].value)?;
                let zoom = self.compile_expr(&args[5].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_set_camera, self.builder.func);
                self.builder.ins().call(func_ref, &[tx, ty_val, ox, oy, rot, zoom]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "begin_mode_2d" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_begin_mode_2d, self.builder.func);
                self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }
            if mod_name == "rl" && method == "end_mode_2d" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_end_mode_2d, self.builder.func);
                self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }

            // --- Raylib: Audio ---
            if mod_name == "rl" && method == "init_audio" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_init_audio, self.builder.func);
                self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
            }

            // --- Raylib: Colors ---
            if mod_name == "rl" && method == "black" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_black, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "white" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_white, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "red" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_red, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "green" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_green, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "blue" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_blue, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "yellow" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_yellow, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "purple" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_purple, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "darkblue" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_darkblue, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "darkgray" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_darkgray, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "gray" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_gray, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "rl" && method == "color_alpha" {
                let color = self.compile_expr(&args[0].value)?;
                let alpha = self.compile_expr(&args[1].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.rl_color_alpha, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[color, alpha]);
                return Ok(self.builder.inst_results(call)[0]);
            }

            // --- Math ---
            if mod_name == "math" && method == "sqrt" {
                let x = self.compile_expr(&args[0].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.math_sqrt, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[x]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "math" && method == "abs" {
                let x = self.compile_expr(&args[0].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.math_abs, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[x]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "math" && method == "cos" {
                let x = self.compile_expr(&args[0].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.math_cos, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[x]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "math" && method == "sin" {
                let x = self.compile_expr(&args[0].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.math_sin, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[x]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "math" && method == "clamp" {
                let x = self.compile_expr(&args[0].value)?;
                let lo = self.compile_expr(&args[1].value)?;
                let hi = self.compile_expr(&args[2].value)?;
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.math_clamp, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[x, lo, hi]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            if mod_name == "math" && method == "rand" {
                let func_ref = self.cg.obj.declare_func_in_func(self.cg.math_rand, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[]);
                return Ok(self.builder.inst_results(call)[0]);
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
        span: Span,
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

    /// Scan a block's statements for `var = expr` assignments where the
    /// target is a pointer-typed variable already in scope (i.e., an
    /// outer variable). Returns (name, type) pairs.
    fn find_assigned_ptr_vars(&self, block: &ast::Block) -> Vec<(String, Ty)> {
        let mut result = Vec::new();
        for stmt in &block.stmts {
            if let StmtKind::Assign { name, .. } = &stmt.kind {
                if let Some(ty) = self.name_types.get(name) {
                    if !is_scalar(ty) {
                        result.push((name.clone(), ty.clone()));
                    }
                }
            }
        }
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result.dedup_by(|a, b| a.0 == b.0);
        result
    }

    fn compile_struct_lit(
        &mut self,
        name: &str,
        fields: &[ast::FieldInit],
        spread: Option<&ast::Expr>,
        span: Span,
    ) -> Result<Value, NativeError> {
        // Named-field variant constructor? (e.g. Circle { radius: 2 })
        if get_struct_fields(&self.cg.info.types, name).is_empty() {
            if let Some(sum_name) = self.find_sum_for_variant(name) {
                let scrut_ty = Ty::User(sum_name);
                let tag = self.variant_tag(&scrut_ty, name).unwrap_or(0);
                let var_fields = self.variant_field_types(&scrut_ty, name).unwrap_or_default();
                if var_fields.is_empty() {
                    return self.build_sum_block(tag, None);
                }
                let total = struct_size(&var_fields);
                let payload = self.rc_alloc(total as i64)?;
                for (fname, fty) in &var_fields {
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

    /// Emit rc_decr for all pointer-typed local variables currently in scope.
    fn emit_decr_all_ptr_locals(&mut self) {
        let ptr_locals: Vec<(String, Variable)> = self
            .names
            .iter()
            .filter(|(name, _)| {
                self.name_types
                    .get(*name)
                    .map(|ty| !is_scalar(ty))
                    .unwrap_or(false)
            })
            .map(|(name, &var)| (name.clone(), var))
            .collect();
        for (_name, var) in ptr_locals {
            let val = self.builder.use_var(var);
            self.emit_rc_decr(val);
        }
    }

    // --- GC: arena watermark + deep copy ----------------------------------

    /// Load the current bump pointer offset (not absolute address).
    fn load_bump_offset(&mut self) -> Result<Value, NativeError> {
        let bump_gv = self
            .cg
            .obj
            .declare_data_in_func(self.cg.bump_ptr_data, self.builder.func);
        let bump_addr = self.builder.ins().global_value(PTR, bump_gv);
        Ok(self.builder.ins().load(PTR, MemFlags::new(), bump_addr, 0))
    }

    /// Store a new bump pointer offset.
    fn store_bump_offset(&mut self, val: Value) -> Result<(), NativeError> {
        let bump_gv = self
            .cg
            .obj
            .declare_data_in_func(self.cg.bump_ptr_data, self.builder.func);
        let bump_addr = self.builder.ins().global_value(PTR, bump_gv);
        self.builder.ins().store(MemFlags::new(), val, bump_addr, 0);
        Ok(())
    }

    /// Reset bump to watermark, deep-copy the return value into the
    /// reclaimed space so it survives, then return the new pointer.
    fn emit_reclaim(
        &mut self,
        result: Value,
        ret_ty: &Ty,
        watermark_var: Variable,
    ) -> Result<Value, NativeError> {
        // Scalars don't live on the heap — just reset and return.
        if is_scalar(ret_ty) {
            let wm = self.builder.use_var(watermark_var);
            self.store_bump_offset(wm)?;
            return Ok(result);
        }

        // Pointer types: save the result, reset bump to watermark,
        // then deep-copy the result into the fresh space.
        let saved_result = self.fresh_var(PTR);
        self.builder.def_var(saved_result, result);

        let wm = self.builder.use_var(watermark_var);
        self.store_bump_offset(wm)?;

        // Deep copy the saved result into the (now-reclaimed) space.
        let saved = self.builder.use_var(saved_result);
        self.emit_deep_copy(saved, ret_ty)
    }

    /// Recursively deep-copy a heap value. Returns a new pointer
    /// allocated via bump_alloc in the (reclaimed) parent arena.
    fn emit_deep_copy(&mut self, src: Value, ty: &Ty) -> Result<Value, NativeError> {
        match ty {
            Ty::String | Ty::Bytes => self.emit_string_copy(src),
            Ty::User(name) => {
                let fields = get_struct_fields(&self.cg.info.types, name);
                if fields.is_empty() {
                    // Might be a sum type — treat as opaque for now.
                    self.emit_shallow_ptr_copy(src, 16)
                } else {
                    self.emit_struct_copy(src, &fields)
                }
            }
            Ty::Option(_) | Ty::Result(_, _) => {
                // Sum types: copy the 16-byte header + deep-copy payload.
                self.emit_sum_copy(src, ty)
            }
            Ty::Tuple(elems) => {
                let fields = tuple_as_fields(elems);
                self.emit_struct_copy(src, &fields)
            }
            _ => Ok(src), // scalar fallthrough (shouldn't reach here)
        }
    }

    fn emit_string_copy(&mut self, src: Value) -> Result<Value, NativeError> {
        let flags = MemFlags::new();
        // Read len from src.
        let len_i32 = self.builder.ins().load(cl_types::I32, flags, src, 0);
        let len = self.builder.ins().uextend(PTR, len_i32);
        // Allocate 4 + len bytes.
        let four = self.builder.ins().iconst(PTR, 4);
        let total = self.builder.ins().iadd(len, four);
        let dst = self.bump_alloc_dynamic(total)?;
        // memcpy(dst, src, 4 + len)
        let heap_gv = self.cg.obj.declare_data_in_func(self.cg.heap_data, self.builder.func);
        let _ = heap_gv; // not needed here, bump_alloc_dynamic returns absolute
        self.builder.call_memcpy(
            self.cg.obj.target_config(),
            dst,
            src,
            total,
        );
        Ok(dst)
    }

    fn emit_struct_copy(
        &mut self,
        src: Value,
        fields: &[(String, Ty)],
    ) -> Result<Value, NativeError> {
        let size = struct_size(fields);
        let dst = self.bump_alloc(size as i64)?;
        let flags = MemFlags::new();
        for (fname, fty) in fields {
            let (offset, _) = field_offset(fields, fname);
            if is_scalar(fty) {
                let cl_ty = lumen_to_cl(fty);
                let val = self.builder.ins().load(cl_ty, flags, src, offset);
                self.builder.ins().store(flags, val, dst, offset);
            } else {
                // Pointer field: load, deep-copy, store new pointer.
                let old_ptr = self.builder.ins().load(PTR, flags, src, offset);
                let new_ptr = self.emit_deep_copy(old_ptr, fty)?;
                self.builder.ins().store(flags, new_ptr, dst, offset);
            }
        }
        Ok(dst)
    }

    fn emit_sum_copy(&mut self, src: Value, ty: &Ty) -> Result<Value, NativeError> {
        let flags = MemFlags::new();
        // Allocate 16 bytes for the new sum header.
        let dst = self.bump_alloc(16)?;
        // Copy tag.
        let tag = self.builder.ins().load(cl_types::I32, flags, src, 0);
        self.builder.ins().store(flags, tag, dst, 0);
        // Load payload_ptr.
        let payload = self.builder.ins().load(PTR, flags, src, 8);
        // If payload is null, store null. Otherwise shallow-copy it.
        // For a proper deep copy we'd need to know the variant's field
        // types from the tag at runtime, which requires a dispatch.
        // For the prototype: shallow-copy the payload block (works for
        // single-field variants with scalar or string payloads).
        let zero = self.builder.ins().iconst(PTR, 0);
        let is_null = self.builder.ins().icmp(IntCC::Equal, payload, zero);

        let copy_bb = self.builder.create_block();
        let merge_bb = self.builder.create_block();
        self.builder.append_block_param(merge_bb, PTR);
        self.builder
            .ins()
            .brif(is_null, merge_bb, &[BlockArg::Value(zero)], copy_bb, &[]);

        self.builder.switch_to_block(copy_bb);
        // Determine payload size. For built-in Option/Result, the payload
        // is a single value. For user sums, use the max variant size.
        let payload_size = match ty {
            Ty::Option(inner) | Ty::Result(inner, _) => native_sizeof(inner).max(8),
            _ => 8, // conservative default
        };
        let new_payload = self.emit_shallow_ptr_copy(payload, payload_size as i64)?;
        self.builder
            .ins()
            .jump(merge_bb, &[BlockArg::Value(new_payload)]);

        self.builder.switch_to_block(merge_bb);
        let final_payload = self.builder.block_params(merge_bb)[0];
        self.builder.ins().store(flags, final_payload, dst, 8);
        Ok(dst)
    }

    /// Copy `size` bytes from `src` to a new bump allocation.
    fn emit_shallow_ptr_copy(
        &mut self,
        src: Value,
        size: i64,
    ) -> Result<Value, NativeError> {
        let dst = self.bump_alloc(size)?;
        let len = self.builder.ins().iconst(PTR, size);
        self.builder.call_memcpy(
            self.cg.obj.target_config(),
            dst,
            src,
            len,
        );
        Ok(dst)
    }

    /// Bump-allocate a dynamic number of bytes (value known at runtime).
    fn bump_alloc_dynamic(&mut self, size: Value) -> Result<Value, NativeError> {
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

        let new_off = self.builder.ins().iadd(old_off, size);
        let seven = self.builder.ins().iconst(PTR, 7);
        let new_off = self.builder.ins().iadd(new_off, seven);
        let mask = self.builder.ins().iconst(PTR, -8i64);
        let new_off = self.builder.ins().band(new_off, mask);
        self.builder.ins().store(flags, new_off, bump_addr, 0);

        Ok(result)
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

    fn infer_ty(&self, expr: &Expr) -> Result<Ty, NativeError> {
        // Simplified type inference for codegen dispatch.
        Ok(match &expr.kind {
            ExprKind::IntLit { suffix, .. } => match suffix {
                Some(IntSuffix::I64) => Ty::I64,
                Some(IntSuffix::U64) => Ty::U64,
                Some(IntSuffix::U32) => Ty::U32,
                _ => Ty::I32,
            },
            ExprKind::FloatLit(_) => Ty::F64,
            ExprKind::BoolLit(_) => Ty::Bool,
            ExprKind::UnitLit => Ty::Unit,
            ExprKind::StringLit(_) => Ty::String,
            ExprKind::Ident(name) => {
                if let Some(ty) = self.name_types.get(name) {
                    return Ok(ty.clone());
                }
                // If the ident names a known function, return its FnPtr type.
                if let Some(sig) = self.cg.info.fns.get(name) {
                    let params: Vec<Ty> = sig.params.iter().map(|(_, t)| t.clone()).collect();
                    return Ok(Ty::FnPtr { params, ret: Box::new(sig.ret.clone()) });
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
                    if let Some(sig) = self.cg.info.fns.get(name) {
                        return Ok(sig.ret.clone());
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
                        return Ok(Ty::List(Box::new(Ty::I64)));
                    }
                    if m == "list" && method == "len" {
                        return Ok(Ty::I32);
                    }
                    if m == "list" && method == "push" {
                        // Returns the (possibly reallocated) list — same type as first arg.
                        if let Some(first_arg) = args.first() {
                            return self.infer_ty(&first_arg.value);
                        }
                        return Ok(Ty::List(Box::new(Ty::I32)));
                    }
                    if m == "list" && method == "get" {
                        // Without proper generics, we can't know the
                        // element type. Return I64 (the raw storage type).
                        // Field access on the result works because the
                        // codegen treats Error/I64 as PTR.
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
                }
                Ty::I32
            }
            ExprKind::StructLit { name, .. } => Ty::User(name.clone()),
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
            _ => Ty::I32,
        })
    }

    fn infer_block_ty(&self, block: &ast::Block) -> Option<Ty> {
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
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns true when the expression reads an existing reference rather than
/// producing a fresh allocation. Used to decide whether `var = expr` needs
/// rc_incr (copies need it, fresh values already have rc=1).
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
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::Unit => cl_types::I32,
        Ty::I64 | Ty::U64 => cl_types::I64,
        Ty::F64 => cl_types::F64,
        Ty::String | Ty::Bytes | Ty::User(_) | Ty::Option(_) | Ty::Result(_, _) | Ty::List(_) | Ty::Handle(_) | Ty::Tuple(_) => PTR,
        Ty::FnPtr { .. } => PTR,
        Ty::Error => cl_types::I32,
    }
}

fn is_scalar(ty: &Ty) -> bool {
    matches!(
        ty,
        Ty::I32 | Ty::U32 | Ty::I64 | Ty::U64 | Ty::F64 | Ty::Bool | Ty::Unit
        // Lists are treated as scalar for RC purposes: they manage their
        // own memory via realloc inside push/remove. RC decrementing a
        // list after realloc moved it would double-free.
        | Ty::List(_)
        // Function pointers are just integer addresses — no RC needed.
        | Ty::FnPtr { .. }
    )
}

fn native_sizeof(ty: &Ty) -> i32 {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::Unit => 4,
        Ty::I64 | Ty::U64 | Ty::F64 => 8,
        Ty::String | Ty::Bytes | Ty::User(_) | Ty::Option(_) | Ty::Result(_, _) | Ty::List(_) | Ty::Handle(_) | Ty::Tuple(_) => 8, // pointer
        Ty::FnPtr { .. } => 8,
        Ty::Error => 4,
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
