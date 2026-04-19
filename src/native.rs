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

use crate::ast::{self, BinOp, Expr, ExprKind, FnDecl, Item, StmtKind, UnaryOp};
use crate::lexer::IntSuffix;
use crate::span::Span;
use crate::types::{ModuleInfo, Ty, TypeInfo};

/// Compile a type-checked module to a native object file (bytes).
pub fn compile_native(
    module: &ast::Module,
    info: &ModuleInfo,
    imported_modules: &[(&str, &ast::Module)],
    debug: bool,
) -> Result<Vec<u8>, NativeError> {
    let mut cg = NativeCodegen::new(info)?;
    cg.debug_mode = debug;
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

// ---------------------------------------------------------------------------
// Native codegen state
// ---------------------------------------------------------------------------

struct NativeCodegen<'a> {
    info: &'a ModuleInfo,
    obj: ObjectModule,

    /// Lumen fn name → Cranelift FuncId.
    fn_ids: HashMap<String, FuncId>,
    /// Core infrastructure FuncIds (not module-managed).
    libc_malloc: FuncId,
    libc_free: FuncId,
    helper_concat: FuncId,
    helper_println: FuncId,
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

    /// Lambda FuncIds keyed by source span (line, col).
    lambda_ids: HashMap<(u32, u32), FuncId>,
    /// Lambda FnSigs (lambdas aren't in ModuleInfo, so we store sigs here).
    lambda_sigs: HashMap<String, crate::types::FnSig>,

    /// Imported module function FuncIds: C link name → FuncId.
    module_fn_ids: HashMap<String, FuncId>,

    /// Uses WASI / io module.
    uses_io: bool,
    /// Debug mode: emit frame chain push/pop for stack traces.
    debug_mode: bool,
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
                b.set("opt_level", "speed").ok();
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
            lambda_ids: HashMap::new(),
            lambda_sigs: HashMap::new(),
            module_fn_ids: HashMap::new(),
            uses_io: false,
            debug_mode: false,
        })
    }

    fn compile_module(&mut self, module: &ast::Module, imported_modules: &[(&str, &ast::Module)]) -> Result<(), NativeError> {
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

        // Declare and compile fn items from imported modules.
        for &(mod_name, mod_ast) in imported_modules {
            // Intern strings from the imported module.
            self.intern_all_strings(mod_ast);
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
                    // Register under the Lumen-facing name so compile_call finds it.
                    if let Some(&id) = self.module_fn_ids.get(ef.link_name.as_deref().unwrap_or(&ef.name)) {
                        self.fn_ids.insert(ef.name.clone(), id);
                    }
                }
            }
            // Declare each fn.
            for item in &mod_ast.items {
                if let Item::Fn(f) = item {
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
            // Register unmangled names for intra-module calls.
            for item in &mod_ast.items {
                if let Item::Fn(f) = item {
                    let mangled = format!("{mod_name}${}", f.name);
                    if let Some(&id) = self.fn_ids.get(&mangled) {
                        self.fn_ids.insert(f.name.clone(), id);
                    }
                }
            }
            // Compile each fn body with mangled name.
            let fn_ids = self.fn_ids.clone();
            for item in &mod_ast.items {
                if let Item::Fn(f) = item {
                    let mangled = format!("{mod_name}${}", f.name);
                    if let Some(&func_id) = fn_ids.get(&mangled) {
                        let synthetic = FnDecl {
                            name: mangled,
                            name_span: f.name_span,
                            params: f.params.clone(),
                            return_type: f.return_type.clone(),
                            effect: f.effect,
                            body: f.body.clone(),
                            span: f.span,
                        };
                        self.define_function(&synthetic, func_id)?;
                    }
                }
            }
        }

        // Define helper bodies (concat/println/itoa are now in rt.c).
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
            };
            let synthetic = FnDecl {
                name: lam_name.clone(),
                name_span: lam.span,
                params: lam.params.clone(),
                return_type: lam.return_type.clone(),
                effect: ast::Effect::Pure,
                body: lam.body.clone(),
                span: lam.span,
            };
            self.lambda_sigs.insert(lam_name, fn_sig);
            self.define_function(&synthetic, func_id)?;
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
        // Look up FnSig from module info, falling back to lambda_sigs.
        let sig_owned;
        let sig: &crate::types::FnSig = if let Some(s) = self.info.fns.get(&f.name) {
            s
        } else {
            sig_owned = self.lambda_sigs[&f.name].clone();
            &sig_owned
        };
        let cl_sig = self.build_sig_from(sig);

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
        let results = self.builder.inst_results(call);
        if results.is_empty() {
            // Void function — return a unit placeholder.
            Ok(self.builder.ins().iconst(cl_types::I32, 0))
        } else {
            Ok(results[0])
        }
    }

    // --- Method call helpers ------------------------------------------------

    /// Compile args, call a builtin FuncId, return the result.
    fn call_builtin(&mut self, func_id: FuncId, args: &[ast::Arg]) -> Result<Value, NativeError> {
        let mut vals = Vec::new();
        for a in args { vals.push(self.compile_expr(&a.value)?); }
        let func_ref = self.cg.obj.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &vals);
        Ok(self.builder.inst_results(call)[0])
    }

    /// Compile args, call a void builtin FuncId, return unit (i32 0).
    fn call_builtin_void(&mut self, func_id: FuncId, args: &[ast::Arg]) -> Result<Value, NativeError> {
        let mut vals = Vec::new();
        for a in args { vals.push(self.compile_expr(&a.value)?); }
        let func_ref = self.cg.obj.declare_func_in_func(func_id, self.builder.func);
        self.builder.ins().call(func_ref, &vals);
        Ok(self.builder.ins().iconst(cl_types::I32, 0))
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

            // debug.print: compile-time specialized formatting
            if mod_name == "debug" && method == "print" {
                let val = self.compile_expr(&args[0].value)?;
                let ty = self.infer_ty(&args[0].value)?;
                self.emit_debug_print(val, &ty)?;
                // newline
                let nl_ref = self.cg.obj.declare_func_in_func(self.cg.debug_newline, self.builder.func);
                self.builder.ins().call(nl_ref, &[]);
                return Ok(self.builder.ins().iconst(cl_types::I32, 0));
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
            // list.new: pass elem_size=8
            if mod_name == "list" && method == "new" {
                let elem_size = self.builder.ins().iconst(cl_types::I32, 8);
                let fid = self.module_func("lumen_list_new");
                let func_ref = self.cg.obj.declare_func_in_func(fid, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[elem_size]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            // list.push: rc_incr pointer-typed values, sextend i32→i64
            if mod_name == "list" && method == "push" {
                let l = self.compile_expr(&args[0].value)?;
                let val_raw = self.compile_expr(&args[1].value)?;
                let val_ty = self.infer_ty(&args[1].value)?;
                if !is_scalar(&val_ty) { self.emit_rc_incr(val_raw); }
                let val64 = if lumen_to_cl(&val_ty) == cl_types::I32 {
                    self.builder.ins().sextend(cl_types::I64, val_raw)
                } else { val_raw };
                let fid = self.module_func("lumen_list_push");
                let func_ref = self.cg.obj.declare_func_in_func(fid, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[l, val64]);
                return Ok(self.builder.inst_results(call)[0]);
            }
            // list.get: ireduce i64→i32 if elem type is i32
            if mod_name == "list" && method == "get" {
                let l = self.compile_expr(&args[0].value)?;
                let i = self.compile_expr(&args[1].value)?;
                let fid = self.module_func("lumen_list_get");
                let func_ref = self.cg.obj.declare_func_in_func(fid, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[l, i]);
                let result = self.builder.inst_results(call)[0];
                let list_ty = self.infer_ty(&args[0].value)?;
                let elem_ty = match list_ty { Ty::List(inner) => *inner, _ => Ty::I64 };
                return Ok(if lumen_to_cl(&elem_ty) == cl_types::I32 {
                    self.builder.ins().ireduce(cl_types::I32, result)
                } else { result });
            }
            // list.set: sextend val if i32
            if mod_name == "list" && method == "set" {
                let l = self.compile_expr(&args[0].value)?;
                let i = self.compile_expr(&args[1].value)?;
                let val_raw = self.compile_expr(&args[2].value)?;
                let val_ty = self.infer_ty(&args[2].value)?;
                let val64 = if lumen_to_cl(&val_ty) == cl_types::I32 {
                    self.builder.ins().sextend(cl_types::I64, val_raw)
                } else { val_raw };
                let fid = self.module_func("lumen_list_set");
                let func_ref = self.cg.obj.declare_func_in_func(fid, self.builder.func);
                let call = self.builder.ins().call(func_ref, &[l, i, val64]);
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
                    return if is_void { self.call_builtin_void(func_id, args) }
                           else { self.call_builtin(func_id, args) };
                }
            }
            // Check Lumen fn by mod_name:method key.
            let fn_key = format!("{mod_name}:{method}");
            if let Some(&func_id) = self.cg.module_fn_ids.get(&fn_key) {
                return if is_void { self.call_builtin_void(func_id, args) }
                       else { self.call_builtin(func_id, args) };
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

    // --- debug.print codegen -----------------------------------------------

    /// Emit a raw string literal to stderr.
    fn emit_debug_raw(&mut self, s: &str) {
        // Create a data object for this string on the fly.
        let name = format!("__dbg_{}", self.cg.string_data.len() + self.cg.debug_data_counter);
        self.cg.debug_data_counter += 1;
        let data_id = self.cg.obj.declare_data(&name, Linkage::Local, false, false).unwrap();
        let mut desc = DataDescription::new();
        desc.define(s.as_bytes().to_vec().into_boxed_slice());
        self.cg.obj.define_data(data_id, &desc).unwrap();
        let gv = self.cg.obj.declare_data_in_func(data_id, self.builder.func);
        let ptr = self.builder.ins().global_value(PTR, gv);
        let len = self.builder.ins().iconst(cl_types::I32, s.len() as i64);
        let func_ref = self.cg.obj.declare_func_in_func(self.cg.debug_raw, self.builder.func);
        self.builder.ins().call(func_ref, &[ptr, len]);
    }

    /// Emit code to print a value of the given type to stderr.
    fn emit_debug_print(&mut self, val: Value, ty: &Ty) -> Result<(), NativeError> {
        match ty {
            Ty::I32 | Ty::U32 => {
                let f = self.cg.obj.declare_func_in_func(self.cg.debug_i32, self.builder.func);
                self.builder.ins().call(f, &[val]);
            }
            Ty::I64 | Ty::U64 => {
                let f = self.cg.obj.declare_func_in_func(self.cg.debug_i64, self.builder.func);
                self.builder.ins().call(f, &[val]);
            }
            Ty::F64 => {
                let f = self.cg.obj.declare_func_in_func(self.cg.debug_f64, self.builder.func);
                self.builder.ins().call(f, &[val]);
            }
            Ty::Bool => {
                let f = self.cg.obj.declare_func_in_func(self.cg.debug_bool, self.builder.func);
                self.builder.ins().call(f, &[val]);
            }
            Ty::String | Ty::Bytes => {
                let f = self.cg.obj.declare_func_in_func(self.cg.debug_str, self.builder.func);
                self.builder.ins().call(f, &[val]);
            }
            Ty::Unit => {
                self.emit_debug_raw("unit");
            }
            Ty::User(name) => {
                let fields = get_struct_fields(&self.cg.info.types, name);
                if fields.is_empty() {
                    // Sum type or unknown — just print the name.
                    self.emit_debug_raw(name);
                } else {
                    self.emit_debug_raw(&format!("{name} {{ "));
                    for (i, (fname, fty)) in fields.iter().enumerate() {
                        if i > 0 { self.emit_debug_raw(", "); }
                        self.emit_debug_raw(&format!("{fname}: "));
                        let (offset, _) = field_offset(&fields, fname);
                        let cl_ty = lumen_to_cl(fty);
                        let fval = self.builder.ins().load(cl_ty, MemFlags::new(), val, offset);
                        self.emit_debug_print(fval, fty)?;
                    }
                    self.emit_debug_raw(" }");
                }
            }
            Ty::Tuple(elems) => {
                self.emit_debug_raw("(");
                let fields = tuple_as_fields(elems);
                for (i, (fname, fty)) in fields.iter().enumerate() {
                    if i > 0 { self.emit_debug_raw(", "); }
                    let (offset, _) = field_offset(&fields, fname);
                    let cl_ty = lumen_to_cl(fty);
                    let fval = self.builder.ins().load(cl_ty, MemFlags::new(), val, offset);
                    self.emit_debug_print(fval, fty)?;
                }
                self.emit_debug_raw(")");
            }
            Ty::List(inner) => {
                // Call list_len, iterate, print each element.
                self.emit_debug_raw("[");
                let len_fid = self.module_func("lumen_list_len");
                let len_ref = self.cg.obj.declare_func_in_func(len_fid, self.builder.func);
                let len_call = self.builder.ins().call(len_ref, &[val]);
                let len = self.builder.inst_results(len_call)[0];
                // Simple loop: for i in 0..len
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
                // Print ", " separator after first element
                let is_first = self.builder.ins().icmp_imm(IntCC::Equal, i, 0);
                let sep_bb = self.builder.create_block();
                let after_sep = self.builder.create_block();
                self.builder.ins().brif(is_first, after_sep, &[], sep_bb, &[]);
                self.builder.switch_to_block(sep_bb);
                self.emit_debug_raw(", ");
                self.builder.ins().jump(after_sep, &[]);
                self.builder.switch_to_block(after_sep);
                let i = self.builder.use_var(counter);
                let elem_call = self.builder.ins().call(get_ref, &[val, i]);
                let elem = self.builder.inst_results(elem_call)[0];
                // If inner type is i32, ireduce
                let elem_val = if lumen_to_cl(inner) == cl_types::I32 {
                    self.builder.ins().ireduce(cl_types::I32, elem)
                } else { elem };
                self.emit_debug_print(elem_val, inner)?;
                let one = self.builder.ins().iconst(cl_types::I32, 1);
                let i = self.builder.use_var(counter);
                let next = self.builder.ins().iadd(i, one);
                self.builder.def_var(counter, next);
                self.builder.ins().jump(header, &[]);
                self.builder.switch_to_block(exit);
                self.emit_debug_raw("]");
            }
            _ => {
                // Fallback: print the type name.
                self.emit_debug_raw(&format!("<{}>", ty.display()));
            }
        }
        Ok(())
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
                    // Fall through to imported module lookup.
                    if let Some(mod_fns) = self.cg.info.modules.get(m.as_str()) {
                        if let Some(sig) = mod_fns.get(method.as_str()) {
                            return Ok(sig.ret.clone());
                        }
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

/// Convert an AST type to a Ty (simplified, for lambda signatures).
fn resolve_type_to_ty(ty: &ast::Type) -> Ty {
    match &ty.kind {
        ast::TypeKind::Named { name, args } if args.is_empty() => match name.as_str() {
            "i32" => Ty::I32,
            "i64" => Ty::I64,
            "u32" => Ty::U32,
            "u64" => Ty::U64,
            "f64" => Ty::F64,
            "bool" => Ty::Bool,
            "unit" => Ty::Unit,
            "String" => Ty::String,
            "Bytes" => Ty::Bytes,
            other => Ty::User(other.to_string()),
        },
        ast::TypeKind::FnPtr { params, ret } => {
            let param_tys = params.iter().map(resolve_type_to_ty).collect();
            Ty::FnPtr { params: param_tys, ret: Box::new(resolve_type_to_ty(ret)) }
        }
        _ => Ty::I32, // fallback
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
