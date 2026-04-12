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
use cranelift_codegen::settings;
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
            .finish(settings::Flags::new(settings::builder()))
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

        // Define user function bodies.
        let fn_ids = self.fn_ids.clone();
        for item in &module.items {
            if let Item::Fn(f) = item {
                let func_id = fn_ids[&f.name];
                self.define_function(f, func_id)?;
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
        // Check if ptr is in the heap (bump-allocated) vs static data.
        // Static data pointers are in .rodata and don't have a refcount
        // header. We distinguish by checking if ptr is in the malloc'd
        // range — but since we use malloc now, ALL rc_alloc'd pointers
        // are in the C heap, and static data is in .rodata. The simplest
        // safe heuristic: check if *(ptr-8) looks like a valid refcount
        // (positive, < 1M). If not, skip. This is a conservative guard.
        // For correctness, we just always do the incr — static data
        // pointers should never reach rc_incr because the codegen only
        // emits incr for values that went through rc_alloc.
        let eight = builder.ins().iconst(PTR, 8);
        let header = builder.ins().isub(ptr, eight);
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

            // Declare params as variables.
            for (i, (pname, pty)) in sig.params.iter().enumerate() {
                let var = fb.fresh_var(lumen_to_cl(pty));
                let val = fb.builder.block_params(entry)[i];
                fb.builder.def_var(var, val);
                fb.names.insert(pname.clone(), var);
                fb.name_types.insert(pname.clone(), pty.clone());
            }

            // Compile body.
            let result = fb.compile_block(&f.body)?;

            // Return. RC note: the return value keeps its current
            // refcount (rc=1 from allocation). The caller receives
            // ownership. rc_decr fires on var reassignment (not on
            // function exit — that's a follow-up for full scope tracking).
            fb.builder.ins().return_(&[result]);
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
        }
    }

    fn fresh_var(&mut self, ty: CLType) -> Variable {
        self.builder.declare_var(ty)
    }

    fn compile_block(&mut self, block: &ast::Block) -> Result<Value, NativeError> {
        for stmt in &block.stmts {
            self.compile_stmt(stmt)?;
        }
        match &block.tail {
            Some(e) => self.compile_expr(e),
            None => Ok(self.builder.ins().iconst(cl_types::I32, 0)),
        }
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
                self.name_types.insert(name.clone(), lumen_ty);
            }
            StmtKind::Assign { name, value } => {
                // RC: decrement the old value before overwriting.
                if let Some(ty) = self.name_types.get(name) {
                    if !is_scalar(ty) {
                        if let Some(&var) = self.names.get(name) {
                            let old = self.builder.use_var(var);
                            self.emit_rc_decr(old);
                        }
                    }
                }
                let val = self.compile_expr(value)?;
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
                        return Err(NativeError {
                            span: expr.span,
                            message: "field access on non-struct".into(),
                        })
                    }
                };
                let fields = get_struct_fields(&self.cg.info.types, &type_name);
                let (offset, fty) = field_offset(&fields, name);
                let cl_ty = lumen_to_cl(&fty);
                Ok(self.builder.ins().load(cl_ty, MemFlags::new(), ptr, offset))
            }
            ExprKind::StructLit { name, fields, .. } => {
                self.compile_struct_lit(name, fields, expr.span)
            }
            ExprKind::Block(b) => self.compile_block(b),
            ExprKind::Match { scrutinee, arms } => {
                self.compile_match(scrutinee, arms, expr.span)
            }
            ExprKind::Try(inner) => self.compile_try(inner, expr.span),
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
            arg_vals.push(self.compile_expr(&a.value)?);
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
        // (sealed later)
        let then_val = self.compile_block(then_block)?;
        self.builder.ins().jump(merge_bb, &[BlockArg::Value(then_val)]);

        self.builder.switch_to_block(else_bb);
        // (sealed later)
        let else_val = self.compile_block(else_block)?;
        self.builder.ins().jump(merge_bb, &[BlockArg::Value(else_val)]);

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

        let def_fields = get_struct_fields(&self.cg.info.types, name);
        let total_size = struct_size(&def_fields);
        let ptr = self.rc_alloc(total_size as i64)?;

        for (fname, fty) in &def_fields {
            let init = fields.iter().find(|fi| &fi.name == fname).ok_or_else(|| {
                NativeError {
                    span,
                    message: format!("missing field `{fname}`"),
                }
            })?;
            let val = self.compile_expr(&init.value)?;
            let (offset, _) = field_offset(&def_fields, fname);
            self.builder.ins().store(MemFlags::new(), val, ptr, offset);
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
            Ty::String => self.emit_string_copy(src),
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
                }
                Ty::I32
            }
            ExprKind::MethodCall { receiver, method, .. } => {
                if let ExprKind::Ident(m) = &receiver.kind {
                    if m == "int" && method == "to_string_i32" {
                        return Ok(Ty::String);
                    }
                    if m == "io" && method == "println" {
                        return Ok(Ty::Unit);
                    }
                }
                Ty::I32
            }
            ExprKind::StructLit { name, .. } => Ty::User(name.clone()),
            ExprKind::Field { receiver, name } => {
                let recv_ty = self.infer_ty(receiver)?;
                if let Ty::User(tn) = recv_ty {
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

fn lumen_to_cl(ty: &Ty) -> CLType {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::Unit => cl_types::I32,
        Ty::I64 | Ty::U64 => cl_types::I64,
        Ty::F64 => cl_types::F64,
        Ty::String | Ty::User(_) | Ty::Option(_) | Ty::Result(_, _) | Ty::List(_) => PTR,
        Ty::Error => cl_types::I32,
    }
}

fn is_scalar(ty: &Ty) -> bool {
    matches!(
        ty,
        Ty::I32 | Ty::U32 | Ty::I64 | Ty::U64 | Ty::F64 | Ty::Bool | Ty::Unit
    )
}

fn native_sizeof(ty: &Ty) -> i32 {
    match ty {
        Ty::I32 | Ty::U32 | Ty::Bool | Ty::Unit => 4,
        Ty::I64 | Ty::U64 | Ty::F64 => 8,
        Ty::String | Ty::User(_) | Ty::Option(_) | Ty::Result(_, _) | Ty::List(_) => 8, // pointer
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
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::String | Ty::User(_) => 8,
            _ => 4,
        };
        offset = (offset + align - 1) & !(align - 1);
        if name == target {
            return (offset, ty.clone());
        }
        let size = match ty {
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::String | Ty::User(_) => 8,
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
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::String | Ty::User(_) => 8,
            _ => 4,
        };
        offset = (offset + align - 1) & !(align - 1);
        let size = match ty {
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::String | Ty::User(_) => 8,
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
        ExprKind::StructLit { fields, .. } => {
            for fi in fields {
                collect_strings_expr(&fi.value, acc);
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
