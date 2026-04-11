//! AST → Wasm codegen. See `lumen-awl`.
//!
//! Targets Wasmtime + WASI. Memory model: a single linear memory with a
//! bump allocator (no free). Strings are `(ptr: i32, len: i32)` pairs held
//! as two i32s on the Wasm stack. Structs and sum-type payloads live in
//! linear memory, pointed to by an i32.
