//! Thin wrapper over `wasm-encoder` for emitting the final `.wasm` binary.
//! Encapsulates section ordering, type-index allocation, and the WASI import
//! table so `codegen` can stay at the Lumen-semantic level.
