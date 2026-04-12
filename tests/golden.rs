#![cfg(feature = "wasm-runtime")]
//! Golden tests: compile each `.lm` example, run it on Wasmtime, and
//! compare stdout against the expected output.

fn compile_and_run(src: &str) -> String {
    let tokens = lumen::lexer::lex(src).unwrap();
    let module = lumen::parser::parse(tokens).unwrap();
    let info = lumen::types::typecheck(&module).unwrap();
    let wasm = lumen::codegen::compile(&module, &info).unwrap();

    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut linker = wasmtime::Linker::new(&engine);
    let stdout = wasmtime_wasi::p2::pipe::MemoryOutputPipe::new(4096);
    let wasi = wasmtime_wasi::WasiCtxBuilder::new()
        .stdout(stdout.clone())
        .build_p1();
    let mut store = wasmtime::Store::new(&engine, wasi);
    wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |s| s).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    // Try main() -> i32.
    if let Ok(f) = instance.get_typed_func::<(), i32>(&mut store, "main") {
        let _ = f.call(&mut store, ());
    }

    drop(store);
    String::from_utf8(stdout.try_into_inner().unwrap().into()).unwrap()
}

#[test]
fn golden_hello() {
    let src = include_str!("../examples/hello.lm");
    assert_eq!(compile_and_run(src), "hello, world\n");
}

#[test]
fn golden_sum_of_squares() {
    let src = include_str!("../examples/sum_of_squares.lm");
    assert_eq!(compile_and_run(src), "sum of squares 1..10 = 385\n");
}

#[test]
fn golden_match_demo() {
    let src = include_str!("../examples/match_demo.lm");
    let out = compile_and_run(src);
    assert!(out.contains("circle area ~ 75"), "got: {out}");
    assert!(out.contains("rect area   = 24"), "got: {out}");
}

#[test]
fn golden_error_chain() {
    let src = include_str!("../examples/error_chain.lm");
    let out = compile_and_run(src);
    assert!(out.contains("at compute"), "got: {out}");
    assert!(out.contains("at main"), "got: {out}");
    assert!(
        !out.contains("result ="),
        "should not print result on error, got: {out}"
    );
}
