//! Lumen CLI.
//!
//! Usage:
//!   lumen build <path.lm>    # compile a Lumen source file to `.wasm`
//!   lumen run   <path.lm>    # compile and run on an embedded Wasmtime

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();

    match argv.as_slice() {
        [_, "build", path] => cmd_build(path),
        [_, "run", path] => cmd_run(path),
        [_, "--version"] | [_, "-V"] => {
            println!("lumen {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "usage:\n  lumen build <path.lm>\n  lumen run   <path.lm>\n  lumen --version"
            );
            ExitCode::from(1)
        }
    }
}

fn compile(path: &str) -> Result<Vec<u8>, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let tokens = lumen::lexer::lex(&src).map_err(|e| format!("lex error: {e}"))?;
    let module = lumen::parser::parse(tokens).map_err(|e| format!("parse error: {e}"))?;
    let info =
        lumen::types::typecheck(&module).map_err(|errs| {
            errs.iter()
                .map(|e| format!("type error: {e}"))
                .collect::<Vec<_>>()
                .join("\n")
        })?;
    let wasm =
        lumen::codegen::compile(&module, &info).map_err(|e| format!("codegen error: {e}"))?;
    Ok(wasm)
}

fn cmd_build(path: &str) -> ExitCode {
    match compile(path) {
        Ok(wasm) => {
            let out = path.replace(".lm", ".wasm");
            if let Err(e) = std::fs::write(&out, &wasm) {
                eprintln!("error writing {out}: {e}");
                return ExitCode::from(2);
            }
            eprintln!("wrote {out} ({} bytes)", wasm.len());
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(1)
        }
    }
}

fn cmd_run(path: &str) -> ExitCode {
    let wasm = match compile(path) {
        Ok(w) => w,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(1);
        }
    };

    // Check if we need WASI by looking for imports in the Wasm module.
    let needs_wasi = wasm
        .windows(4)
        .any(|w| w == b"wasi");

    let engine = wasmtime::Engine::default();
    let module = match wasmtime::Module::new(&engine, &wasm) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("wasmtime module error: {e}");
            return ExitCode::from(2);
        }
    };

    if needs_wasi {
        let mut linker = wasmtime::Linker::new(&engine);
        let wasi = wasmtime_wasi::WasiCtxBuilder::new()
            .inherit_stdio()
            .build_p1();
        let mut store = wasmtime::Store::new(&engine, wasi);
        wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |s| s).unwrap();

        let instance = match linker.instantiate(&mut store, &module) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("instantiation error: {e}");
                return ExitCode::from(2);
            }
        };

        if let Ok(f) = instance.get_typed_func::<(), i32>(&mut store, "main") {
            match f.call(&mut store, ()) {
                Ok(_) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("runtime error: {e}");
                    ExitCode::from(2)
                }
            }
        } else {
            eprintln!("no `main` function found");
            ExitCode::from(1)
        }
    } else {
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = match wasmtime::Instance::new(&mut store, &module, &[]) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("instantiation error: {e}");
                return ExitCode::from(2);
            }
        };

        if let Ok(f) = instance.get_typed_func::<(), i32>(&mut store, "main") {
            match f.call(&mut store, ()) {
                Ok(ret) => {
                    if ret != 0 {
                        eprintln!("main returned {ret}");
                    }
                    ExitCode::from(ret as u8)
                }
                Err(e) => {
                    eprintln!("runtime error: {e}");
                    ExitCode::from(2)
                }
            }
        } else {
            eprintln!("no `main` function found");
            ExitCode::from(1)
        }
    }
}
