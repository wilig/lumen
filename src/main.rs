//! Lumen CLI.
//!
//! Usage:
//!   lumen build <path.lm>    # compile to a native executable
//!   lumen run   <path.lm>    # compile and run
//!   lumen build --wasm <path.lm>  # compile to .wasm (legacy)

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();

    match argv.as_slice() {
        [_, "build", "--wasm", path] => cmd_build_wasm(path),
        [_, "build", path] => cmd_build_native(path),
        [_, "run", path] => cmd_run(path),
        [_, "--version"] | [_, "-V"] => {
            println!("lumen {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "usage:\n  lumen build <path.lm>          # native executable\n  lumen build --wasm <path.lm>   # WebAssembly\n  lumen run   <path.lm>          # compile + run\n  lumen --version"
            );
            ExitCode::from(1)
        }
    }
}

fn compile_wasm(path: &str) -> Result<Vec<u8>, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let tokens = lumen::lexer::lex(&src).map_err(|e| format!("lex error: {e}"))?;
    let module = lumen::parser::parse(tokens).map_err(|e| format!("parse error: {e}"))?;
    let info = lumen::types::typecheck(&module).map_err(|errs| {
        errs.iter()
            .map(|e| format!("type error: {e}"))
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    let wasm =
        lumen::codegen::compile(&module, &info).map_err(|e| format!("codegen error: {e}"))?;
    Ok(wasm)
}

fn compile_to_object(path: &str) -> Result<(Vec<u8>, String), String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let tokens = lumen::lexer::lex(&src).map_err(|e| format!("lex error: {e}"))?;
    let module = lumen::parser::parse(tokens).map_err(|e| format!("parse error: {e}"))?;
    let info = lumen::types::typecheck(&module).map_err(|errs| {
        errs.iter()
            .map(|e| format!("type error: {e}"))
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    let obj = lumen::native::compile_native(&module, &info)
        .map_err(|e| format!("native codegen error: {e}"))?;
    let stem = path.strip_suffix(".lm").unwrap_or(path);
    Ok((obj, stem.to_string()))
}

fn link(obj_bytes: &[u8], stem: &str) -> Result<String, String> {
    let obj_path = format!("{stem}.o");
    let exe_path = stem.to_string();
    std::fs::write(&obj_path, obj_bytes).map_err(|e| format!("write {obj_path}: {e}"))?;

    // Compile the Lumen runtime (runtime/rt.c) if present.
    let rt_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime");
    let rt_src = rt_dir.join("rt.c");
    let rt_obj = format!("{stem}_rt.o");
    let mut link_args = vec![obj_path.clone(), "-o".to_string(), exe_path.clone(), "-lc".to_string()];
    if rt_src.exists() {
        let cc_status = std::process::Command::new("cc")
            .args(["-c", "-O2", rt_src.to_str().unwrap(), "-o", &rt_obj])
            .status()
            .map_err(|e| format!("failed to compile runtime: {e}"))?;
        if !cc_status.success() {
            return Err("runtime compilation failed".into());
        }
        link_args.insert(1, rt_obj.clone());
    }

    let status = std::process::Command::new("cc")
        .args(&link_args)
        .status()
        .map_err(|e| format!("failed to run cc: {e}"))?;
    let _ = std::fs::remove_file(&obj_path);
    let _ = std::fs::remove_file(&rt_obj);
    if !status.success() {
        return Err(format!("linker failed with {status}"));
    }
    Ok(exe_path)
}

fn cmd_build_wasm(path: &str) -> ExitCode {
    match compile_wasm(path) {
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

fn cmd_build_native(path: &str) -> ExitCode {
    match compile_to_object(path) {
        Ok((obj, stem)) => match link(&obj, &stem) {
            Ok(exe) => {
                eprintln!("wrote {exe}");
                ExitCode::SUCCESS
            }
            Err(msg) => {
                eprintln!("{msg}");
                ExitCode::from(2)
            }
        },
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(1)
        }
    }
}

fn cmd_run(path: &str) -> ExitCode {
    let (obj, stem) = match compile_to_object(path) {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(1);
        }
    };
    let exe = match link(&obj, &stem) {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    let status = std::process::Command::new(&format!("./{exe}"))
        .status()
        .map_err(|e| format!("failed to run {exe}: {e}"));
    let _ = std::fs::remove_file(&exe);
    match status {
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(2)
        }
    }
}
