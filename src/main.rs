//! Lumen CLI.
//!
//! Usage:
//!   lumen build <path.lm>    # compile to a native executable
//!   lumen run   <path.lm>    # compile and run

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        return ExitCode::from(1);
    }
    match args[0].as_str() {
        "build" => {
            let mut debug = false;
            let mut path = None;
            let mut extra_link_flags = Vec::new();
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--debug" => debug = true,
                    s if s.starts_with("-l") || s.starts_with("-L") || s.starts_with("-framework") || s.starts_with("-Wl,") => {
                        extra_link_flags.push(args[i].clone());
                    }
                    "-framework" => {
                        extra_link_flags.push(args[i].clone());
                        if i + 1 < args.len() {
                            i += 1;
                            extra_link_flags.push(args[i].clone());
                        }
                    }
                    _ if !args[i].starts_with('-') && path.is_none() => {
                        path = Some(args[i].as_str());
                    }
                    other => {
                        eprintln!("unknown option: {other}");
                        return ExitCode::from(1);
                    }
                }
                i += 1;
            }
            match path {
                Some(p) => cmd_build_native(p, debug, &extra_link_flags),
                None => { eprintln!("error: no input file"); ExitCode::from(1) }
            }
        }
        "run" => {
            if let Some(path) = args.get(1) {
                cmd_run(path)
            } else {
                eprintln!("error: no input file");
                ExitCode::from(1)
            }
        }
        "lsp" => {
            // Speaks LSP over stdin/stdout; editor clients launch
            // `lumen lsp` as a subprocess.
            ExitCode::from(lumen::lsp::run() as u8)
        }
        "--version" | "-V" => {
            println!("lumen {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => { print_usage(); ExitCode::from(1) }
    }
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!("  lumen build [--debug] <path.lm> [-lname] [-Lpath] [-framework name]");
    eprintln!("  lumen run   <path.lm>");
    eprintln!("  lumen lsp                                     # language server on stdin/stdout");
    eprintln!("  lumen --version");
}

/// Format an error with source context: the offending line + a caret.
fn format_error(src: &str, file: &str, kind: &str, line: u32, col: u32, end: u32, message: &str) -> String {
    let mut out = format!("{kind}: {file}:{line}:{col}: {message}");
    if line > 0 {
        if let Some(src_line) = src.lines().nth((line - 1) as usize) {
            out.push_str(&format!("\n  {line} | {src_line}"));
            // Caret(s) under the error location.
            let pad = format!("{line}").len() + 3 + (col as usize).saturating_sub(1);
            let width = if end > 0 {
                let span_len = (end - (col - 1) as u32) as usize;
                span_len.max(1).min(src_line.len().saturating_sub((col - 1) as usize)).max(1)
            } else { 1 };
            out.push_str(&format!("\n  {}{}",
                " ".repeat(pad),
                "^".repeat(width),
            ));
        }
    }
    out
}

fn compile_to_object(path: &str, debug: bool) -> Result<(Vec<u8>, String, Vec<String>), String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let tokens = lumen::lexer::lex(&src).map_err(|e| {
        format_error(&src, path, "lex error", e.span.line, e.span.col, 0, &e.message)
    })?;
    let module = lumen::parser::parse(tokens).map_err(|e| {
        format_error(&src, path, "parse error", e.span.line, e.span.col, 0, &e.message)
    })?;

    // Resolve imports transitively. Any load/lex/parse failures get
    // surfaced as top-level errors so the CLI stays strict.
    let base_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let resolved = lumen::imports::resolve(&module, base_dir);
    if let Some(first) = resolved.errors.first() {
        return Err(first.clone());
    }
    let imported = resolved.imported;
    let module_paths = resolved.paths;

    let mut info = lumen::types::typecheck(&module, &imported).map_err(|errs| {
        errs.iter()
            .map(|e| format_error(&src, path, "type error", e.span.line, e.span.col, e.span.end, &e.message))
            .collect::<Vec<_>>()
            .join("\n\n")
    })?;
    let imports = info.imports.clone();
    let imported_refs: Vec<(&str, &lumen::ast::Module)> = imported.iter()
        .map(|i| (i.name.as_str(), &i.module))
        .collect();
    let obj = lumen::native::compile_native(&module, &mut info, &imported_refs, &module_paths, debug, path)
        .map_err(|e| format_error(&src, path, "codegen error", e.span.line, e.span.col, e.span.end, &e.message))?;
    let stem = path.strip_suffix(".lm").unwrap_or(path);
    Ok((obj, stem.to_string(), imports))
}

fn link(obj_bytes: &[u8], stem: &str, imports: &[String], extra_link_flags: &[String]) -> Result<String, String> {
    let obj_path = format!("{stem}.o");
    let exe_path = stem.to_string();
    std::fs::write(&obj_path, obj_bytes).map_err(|e| format!("write {obj_path}: {e}"))?;

    // Compile the Lumen runtime (runtime/rt.c) if present.
    let rt_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime");
    let rt_src = rt_dir.join("rt.c");
    let rt_obj = format!("{stem}_rt.o");
    // -rdynamic exports all symbols so libc's backtrace_symbols_fd
    // can resolve function names in stack traces (without it, only
    // symbols in the dynamic table are visible — most Lumen fns are
    // Local-linkage and wouldn't appear).
    let mut link_args = vec![
        obj_path.clone(),
        "-o".to_string(),
        exe_path.clone(),
        "-rdynamic".to_string(),
        "-lc".to_string(),
        "-lm".to_string(),
    ];

    // Compile the main runtime.
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

    // Compile the raylib bridge when vendor/raylib is imported.
    let rl_bridge_src = rt_dir.join("raylib_bridge.c");
    let rl_bridge_obj = format!("{stem}_rl.o");
    let uses_raylib = imports.iter().any(|i| i == "vendor/raylib" || i == "std/rl" || i == "std/raylib");
    if uses_raylib && rl_bridge_src.exists() {
        let cc_status = std::process::Command::new("cc")
            .args(["-c", "-O2", rl_bridge_src.to_str().unwrap(), "-o", &rl_bridge_obj])
            .status()
            .map_err(|e| format!("failed to compile raylib bridge: {e}"))?;
        if cc_status.success() {
            link_args.insert(2, rl_bridge_obj.clone());
        }
    }

    // Append user-provided linker flags (e.g. -lraylib -L/path/to/lib).
    for flag in extra_link_flags {
        link_args.push(flag.clone());
    }

    let status = std::process::Command::new("cc")
        .args(&link_args)
        .status()
        .map_err(|e| format!("failed to run cc: {e}"))?;
    let _ = std::fs::remove_file(&obj_path);
    let _ = std::fs::remove_file(&rt_obj);
    let _ = std::fs::remove_file(&rl_bridge_obj);
    if !status.success() {
        return Err(format!("linker failed with {status}"));
    }
    Ok(exe_path)
}

fn cmd_build_native(path: &str, debug: bool, extra_link_flags: &[String]) -> ExitCode {
    match compile_to_object(path, debug) {
        Ok((obj, stem, imports)) => match link(&obj, &stem, &imports, extra_link_flags) {
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
    let (obj, stem, imports) = match compile_to_object(path, false) {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(1);
        }
    };
    let exe = match link(&obj, &stem, &imports, &[]) {
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
