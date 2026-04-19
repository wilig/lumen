//! Lumen CLI.
//!
//! Usage:
//!   lumen build <path.lm>    # compile to a native executable
//!   lumen run   <path.lm>    # compile and run

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();

    match argv.as_slice() {
        [_, "build", path] => cmd_build_native(path),
        [_, "run", path] => cmd_run(path),
        [_, "--version"] | [_, "-V"] => {
            println!("lumen {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "usage:\n  lumen build <path.lm>   # native executable\n  lumen run   <path.lm>   # compile + run\n  lumen --version"
            );
            ExitCode::from(1)
        }
    }
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

fn compile_to_object(path: &str) -> Result<(Vec<u8>, String, Vec<String>), String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let tokens = lumen::lexer::lex(&src).map_err(|e| {
        format_error(&src, path, "lex error", e.span.line, e.span.col, 0, &e.message)
    })?;
    let module = lumen::parser::parse(tokens).map_err(|e| {
        format_error(&src, path, "parse error", e.span.line, e.span.col, 0, &e.message)
    })?;

    // Resolve imports to stdlib .lm files, recursively following
    // transitive imports (e.g. std/http imports std/bytes).
    let std_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("std");
    let mut imported = Vec::new();
    let mut loaded: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Queue: (file_name, registered_name). Alias only applies to direct user imports.
    let mut queue: Vec<(String, String)> = module.imports.iter()
        .map(|imp| {
            let file_name = imp.path.last().cloned().unwrap_or_default();
            let reg_name = imp.alias.clone().unwrap_or_else(|| file_name.clone());
            (file_name, reg_name)
        })
        .collect();
    while let Some((file_name, reg_name)) = queue.pop() {
        if loaded.contains(&reg_name) { continue; }
        loaded.insert(reg_name.clone());
        let mod_path = std_dir.join(format!("{file_name}.lm"));
        if mod_path.exists() {
            let mod_src = std::fs::read_to_string(&mod_path)
                .map_err(|e| format!("read {}: {e}", mod_path.display()))?;
            let mod_tokens = lumen::lexer::lex(&mod_src)
                .map_err(|e| format!("lex error in std/{file_name}.lm: {e}"))?;
            let mod_ast = lumen::parser::parse(mod_tokens)
                .map_err(|e| format!("parse error in std/{file_name}.lm: {e}"))?;
            // Queue transitive imports (no alias — use file name directly).
            for imp in &mod_ast.imports {
                let dep = imp.path.last().cloned().unwrap_or_default();
                let dep_name = imp.alias.clone().unwrap_or_else(|| dep.clone());
                if !loaded.contains(&dep_name) { queue.push((dep, dep_name)); }
            }
            imported.push(lumen::types::ParsedImport { name: reg_name, module: mod_ast });
        }
    }

    let info = lumen::types::typecheck(&module, &imported).map_err(|errs| {
        errs.iter()
            .map(|e| format_error(&src, path, "type error", e.span.line, e.span.col, e.span.end, &e.message))
            .collect::<Vec<_>>()
            .join("\n\n")
    })?;
    let imports = info.imports.clone();
    let imported_refs: Vec<(&str, &lumen::ast::Module)> = imported.iter()
        .map(|i| (i.name.as_str(), &i.module))
        .collect();
    let obj = lumen::native::compile_native(&module, &info, &imported_refs)
        .map_err(|e| format_error(&src, path, "codegen error", e.span.line, e.span.col, e.span.end, &e.message))?;
    let stem = path.strip_suffix(".lm").unwrap_or(path);
    Ok((obj, stem.to_string(), imports))
}

fn link(obj_bytes: &[u8], stem: &str, imports: &[String]) -> Result<String, String> {
    let obj_path = format!("{stem}.o");
    let exe_path = stem.to_string();
    std::fs::write(&obj_path, obj_bytes).map_err(|e| format!("write {obj_path}: {e}"))?;

    // Compile the Lumen runtime (runtime/rt.c) if present.
    let rt_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime");
    let rt_src = rt_dir.join("rt.c");
    let rt_obj = format!("{stem}_rt.o");
    let mut link_args = vec![obj_path.clone(), "-o".to_string(), exe_path.clone(), "-lc".to_string(), "-lm".to_string()];

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

    // Compile and link the raylib bridge only when the source imports std/raylib.
    let rl_bridge_src = rt_dir.join("raylib_bridge.c");
    let rl_bridge_obj = format!("{stem}_rl.o");
    let uses_raylib = imports.iter().any(|i| i == "std/rl" || i == "std/raylib");
    if uses_raylib && rl_bridge_src.exists() {
        let cc_status = std::process::Command::new("cc")
            .args(["-c", "-O2", rl_bridge_src.to_str().unwrap(), "-o", &rl_bridge_obj])
            .status()
            .map_err(|e| format!("failed to compile raylib bridge: {e}"))?;
        if cc_status.success() {
            link_args.insert(2, rl_bridge_obj.clone());
            // Add raylib link flags (platform-specific).
            #[cfg(target_os = "linux")]
            for flag in [
                "-L/usr/lib/odin/vendor/raylib/linux",
                "-Wl,-rpath,/usr/lib/odin/vendor/raylib/linux",
                "-lraylib", "-lm", "-lGL", "-ldl", "-lpthread",
            ] {
                link_args.push(flag.to_string());
            }
            #[cfg(target_os = "macos")]
            {
                // Check for homebrew raylib.
                if let Ok(out) = std::process::Command::new("brew")
                    .args(["--prefix", "raylib"])
                    .output()
                {
                    if out.status.success() {
                        let prefix = String::from_utf8_lossy(&out.stdout).trim().to_string();
                        link_args.push(format!("-L{prefix}/lib"));
                        link_args.push(format!("-I{prefix}/include"));
                    }
                }
                for flag in [
                    "-lraylib", "-lm",
                    "-framework", "OpenGL",
                    "-framework", "Cocoa",
                    "-framework", "IOKit",
                    "-framework", "CoreVideo",
                ] {
                    link_args.push(flag.to_string());
                }
            }
        }
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

fn cmd_build_native(path: &str) -> ExitCode {
    match compile_to_object(path) {
        Ok((obj, stem, imports)) => match link(&obj, &stem, &imports) {
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
    let (obj, stem, imports) = match compile_to_object(path) {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(1);
        }
    };
    let exe = match link(&obj, &stem, &imports) {
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
