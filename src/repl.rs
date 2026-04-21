//! Interactive read-eval-print loop (lumen-hlm).
//!
//! Each input is wrapped as a one-shot Lumen program that prints
//! the expression, compiled via the same AOT pipeline the CLI uses,
//! and run as a subprocess. No persistent bindings yet — each input
//! is independent. `let` / `fn` at the REPL DO carry through the
//! session via an accumulating "preamble" string.
//!
//! Multi-line input: unmatched braces trigger a continuation prompt.
//! `:quit` / Ctrl-D / EOF exits. `:clear` resets the preamble.

use std::io::{self, BufRead, Write};
use std::process::{Command, Stdio};

pub fn run() -> i32 {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut preamble = String::new();
    writeln!(out, "lumen repl. :quit to exit, :clear to reset.").ok();
    out.flush().ok();

    loop {
        write!(out, "> ").ok();
        out.flush().ok();
        let Some(input) = read_one_input(&mut reader) else { break; };
        let trimmed = input.trim();
        if trimmed.is_empty() { continue; }
        if trimmed == ":quit" || trimmed == ":q" { break; }
        if trimmed == ":clear" {
            preamble.clear();
            writeln!(out, "(cleared session state)").ok();
            continue;
        }

        match eval_input(&preamble, &input) {
            Ok(EvalResult::Value(s)) => {
                writeln!(out, "{s}").ok();
            }
            Ok(EvalResult::Decl(new_preamble)) => {
                preamble = new_preamble;
                writeln!(out, "(ok)").ok();
            }
            Err(e) => {
                writeln!(out, "error: {e}").ok();
            }
        }
    }
    0
}

enum EvalResult {
    Value(String),
    Decl(String),
}

/// Decide whether the input is a declaration (let / fn / type / import)
/// that should persist in the preamble, or an expression to evaluate
/// and print. Compile + run accordingly.
fn eval_input(preamble: &str, input: &str) -> Result<EvalResult, String> {
    let trimmed = input.trim_start();
    let is_decl = trimmed.starts_with("let ") || trimmed.starts_with("var ")
        || trimmed.starts_with("fn ") || trimmed.starts_with("type ")
        || trimmed.starts_with("import ") || trimmed.starts_with("extern ")
        || trimmed.starts_with("actor ") || trimmed.starts_with("msg ");

    if is_decl {
        // Append to the preamble and validate by compiling a
        // synthetic main.
        let mut new_preamble = preamble.to_string();
        if !new_preamble.is_empty() && !new_preamble.ends_with('\n') {
            new_preamble.push('\n');
        }
        new_preamble.push_str(input);
        // Validate by compiling a trivial main() that imports the
        // declarations implicitly.
        let program = wrap_for_validation(&new_preamble);
        validate_only(&program)?;
        Ok(EvalResult::Decl(new_preamble))
    } else {
        // Evaluate as an expression: wrap in `fn main` that prints it.
        let program = wrap_expr(preamble, input.trim_end());
        let output = compile_and_run(&program)?;
        Ok(EvalResult::Value(output.trim_end().to_string()))
    }
}

fn wrap_expr(preamble: &str, expr: &str) -> String {
    let mut s = String::new();
    if !preamble.contains("import std/io") {
        s.push_str("import std/io\n");
    }
    s.push_str(preamble);
    if !s.ends_with('\n') { s.push('\n'); }
    s.push_str("fn __repl_main(): i32 io {\n    io.println(");
    s.push_str(expr);
    s.push_str(")\n    return 0\n}\n");
    // Rename into main() for the CLI (which expects `main`).
    s.replace("__repl_main", "main")
}

fn wrap_for_validation(preamble: &str) -> String {
    let mut s = String::new();
    if !preamble.contains("import std/io") {
        s.push_str("import std/io\n");
    }
    s.push_str(preamble);
    if !s.ends_with('\n') { s.push('\n'); }
    if !preamble.contains("fn main(") {
        s.push_str("fn main(): i32 io { return 0 }\n");
    }
    s
}

/// Compile-only path. Writes a temp file, runs `lumen build --check`
/// equivalent. Since we don't have a build --check, we just compile
/// to a temp output and delete on success.
fn validate_only(program: &str) -> Result<(), String> {
    let tmp_src = std::env::temp_dir().join(format!("lumen_repl_{}.lm", std::process::id()));
    std::fs::write(&tmp_src, program).map_err(|e| format!("write tmp: {e}"))?;
    let lumen_bin = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let out = Command::new(lumen_bin)
        .args(["build", tmp_src.to_str().unwrap()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("spawn: {e}"))?;
    let tmp_exe = tmp_src.with_extension("");
    let _ = std::fs::remove_file(&tmp_src);
    let _ = std::fs::remove_file(&tmp_exe);
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(())
}

/// Compile the program and run it; return its stdout.
fn compile_and_run(program: &str) -> Result<String, String> {
    let tmp_src = std::env::temp_dir().join(format!("lumen_repl_{}.lm", std::process::id()));
    std::fs::write(&tmp_src, program).map_err(|e| format!("write tmp: {e}"))?;
    let lumen_bin = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let build = Command::new(&lumen_bin)
        .args(["build", tmp_src.to_str().unwrap()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("spawn build: {e}"))?;
    let tmp_exe = tmp_src.with_extension("");
    let _ = std::fs::remove_file(&tmp_src);
    if !build.status.success() {
        let _ = std::fs::remove_file(&tmp_exe);
        return Err(String::from_utf8_lossy(&build.stderr).trim().to_string());
    }
    let run = Command::new(&tmp_exe)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("spawn run: {e}"))?;
    let _ = std::fs::remove_file(&tmp_exe);
    if !run.status.success() {
        return Err(format!("runtime error:\n{}", String::from_utf8_lossy(&run.stderr)));
    }
    Ok(String::from_utf8_lossy(&run.stdout).into_owned())
}

/// Read one logical input from the reader. If the first line has
/// unbalanced braces, keep reading until they balance.
fn read_one_input(reader: &mut impl BufRead) -> Option<String> {
    let mut buf = String::new();
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).ok()?;
        if n == 0 {
            return if buf.is_empty() { None } else { Some(buf) };
        }
        buf.push_str(&line);
        if is_balanced(&buf) {
            return Some(buf);
        }
    }
}

/// Very loose bracket balance: counts `{`/`}` and `(`/`)` ignoring
/// string/char literals + comments. Returns true if the counts
/// match (we're ready to compile).
fn is_balanced(src: &str) -> bool {
    let bytes = src.as_bytes();
    let mut brace = 0i32;
    let mut paren = 0i32;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        // String literal.
        if b == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() { i += 1; }
                i += 1;
            }
            if i < bytes.len() { i += 1; }
            continue;
        }
        if b == b'\'' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'\'' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() { i += 1; }
                i += 1;
            }
            if i < bytes.len() { i += 1; }
            continue;
        }
        // Line comment.
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' { i += 1; }
            continue;
        }
        match b {
            b'{' => brace += 1,
            b'}' => brace -= 1,
            b'(' => paren += 1,
            b')' => paren -= 1,
            _ => {}
        }
        i += 1;
    }
    brace <= 0 && paren <= 0
}
