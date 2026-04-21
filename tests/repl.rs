// REPL smoke tests: drive `lumen repl` as a subprocess, feed inputs
// via stdin, verify stdout.

use std::io::Write;
use std::process::{Command, Stdio};

fn run_repl(inputs: &str) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .arg("repl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lumen repl");
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(inputs.as_bytes()).unwrap();
    stdin.write_all(b":quit\n").unwrap();
    drop(stdin);
    let out = child.wait_with_output().expect("wait");
    assert!(out.status.success(), "repl exited nonzero; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn repl_evaluates_integer_expression() {
    let out = run_repl("1 + 2\n");
    // REPL prompt+result comes out as `> 3\n`; just require "3" on
    // its own token somewhere in the stdout.
    assert!(out.contains("> 3"), "expected `> 3` in output, got:\n{out}");
}

#[test]
fn repl_evaluates_string_concat() {
    let out = run_repl(r#""hello " + "world"
"#);
    assert!(out.contains("hello world"),
        "expected 'hello world' in output, got:\n{out}");
}

#[test]
fn repl_declarations_persist() {
    let out = run_repl("fn triple(n: i32): i32 { return n * 3 }\ntriple(14)\n");
    assert!(out.contains("42"), "expected 42 (triple of 14), got:\n{out}");
}

#[test]
fn repl_clear_resets_session() {
    // Declare `x`, clear, then reference `x` — should fail.
    let out = run_repl("let x: i32 = 10\n:clear\nx + 1\n");
    assert!(out.contains("error"), "expected error after :clear, got:\n{out}");
}
