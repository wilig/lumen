// End-to-end `lumen fmt` tests. Drives the subcommand as a subprocess
// so we cover the CLI wiring in addition to the fmt module.

use std::fs;
use std::process::Command;

fn write_tmp(name: &str, content: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("lumen_fmt_{name}.lm"));
    fs::write(&path, content).unwrap();
    path
}

#[test]
fn fmt_rewrites_ugly_source() {
    let path = write_tmp("ugly", "fn   main(  ):i32{return    42}");
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(["fmt", path.to_str().unwrap()])
        .output()
        .expect("spawn lumen fmt");
    assert!(out.status.success(), "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr));
    let result = fs::read_to_string(&path).unwrap();
    assert_eq!(result, "fn main(): i32 { return 42 }\n");
    let _ = fs::remove_file(&path);
}

#[test]
fn fmt_check_exits_zero_on_canonical() {
    let path = write_tmp("clean", "fn main(): i32 { return 0 }\n");
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(["fmt", "--check", path.to_str().unwrap()])
        .output()
        .expect("spawn lumen fmt");
    assert!(out.status.success(), "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr));
    let _ = fs::remove_file(&path);
}

#[test]
fn fmt_check_exits_nonzero_on_bad() {
    let path = write_tmp("bad", "fn   main():i32{return 0}");
    let out = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(["fmt", "--check", path.to_str().unwrap()])
        .output()
        .expect("spawn lumen fmt");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not canonical"), "stderr:\n{stderr}");
    let _ = fs::remove_file(&path);
}

#[test]
fn fmt_is_idempotent() {
    // Format once to get the canonical form, then format again and
    // verify nothing changes.
    let path = write_tmp("idem", "import std/io\n\nfn f(): i32 io {\n    io.println(\"x\")\n    return 0\n}\n");
    let run_fmt = || {
        Command::new(env!("CARGO_BIN_EXE_lumen"))
            .args(["fmt", path.to_str().unwrap()])
            .output().unwrap()
    };
    let _ = run_fmt();
    let first = fs::read_to_string(&path).unwrap();
    let _ = run_fmt();
    let second = fs::read_to_string(&path).unwrap();
    assert_eq!(first, second, "fmt not idempotent");
    let _ = fs::remove_file(&path);
}
