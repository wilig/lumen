// Integer overflow traps per lumen-ika. Each case compiles a tiny
// program whose only behavior is an overflowing op and asserts the
// process aborts with a SIGTRAP (128 + 5 = 133 on Linux, or a
// SIGTRAP-derived core-dump signal).

use std::process::Command;

fn build_and_run(src: &str, stem: &str) -> std::process::Output {
    let workspace = env!("CARGO_MANIFEST_DIR");
    let tmp = std::path::Path::new(workspace).join("target").join(stem);
    std::fs::create_dir_all(tmp.parent().unwrap()).unwrap();
    let src_path = tmp.with_extension("lm");
    std::fs::write(&src_path, src).unwrap();

    let build = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(["build", src_path.to_str().unwrap()])
        .output()
        .expect("spawn lumen build");
    assert!(
        build.status.success(),
        "compile failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr),
    );

    let run = Command::new(&tmp)
        .output()
        .expect("spawn test binary");
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&tmp);
    run
}

/// Assert the program aborted via signal (no exit code) or with the
/// conventional 128+SIGTRAP / 128+SIGABRT code. Any "expected crash"
/// satisfies the test — we don't care which flavor.
fn assert_aborted(out: &std::process::Output, label: &str) {
    let code = out.status.code();
    let signaled = code.is_none();
    let trap_like = code.map_or(false, |c| c == 128 + 5 || c == 128 + 6);
    let combined = String::from_utf8_lossy(&out.stdout).to_string()
        + &String::from_utf8_lossy(&out.stderr);
    assert!(
        signaled || trap_like,
        "{label}: expected crash, got exit {code:?}.\nOutput:\n{combined}",
    );
}

#[test]
fn overflow_i32_add() {
    let out = build_and_run(
        r#"
fn main(): i32 {
    let a: i32 = 2147483647
    let b: i32 = 1
    return a + b
}
"#,
        "overflow_i32_add",
    );
    assert_aborted(&out, "i32 max + 1");
}

#[test]
fn overflow_i32_sub() {
    let out = build_and_run(
        r#"
fn main(): i32 {
    let a: i32 = 0 - 2147483648
    let b: i32 = 1
    return a - b
}
"#,
        "overflow_i32_sub",
    );
    assert_aborted(&out, "i32 min - 1");
}

#[test]
fn overflow_i32_mul() {
    let out = build_and_run(
        r#"
fn main(): i32 {
    let a: i32 = 2147483647
    let b: i32 = 2
    return a * b
}
"#,
        "overflow_i32_mul",
    );
    assert_aborted(&out, "i32 max * 2");
}

#[test]
fn overflow_u32_underflow() {
    let out = build_and_run(
        r#"
fn main(): i32 {
    let a: u32 = 0u32
    let b: u32 = 1u32
    let c: u32 = a - b
    return 0
}
"#,
        "overflow_u32_underflow",
    );
    assert_aborted(&out, "u32 0 - 1");
}

#[test]
fn overflow_divide_by_zero() {
    let out = build_and_run(
        r#"
fn main(): i32 {
    let a: i32 = 10
    let b: i32 = 0
    return a / b
}
"#,
        "overflow_div_zero",
    );
    assert_aborted(&out, "i32 /0");
}

#[test]
fn overflow_ok_adds_at_boundary() {
    // A non-overflowing add should NOT trap. Use small values so
    // the return code fits in the shell's 8-bit exit slot.
    let out = build_and_run(
        r#"
fn main(): i32 {
    let a: i32 = 40
    let b: i32 = 2
    return a + b
}
"#,
        "overflow_ok_add",
    );
    assert_eq!(
        out.status.code(),
        Some(42),
        "expected exit 42, stderr:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
}
