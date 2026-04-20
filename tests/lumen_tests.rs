// End-to-end harness for in-language test files under tests/lumen/.
// Each .lm file uses std/test, runs assertions, and calls
// test.summary() from main — so the exit code is the failure count.
// The harness just compiles, runs, and asserts exit code == 0.
//
// Unlike tests/programs (which diffs stdout), this decouples failure
// detection from exact phrasing — the std/test module is free to
// evolve its output format.

use std::process::Command;

include!(concat!(env!("OUT_DIR"), "/lumen_cases.rs"));

fn run_case(name: &str) {
    let workspace = env!("CARGO_MANIFEST_DIR");
    let src = format!("{workspace}/tests/lumen/{name}.lm");
    let bin = format!("{workspace}/tests/lumen/{name}");

    let build = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .args(["build", &src])
        .current_dir(workspace)
        .output()
        .expect("failed to invoke compiler");
    assert!(
        build.status.success(),
        "{name}: compile failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr),
    );

    let run = Command::new(&bin)
        .output()
        .expect("failed to run test binary");
    let _ = std::fs::remove_file(&bin);

    let code = run.status.code().unwrap_or(-1);
    assert_eq!(
        code, 0,
        "{name}: {code} assertion(s) failed\n--- stdout ---\n{}",
        String::from_utf8_lossy(&run.stdout),
    );
}
