// Snapshot test corpus: discovers every `.lm` file under tests/programs/,
// compiles + runs each, and diffs stdout against the matching `.expected`
// file. One libtest case per program, so a single failure isolates to the
// offending file (vs. the kitchen_sink monolith).
//
// To accept new output as the new expected:
//   cp tests/programs/<name>.actual tests/programs/<name>.expected
//
// To add a new test: drop `tests/programs/<name>.lm` + `.expected`. No
// harness changes needed — `cargo test` rediscovers via build script.

use std::process::Command;

include!(concat!(env!("OUT_DIR"), "/programs_cases.rs"));

fn run_case(name: &str) {
    let workspace = env!("CARGO_MANIFEST_DIR");
    let src = format!("{workspace}/tests/programs/{name}.lm");
    let bin = format!("{workspace}/tests/programs/{name}");
    let expected_path = format!("{workspace}/tests/programs/{name}.expected");
    let actual_path = format!("{workspace}/tests/programs/{name}.actual");

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
        .expect("failed to run compiled program binary");
    let _ = std::fs::remove_file(&bin);
    assert!(
        run.status.success(),
        "{name}: binary exited non-zero ({:?}):\n--- stderr ---\n{}",
        run.status.code(),
        String::from_utf8_lossy(&run.stderr),
    );

    let actual = String::from_utf8(run.stdout)
        .unwrap_or_else(|_| panic!("{name}: stdout not valid UTF-8"));
    let expected = std::fs::read_to_string(&expected_path).unwrap_or_else(|e| {
        panic!("{name}: missing {expected_path}: {e}");
    });

    if actual != expected {
        std::fs::write(&actual_path, &actual).ok();
        panic!(
            "{name}: output mismatch.\n  diff {expected_path} {actual_path}\n  \
             If new output is intentional: cp {actual_path} {expected_path}",
        );
    }
}
