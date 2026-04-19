// End-to-end snapshot test: compiles tests/kitchen_sink.lm, runs the
// resulting binary, and diffs stdout against tests/kitchen_sink.expected.
//
// On mismatch, the actual output is written to tests/kitchen_sink.actual
// for easy diffing. To accept new output as the new expected:
//
//   cp tests/kitchen_sink.actual tests/kitchen_sink.expected
//
// The kitchen_sink program combines deterministic example demos
// (hello, generics, interpolation, list utilities, Map ops, file I/O,
// actors, etc.) so a single test covers most user-facing language
// surface. Anything that's nondeterministic or external (timing,
// network, raylib) is excluded.

use std::process::Command;

#[test]
fn kitchen_sink_matches_snapshot() {
    let workspace = env!("CARGO_MANIFEST_DIR");
    let src = format!("{workspace}/tests/kitchen_sink.lm");
    let bin = format!("{workspace}/tests/kitchen_sink");
    let expected_path = format!("{workspace}/tests/kitchen_sink.expected");
    let actual_path = format!("{workspace}/tests/kitchen_sink.actual");
    let scratch_file = "/tmp/lumen_kitchen_sink.txt";

    // Compile via the regular CLI (matches what users run).
    let build = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--", "build", &src])
        .current_dir(workspace)
        .output()
        .expect("failed to invoke compiler");
    assert!(
        build.status.success(),
        "kitchen_sink compile failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr),
    );

    // Clean up any leftover scratch file from a previous run so the
    // file_io section starts fresh.
    let _ = std::fs::remove_file(scratch_file);

    let run = Command::new(&bin)
        .output()
        .expect("failed to run compiled kitchen_sink binary");
    assert!(
        run.status.success(),
        "kitchen_sink binary exited non-zero ({:?}):\n--- stderr ---\n{}",
        run.status.code(),
        String::from_utf8_lossy(&run.stderr),
    );

    let actual = String::from_utf8(run.stdout)
        .expect("kitchen_sink stdout was not valid UTF-8");
    let expected = std::fs::read_to_string(&expected_path).unwrap_or_else(|e| {
        panic!(
            "missing or unreadable {expected_path}: {e}\n\
             To bootstrap, run the binary once and capture stdout there:\n  \
             ./tests/kitchen_sink > {expected_path}",
        )
    });

    if actual != expected {
        std::fs::write(&actual_path, &actual).ok();
        panic!(
            "kitchen_sink output mismatch.\n\
             diff {expected_path} {actual_path}\n\
             If the new output is intentional:\n  \
             cp {actual_path} {expected_path}",
        );
    }
}
