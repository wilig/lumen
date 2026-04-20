// Generates a #[test] case per file in tests/programs/*.lm so each
// program shows up as its own libtest entry. Re-runs whenever a file
// in that directory is added, removed, or renamed.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let programs_dir = PathBuf::from(&manifest).join("tests/programs");
    println!("cargo:rerun-if-changed=tests/programs");

    let mut cases = Vec::new();
    if programs_dir.is_dir() {
        for entry in fs::read_dir(&programs_dir).expect("read tests/programs") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|s| s.to_str()) != Some("lm") {
                continue;
            }
            let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
            // Filenames must be valid Rust identifiers.
            if !stem.chars().all(|c| c.is_alphanumeric() || c == '_') {
                panic!(
                    "tests/programs/{stem}.lm — name must be alphanumeric/underscore",
                );
            }
            cases.push(stem);
        }
    }
    cases.sort();

    // Programs that document known-failing behavior. Listed here to
    // keep the test surface visible (run with `--ignored`) while
    // letting `cargo test` stay green. Add an entry alongside the
    // .lm/.expected files; remove it once the underlying bug is fixed.
    let known_failing: &[&str] = &[
        // lumen-0i2: generic fn params Maybe<T>/Pair<A,B> don't unify
        // with already-monomorphized argument types.
        "generic_maybe",
        "generic_pair",
    ];

    let mut out = String::new();
    for name in &cases {
        let attrs = if known_failing.contains(&name.as_str()) {
            "#[test]\n#[ignore]\n"
        } else {
            "#[test]\n"
        };
        out.push_str(&format!(
            "{attrs}fn program_{name}() {{ run_case(\"{name}\"); }}\n",
        ));
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest = PathBuf::from(out_dir).join("programs_cases.rs");
    fs::write(&dest, out).expect("write programs_cases.rs");
}
