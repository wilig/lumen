// Generates a #[test] case per file in tests/programs/*.lm so each
// program shows up as its own libtest entry. Re-runs whenever a file
// in that directory is added, removed, or renamed.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    generate_cases(
        &PathBuf::from(&manifest).join("tests/programs"),
        &out_dir.join("programs_cases.rs"),
        "program_",
        // Programs that document known-failing behavior. Listed here
        // to keep the test surface visible (run with `--ignored`)
        // while letting `cargo test` stay green.
        &[],
        "tests/programs",
    );

    generate_cases(
        &PathBuf::from(&manifest).join("tests/lumen"),
        &out_dir.join("lumen_cases.rs"),
        "lumen_",
        &[],
        "tests/lumen",
    );
}

fn generate_cases(
    dir: &PathBuf,
    dest: &PathBuf,
    prefix: &str,
    known_failing: &[&str],
    rerun_path: &str,
) {
    println!("cargo:rerun-if-changed={rerun_path}");
    let mut cases: Vec<String> = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(dir).expect("read test dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|s| s.to_str()) != Some("lm") {
                continue;
            }
            let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
            if !stem.chars().all(|c| c.is_alphanumeric() || c == '_') {
                panic!("{rerun_path}/{stem}.lm — name must be alphanumeric/underscore");
            }
            cases.push(stem);
        }
    }
    cases.sort();

    let mut out = String::new();
    for name in &cases {
        let attrs = if known_failing.contains(&name.as_str()) {
            "#[test]\n#[ignore]\n"
        } else {
            "#[test]\n"
        };
        out.push_str(&format!(
            "{attrs}fn {prefix}{name}() {{ run_case(\"{name}\"); }}\n",
        ));
    }
    fs::write(dest, out).expect("write cases file");
}
