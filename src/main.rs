//! Lumen CLI.
//!
//! Usage:
//!   lumen build <path.lm>    # compile a Lumen source file to `.wasm`
//!   lumen run   <path.lm>    # compile and run on an embedded Wasmtime
//!
//! v0.1: stub. Real pipeline lands as each compiler stage is implemented.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();

    match argv.as_slice() {
        [_, "build", path] => {
            eprintln!("lumen build {path}: not yet implemented");
            ExitCode::from(2)
        }
        [_, "run", path] => {
            eprintln!("lumen run {path}: not yet implemented");
            ExitCode::from(2)
        }
        [_, "--version"] | [_, "-V"] => {
            println!("lumen {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "usage:\n  lumen build <path.lm>\n  lumen run   <path.lm>\n  lumen --version"
            );
            ExitCode::from(1)
        }
    }
}
