// SPDX-License-Identifier: Apache-2.0
//! `cargo run -p krishiv-functions --bin gen-python-functions [-- --check]`: writes the Python wrappers
//! of the function registry (`python/krishiv/sql/_generated_functions.py` and `.pyi`), or with
//! `--check` fails when they are out of date.

#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "CLI binary: stdout/stderr are its interface"
)]

use std::path::Path;
use std::process::ExitCode;

use krishiv_functions::python_wrappers::{PY_PATH, PYI_PATH, generate_for_registry, out_of_date};

fn run(check: bool) -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if check {
        let stale = out_of_date(&root)?;
        return if stale.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} out of date; run `cargo run -p krishiv-functions --bin gen-python-functions`",
                stale.join(", ")
            ))
        };
    }
    let generated = generate_for_registry()?;
    for (path, text) in [(PY_PATH, &generated.py), (PYI_PATH, &generated.pyi)] {
        std::fs::write(root.join(path), text).map_err(|e| format!("cannot write {path}: {e}"))?;
        println!("wrote {path}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let check = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] => false,
        ["--check"] => true,
        _ => {
            eprintln!("usage: gen-python-functions [--check]");
            return ExitCode::FAILURE;
        }
    };
    match run(check) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
