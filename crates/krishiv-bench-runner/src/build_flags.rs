// SPDX-License-Identifier: Apache-2.0
//! Reads the build settings `--build-info` reports from rustc's flags (`CARGO_ENCODED_RUSTFLAGS`,
//! `\x1f`-separated). Used by `build.rs`; compiled into the library only for its tests.

/// The LLVM option that keeps jumps off 32-byte boundaries (x86-64).
const BRANCH_ALIGN: &str = "x86-branches-within-32B-boundaries";

/// The values of every `-C <name>=<value>` codegen option in the flags, in order, in each of
/// rustc's spellings: `-C name=v`, `-Cname=v`, `--codegen name=v` and `--codegen=name=v`.
fn codegen_values<'a>(encoded_flags: &'a str, name: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut args = encoded_flags.split('\u{1f}');
    while let Some(a) = args.next() {
        let option = if a == "-C" || a == "--codegen" {
            args.next().unwrap_or_default()
        } else if let Some(rest) = a.strip_prefix("--codegen=") {
            rest
        } else if let Some(rest) = a.strip_prefix("-C") {
            rest
        } else {
            continue;
        };
        if let Some(v) = option.strip_prefix(name).and_then(|r| r.strip_prefix('=')) {
            out.push(v);
        }
    }
    out
}

/// The last `target-cpu` named in the flags, or `default`.
pub(crate) fn target_cpu(encoded_flags: &str) -> String {
    codegen_values(encoded_flags, "target-cpu")
        .last()
        .map_or_else(|| "default".to_owned(), |v| (*v).to_owned())
}

/// `32B` when the last LLVM `-x86-branches-within-32B-boundaries` in the flags turns it on (bare,
/// `=true` or `=1`; one `llvm-args` value may hold several space-separated options), else `off`.
pub(crate) fn branch_align(encoded_flags: &str) -> &'static str {
    let mut on = false;
    for value in codegen_values(encoded_flags, "llvm-args") {
        // LLVM takes its options with one dash or two.
        for opt in value.split_whitespace().map(|o| o.trim_start_matches('-')) {
            if opt == BRANCH_ALIGN {
                on = true;
            } else if let Some(v) = opt
                .strip_prefix(BRANCH_ALIGN)
                .and_then(|r| r.strip_prefix('='))
            {
                on = matches!(v, "true" | "1" | "TRUE" | "True");
            }
        }
    }
    if on { "32B" } else { "off" }
}

#[cfg(test)]
mod tests {
    use super::{branch_align, target_cpu};

    fn enc(flags: &[&str]) -> String {
        flags.join("\u{1f}")
    }

    #[test]
    fn branch_alignment_in_every_spelling() {
        let flag = "llvm-args=-x86-branches-within-32B-boundaries";
        for f in [
            enc(&["-C", flag]),
            enc(&[&format!("-C{flag}")]),
            enc(&["--codegen", flag]),
            enc(&[&format!("--codegen={flag}")]),
            enc(&["-C", "target-cpu=x86-64-v3", "-C", flag]),
            enc(&["-C", "llvm-args=-x86-branches-within-32B-boundaries=true"]),
            enc(&["-C", "llvm-args=-x86-branches-within-32B-boundaries=1"]),
            enc(&[
                "-C",
                "llvm-args=-foo -x86-branches-within-32B-boundaries -bar",
            ]),
        ] {
            assert_eq!(branch_align(&f), "32B", "{f:?}");
        }
    }

    #[test]
    fn branch_alignment_off_when_absent_or_turned_off() {
        for f in [
            String::new(),
            enc(&["-C", "target-cpu=x86-64-v3"]),
            enc(&["-C", "llvm-args=-x86-branches-within-32B-boundaries=false"]),
            enc(&[
                "-C",
                "llvm-args=-x86-branches-within-32B-boundaries",
                "-C",
                "llvm-args=-x86-branches-within-32B-boundaries=false",
            ]),
            // Not a codegen option, and a different LLVM option with the same prefix.
            enc(&["-x86-branches-within-32B-boundaries"]),
            enc(&["-C", "llvm-args=-x86-branches-within-32B-boundariesX"]),
        ] {
            assert_eq!(branch_align(&f), "off", "{f:?}");
        }
    }

    #[test]
    fn target_cpu_is_the_last_one_named() {
        assert_eq!(target_cpu(""), "default");
        assert_eq!(
            target_cpu(&enc(&["-C", "target-cpu=x86-64-v3"])),
            "x86-64-v3"
        );
        assert_eq!(target_cpu(&enc(&["-Ctarget-cpu=native"])), "native");
        assert_eq!(
            target_cpu(&enc(&[
                "-C",
                "target-cpu=x86-64-v2",
                "--codegen=target-cpu=znver2"
            ])),
            "znver2"
        );
        assert_eq!(target_cpu(&enc(&["-C", "opt-level=3"])), "default");
    }
}
