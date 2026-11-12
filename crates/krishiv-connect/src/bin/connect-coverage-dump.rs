// SPDX-License-Identifier: Apache-2.0
//! Prints the coverage table (`krishiv_connect::coverage`) as JSON, one object per arm. It is the
//! input of `cargo xtask connect-coverage`, which writes the generated documents.
#![forbid(unsafe_code)]

use std::fmt::Write as _;

use krishiv_connect::coverage::all;

fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() {
    let rows = all();
    let mut out = String::from("[\n");
    for (i, r) in rows.iter().enumerate() {
        let _ = write!(
            out,
            "  {{\"oneof\": {}, \"field\": {}, \"number\": {}, \"status\": {}, \"refused\": {}, \"target\": {}, \"task\": {}, \"group\": {}}}",
            quote(r.oneof),
            quote(r.field),
            r.number,
            quote(r.status.word()),
            quote(r.refused),
            quote(r.target),
            quote(r.task),
            r.group
        );
        out.push_str(if i + 1 < rows.len() { ",\n" } else { "\n" });
    }
    out.push_str("]\n");
    // A generator whose stdout is the product; there is no logger to use.
    #[expect(clippy::print_stdout, reason = "the dump is the binary's output")]
    {
        print!("{out}");
    }
}
