// SPDX-License-Identifier: Apache-2.0
//! Prints the coverage table (`krishiv_connect::coverage`) as JSON, one object per arm. It is the
//! input of `cargo xtask connect-coverage`, which writes the generated documents.
#![forbid(unsafe_code)]

fn main() {
    // A generator whose stdout is the product; there is no logger to use.
    #[expect(clippy::print_stdout, reason = "the dump is the binary's output")]
    {
        print!("{}", krishiv_connect::coverage::to_json());
    }
}
