// SPDX-License-Identifier: Apache-2.0
//! A deliberately broken target that proves the crash pipeline (P2-T108): it
//! panics on a four-byte prefix that libFuzzer finds in seconds with the
//! dictionary entry in `planted.dict`. The weekly workflow runs it, expects
//! the crash, minimizes it into a regression file and replays that file. It
//! is never part of the nightly or weekly target matrix.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.starts_with(b"KRSH") {
        panic!("planted panic");
    }
});
