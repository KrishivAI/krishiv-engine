// SPDX-License-Identifier: Apache-2.0
//! The Iceberg REST client's response parsers (P3-T022b, PR-17 leftover; design `P3-lakehouse.md` §5.6): a catalog
//! server is input, not a trusted party. The first byte picks the parser (0 bare JSON, 1 config, 2 namespace page,
//! 3 table page, 4 load result, 5 credentials, 6 error model + OAuth error, 7 token response), the second the
//! nesting limit (`1 + b % 128`, the range the client allows); each must return a value or an error
//! for any bytes, never panic, loop or allocate without bound. Seeds: the response bodies in the crate's tests.
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let _ = krishiv_catalogs::iceberg_rest::fuzz_entry::parse_input(data);
});
