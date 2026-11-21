// SPDX-License-Identifier: Apache-2.0
//! The Iceberg REST client's response parsers (P3-T022b, PR-17 leftover; design `P3-lakehouse.md` §5.6): a catalog
//! server is input, not a trusted party. The first byte picks the parser (config, namespace page, table page, load
//! result, credentials, error model, OAuth error, token response, bare JSON); each must return a value or an error
//! for any bytes, never panic, loop or allocate without bound. Seeds: the response bodies in the crate's tests.
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let Some((&which, body)) = data.split_first() else {
        return;
    };
    // The client's default nesting limit.
    let _ = krishiv_catalogs::iceberg_rest::fuzz_entry::parse_response(which, body, 64);
});
