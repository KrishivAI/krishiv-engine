// SPDX-License-Identifier: Apache-2.0
//! The SQL parser on arbitrary input, under every parser option: it must
//! return a tree or an error, never panic or overflow the stack, and a tree's
//! spans must lie within the input.
#![no_main]

use krishiv_sql::{ParserOptions, parse_query};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let sql = String::from_utf8_lossy(data);
    for bits in 0..16_u8 {
        let options = ParserOptions {
            enforce_reserved_keywords: bits & 1 != 0,
            double_quoted_identifiers: bits & 2 != 0,
            legacy_setops_precedence: bits & 4 != 0,
            escaped_string_literals: bits & 8 != 0,
        };
        match parse_query(&sql, &options) {
            Ok(q) => assert!(q.span.end <= sql.len() && q.span.start <= q.span.end),
            Err(e) => assert!(e.span.end <= sql.len()),
        }
    }
});
