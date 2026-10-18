// SPDX-License-Identifier: Apache-2.0
//! Every case of the Spark `hash()`/`xxhash64()` fixture on the hash kernels
//! (P2-T027): the argument types and both hashes must equal Spark's, each
//! case under the configuration it was generated with, and the expression
//! Spark rejects by default must be rejected.
#![forbid(unsafe_code)]

use krishiv_testkit::hash_fixture::HashFixture;
use krishiv_testkit::kernel_eval::hash::{check_case, default_conf_error};

type R = Result<(), Box<dyn std::error::Error>>;

#[test]
fn murmur3_and_xxhash64_match_spark() -> R {
    let fx = HashFixture::load_default()?;
    assert!(fx.cases.len() >= 90, "fixture shrank to {}", fx.cases.len());
    let failures: Vec<String> = fx
        .cases
        .iter()
        .filter_map(|c| {
            check_case(c)
                .err()
                .map(|e| format!("{} `{}`: {e}", c.id, c.expr))
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn map_hashing_needs_the_legacy_switch() -> R {
    let fx = HashFixture::load_default()?;
    assert!(!fx.default_conf_errors.is_empty());
    for e in &fx.default_conf_errors {
        assert_eq!(
            default_conf_error(&e.expr)?,
            Some(e.error_class.as_str()),
            "{}",
            e.expr
        );
    }
    // A non-map argument list is accepted.
    assert_eq!(default_conf_error("1, 'a'")?, None);
    Ok(())
}
