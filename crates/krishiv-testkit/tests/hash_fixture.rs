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

#[test]
fn fixture_expressions_that_cannot_be_built_are_reported() {
    use krishiv_testkit::kernel_eval::hash::default_conf_error;
    // Spark: NULL_MAP_KEY; arity, mixed element types and unknown casts have no value.
    for (expr, why) in [
        ("map('a', 1, CAST(NULL AS STRING), 2)", "NULL_MAP_KEY"),
        ("map('a')", "map arity"),
        ("named_struct('a')", "named_struct arity"),
        ("named_struct(1, 2)", "named_struct field name"),
        ("array(1, 'a')", "elements of"),
        (
            "CAST(array(1) AS ARRAY<STRING>)",
            "cast array<int> to array<string>",
        ),
        ("CAST(1 AS ARRAY<INT>)", "cast int to array<int>"),
        ("1 +", "does not parse"),
    ] {
        let e = default_conf_error(expr).unwrap_err();
        assert!(e.contains(why), "{expr}: {e}");
    }
    // Empty containers cast to their types hash like Spark's.
    assert_eq!(
        default_conf_error("CAST(array() AS ARRAY<INT>)").unwrap(),
        None
    );
    assert_eq!(
        default_conf_error("CAST(map() AS MAP<STRING,INT>)").unwrap(),
        Some("DATATYPE_MISMATCH.HASH_MAP_TYPE")
    );
    assert_eq!(
        default_conf_error("CAST(NULL AS STRUCT<a:INT>)").unwrap(),
        None
    );
    assert_eq!(default_conf_error("CAST('abc' AS BINARY)").unwrap(), None);
    assert!(default_conf_error("CAST(1 AS BINARY)").is_err());
    assert_eq!(default_conf_error("CAST(NULL AS BINARY)").unwrap(), None);
}
