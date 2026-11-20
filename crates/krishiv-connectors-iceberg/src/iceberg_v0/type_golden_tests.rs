// SPDX-License-Identifier: Apache-2.0
//! The Spark type of every Iceberg type against a real Spark 4.1.0 + iceberg-spark-runtime 1.12.0 run
//! (`tests/scripts/gen_iceberg_type_goldens.py`, fixture `tests/fixtures/iceberg-1.12.0/types.json`): Iceberg's
//! own `SparkSchemaUtil.convert` of each schema, or the message it refused with. [`spark_schema`] gives the same
//! Spark schema (compared through Spark's JSON form) or refuses with the same message.

use std::path::Path;

use iceberg::spec::Schema;
use krishiv_types::DataType;
use serde_json::Value;

use super::types::spark_schema;

/// Types iceberg-rust 0.11 cannot parse (case-name prefixes after `optional `/`required `): `unknown` (Spark reads
/// it as `void`), `geometry` and `geography`, so a table holding one fails in iceberg-rust's metadata parse before
/// any mapping. If one starts to parse, this test fails so the mapping gets its arm.
const NOT_PARSED_BY_ICEBERG_RUST: &[&str] = &["unknown", "geometry", "geography"];

fn not_parsed(name: &str) -> bool {
    let ty = name
        .strip_prefix("optional ")
        .or_else(|| name.strip_prefix("required "))
        .unwrap_or(name);
    NOT_PARSED_BY_ICEBERG_RUST.iter().any(|p| ty.starts_with(p))
}

/// A documented difference: Spark shows a format v3 column default as `EXISTS_DEFAULT`/`CURRENT_DEFAULT`
/// metadata and fills missing columns from it; Krishiv refuses the table until it does both (`types.rs`).
const REFUSED_BY_KRISHIV: &[&str] = &["column default values"];

fn goldens() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/iceberg-1.12.0/types.json");
    let text = std::fs::read_to_string(path).expect("types.json");
    serde_json::from_str(&text).expect("types.json is JSON")
}

#[test]
fn every_iceberg_type_maps_as_spark_maps_it() {
    let goldens = goldens();
    assert_eq!(goldens["spark"], "4.1.0");
    assert_eq!(goldens["iceberg"], "1.12.0");
    let cases = goldens["cases"].as_array().expect("cases");
    assert!(cases.len() >= 60, "{} cases", cases.len());
    // A floor on the cases compared with Spark's schema, so a too-wide `NOT_PARSED_BY_ICEBERG_RUST`
    // prefix cannot skip them unnoticed.
    let mut compared = 0;
    for case in cases {
        let name = case["case"].as_str().expect("case name");
        assert!(
            case.get("parse_error").is_none(),
            "{name}: Iceberg refused the schema"
        );
        let parsed = serde_json::from_value::<Schema>(case["iceberg"].clone());
        if not_parsed(name) {
            assert!(parsed.is_err(), "{name}: now parses, map it");
            continue;
        }
        let schema = parsed.unwrap_or_else(|e| panic!("{name}: {e}"));
        let ours = spark_schema(&schema);
        compared += 1;
        if REFUSED_BY_KRISHIV.contains(&name) {
            let e = ours.expect_err(name);
            assert!(
                e.to_string().contains("column default values"),
                "{name}: {e}"
            );
            assert!(
                case.get("spark").is_some(),
                "{name}: Spark now refuses it too"
            );
            continue;
        }
        match (case.get("spark"), case.get("error")) {
            (Some(spark), None) => {
                let expected = DataType::from_json(&spark.to_string())
                    .unwrap_or_else(|e| panic!("{name}: Spark's JSON: {e}"));
                let ours = ours.unwrap_or_else(|e| panic!("{name}: refused: {e}"));
                assert_eq!(DataType::Struct(ours), expected, "{name}");
            }
            (None, Some(error)) => {
                let message = error["message"].as_str().expect("message");
                let e = ours.expect_err(name);
                assert!(
                    e.to_string().contains(message),
                    "{name}: {e} vs Spark's {message}"
                );
                assert!(
                    e.to_string().contains("KRISHIV_UNSUPPORTED_FEATURE"),
                    "{name}: {e}"
                );
            }
            other => panic!("{name}: neither a schema nor an error: {other:?}"),
        }
    }
    assert!(
        compared >= 48,
        "{compared} cases compared with spark_schema"
    );
}
