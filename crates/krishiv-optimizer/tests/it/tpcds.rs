// SPDX-License-Identifier: Apache-2.0
//! The 99 TPC-DS queries, parsed, resolved and optimised: plan snapshots,
//! unchanged outputs and idempotence. Plan snapshots need only table
//! schemas, not data.

use std::collections::BTreeMap;

use krishiv_optimizer::optimize;
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};
use serde_json::Value;

use crate::support;

/// A TPC-DS table type the engine does not have yet, read as its closest
/// supported one (`CHAR`/`VARCHAR` as `STRING`).
fn map_type(ty: &str) -> String {
    let upper = ty.to_ascii_uppercase();
    if upper.starts_with("CHAR") || upper.starts_with("VARCHAR") {
        return "string".to_owned();
    }
    ty.to_owned()
}

/// The 24 TPC-DS tables, as the testkit fixture (`tpcds-schemas.json`)
/// types them.
pub(crate) fn catalog() -> MemoryCatalog {
    let path = support::root().join("tests/fixtures/spark-4.2.0/golden-tables/tpcds-schemas.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut cat = MemoryCatalog::new("spark_catalog");
    for table in doc["tables"].as_array().unwrap() {
        let columns = table["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| TableColumn {
                name: c["name"].as_str().unwrap().to_owned(),
                data_type: krishiv_types::parse_data_type(&map_type(c["type"].as_str().unwrap()))
                    .unwrap(),
                nullable: c["nullable"].as_bool().unwrap(),
                char_varchar: None,
            })
            .collect();
        cat = cat.with_table(&["default"], table["name"].as_str().unwrap(), columns, None);
    }
    cat
}

/// The text of TPC-DS query `q` (1-based) from the benchmark harness, as Spark's own TPC-DS files write it: the
/// harness copies quote column aliases with double quotes (`AS "order count"`), which Spark reads as strings, so
/// they become backtick-quoted identifiers. (Single quotes are the literals.) The harness's q30 reads the
/// TPC-DS v2 column `c_last_review_date_sk`; Spark's schema and its own q30 name it `c_last_review_date`.
pub(crate) fn sql(q: usize) -> String {
    let text =
        std::fs::read_to_string(support::root().join(format!("bench/queries/tpcds/q{q:02}.sql")))
            .unwrap();
    let text = if q == 30 {
        text.replace("c_last_review_date_sk", "c_last_review_date")
    } else {
        text
    };
    let mut out = String::with_capacity(text.len());
    let mut quoted = false;
    for c in text.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                out.push('`');
            }
            c => out.push(c),
        }
    }
    assert!(!quoted, "q{q:02}: an unclosed double quote");
    out
}

/// TPC-DS queries that do not resolve yet (with the reason): snapshots
/// cannot cover them until their features land.
pub(crate) const UNSUPPORTED: &[(usize, &str)] = &[];

#[test]
fn tpcds_plans() {
    let ctx = support::ctx_over(catalog());
    let mut unsupported: BTreeMap<usize, String> = BTreeMap::new();
    let mut snapshotted = 0;
    for q in 1..=99 {
        let text = sql(q);
        let Some(plan) = support::try_resolve(&text, &ctx) else {
            let why = UNSUPPORTED
                .iter()
                .find_map(|(u, why)| (*u == q).then_some(*why))
                .unwrap_or("unlisted");
            unsupported.insert(q, why.to_owned());
            continue;
        };
        let optimized = optimize(&plan).unwrap();
        assert_eq!(
            optimized.root().resolved_output(),
            plan.root().resolved_output(),
            "q{q:02} keeps its output"
        );
        let again = optimize(&optimized).unwrap();
        assert!(
            again.structurally_eq(&optimized),
            "q{q:02} is a fixed point:\n{}",
            again.root()
        );
        insta::assert_snapshot!(format!("q{q:02}"), optimized.root().to_string());
        snapshotted += 1;
    }
    let expected: BTreeMap<usize, String> = UNSUPPORTED
        .iter()
        .map(|(q, why)| (*q, (*why).to_owned()))
        .collect();
    assert_eq!(
        unsupported, expected,
        "update UNSUPPORTED with the queries that do not resolve"
    );
    assert!(snapshotted == 99 - UNSUPPORTED.len());
}
