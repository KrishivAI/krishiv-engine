// SPDX-License-Identifier: Apache-2.0
//! Planning time: the 22 TPC-H queries, parsed and resolved once, then
//! optimised (rule batches, join ordering, pruning); run under CodSpeed in
//! CI. Also per join enumerator (`krishiv.optimizer.joinEnumerator`, P2-T113):
//! the TPC-DS queries, and stars and chains of 10 and 12 inputs, which only
//! the join order of differs between enumerators.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_common::config::Config;
use krishiv_optimizer::{NoStatistics, optimize_with_udfs};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, SourceId};
use krishiv_types::parse_data_type;
use krishiv_udf_sdk::UdfLibrary;
use serde_json::Value;

/// The TPC-H tables, columns as `name type` pairs.
const TABLES: [(&str, &str); 8] = [
    (
        "region",
        "r_regionkey bigint, r_name string, r_comment string",
    ),
    (
        "nation",
        "n_nationkey bigint, n_name string, n_regionkey bigint, n_comment string",
    ),
    (
        "supplier",
        "s_suppkey bigint, s_name string, s_address string, s_nationkey bigint, s_phone string, s_acctbal decimal(15,2), s_comment string",
    ),
    (
        "customer",
        "c_custkey bigint, c_name string, c_address string, c_nationkey bigint, c_phone string, c_acctbal decimal(15,2), c_mktsegment string, c_comment string",
    ),
    (
        "part",
        "p_partkey bigint, p_name string, p_mfgr string, p_brand string, p_type string, p_size int, p_container string, p_retailprice decimal(15,2), p_comment string",
    ),
    (
        "partsupp",
        "ps_partkey bigint, ps_suppkey bigint, ps_availqty int, ps_supplycost decimal(15,2), ps_comment string",
    ),
    (
        "orders",
        "o_orderkey bigint, o_custkey bigint, o_orderstatus string, o_totalprice decimal(15,2), o_orderdate date, o_orderpriority string, o_clerk string, o_shippriority int, o_comment string",
    ),
    (
        "lineitem",
        "l_orderkey bigint, l_partkey bigint, l_suppkey bigint, l_linenumber int, l_quantity decimal(15,2), l_extendedprice decimal(15,2), l_discount decimal(15,2), l_tax decimal(15,2), l_returnflag string, l_linestatus string, l_shipdate date, l_commitdate date, l_receiptdate date, l_shipinstruct string, l_shipmode string, l_comment string",
    ),
];

/// `sql` parsed and resolved over `catalog`.
fn resolve(sql: &str, ctx: &krishiv_plan::analyzer::AnalysisContext) -> Option<Plan> {
    let plan =
        krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .ok()?;
    Some(block_on(DefaultAnalyzer.analyze(&plan, ctx)).ok()?.plan)
}

fn tpch_catalog() -> MemoryCatalog {
    let mut catalog = MemoryCatalog::new("spark_catalog");
    for (name, cols) in TABLES {
        let columns = cols
            .split(", ")
            .map(|c| {
                let (n, t) = c.split_once(' ').expect("name type");
                TableColumn {
                    name: n.to_owned(),
                    data_type: parse_data_type(t).expect("type"),
                    nullable: true,
                }
            })
            .collect();
        catalog = catalog.with_table(&["default"], name, columns, None);
    }
    catalog
}

fn plans() -> Vec<Plan> {
    let ctx = context(tpch_catalog(), ResolutionConf::default());
    (1..=22)
        .map(|q| {
            let path = format!(
                "{}/../../bench/queries/tpch/q{q:02}.sql",
                env!("CARGO_MANIFEST_DIR")
            );
            let sql = std::fs::read_to_string(path).expect("query");
            resolve(&sql, &ctx).expect("resolves")
        })
        .collect()
}

/// The TPC-DS queries that resolve over the testkit's schemas (`CHAR` and
/// `VARCHAR` read as strings; the harness copies' double-quoted aliases as
/// backtick identifiers, as Spark's own files write them).
fn tpcds_plans() -> Vec<Plan> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let schemas = std::fs::read_to_string(format!(
        "{root}/tests/fixtures/spark-4.2.0/golden-tables/tpcds-schemas.json"
    ))
    .expect("schemas");
    let doc: Value = serde_json::from_str(&schemas).expect("json");
    let mut catalog = MemoryCatalog::new("spark_catalog");
    for table in doc["tables"].as_array().expect("tables") {
        let columns = table["columns"]
            .as_array()
            .expect("columns")
            .iter()
            .map(|c| {
                let ty = c["type"].as_str().expect("type");
                let upper = ty.to_ascii_uppercase();
                let ty = if upper.starts_with("CHAR") || upper.starts_with("VARCHAR") {
                    "string"
                } else {
                    ty
                };
                TableColumn {
                    name: c["name"].as_str().expect("name").to_owned(),
                    data_type: parse_data_type(ty).expect("type"),
                    nullable: true,
                }
            })
            .collect();
        let name = table["name"].as_str().expect("name");
        catalog = catalog.with_table(&["default"], name, columns, None);
    }
    let ctx = context(catalog, ResolutionConf::default());
    (1..=99)
        .filter_map(|q| {
            let sql = std::fs::read_to_string(format!("{root}/bench/queries/tpcds/q{q:02}.sql"))
                .expect("query")
                .replace('"', "`");
            resolve(&sql, &ctx)
        })
        .collect()
}

/// A star (one fact table joined to `n - 1` dimensions) or a chain of `n`
/// inputs over the TPC-H `lineitem` table.
fn synthetic(n: usize, star: bool) -> Plan {
    let mut sql = "SELECT t1.l_comment FROM lineitem t1".to_owned();
    for i in 2..=n {
        let other = if star { 1 } else { i - 1 };
        sql.push_str(&format!(
            " JOIN lineitem t{i} ON t{i}.l_orderkey = t{other}.l_partkey"
        ));
    }
    resolve(&sql, &context(tpch_catalog(), ResolutionConf::default())).expect("resolves")
}

fn enumerator(name: &str) -> Config {
    let mut config = Config::new();
    config
        .set("krishiv.optimizer.joinEnumerator", name)
        .expect("a known enumerator");
    config
}

fn optimize(c: &mut Criterion) {
    let plans = plans();
    c.bench_function("optimize_tpch_22", |b| {
        b.iter(|| {
            for p in &plans {
                black_box(krishiv_optimizer::optimize(black_box(p)).expect("optimises"));
            }
        });
    });
}

fn enumerators(c: &mut Criterion) {
    let tpcds = tpcds_plans();
    let lib = UdfLibrary::new();
    let shapes = [
        ("star_10", synthetic(10, true)),
        ("chain_10", synthetic(10, false)),
        ("star_12", synthetic(12, true)),
        ("chain_12", synthetic(12, false)),
    ];
    for name in ["dpsub", "dphyp"] {
        let config = enumerator(name);
        c.bench_function(&format!("optimize_tpcds_{name}"), |b| {
            b.iter(|| {
                for p in &tpcds {
                    black_box(
                        optimize_with_udfs(black_box(p), &NoStatistics, &lib, &config)
                            .expect("optimises"),
                    );
                }
            });
        });
        for (shape, plan) in &shapes {
            c.bench_function(&format!("join_{shape}_{name}"), |b| {
                b.iter(|| {
                    black_box(
                        optimize_with_udfs(black_box(plan), &NoStatistics, &lib, &config)
                            .expect("optimises"),
                    )
                });
            });
        }
    }
}

criterion_group!(benches, optimize, enumerators);
criterion_main!(benches);
