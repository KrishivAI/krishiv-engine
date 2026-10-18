// SPDX-License-Identifier: Apache-2.0
//! Planning time: the 22 TPC-H queries, parsed and resolved once, then
//! optimised (rule batches, join ordering, pruning); run under CodSpeed in
//! CI.

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
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf};
use krishiv_plan::{Plan, SourceId};
use krishiv_types::parse_data_type;

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

fn plans() -> Vec<Plan> {
    let mut catalog = MemoryCatalog::new("spark_catalog");
    for (name, cols) in TABLES {
        let columns = cols
            .split(", ")
            .map(|c| {
                let (n, t) = c.split_once(' ').expect("name type");
                krishiv_plan::analyzer::TableColumn {
                    name: n.to_owned(),
                    data_type: parse_data_type(t).expect("type"),
                    nullable: true,
                }
            })
            .collect();
        catalog = catalog.with_table(&["default"], name, columns, None);
    }
    let ctx = context(catalog, ResolutionConf::default());
    (1..=22)
        .map(|q| {
            let path = format!(
                "{}/../../bench/queries/tpch/q{q:02}.sql",
                env!("CARGO_MANIFEST_DIR")
            );
            let sql = std::fs::read_to_string(path).expect("query");
            let plan = krishiv_sql::parse_and_lower(
                &sql,
                &krishiv_sql::ParserOptions::default(),
                SourceId(0),
            )
            .expect("parses");
            block_on(DefaultAnalyzer.analyze(&plan, &ctx))
                .expect("resolves")
                .plan
        })
        .collect()
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

criterion_group!(benches, optimize);
criterion_main!(benches);
