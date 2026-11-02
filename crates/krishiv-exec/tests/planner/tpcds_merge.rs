// SPDX-License-Identifier: Apache-2.0
//! The planner never chooses a sort-merge join for unsorted inputs unless
//! told to (P2-T056 acceptance #5): no `SortMergeJoin` in the default plans
//! of the TPC-DS queries over tables of SF10's sizes, and one wherever an
//! equi-join is planned with `krishiv.exec.join.strategy=sortMerge`.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_exec::planner::{
    DefaultPlanner, MemoryTable, MemoryTables, PhysicalPlanner as _, PlanningContext,
};
use krishiv_optimizer::TableStats;
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use serde_json::Value;

/// TPC-DS SF10 row counts of the tables a join's size estimate reads; the
/// others hold a few thousand rows.
const ROWS: &[(&str, u64)] = &[
    ("store_sales", 28_800_991),
    ("catalog_sales", 14_401_261),
    ("web_sales", 7_197_566),
    ("store_returns", 2_875_432),
    ("catalog_returns", 1_439_749),
    ("web_returns", 719_217),
    ("inventory", 133_110_000),
    ("customer", 500_000),
    ("customer_address", 250_000),
    ("customer_demographics", 1_920_800),
    ("date_dim", 73_049),
    ("time_dim", 86_400),
    ("item", 102_000),
    ("household_demographics", 7_200),
];

/// The TPC-DS tables, empty, with SF10's row counts as their statistics.
fn tables() -> (MemoryCatalog, Arc<MemoryTables>) {
    let path =
        super::tpch::root().join("tests/fixtures/spark-4.2.0/golden-tables/tpcds-schemas.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut catalog = MemoryCatalog::new("spark_catalog");
    let mut tables = MemoryTables::new();
    for table in doc["tables"].as_array().unwrap() {
        let name = table["name"].as_str().unwrap();
        let columns: Vec<TableColumn> = table["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                // `CHAR`/`VARCHAR` read as `STRING`, as the optimizer's
                // TPC-DS snapshots do.
                let ty = c["type"].as_str().unwrap();
                let upper = ty.to_ascii_uppercase();
                let ty = if upper.starts_with("CHAR") || upper.starts_with("VARCHAR") {
                    "string"
                } else {
                    ty
                };
                TableColumn {
                    name: c["name"].as_str().unwrap().to_owned(),
                    data_type: krishiv_types::parse_data_type(ty).unwrap(),
                    nullable: c["nullable"].as_bool().unwrap(),
                }
            })
            .collect();
        let schema = Arc::new(arrow_schema::Schema::new(
            columns
                .iter()
                .map(|c| c.data_type.to_arrow_field(c.name.clone(), c.nullable))
                .collect::<Vec<_>>(),
        ));
        let rows = ROWS
            .iter()
            .find_map(|(t, rows)| (*t == name).then_some(*rows))
            .unwrap_or(5_000);
        let stats = TableStats {
            rows,
            columns: HashMap::new(),
        };
        catalog = catalog.with_table(&["default"], name, columns, None);
        tables = tables.with_table(
            &["spark_catalog", "default", name],
            Arc::new(MemoryTable::new(schema, Vec::new()).with_statistics(Some(stats))),
        );
    }
    (catalog, Arc::new(tables))
}

/// The physical plan of TPC-DS query `q` as text, `None` for a query that
/// does not parse or resolve yet.
fn plan(
    q: usize,
    ctx: &krishiv_plan::analyzer::AnalysisContext,
    planning: &PlanningContext,
) -> Option<String> {
    let path = super::tpch::root().join(format!("bench/queries/tpcds/q{q:02}.sql"));
    let text = std::fs::read_to_string(path).unwrap();
    let parsed =
        krishiv_sql::parse_and_lower(&text, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .ok()?;
    let analyzed = block_on(DefaultAnalyzer.analyze(&parsed, ctx)).ok()?;
    let optimized = krishiv_optimizer::optimize(&analyzed.plan).unwrap();
    let physical = DefaultPlanner
        .plan(&optimized, planning)
        .unwrap_or_else(|e| panic!("q{q:02}: {e}"));
    Some(physical.to_string())
}

#[test]
fn tpcds_default_plans_have_no_sort_merge_join() {
    let (catalog, tables) = tables();
    let ctx = context(catalog, ResolutionConf::default());
    let default = PlanningContext::new(Arc::clone(&tables) as _, Arc::new(Config::new()));
    let mut forced = Config::new();
    forced
        .set("krishiv.exec.join.strategy", "sortMerge")
        .unwrap();
    let forced = PlanningContext::new(Arc::clone(&tables) as _, Arc::new(forced));
    let (mut planned, mut merges) = (0, 0);
    for q in 1..=99 {
        let Some(unforced) = plan(q, &ctx, &default) else {
            continue;
        };
        planned += 1;
        assert!(!unforced.contains("SortMergeJoin"), "q{q:02}\n{unforced}");
        let merged = plan(q, &ctx, &forced).unwrap_or_else(|| panic!("q{q:02} forced"));
        if unforced.contains("HashJoin") {
            merges += usize::from(merged.contains("SortMergeJoin"));
        }
    }
    // The queries the optimizer's snapshots list as unsupported do not plan;
    // of the others only q09 and q28 have no hash join (scalar subqueries and
    // cross joins), and every one that has one merges when forced.
    assert!(planned >= 90, "{planned} queries planned");
    assert!(
        merges >= 88,
        "{merges} queries plan a sort-merge join when forced"
    );
}
