// SPDX-License-Identifier: Apache-2.0
//! `RollupFromParent` (`krishiv-optimizer` `rollup_chain`, TPC-DS q67's shape): the grouping sets
//! of a pre-aggregated expansion are each merged from their smallest computed parent set, with
//! the rows of the aggregate over the expanded rows.

use std::collections::HashMap;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use krishiv_common::config::Config;
use krishiv_optimizer::{ColumnStats, Statistics, TableStats};
use krishiv_plan::{Literal, ReadSource, Scan};

use super::*;

/// `t` as a 2M-row table whose columns are nearly unique, so the pre-aggregate below the
/// expansion does not reduce, and whose integer ranges prove the sums cannot overflow (eager
/// aggregation moves an `int` sum or average only with them, never a `bigint` one); statistics
/// choose the plan, never its rows.
#[derive(Debug)]
struct T;

impl Statistics for T {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let column = |n: u64, range: Option<(Literal, Literal)>| {
            let (min, max) = range.unzip();
            ColumnStats {
                distinct: Some(n),
                nulls: Some(0),
                min,
                max,
                ..ColumnStats::default()
            }
        };
        (name.last()? == "t").then(|| TableStats {
            rows: 2_000_000,
            columns: HashMap::from([
                (
                    "a".to_owned(),
                    column(
                        1_500_000,
                        Some((Literal::Integer(0), Literal::Integer(2_000_000))),
                    ),
                ),
                (
                    "b".to_owned(),
                    column(
                        2_000_000,
                        Some((Literal::Long(0), Literal::Long(20_000_000))),
                    ),
                ),
                ("s".to_owned(), column(2_000_000, None)),
                ("d".to_owned(), column(1_000_000, None)),
            ]),
        })
    }
}

const FIRES: [&str; 7] = [
    // Every merged call over a ROLLUP of the table's columns, the grand total included.
    "SELECT a, b, s, sum(a), count(*), count(a), min(s), max(b), CAST(avg(a) AS STRING) FROM t \
     GROUP BY ROLLUP(a, b, s)",
    // A grouping expression with repeated values, `grouping_id`, `grouping` and `HAVING`.
    "SELECT a % 5 AS p, b, s, grouping_id() AS g, CAST(grouping(s) AS INT) AS gs, sum(a), count(*) FROM t \
     GROUP BY CUBE(a % 5, b, s) HAVING count(*) > 0",
    // Equal sets keep their own rows.
    "SELECT a, b, s, sum(a) FROM t GROUP BY GROUPING SETS ((a, b, s), (a, b), (a, b), ())",
    // Sets with no common finest set read the pre-aggregate.
    "SELECT a, b, s, count(*), max(b) FROM t GROUP BY GROUPING SETS ((a, b), (s), (a))",
    // Decimal states (q67's `sum` of a decimal: the state is the sum and whether a value was
    // seen), merged through every level: `d` is NULL on some rows, and the CASE column is NULL on
    // every row of some groups, so a set's sum is NULL there, never 0.
    // The CASE conditions are comparisons that cannot raise (a `%` may, so the pre-aggregate
    // refuses it) and share no subexpression, which CSE would compute above the expansion, where
    // the pre-aggregate refuses an argument.
    "SELECT a % 7 AS p, b % 3 AS q, s, CAST(sum(d) AS STRING), CAST(avg(d) AS STRING), \
     CAST(sum(CASE WHEN a < 50 THEN NULL ELSE d END) AS STRING), \
     CAST(avg(CASE WHEN a >= 120 THEN NULL ELSE d END) AS STRING), count(d) FROM t \
     GROUP BY ROLLUP(a % 7, b % 3, s)",
    // A set whose every input `d` is NULL (`a` is NULL on every fourth row and `d` then too).
    "SELECT a, s, CAST(sum(d) AS STRING), CAST(avg(d) AS STRING) FROM t \
     GROUP BY GROUPING SETS ((a, s), (a), ())",
    // q67's argument: exact decimal arithmetic under `coalesce`, NULL where `d` or `a` is.
    "SELECT a % 7 AS p, s, CAST(sum(coalesce(d * a, 0)) AS STRING) FROM t GROUP BY ROLLUP(a % 7, s)",
];

/// `t` as [`env_of`]'s table plus `d decimal(7,2)` (on its canonical `Decimal64` carrier), NULL
/// where `a` is NULL and on every fifth row. A decimal `sum` that overflows cannot reach this
/// rule: eager aggregation moves only a `sum` of precision 28 or less, whose `decimal(38)` state
/// holds at least 10^10 summed values, and every set merges a split of the same rows as the
/// unsplit sum.
fn env_with_decimal(n: i32) -> Env {
    use arrow_array::Decimal64Array;
    let (schema, batches) = table(n);
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new("d", ArrowType::Decimal64(7, 2), true));
    let schema = Arc::new(Schema::new(fields));
    let mut row = 0;
    let batches: Vec<Batch> = batches
        .iter()
        .map(|b| {
            let b = b.data();
            let a = b.column(0).as_primitive::<arrow_array::types::Int32Type>();
            let d: Decimal64Array = (0..b.num_rows())
                .map(|r| {
                    let v = i64::from(row + i32::try_from(r).unwrap());
                    (!a.is_null(r) && v % 5 != 4).then_some(v * 137 - 4_000)
                })
                .collect::<Decimal64Array>()
                .with_precision_and_scale(7, 2)
                .unwrap();
            row += i32::try_from(b.num_rows()).unwrap();
            let mut columns = b.columns().to_vec();
            columns.push(Arc::new(d));
            Batch::new(RecordBatch::try_new(Arc::clone(&schema), columns).unwrap())
        })
        .collect();
    let columns = schema
        .fields()
        .iter()
        .map(|f| TableColumn {
            name: f.name().clone(),
            data_type: krishiv_types::DataType::from_arrow(f.data_type()).unwrap(),
            nullable: f.is_nullable(),
            char_varchar: None,
        })
        .collect();
    let catalog = MemoryCatalog::new("spark_catalog").with_table(&["default"], "t", columns, None);
    let tables = MemoryTables::new().with_table(
        &["spark_catalog", "default", "t"],
        Arc::new(MemoryTable::new(schema, batches)),
    );
    Env {
        ctx: context(catalog, ResolutionConf::default()),
        tables: Arc::new(tables),
    }
}

fn fired(plan: &Plan) -> bool {
    plan.root().to_string().contains("__shared_rollup_")
}

fn rows_under(env: &Env, sql: &str, on: bool) -> Vec<String> {
    let mut config = Config::new();
    config
        .set("krishiv.sql.optimizer.shareCteInstances", "true")
        .unwrap();
    config
        .set(
            "krishiv.optimizer.rollupFromParent",
            if on { "true" } else { "false" },
        )
        .unwrap();
    let plan = krishiv_optimizer::optimize_with_udfs(
        &env.analyze(sql),
        &T,
        &krishiv_udf_sdk::UdfLibrary::new(),
        &config,
    )
    .unwrap();
    assert_eq!(fired(&plan), on, "{sql}\n{}", plan.root());
    let cx = PlanningContext::new(Arc::clone(&env.tables) as _, Arc::new(config));
    let runnable = env.runnable(DefaultPlanner.plan(&plan, &cx).unwrap());
    let mut rows = Vec::new();
    for b in &run(&runnable).unwrap_or_else(|e| panic!("{sql}: {e}")) {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    rows
}

/// On or off, the rows are the same; on, the sets read their parents.
#[test]
fn each_set_from_its_parent_keeps_the_rows() {
    let env = env_with_decimal(200);
    for q in FIRES {
        let off = rows_under(&env, q, false);
        assert!(!off.is_empty(), "{q}");
        assert_eq!(rows_under(&env, q, true), off, "{q}");
    }
}

/// An empty table: the grand total's row (or none) is what the expansion gives.
#[test]
fn an_empty_input_keeps_the_rows() {
    let env = env_with_decimal(0);
    for q in FIRES {
        assert_eq!(rows_under(&env, q, true), rows_under(&env, q, false), "{q}");
    }
}
