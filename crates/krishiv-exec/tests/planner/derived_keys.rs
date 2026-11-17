// SPDX-License-Identifier: Apache-2.0
//! `DerivedKeysAbove` (P3-A part 2c, ClickBench q36) through the planner and the executor:
//! `GROUP BY c, c - 1, c - 2` grouped by `c` alone returns the rows of the plan with the switch
//! off over data with NULL keys and duplicates; with a key at `int`'s maximum, ANSI refuses the
//! rewrite and both plans raise the same error, without ANSI the rewrite runs and both wrap alike.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_optimizer::{ColumnStats, Statistics, TableStats};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Literal, Plan, ReadSource, Scan, SourceId};
use krishiv_types::batch::Batch;
use krishiv_udf_sdk::UdfLibrary;

use krishiv_exec::planner::{
    DefaultPlanner, MemoryTable, MemoryTables, PhysicalPlanner, PlanningContext,
};

use super::{Runnable, run};

const QUERY: &str = "SELECT c, c - 1 AS d1, c - 2 AS d2, count(*) AS n, sum(v) AS sv FROM h \
                     GROUP BY c, c - 1, c - 2";

/// What a plan returned: its rows sorted, or the error class.
type Outcome = Result<Vec<String>, ErrorClass>;

/// Statistics of `h`: a million rows, the range of `c` as given.
#[derive(Debug)]
struct Range(i32, i32);

impl Statistics for Range {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        if name.last()? != "h" {
            return None;
        }
        let columns = scan.output.iter().map(|c| {
            let (min, max) = if c.name == "c" {
                (
                    Some(Literal::Integer(self.0)),
                    Some(Literal::Integer(self.1)),
                )
            } else {
                (None, None)
            };
            let stats = ColumnStats {
                distinct: Some(1_000),
                nulls: Some(0),
                min,
                max,
                ..ColumnStats::default()
            };
            (c.name.clone(), stats)
        });
        Some(TableStats {
            rows: 1_000_000,
            columns: columns.collect(),
        })
    }
}

struct Db {
    ctx: krishiv_plan::analyzer::AnalysisContext,
    tables: Arc<MemoryTables>,
    ansi: bool,
}

/// `h(c int, v bigint)` over `rows`, in batches of three.
fn db(rows: &[(Option<i32>, i64)], ansi: bool) -> Db {
    let schema = Arc::new(Schema::new(vec![
        Field::new("c", ArrowType::Int32, true),
        Field::new("v", ArrowType::Int64, true),
    ]));
    let batches = rows
        .chunks(3)
        .map(|chunk| {
            let cols: Vec<ArrayRef> = vec![
                Arc::new(Int32Array::from_iter(chunk.iter().map(|r| r.0))),
                Arc::new(Int64Array::from_iter(chunk.iter().map(|r| Some(r.1)))),
            ];
            Batch::new(RecordBatch::try_new(Arc::clone(&schema), cols).unwrap())
        })
        .collect();
    let columns = schema
        .fields()
        .iter()
        .map(|f| TableColumn {
            name: f.name().clone(),
            data_type: krishiv_types::DataType::from_arrow(f.data_type()).unwrap(),
            nullable: true,
            char_varchar: None,
        })
        .collect();
    let catalog = MemoryCatalog::new("spark_catalog").with_table(&["default"], "h", columns, None);
    let tables = MemoryTables::new().with_table(
        &["spark_catalog", "default", "h"],
        Arc::new(MemoryTable::new(schema, batches)),
    );
    let conf = ResolutionConf::from_conf(|k| {
        (k == "spark.sql.ansi.enabled").then_some(if ansi { "true" } else { "false" })
    });
    Db {
        ctx: context(catalog, conf),
        tables: Arc::new(tables),
        ansi,
    }
}

impl Db {
    fn config(&self, on: bool) -> Config {
        let mut config = Config::new();
        config
            .set(
                "spark.sql.ansi.enabled",
                if self.ansi { "true" } else { "false" },
            )
            .unwrap();
        config
            .set(
                "krishiv.optimizer.derivedKeysAbove",
                if on { "true" } else { "false" },
            )
            .unwrap();
        config
    }

    fn analyze(&self, sql: &str) -> Plan {
        let plan =
            krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .unwrap_or_else(|e| panic!("{sql}: {e}"));
        block_on(DefaultAnalyzer.analyze(&plan, &self.ctx))
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .plan
    }

    /// `sql` optimized with the switch `on`, under statistics `range`.
    fn optimize(&self, sql: &str, range: &Range, on: bool) -> Plan {
        krishiv_optimizer::optimize_with_udfs(
            &self.analyze(sql),
            range,
            &UdfLibrary::new(),
            &self.config(on),
        )
        .unwrap()
    }

    fn execute(&self, plan: &Plan) -> Outcome {
        let cx = PlanningContext::new(Arc::clone(&self.tables) as _, Arc::new(self.config(false)));
        let physical = DefaultPlanner
            .plan(plan, &cx)
            .map_err(|e: KrishivError| e.error_class())?;
        let runnable = Runnable::new(physical, Arc::clone(&self.tables) as _);
        let batches = run(&runnable).map_err(|e| e.error_class())?;
        let mut rows = Vec::new();
        for b in &batches {
            for r in 0..b.num_rows() {
                let cells: Vec<String> = b.columns().iter().map(|c| show(c, r)).collect();
                rows.push(cells.join("|"));
            }
        }
        rows.sort();
        Ok(rows)
    }
}

fn show(a: &ArrayRef, row: usize) -> String {
    if a.is_null(row) {
        return "NULL".to_owned();
    }
    match a.data_type() {
        ArrowType::Int32 => a.as_primitive::<Int32Type>().value(row).to_string(),
        ArrowType::Int64 => a.as_primitive::<Int64Type>().value(row).to_string(),
        other => panic!("no test formatting for {other}"),
    }
}

/// The grouping line of the first aggregate in the plan's text.
fn keys(plan: &Plan) -> String {
    plan.root()
        .to_string()
        .lines()
        .find(|l| l.contains("Aggregate"))
        .unwrap()
        .to_owned()
}

fn small() -> Vec<(Option<i32>, i64)> {
    vec![
        (Some(5), 1),
        (None, 2),
        (Some(5), 3),
        (Some(-7), 4),
        (None, 5),
        (Some(0), 6),
        (Some(5), 7),
        (Some(1_000_000), 8),
    ]
}

/// Over a range the proof holds for, the rule fires; its rows are the switch-off plan's.
#[test]
fn rewritten_keys_return_the_rows_of_the_plan_without_the_rule() {
    let db = db(&small(), true);
    let range = Range(-1_000_000, 1_000_000);
    let on = db.optimize(QUERY, &range, true);
    let off = db.optimize(QUERY, &range, false);
    assert!(keys(&on).contains("Aggregate [c#0], "), "{}", on.root());
    assert!(keys(&off).contains("c#0 - 1"), "{}", off.root());
    let want = db.execute(&off);
    assert_eq!(db.execute(&on), want);
    assert_eq!(db.execute(&db.analyze(QUERY)), want);
    // Four non-NULL keys and the NULL one, each group once.
    let rows = want.unwrap();
    assert_eq!(rows.len(), 5, "{rows:?}");
    assert!(rows.contains(&"5|4|3|3|11".to_owned()), "{rows:?}");
    assert!(rows.contains(&"NULL|NULL|NULL|2|7".to_owned()), "{rows:?}");
}

/// A key at `int`'s maximum under ANSI: the range proves nothing, the rule refuses and both plans
/// raise the overflow.
#[test]
fn ansi_with_a_key_at_int_max_refuses_and_both_plans_raise() {
    let mut rows = small();
    rows.push((Some(i32::MIN), 9));
    let db = db(&rows, true);
    let range = Range(i32::MIN, i32::MAX);
    let on = db.optimize(QUERY, &range, true);
    let off = db.optimize(QUERY, &range, false);
    assert!(keys(&on).contains("c#0 - 1"), "{}", on.root());
    let want = db.execute(&off);
    assert_eq!(want, Err(ErrorClass::ArithmeticOverflow));
    assert_eq!(db.execute(&on), want);
    assert_eq!(db.execute(&db.analyze(QUERY)), want);
}

/// Without ANSI the same data and range rewrite, and both plans wrap alike.
#[test]
fn without_ansi_a_key_at_int_min_wraps_alike_after_the_rewrite() {
    let mut rows = small();
    rows.push((Some(i32::MIN), 9));
    rows.push((Some(i32::MAX), 10));
    let db = db(&rows, false);
    let range = Range(i32::MIN, i32::MAX);
    let on = db.optimize(QUERY, &range, true);
    let off = db.optimize(QUERY, &range, false);
    assert!(keys(&on).contains("Aggregate [c#0], "), "{}", on.root());
    let want = db.execute(&off);
    assert!(want.is_ok(), "{want:?}");
    assert_eq!(db.execute(&on), want);
    assert_eq!(db.execute(&db.analyze(QUERY)), want);
    // `int_min - 1` wrapped to `int_max`.
    assert!(
        want.unwrap()
            .iter()
            .any(|r| r.starts_with("-2147483648|2147483647|2147483646|")),
    );
}

/// The derived column of a derived table, a `HAVING` on it, and an ordered limit.
#[test]
fn a_derived_table_column_with_having_and_limit() {
    let db = db(&small(), true);
    let range = Range(-1_000_000, 1_000_000);
    let sql = "SELECT c, d, count(*) AS n FROM (SELECT c, c - 1 AS d, v FROM h) \
               GROUP BY c, d HAVING d > 0 ORDER BY n DESC, c LIMIT 2";
    let on = db.optimize(sql, &range, true);
    let off = db.optimize(sql, &range, false);
    assert!(!on.structurally_eq(&off), "{}", on.root());
    let want = db.execute(&off);
    assert_eq!(db.execute(&on), want);
    // The rows come back sorted as text.
    assert_eq!(want.unwrap(), ["1000000|999999|1", "5|4|3"]);
}
