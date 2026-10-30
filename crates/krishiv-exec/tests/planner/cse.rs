// SPDX-License-Identifier: Apache-2.0
//! Common-subexpression elimination under ANSI and legacy arithmetic: the
//! optimised plan returns what the analyzed plan does (the same rows, or
//! the same error class), and hoisting never moves an error to rows Spark
//! would not evaluate.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, SourceId};
use krishiv_types::batch::Batch;

use krishiv_exec::planner::{
    DefaultPlanner, MemoryTable, MemoryTables, PhysicalPlanner, PlanningContext,
};

use super::{Runnable, run};

/// One row of `t(a int, b int)`: a zero and a NULL divisor exercise the
/// guarded and unguarded divisions alike.
type Row = (Option<i32>, Option<i32>);

struct Db {
    ctx: krishiv_plan::analyzer::AnalysisContext,
    tables: Arc<MemoryTables>,
    config: Arc<Config>,
}

fn make_db(rows: &[Row], batch: usize, ansi: bool) -> Db {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, true),
        Field::new("b", ArrowType::Int32, true),
    ]));
    let mut batches = Vec::new();
    for chunk in rows.chunks(batch.max(1)) {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter(chunk.iter().map(|r| r.0))),
            Arc::new(Int32Array::from_iter(chunk.iter().map(|r| r.1))),
        ];
        batches.push(Batch::new(
            RecordBatch::try_new(Arc::clone(&schema), cols).unwrap(),
        ));
    }
    let columns = schema
        .fields()
        .iter()
        .map(|f| TableColumn {
            name: f.name().clone(),
            data_type: krishiv_types::DataType::from_arrow(f.data_type()).unwrap(),
            nullable: true,
        })
        .collect();
    let catalog = MemoryCatalog::new("spark_catalog").with_table(&["default"], "t", columns, None);
    let tables = MemoryTables::new().with_table(
        &["spark_catalog", "default", "t"],
        Arc::new(MemoryTable::new(schema, batches)),
    );
    let mut config = Config::new();
    config
        .set(
            "spark.sql.ansi.enabled",
            if ansi { "true" } else { "false" },
        )
        .unwrap();
    let conf = ResolutionConf::from_conf(move |k| {
        (k == "spark.sql.ansi.enabled").then_some(if ansi { "true" } else { "false" })
    });
    Db {
        ctx: context(catalog, conf),
        tables: Arc::new(tables),
        config: Arc::new(config),
    }
}

/// What a plan returned: its rows sorted, or the error class.
type Outcome = Result<Vec<String>, ErrorClass>;

impl Db {
    fn analyze(&self, sql: &str) -> Plan {
        let plan =
            krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .unwrap_or_else(|e| panic!("{sql}: {e}"));
        block_on(DefaultAnalyzer.analyze(&plan, &self.ctx))
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .plan
    }

    fn execute(&self, plan: &Plan) -> Outcome {
        let cx = PlanningContext::new(Arc::clone(&self.tables) as _, Arc::clone(&self.config));
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

    /// The rows of `sql` as analyzed, and as optimised (also returned as
    /// text).
    fn both(&self, sql: &str) -> (Outcome, Outcome, String) {
        let plan = self.analyze(sql);
        let optimized = krishiv_optimizer::optimize(&plan).unwrap();
        (
            self.execute(&plan),
            self.execute(&optimized),
            optimized.root().to_string(),
        )
    }
}

fn show(a: &ArrayRef, row: usize) -> String {
    if a.is_null(row) {
        return "NULL".to_owned();
    }
    match a.data_type() {
        ArrowType::Int32 => a.as_primitive::<Int32Type>().value(row).to_string(),
        ArrowType::Int64 => a.as_primitive::<Int64Type>().value(row).to_string(),
        ArrowType::Boolean => a.as_boolean().value(row).to_string(),
        ArrowType::Float64 => a
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(row)
            .to_string(),
        other => panic!("no test formatting for {other}"),
    }
}

fn rows() -> Vec<Row> {
    vec![
        (Some(4), Some(2)),
        (Some(4), Some(0)),
        (None, Some(2)),
        (Some(4), None),
        (Some(-4), Some(0)),
    ]
}

/// A guarded division repeated twice is not hoisted: with `b = 0` rows
/// present, nothing raises in either mode. The query also repeats `a + b`,
/// which is hoisted, so the test fails without the rule too.
#[test]
fn guarded_division_never_hoists_nor_raises() {
    let sql = "SELECT CASE WHEN b <> 0 THEN a / b ELSE NULL END AS x, CASE WHEN b <> 0 THEN a / b ELSE NULL END AS y, (a + b) + (a + b) AS z FROM t";
    for ansi in [false, true] {
        let db = make_db(&rows(), 2, ansi);
        let (want, got, text) = db.both(sql);
        assert_eq!(got, want, "ansi={ansi}");
        assert!(text.contains("AS cse_"), "{text}");
        assert_eq!(text.matches("CASE WHEN").count(), 1, "{text}");
    }
}

/// A surely-evaluated division hoisted next to a conditional use still
/// raises where it raised: `a / b` is computed once per row, so under
/// ANSI both plans raise `DIVIDE_BY_ZERO` on the `b = 0` rows, while
/// legacy mode gives NULLs in both.
#[test]
fn surely_evaluated_division_still_raises() {
    let sql = "SELECT a / b AS x, CASE WHEN a > 0 THEN a / b ELSE NULL END AS y FROM t";
    let db = make_db(&rows(), 2, true);
    let (want, got, text) = db.both(sql);
    assert_eq!(want, Err(ErrorClass::DivideByZero), "ansi raises: {want:?}");
    assert_eq!(got, want, "ansi raises alike: {text}");
    assert!(text.contains("AS cse_"), "{text}");
    let db = make_db(&rows(), 2, false);
    let (want, got, _) = db.both(sql);
    assert_eq!(got, want, "legacy nulls alike");
    assert!(want.is_ok(), "{want:?}");
}

/// A `try_divide` repeated twice is never extracted: the results match in
/// both modes, and no `cse_*` column computes a `try_*` call. The query
/// also repeats `a + b`, which is hoisted, so the test fails without the
/// rule too.
#[test]
fn try_divide_is_never_extracted() {
    let sql = "SELECT try_divide(a, b) AS x, try_divide(a, b) AS y, (a + b) + (a + b) AS z FROM t";
    for ansi in [false, true] {
        let db = make_db(&rows(), 2, ansi);
        let (want, got, text) = db.both(sql);
        assert_eq!(got, want, "ansi={ansi}");
        assert_eq!(text.matches("try_divide").count(), 2, "{text}");
        assert!(text.contains("AS cse_"), "{text}");
    }
}

/// A rewritten filter under a `UNION ALL` branch still plans and returns
/// the same rows: the union reads its inputs by position, so the
/// extracted columns must not reach it.
#[test]
fn a_filtered_union_branch_plans_and_matches() {
    let sql = "SELECT a, b FROM t WHERE (a + b) > 1 AND (a + b) < 9 UNION ALL SELECT a, b FROM t";
    for ansi in [false, true] {
        let db = make_db(&rows(), 2, ansi);
        let (want, got, text) = db.both(sql);
        assert!(want.is_ok(), "{want:?}");
        assert_eq!(got, want, "ansi={ansi}: {text}");
        assert!(text.contains("AS cse_"), "{text}");
    }
}

/// A post-aggregation scalar over a grouping key is evaluated above the
/// aggregate, so it never reads an extracted column.
#[test]
fn post_aggregation_scalars_plan_and_match() {
    let sql = "SELECT b + 1 AS k, sum(b + 1) AS s, count(b + 1) AS n FROM t GROUP BY b";
    for ansi in [false, true] {
        let db = make_db(&rows(), 2, ansi);
        let (want, got, text) = db.both(sql);
        assert!(want.is_ok(), "{want:?}");
        assert_eq!(got, want, "ansi={ansi}: {text}");
        assert!(text.contains("AS cse_"), "{text}");
    }
}

/// `IN` list items are evaluated only when the value is not NULL and no
/// earlier item matched, so a repeated division there is not hoisted: with
/// `a = 0` or `b = 0` rows nothing raises under ANSI, as in the analyzed
/// plan.
#[test]
fn in_items_never_hoist_a_division_that_raises() {
    let rows = vec![
        (Some(0), Some(0)),
        (Some(0), Some(2)),
        (None, Some(0)),
        (Some(1), Some(0)),
    ];
    let sql = "SELECT a IN (0, 1 / b) AS p, b IN (0, 1 / b) AS q FROM t";
    for ansi in [false, true] {
        let db = make_db(&rows, 2, ansi);
        let (want, got, text) = db.both(sql);
        assert_eq!(got, want, "ansi={ansi}: {text}");
    }
}
