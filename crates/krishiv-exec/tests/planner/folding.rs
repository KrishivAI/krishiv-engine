// SPDX-License-Identifier: Apache-2.0
//! Foldable expressions end to end, as Spark 4.2.0 answers them: a `LIMIT`
//! over casts, dates and datetime calls returns rows (the values the
//! analyzer folds, or what the downstream folders reduce to a literal);
//! the legacy `date_add` day count reads through a cast to string; `NaN`
//! and `-0.0` keep their values; `AND`/`OR` short-circuit like Spark's
//! Kleene logic. Each case runs with ANSI on and off.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, SourceId};

use krishiv_exec::planner::{
    DefaultPlanner, MemoryTable, MemoryTables, PhysicalPlanner, PlanningContext,
};

use super::tpch::render;
use super::{Runnable, run};

struct Db {
    ctx: krishiv_plan::analyzer::AnalysisContext,
    tables: Arc<MemoryTables>,
    config: Arc<Config>,
}

fn db(ansi: bool) -> Db {
    let schema = Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int32, true)]));
    let a: ArrayRef = Arc::new(Int32Array::from_iter((0..10).map(Some)));
    let batches = vec![krishiv_types::batch::Batch::new(
        RecordBatch::try_new(Arc::clone(&schema), vec![a]).unwrap(),
    )];
    let columns = schema
        .fields()
        .iter()
        .map(|f| TableColumn {
            name: f.name().clone(),
            data_type: krishiv_types::DataType::from_arrow(f.data_type()).unwrap(),
            nullable: f.is_nullable(),
        })
        .collect();
    let catalog = MemoryCatalog::new("spark_catalog").with_table(&["default"], "t", columns, None);
    let tables = MemoryTables::new().with_table(
        &["spark_catalog", "default", "t"],
        Arc::new(MemoryTable::new(schema, batches)),
    );
    let flag = if ansi { "true" } else { "false" };
    let mut config = Config::new();
    config.set("spark.sql.ansi.enabled", flag).unwrap();
    let conf = ResolutionConf::from_conf(|k| (k == "spark.sql.ansi.enabled").then_some(flag));
    Db {
        ctx: context(catalog, conf),
        tables: Arc::new(tables),
        config: Arc::new(config),
    }
}

/// What a query returned: its rows rendered as Spark renders them, or the
/// error class.
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
                let cells: Vec<String> =
                    b.columns().iter().map(|c| render(c.as_ref(), r)).collect();
                rows.push(cells.join("|"));
            }
        }
        Ok(rows)
    }

    fn rows(&self, sql: &str) -> Outcome {
        let plan = self.analyze(sql);
        let optimized = krishiv_optimizer::optimize(&plan).unwrap();
        self.execute(&optimized)
    }
}

fn digits(n: i32) -> Vec<String> {
    (0..n).map(|v| v.to_string()).collect()
}

/// `LIMIT` over a foldable expression returns rows: the analyzer folds a
/// cast to string, a cast to date, and the datetime calls over them.
#[test]
fn limits_over_foldable_expressions_return_rows() {
    for ansi in [true, false] {
        let db = db(ansi);
        for (sql, want) in [
            (
                "SELECT a FROM t LIMIT length(CAST(12 AS STRING))",
                digits(2),
            ),
            (
                "SELECT a FROM t LIMIT year(DATE'2024-01-01') - 2020",
                digits(4),
            ),
            (
                "SELECT a FROM t LIMIT year(CAST('2024-01-01' AS DATE)) - 2020",
                digits(4),
            ),
            (
                "SELECT a FROM t LIMIT length(concat('a', CAST(1 AS STRING)))",
                digits(2),
            ),
        ] {
            assert_eq!(db.rows(sql), Ok(want), "{sql} (ansi {ansi})");
        }
    }
}

/// A foldable `LIMIT` the analyzer folder cannot evaluate is not an
/// error: it passes through, and what the downstream folders reduce runs.
/// (`IN` folds in the optimizer but not in the analyzer folder.)
#[test]
fn an_unevaluated_foldable_limit_passes_through() {
    for ansi in [true, false] {
        let db = db(ansi);
        assert_eq!(
            db.rows("SELECT a FROM t LIMIT CASE WHEN 1 IN (1, 2) THEN 2 ELSE 3 END"),
            Ok(digits(2)),
            "(ansi {ansi})"
        );
    }
}

/// The legacy day count reads through a cast to string, as Spark's
/// `StringLiteralCoercion` folds it.
#[test]
fn legacy_day_counts_read_through_casts() {
    for ansi in [true, false] {
        let db = db(ansi);
        assert_eq!(
            db.rows("SELECT date_add(DATE'2024-01-01', CAST(1 AS STRING))"),
            Ok(vec!["2024-01-02".to_owned()]),
            "(ansi {ansi})"
        );
    }
}

/// `NaN` and `-0.0` keep their values through casts and comparisons.
#[test]
fn nan_and_signed_zero_keep_their_values() {
    for ansi in [true, false] {
        let db = db(ansi);
        for (sql, want) in [
            ("SELECT CAST('NaN' AS DOUBLE)", vec!["nan"]),
            ("SELECT CAST('-0.0' AS DOUBLE)", vec!["-0.0"]),
            ("SELECT CAST('Infinity' AS DOUBLE)", vec!["inf"]),
            ("SELECT 0.0 = -0.0", vec!["true"]),
        ] {
            let want: Vec<String> = want.into_iter().map(str::to_owned).collect();
            assert_eq!(db.rows(sql), Ok(want), "{sql} (ansi {ansi})");
        }
    }
}

/// `AND`/`OR` short-circuit as Spark's Kleene logic does: `false AND ...`
/// never raises, while `NULL AND ...` raises what the right side raises.
#[test]
fn and_or_short_circuit() {
    for ansi in [true, false] {
        let db = db(ansi);
        assert_eq!(
            db.rows("SELECT false AND 1/0 = 1"),
            Ok(vec!["false".to_owned()]),
            "(ansi {ansi})"
        );
        assert_eq!(
            db.rows("SELECT true OR 1/0 = 1"),
            Ok(vec!["true".to_owned()]),
            "(ansi {ansi})"
        );
    }
    assert_eq!(
        db(true).rows("SELECT NULL AND 1/0 = 1"),
        Err(ErrorClass::DivideByZero),
    );
    assert_eq!(
        db(false).rows("SELECT NULL AND 1/0 = 1"),
        Ok(vec!["NULL".to_owned()]),
    );
}
