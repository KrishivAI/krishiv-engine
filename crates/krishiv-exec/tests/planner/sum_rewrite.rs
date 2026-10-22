// SPDX-License-Identifier: Apache-2.0
//! `SUM(x + c)` rewritten to `SUM(x) + c * COUNT(x)` by the optimizer: the
//! rows (or the error) of the optimised plan equal those of the plan as
//! analyzed, for generated data with NULLs and the extremes of each type,
//! under ANSI and legacy arithmetic, with and without `GROUP BY`, for
//! constants inside and outside the rewrite's bounds.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int8Array, Int16Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{Plan, SourceId};
use krishiv_types::batch::Batch;
use proptest::prelude::*;

use krishiv_exec::planner::{
    DefaultPlanner, MemoryTable, MemoryTables, PhysicalPlanner, PlanningContext,
};

use super::{Runnable, run};

/// One generated row of `s(g, ti, si, i, l)`.
type Row = (
    Option<i32>,
    Option<i8>,
    Option<i16>,
    Option<i32>,
    Option<i64>,
);

struct Db {
    ctx: krishiv_plan::analyzer::AnalysisContext,
    tables: Arc<MemoryTables>,
    config: Arc<Config>,
}

fn db(rows: &[Row], batch: usize, ansi: bool) -> Db {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", ArrowType::Int32, true),
        Field::new("ti", ArrowType::Int8, true),
        Field::new("si", ArrowType::Int16, true),
        Field::new("i", ArrowType::Int32, true),
        Field::new("l", ArrowType::Int64, true),
    ]));
    let mut batches = Vec::new();
    for chunk in rows.chunks(batch.max(1)) {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter(chunk.iter().map(|r| r.0))),
            Arc::new(Int8Array::from_iter(chunk.iter().map(|r| r.1))),
            Arc::new(Int16Array::from_iter(chunk.iter().map(|r| r.2))),
            Arc::new(Int32Array::from_iter(chunk.iter().map(|r| r.3))),
            Arc::new(Int64Array::from_iter(chunk.iter().map(|r| r.4))),
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
    let catalog = MemoryCatalog::new("spark_catalog").with_table(&["default"], "s", columns, None);
    let tables = MemoryTables::new().with_table(
        &["spark_catalog", "default", "s"],
        Arc::new(MemoryTable::new(schema, batches)),
    );
    let mut config = Config::new();
    config
        .set(
            "spark.sql.ansi.enabled",
            if ansi { "true" } else { "false" },
        )
        .unwrap();
    let conf = ResolutionConf::from_conf(|k| {
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
        other => panic!("no test formatting for {other}"),
    }
}

fn int8() -> impl Strategy<Value = Option<i8>> {
    prop_oneof![
        1 => Just(None),
        2 => Just(Some(i8::MIN)),
        2 => Just(Some(i8::MAX)),
        6 => any::<i8>().prop_map(Some),
    ]
}

fn int16() -> impl Strategy<Value = Option<i16>> {
    prop_oneof![
        1 => Just(None),
        2 => Just(Some(i16::MIN)),
        2 => Just(Some(i16::MAX)),
        6 => any::<i16>().prop_map(Some),
    ]
}

fn int32() -> impl Strategy<Value = Option<i32>> {
    prop_oneof![
        1 => Just(None),
        2 => Just(Some(i32::MIN)),
        2 => Just(Some(i32::MAX)),
        6 => any::<i32>().prop_map(Some),
    ]
}

fn int64() -> impl Strategy<Value = Option<i64>> {
    prop_oneof![
        1 => Just(None),
        2 => Just(Some(i64::MIN)),
        2 => Just(Some(i64::MAX)),
        6 => any::<i64>().prop_map(Some),
    ]
}

fn rows() -> impl Strategy<Value = Vec<Row>> {
    let group = prop_oneof![1 => Just(None), 4 => (0..4_i32).prop_map(Some)];
    prop::collection::vec((group, int8(), int16(), int32(), int64()), 0..24)
}

/// A constant inside the bound, or around it.
fn constant() -> impl Strategy<Value = i32> {
    prop_oneof![
        6 => -100..=100_i32,
        2 => Just(65_536),
        2 => Just(-65_536),
        1 => Just(65_537),
        1 => Just(-65_537),
        1 => Just(i32::MAX),
        1 => Just(1_000_000),
    ]
}

/// A column expression, whether the rewrite takes it or not.
const COLUMNS: &[&str] = &[
    "si",
    "ti",
    "CAST(si AS BIGINT)",
    "CAST(ti AS INT)",
    "i",
    "l",
    "CAST(i AS BIGINT)",
    "g",
];

/// A sum of `x` shifted by `c` in one of the four shapes.
fn sum_of(shape: u8, x: &str, c: i32) -> String {
    match shape % 4 {
        0 => format!("SUM({x} + {c})"),
        1 => format!("SUM({c} + {x})"),
        2 => format!("SUM({x} - {c})"),
        _ => format!("SUM({c} - {x})"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// The optimised plan returns what the analyzed plan does: the same
    /// rows, or the same error class, whether the sums were rewritten.
    #[test]
    fn rewritten_sums_equal_the_original(
        data in rows(),
        batch in 1..8_usize,
        ansi in any::<bool>(),
        grouped in any::<bool>(),
        picks in prop::collection::vec((0..COLUMNS.len(), any::<u8>(), constant()), 1..4),
    ) {
        let db = db(&data, batch, ansi);
        let items: Vec<String> = picks
            .iter()
            .map(|(col, shape, c)| sum_of(*shape, COLUMNS[*col], *c))
            .collect();
        let sql = if grouped {
            format!("SELECT g, {} FROM s GROUP BY g", items.join(", "))
        } else {
            format!("SELECT {} FROM s", items.join(", "))
        };
        let (original, optimized, _) = db.both(&sql);
        prop_assert_eq!(original, optimized, "{} (ansi {})", sql, ansi);
    }
}

/// The rewrite really runs on the shapes it takes (the property above
/// would also hold if it never applied).
#[test]
fn the_rewrite_applies_and_agrees() {
    for ansi in [true, false] {
        let rows: Vec<Row> = vec![
            (Some(1), Some(i8::MAX), Some(i16::MAX), Some(1), Some(1)),
            (Some(1), Some(i8::MIN), Some(i16::MIN), None, None),
            (Some(2), None, None, Some(i32::MAX), Some(i64::MAX)),
            (None, Some(7), Some(-7), Some(i32::MIN), Some(i64::MIN)),
        ];
        let db = db(&rows, 2, ansi);
        for sql in [
            "SELECT SUM(si + 65536), SUM(si - 65536), SUM(65536 - si), SUM(si + 1), SUM(si) FROM s",
            "SELECT g, SUM(ti + 1), SUM(CAST(si AS BIGINT) - 3) FROM s GROUP BY g",
            "SELECT SUM(si + 1) FROM s WHERE si > 100000",
        ] {
            let (original, optimized, text) = db.both(sql);
            assert!(text.contains("count("), "{sql}\n{text}");
            assert_eq!(original, optimized, "{sql} (ansi {ansi})");
            assert!(original.is_ok(), "{sql}: {original:?}");
        }
    }
    // No rows: the sum is NULL, as the shifted sum is.
    let empty = db(&[], 1, true);
    let (original, optimized, text) = empty.both("SELECT SUM(si + 1) FROM s");
    assert!(text.contains("count("), "{text}");
    assert_eq!(original, Ok(vec!["NULL".to_owned()]));
    assert_eq!(optimized, original);
}

/// Where `x + c` can overflow per row the sum is left alone, so Spark's
/// error (ANSI) or wrap (legacy) is kept.
#[test]
fn per_row_overflow_is_kept() {
    let rows: Vec<Row> = vec![(Some(0), None, None, Some(i32::MAX), Some(i64::MAX))];
    for sql in ["SELECT SUM(i + 1) FROM s", "SELECT SUM(l + 1) FROM s"] {
        let ansi = db(&rows, 1, true);
        let (original, optimized, text) = ansi.both(sql);
        assert!(!text.contains("count("), "{sql}\n{text}");
        assert_eq!(original, Err(ErrorClass::ArithmeticOverflow), "{sql}");
        assert_eq!(optimized, original, "{sql}");
        let legacy = db(&rows, 1, false);
        let (original, optimized, _) = legacy.both(sql);
        assert!(original.is_ok(), "{sql}: {original:?}");
        assert_eq!(optimized, original, "{sql}");
    }
}
