// SPDX-License-Identifier: Apache-2.0
//! Group keys computed from other column keys (`GROUP BY g, g - 1`) leave
//! the keys (the optimizer's `dependent_keys`): the rows (or the error) of
//! the optimised plan equal those of the plan as analyzed, for generated
//! data with NULLs and the extremes of each type, under ANSI and legacy
//! arithmetic, so a key that overflows for some row raises as before.

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
            char_varchar: None,
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
        ArrowType::Utf8 => a.as_string::<i32>().value(row).to_owned(),
        ArrowType::Utf8View => a.as_string_view().value(row).to_owned(),
        other => panic!("no test formatting for {other}"),
    }
}

fn int32() -> impl Strategy<Value = Option<i32>> {
    prop_oneof![
        1 => Just(None),
        2 => Just(Some(i32::MIN)),
        2 => Just(Some(i32::MAX)),
        6 => (-3..3_i32).prop_map(Some),
    ]
}

fn int64() -> impl Strategy<Value = Option<i64>> {
    prop_oneof![
        1 => Just(None),
        2 => Just(Some(i64::MIN)),
        2 => Just(Some(i64::MAX)),
        6 => (-3..3_i64).prop_map(Some),
    ]
}

fn rows() -> impl Strategy<Value = Vec<Row>> {
    let group = prop_oneof![1 => Just(None), 4 => (0..4_i32).prop_map(Some)];
    let small8 = prop_oneof![1 => Just(None), 4 => (-2..2_i8).prop_map(Some)];
    let small16 = prop_oneof![1 => Just(None), 4 => (-2..2_i16).prop_map(Some)];
    prop::collection::vec((group, small8, small16, int32(), int64()), 0..24)
}

/// Dependent keys over `i` (an `INT` holding its extremes) and `l`, some of
/// which overflow under ANSI for some rows.
const DEPENDENT: &[&str] = &[
    "i - 1",
    "i + 1",
    "i * 2",
    "abs(i)",
    "-i",
    "CAST(i AS BIGINT) + 1",
    "l - 1",
    "i + l",
    "i % 3",
    "CAST(i AS STRING)",
    "coalesce(i, 0)",
    "'k'",
];

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// The optimised plan returns what the analyzed plan does: the same
    /// rows, or the same error class.
    #[test]
    fn dropped_keys_give_the_same_rows_or_error(
        data in rows(),
        batch in 1..8_usize,
        ansi in any::<bool>(),
        picks in prop::collection::vec(0..DEPENDENT.len(), 1..4),
        having in any::<bool>(),
    ) {
        let db = db(&data, batch, ansi);
        let dependent: Vec<&str> = picks.iter().map(|k| DEPENDENT[*k]).collect();
        let keys = format!("i, l, {}", dependent.join(", "));
        // A filter that drops groups above the aggregate must not hide an
        // error the dropped key raises.
        let filter = if having { " HAVING COUNT(*) > 1" } else { "" };
        let sql = format!(
            "SELECT {keys}, COUNT(*), SUM(ti) FROM s GROUP BY {keys}{filter}"
        );
        let (original, optimized, _) = db.both(&sql);
        prop_assert_eq!(original, optimized, "{} (ansi {})", sql, ansi);
    }
}

#[test]
fn q36_shape_drops_three_keys_and_raises_on_overflow() {
    let sql = "SELECT i, i - 1, i - 2, i - 3, COUNT(*) AS c FROM s \
               GROUP BY i, i - 1, i - 2, i - 3 HAVING COUNT(*) > 5 ORDER BY c DESC LIMIT 10";
    let rows = vec![
        (Some(0), Some(0), Some(0), Some(7), Some(0)),
        (Some(0), Some(0), Some(0), Some(i32::MIN + 1), Some(0)),
    ];
    let db = db(&rows, 1, true);
    let (original, optimized, _) = db.both(sql);
    // `i32::MIN + 1 - 2` overflows: an error both ways, although the
    // group's count of 1 fails the HAVING.
    assert!(original.is_err(), "{original:?}");
    assert_eq!(original, optimized);
    let legacy = db_legacy(&rows);
    let (original, optimized, _) = legacy.both(sql);
    assert!(original.is_ok(), "{original:?}");
    assert_eq!(original, optimized);
}

fn db_legacy(rows: &[Row]) -> Db {
    db(rows, 1, false)
}
