// SPDX-License-Identifier: Apache-2.0
//! Comparisons and value-merging operators over one Spark `STRING` stored
//! in different Arrow encodings: `u8` keeps its `s` column as `Utf8`,
//! `uv` as `Utf8View` (string literals always arrive as `Utf8View`).
//! Every operator below used to fail planning with
//! `comparison of Utf8 with Utf8View` (or the join's
//! `join keys differ in type`); mixed operands now meet in the canonical
//! carrier, and operands of different Spark types still fail.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BinaryArray, Int32Array, RecordBatch, StringArray, StringViewArray,
};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_plan::{
    BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, JoinCondition, JoinKind,
    LogicalPlan, Plan, PlanNode, ResolvedExpr, ResolvedFunction,
};

use super::{Runnable, run};
use krishiv_exec::planner::{
    DefaultPlanner, MemoryTable, MemoryTables, PhysicalPlanner, PlanningContext,
    TableProviderResolver,
};
use krishiv_types::DataType;
use krishiv_types::batch::Batch;

/// An analysis and planning context with name resolution for `tables`.
struct Cx {
    ctx: krishiv_plan::analyzer::AnalysisContext,
    tables: Arc<dyn TableProviderResolver>,
    config: Arc<Config>,
}

/// `Cx` over the `u8` (`Utf8`) and `uv` (`Utf8View`) tables, ANSI on.
fn cx() -> Cx {
    fn table(
        storage: &ArrowType,
        strings: Vec<Option<String>>,
    ) -> (Vec<TableColumn>, Arc<MemoryTable>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", ArrowType::Int32, true),
            Field::new("s", storage.clone(), true),
        ]));
        let a: Vec<Option<i32>> = (1..=i32::try_from(strings.len()).unwrap())
            .map(Some)
            .collect();
        let s: ArrayRef = match storage {
            ArrowType::Utf8 => Arc::new(StringArray::from(strings)),
            _ => Arc::new(StringViewArray::from(strings)),
        };
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(a)), s])
                .unwrap();
        let columns = vec![
            TableColumn {
                name: "a".to_owned(),
                data_type: krishiv_types::DataType::Integer,
                nullable: true,
            },
            TableColumn {
                name: "s".to_owned(),
                data_type: krishiv_types::DataType::String,
                nullable: true,
            },
        ];
        (
            columns,
            Arc::new(MemoryTable::new(schema, vec![Batch::new(batch)])),
        )
    }

    let strings8 = vec![
        Some("apple".to_owned()),
        Some("banana".to_owned()),
        None,
        Some("cherry".to_owned()),
    ];
    let strings_view = vec![
        Some("banana".to_owned()),
        Some("x".to_owned()),
        None,
        Some("cherry".to_owned()),
    ];
    let (cols8, t8) = table(&ArrowType::Utf8, strings8);
    let (cols_view, tview) = table(&ArrowType::Utf8View, strings_view);
    // `ub(a INT, b BINARY)`: a binary key carrier for joins against the
    // string tables below.
    let b_schema = Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, true),
        Field::new("b", ArrowType::Binary, true),
    ]));
    let b_batch = RecordBatch::try_new(
        Arc::clone(&b_schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(2), None, Some(4)])),
            Arc::new(BinaryArray::from(vec![
                Some(b"apple".as_slice()),
                Some(b"banana".as_slice()),
                None,
                Some(b"cherry".as_slice()),
            ])) as ArrayRef,
        ],
    )
    .unwrap();
    let b_cols = vec![
        TableColumn {
            name: "a".to_owned(),
            data_type: DataType::Integer,
            nullable: true,
        },
        TableColumn {
            name: "b".to_owned(),
            data_type: DataType::Binary,
            nullable: true,
        },
    ];
    let catalog = MemoryCatalog::new("spark_catalog")
        .with_table(&["default"], "u8", cols8, None)
        .with_table(&["default"], "uv", cols_view, None)
        .with_table(&["default"], "ub", b_cols, None);
    let mut config = Config::new();
    config.set("spark.sql.ansi.enabled", "true").unwrap();
    Cx {
        ctx: context(
            catalog,
            ResolutionConf {
                ansi: true,
                ..ResolutionConf::default()
            },
        ),
        tables: Arc::new(
            MemoryTables::new()
                .with_table(&["spark_catalog", "default", "u8"], t8)
                .with_table(&["spark_catalog", "default", "uv"], tview)
                .with_table(
                    &["spark_catalog", "default", "ub"],
                    Arc::new(MemoryTable::new(b_schema, vec![Batch::new(b_batch)])),
                ),
        ),
        config: Arc::new(config),
    }
}

/// Parses, analyzes, optimizes and plans `sql`.
fn plan(cx: &Cx, sql: &str) -> Result<Runnable, KrishivError> {
    let plan =
        krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .map_err(KrishivError::from)?;
    let plan = block_on(DefaultAnalyzer.analyze(&plan, &cx.ctx))?.plan;
    let plan = krishiv_optimizer::optimize(&plan)?;
    Ok(Runnable::new(
        DefaultPlanner.plan(
            &plan,
            &PlanningContext::new(Arc::clone(&cx.tables), Arc::clone(&cx.config)),
        )?,
        Arc::clone(&cx.tables),
    ))
}

/// One cell as Spark's golden files render it (`NULL` for null).
fn cell(a: &dyn Array, row: usize) -> String {
    if a.is_null(row) || a.data_type() == &ArrowType::Null {
        return "NULL".to_owned();
    }
    match a.data_type() {
        ArrowType::Boolean => a.as_boolean().value(row).to_string(),
        ArrowType::Int32 => a
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(row)
            .to_string(),
        ArrowType::Utf8View => a.as_string_view().value(row).to_owned(),
        ArrowType::Utf8 => a.as_string::<i32>().value(row).to_owned(),
        other => panic!("no test formatting for {other}"),
    }
}

/// The rows of `sql` in `cx`, each row tab-joined and sorted.
fn rows(cx: &Cx, sql: &str) -> Vec<String> {
    let runnable = plan(cx, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = run(&runnable).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r))
                    .collect::<Vec<_>>()
                    .join("\t"),
            );
        }
    }
    out.sort();
    out
}

/// Every comparison operator between a `Utf8` column and a `Utf8View`
/// literal, both orders, with NULLs exercising three-valued logic.
#[test]
fn comparisons_mix_utf8_and_utf8view() {
    let cx = cx();
    // Against a literal.
    assert_eq!(rows(&cx, "SELECT s FROM u8 WHERE s = 'banana'"), ["banana"]);
    assert_eq!(
        rows(&cx, "SELECT s FROM u8 WHERE s <> 'banana'"),
        ["apple", "cherry"]
    );
    assert_eq!(rows(&cx, "SELECT s FROM u8 WHERE s < 'banana'"), ["apple"]);
    assert_eq!(
        rows(&cx, "SELECT s FROM u8 WHERE s <= 'banana'"),
        ["apple", "banana"]
    );
    assert_eq!(rows(&cx, "SELECT s FROM u8 WHERE s > 'banana'"), ["cherry"]);
    assert_eq!(
        rows(&cx, "SELECT s FROM u8 WHERE s >= 'banana'"),
        ["banana", "cherry"]
    );
    assert_eq!(rows(&cx, "SELECT a FROM u8 WHERE s <=> 'banana'"), ["2"]);
    assert_eq!(rows(&cx, "SELECT a FROM u8 WHERE s <=> NULL"), ["3"]);
    assert_eq!(rows(&cx, "SELECT a FROM u8 WHERE NULL <=> s"), ["3"]);
    // The `Utf8View` table reads the same queries.
    assert_eq!(rows(&cx, "SELECT s FROM uv WHERE s = 'x'"), ["x"]);
    assert_eq!(
        rows(&cx, "SELECT s FROM uv WHERE s <> 'x'"),
        ["banana", "cherry"]
    );
    // Column against column across the encodings, both orders.
    assert_eq!(
        rows(
            &cx,
            "SELECT u8.a, uv.a FROM u8 JOIN uv ON u8.s = uv.s ORDER BY u8.a"
        ),
        ["2\t1", "4\t4"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT u8.a, uv.a FROM u8 JOIN uv ON uv.s = u8.s ORDER BY u8.a"
        ),
        ["2\t1", "4\t4"]
    );
    assert_eq!(
        rows(&cx, "SELECT u8.a FROM u8 JOIN uv ON u8.s > uv.s"),
        ["4"]
    );
    // A `Utf8` literal is still a view: two literals compare as before.
    assert_eq!(rows(&cx, "SELECT 'a' = 'a', 'a' < 'b'"), ["true\ttrue"]);
}

/// `IN` lists (and `IN` subqueries) mixing a `Utf8` column with `Utf8View`
/// elements, with NULLs following Spark's three-valued rules.
#[test]
fn in_lists_mix_utf8_and_utf8view() {
    let cx = cx();
    assert_eq!(
        rows(
            &cx,
            "SELECT a FROM u8 WHERE s IN ('banana', 'fig') ORDER BY a"
        ),
        ["2"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT a FROM u8 WHERE s IN ('banana', NULL) ORDER BY a"
        ),
        ["2"]
    );
    assert_eq!(
        rows(&cx, "SELECT s FROM u8 WHERE s IN ('apple', 'cherry')"),
        ["apple", "cherry"]
    );
    // `NOT IN` with a NULL element matches nothing.
    assert_eq!(
        rows(&cx, "SELECT a FROM u8 WHERE s NOT IN ('banana', NULL)"),
        Vec::<String>::new()
    );
    assert_eq!(
        rows(&cx, "SELECT a FROM u8 WHERE s NOT IN ('banana') ORDER BY a"),
        ["1", "4"]
    );
    // `IN` over a subquery of the other encoding (a semi join).
    assert_eq!(
        rows(
            &cx,
            "SELECT a FROM u8 WHERE s IN (SELECT s FROM uv) ORDER BY a"
        ),
        ["2", "4"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT a FROM uv WHERE s IN (SELECT s FROM u8) ORDER BY a"
        ),
        ["1", "4"]
    );
}

/// `BETWEEN` over a `Utf8` column with `Utf8View` bounds.
#[test]
fn between_mixes_utf8_and_utf8view() {
    let cx = cx();
    assert_eq!(
        rows(
            &cx,
            "SELECT a FROM u8 WHERE s BETWEEN 'avocado' AND 'coconut' ORDER BY a"
        ),
        ["2", "4"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT a FROM u8 WHERE s NOT BETWEEN 'avocado' AND 'coconut' ORDER BY a"
        ),
        ["1"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT a FROM uv WHERE s BETWEEN 'a' AND 'c' ORDER BY a"
        ),
        ["1"]
    );
}

/// `CASE`, `COALESCE`, `IF` and `NULLIF` with branches in both encodings.
#[test]
fn conditionals_mix_utf8_and_utf8view() {
    let cx = cx();
    assert_eq!(
        rows(
            &cx,
            "SELECT a, CASE WHEN a > 2 THEN s ELSE 'other' END FROM u8 ORDER BY a"
        ),
        ["1\tother", "2\tother", "3\tNULL", "4\tcherry"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT a, CASE s WHEN 'banana' THEN 'found' ELSE 'other' END FROM u8 ORDER BY a"
        ),
        ["1\tother", "2\tfound", "3\tother", "4\tother"]
    );
    assert_eq!(
        rows(&cx, "SELECT a, COALESCE(s, 'none') FROM u8 ORDER BY a"),
        ["1\tapple", "2\tbanana", "3\tnone", "4\tcherry"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT u8.a, COALESCE(u8.s, uv.s) FROM u8 JOIN uv ON u8.a = uv.a ORDER BY u8.a"
        ),
        ["1\tapple", "2\tbanana", "3\tNULL", "4\tcherry"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT a, IF(s IS NULL, 'was null', s) FROM u8 ORDER BY a"
        ),
        ["1\tapple", "2\tbanana", "3\twas null", "4\tcherry"]
    );
    assert_eq!(
        rows(&cx, "SELECT a, NULLIF(s, 'banana') FROM u8 ORDER BY a"),
        ["1\tapple", "2\tNULL", "3\tNULL", "4\tcherry"]
    );
    // A `Utf8View` value against a `Utf8` other, through a join.
    assert_eq!(
        rows(
            &cx,
            "SELECT uv.a, NULLIF(uv.s, u8.s) FROM uv JOIN u8 ON uv.a = u8.a ORDER BY uv.a"
        ),
        ["1\tbanana", "2\tx", "3\tNULL", "4\tNULL"]
    );
}

/// Joins on string keys in different encodings: inner (both key orders),
/// null-safe, null-aware anti and a mixed-encoding residual.
#[test]
fn joins_mix_utf8_and_utf8view_keys() {
    let cx = cx();
    assert_eq!(
        rows(
            &cx,
            "SELECT u8.a, uv.a FROM u8 JOIN uv ON u8.s = uv.s ORDER BY u8.a"
        ),
        ["2\t1", "4\t4"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT u8.a, uv.a FROM u8 JOIN uv ON uv.s = u8.s ORDER BY u8.a"
        ),
        ["2\t1", "4\t4"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT u8.a FROM u8 LEFT SEMI JOIN uv ON u8.s = uv.s ORDER BY u8.a"
        ),
        ["2", "4"]
    );
    // Null-safe equality also matches the NULL keys.
    assert_eq!(
        rows(
            &cx,
            "SELECT u8.a, uv.a FROM u8 JOIN uv ON u8.s <=> uv.s ORDER BY u8.a"
        ),
        ["2\t1", "3\t3", "4\t4"]
    );
    // Null-aware anti join: a NULL in the subquery matches nothing, so
    // `NOT IN` over the whole table is empty; without the NULL it keeps
    // only `apple`.
    assert_eq!(
        rows(
            &cx,
            "SELECT a FROM u8 WHERE s NOT IN (SELECT s FROM uv) ORDER BY a"
        ),
        Vec::<String>::new()
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT a FROM u8 WHERE s NOT IN (SELECT s FROM uv WHERE s IS NOT NULL) ORDER BY a"
        ),
        ["1"]
    );
    // Equality keys on `a` with a mixed-encoding residual on `s`.
    assert_eq!(
        rows(
            &cx,
            "SELECT u8.a FROM u8 JOIN uv ON u8.a = uv.a AND u8.s = uv.s ORDER BY u8.a"
        ),
        ["4"]
    );
    assert_eq!(
        rows(
            &cx,
            "SELECT u8.a FROM u8 JOIN uv ON u8.a = uv.a AND u8.s <> uv.s ORDER BY u8.a"
        ),
        ["1", "2"]
    );
}

/// `INTERSECT` and `EXCEPT` (distinct) over the two encodings; the ALL
/// variants go through the `UNION ALL` path of PR #220.
#[test]
fn set_operations_mix_utf8_and_utf8view() {
    let cx = cx();
    assert_eq!(
        rows(&cx, "SELECT s FROM u8 INTERSECT SELECT s FROM uv"),
        ["NULL", "banana", "cherry"]
    );
    assert_eq!(
        rows(&cx, "SELECT s FROM u8 EXCEPT SELECT s FROM uv"),
        ["apple"]
    );
    assert_eq!(rows(&cx, "SELECT s FROM uv EXCEPT SELECT s FROM u8"), ["x"]);
}

/// Operands of different Spark types still fail as before: the analyzer
/// casts the `STRING` side to `BIGINT`, which raises the same cast error
/// for both carriers.
#[test]
fn different_spark_types_still_fail() {
    let cx = cx();
    for sql in [
        "SELECT s FROM u8 WHERE s = a",
        "SELECT s FROM uv WHERE s = a",
    ] {
        let runnable = plan(&cx, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        let err = run(&runnable).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::CastInvalidInput,
            "{sql}: {err}"
        );
    }
}

/// The analyzed plan of `sql`, without optimization or physical planning.
fn analyzed(cx: &Cx, sql: &str) -> Arc<LogicalPlan> {
    let parsed =
        krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .unwrap();
    let plan = block_on(DefaultAnalyzer.analyze(&parsed, &cx.ctx))
        .unwrap()
        .plan;
    Arc::clone(plan.root())
}

/// A resolved `l = r` over two analyzed branches' output columns.
fn eq(l: BoundColumn, r: BoundColumn) -> Arc<Expr> {
    let col = |c: BoundColumn| Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c))));
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            within_group: Vec::new(),
            id: FunctionId::Builtin(Builtin::Eq),
            kind: FunctionKind::Scalar,
            args: vec![col(l), col(r)],
            distinct: false,
            filter: None,
            data_type: DataType::Boolean,
            nullable: true,
            deterministic: true,
        },
    ))))
}

/// Join keys of different Spark types are left for the join builder to
/// reject, never silently coerced: `INT` against `STRING` (one side has
/// no string/binary carrier) and `STRING` against `BINARY` (carriers of
/// different families). The analyzer would cast these pairs, so the test
/// joins the analyzed branches directly.
#[test]
fn joins_with_keys_of_other_spark_types_still_fail() {
    let cx = cx();
    let branch = |sql: &str| {
        let plan = analyzed(&cx, sql);
        let col = plan
            .resolved_output()
            .and_then(|mut cols| cols.pop())
            .unwrap();
        (plan, col)
    };
    for (left, right) in [
        ("SELECT a FROM u8", "SELECT s FROM uv"),
        ("SELECT s FROM u8", "SELECT b FROM ub"),
    ] {
        let ((l, lc), (r, rc)) = (branch(left), branch(right));
        let root = LogicalPlan::new(PlanNode::Join {
            left: l,
            right: r,
            kind: JoinKind::Inner,
            condition: JoinCondition::On(eq(lc, rc)),
        });
        let err = DefaultPlanner
            .plan(
                &Plan::try_new(root).unwrap(),
                &PlanningContext::new(Arc::clone(&cx.tables), Arc::clone(&cx.config)),
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("join keys differ in type"),
            "{left} JOIN {right}: {err}"
        );
    }
}
