// SPDX-License-Identifier: Apache-2.0
//! The DataFrame transforms of P2-T186 part A: distinct, unions,
//! intersections and excepts, hints, renames, sampling
//! (unsupported) and unpivoting, through analysis and execution.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::functions::{col, lit};
use krishiv_session::{Column, Session};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int32, false),
        Field::new("name", ArrowType::Utf8, true),
    ]))
}

fn batch(ids: &[i32], names: &[Option<&str>]) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(names.to_vec())),
        ],
    )
    .unwrap()
}

fn session() -> Session {
    let s = Session::new().unwrap();
    s.register_batches(
        "t1",
        schema(),
        vec![batch(
            &[1, 2, 2, 3],
            &[Some("a"), Some("b"), Some("b"), None],
        )],
    )
    .unwrap();
    s.register_batches(
        "t2",
        schema(),
        vec![batch(&[2, 3, 4], &[Some("b"), None, Some("d")])],
    )
    .unwrap();
    s
}

/// `(id, name)` rows, `None` for null, sorted, so set results compare
/// without depending on the execution order.
fn rows(batches: &[RecordBatch]) -> Vec<(i32, Option<String>)> {
    let mut out: Vec<(i32, Option<String>)> = batches
        .iter()
        .flat_map(|b| {
            let ids = b.column(0).as_primitive::<Int32Type>();
            let names = b.column(1).as_string::<i32>();
            (0..b.num_rows()).map(|i| {
                (
                    ids.value(i),
                    if names.is_null(i) {
                        None
                    } else {
                        Some(names.value(i).to_owned())
                    },
                )
            })
        })
        .collect();
    out.sort();
    out
}

fn class<T: std::fmt::Debug>(r: Result<T, KrishivError>) -> ErrorClass {
    r.unwrap_err().error_class()
}

#[test]
fn distinct_dedupes_nulls_like_sql() {
    let s = session();
    let df = s.table("t1").distinct();
    assert_eq!(
        rows(&block_on(df.collect()).unwrap()),
        [
            (1, Some("a".to_owned())),
            (2, Some("b".to_owned())),
            (3, None),
        ]
    );
}

#[test]
fn union_matches_by_position_and_widens() {
    let s = session();
    let df = s.table("t1").union(&s.table("t2"));
    assert_eq!(block_on(df.count()).unwrap(), 7);
    // A narrower type widens to the common one, as in SQL.
    s.register_batches(
        "narrow",
        Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int32, false)])),
        vec![
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int32, false)])),
                vec![Arc::new(Int32Array::from(vec![1]))],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    s.register_batches(
        "wide",
        Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int64, false)])),
        vec![
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int64, false)])),
                vec![Arc::new(Int64Array::from(vec![2]))],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let df = s.table("narrow").union(&s.table("wide"));
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(schema.field(0).data_type(), &ArrowType::Int64);
}

#[test]
fn union_rejects_mismatched_shapes() {
    let s = session();
    let tall = block_on(s.sql("SELECT id FROM t1")).unwrap();
    let wide = block_on(s.sql("SELECT id, name FROM t1")).unwrap();
    assert_eq!(
        class(block_on(tall.union(&wide).collect())),
        ErrorClass::NumColumnsMismatch
    );
    let dates = block_on(s.sql("SELECT DATE '2020-01-01' AS id")).unwrap();
    assert_eq!(
        class(block_on(tall.union(&dates).collect())),
        ErrorClass::IncompatibleColumnType
    );
}

#[test]
fn union_by_name_matches_by_name() {
    let s = session();
    let a = block_on(s.sql("SELECT id, name FROM t1 ORDER BY id LIMIT 1")).unwrap();
    let b = block_on(s.sql("SELECT name, id FROM t1 ORDER BY id LIMIT 1 OFFSET 1")).unwrap();
    let df = a.union_by_name(&b, false).order_by([col("id")]);
    assert_eq!(
        rows(&block_on(df.collect()).unwrap()),
        [(1, Some("a".to_owned())), (2, Some("b".to_owned())),]
    );
    // A missing column fails without `allow_missing_columns`.
    let c = block_on(s.sql("SELECT id FROM t1 ORDER BY id LIMIT 1")).unwrap();
    let df = a.union_by_name(&c, false);
    assert_eq!(
        class(block_on(df.collect())),
        ErrorClass::UnresolvedColumnAmongFieldNames
    );
}

/// With `allow_missing_columns`, a column one side lacks reads NULL there
/// (Spark 4.2 rows). Ignored until the physical planner unions a
/// null-filled string branch with a `Utf8View` one: it rejects that union
/// with `InternalError` today, as it does the SQL form `SELECT id, NULL AS
/// name FROM t1 UNION ALL SELECT id, name FROM t1`.
#[test]
#[ignore = "physical planner rejects a union of Utf8 and Utf8View branches"]
fn union_by_name_fills_missing_columns() {
    let s = session();
    let a = block_on(s.sql("SELECT id, name FROM t1 ORDER BY id LIMIT 1")).unwrap();
    let c = block_on(s.sql("SELECT id FROM t1 ORDER BY id LIMIT 1")).unwrap();
    let df = a.union_by_name(&c, true).order_by([col("name")]);
    assert_eq!(
        rows(&block_on(df.collect()).unwrap()),
        [(1, None), (1, Some("a".to_owned()))]
    );
}

#[test]
fn set_operations_match_spark() {
    let s = session();
    let l = s.table("t1");
    let r = s.table("t2");
    // Distinct forms drop duplicates; ALL forms keep the overlap counts.
    assert_eq!(
        rows(&block_on(l.intersect(&r).collect()).unwrap()),
        [(2, Some("b".to_owned())), (3, None)]
    );
    assert_eq!(
        rows(&block_on(l.intersect_all(&r).collect()).unwrap()),
        [(2, Some("b".to_owned())), (3, None)]
    );
    assert_eq!(
        rows(&block_on(l.subtract(&r).collect()).unwrap()),
        [(1, Some("a".to_owned()))]
    );
    assert_eq!(
        rows(&block_on(l.except_all(&r).collect()).unwrap()),
        [(1, Some("a".to_owned())), (2, Some("b".to_owned())),]
    );
}

#[test]
fn hints_pass_through_and_warn_like_spark() {
    let s = session();
    // An unknown hint warns and changes nothing.
    let df = s.table("t1").hint("NOPE", [lit("t1")]);
    assert_eq!(block_on(df.count()).unwrap(), 4);
    let warnings = block_on(df.warnings()).unwrap();
    assert!(
        warnings
            .iter()
            .any(|w| w.starts_with("Unrecognized hint: NOPE")),
        "{warnings:?}"
    );
    // A join hint on a single relation warns and changes nothing.
    let df = s.table("t1").hint("BROADCAST", [lit("t1")]);
    assert_eq!(block_on(df.count()).unwrap(), 4);
    // A partitioning hint is checked: a bad partition count fails analysis.
    let df = s.table("t1").hint("COALESCE", [lit(0)]);
    assert_eq!(
        class(block_on(df.collect())),
        ErrorClass::KrishivInvalidPartitionNumber
    );
    // A parameter that names no relation fails analysis.
    let df = s.table("t1").hint("BROADCAST", [lit(1)]);
    assert_eq!(
        class(block_on(df.collect())),
        ErrorClass::LegacyErrorTemp1046
    );
}

#[test]
fn to_df_renames_positionally() {
    let s = session();
    let df = s.table("t1").to_df(["x", "y"]);
    assert_eq!(block_on(df.schema()).unwrap().field(0).name(), "x",);
    assert_eq!(block_on(df.count()).unwrap(), 4);
    let df = s.table("t1").to_df(["only"]);
    assert_eq!(
        class(block_on(df.collect())),
        ErrorClass::AssignmentArityMismatch
    );
}

#[test]
fn offset_skips_and_rejects_negatives() {
    let s = session();
    let df = s.table("t1").order_by([col("id")]).offset(2);
    let ids: Vec<i32> = block_on(df.collect())
        .unwrap()
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    assert_eq!(ids, [2, 3]);
    let df = s.table("t1").offset(-1);
    assert_eq!(
        class(block_on(df.collect())),
        ErrorClass::InvalidLimitLikeExpressionIsNegative
    );
}

#[test]
fn sample_is_unsupported() {
    let s = session();
    let df = s.table("t1").sample(false, 0.5, 3);
    let e = block_on(df.count()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("DataFrame.sample"), "{e}");
}

#[test]
fn unpivot_stacks_value_columns() {
    let s = session();
    s.register_batches(
        "wide",
        Arc::new(Schema::new(vec![
            Field::new("id", ArrowType::Int32, false),
            Field::new("a", ArrowType::Int32, true),
            Field::new("b", ArrowType::Float64, true),
        ])),
        vec![
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("id", ArrowType::Int32, false),
                    Field::new("a", ArrowType::Int32, true),
                    Field::new("b", ArrowType::Float64, true),
                ])),
                vec![
                    Arc::new(Int32Array::from(vec![1, 2])),
                    Arc::new(Int32Array::from(vec![Some(10), None])),
                    Arc::new(Float64Array::from(vec![Some(1.5), Some(2.5)])),
                ],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let df = s.table("wide").unpivot(
        [col("id")],
        [("a".to_owned(), col("a")), ("b".to_owned(), col("b"))],
        "var",
        "val",
    );
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>(),
        ["id", "var", "val"]
    );
    // Values widen to the common type, and NULLs are kept.
    assert_eq!(schema.field(2).data_type(), &ArrowType::Float64);
    let batches = block_on(df.collect()).unwrap();
    let mut got: Vec<(i32, String, Option<f64>)> = batches
        .iter()
        .flat_map(|b| {
            let ids = b.column(0).as_primitive::<Int32Type>();
            let vars = b.column(1);
            let vals = b.column(2).as_primitive::<Float64Type>();
            (0..b.num_rows())
                .map(|i| {
                    let var = match vars.data_type() {
                        ArrowType::Utf8View => vars.as_string_view().value(i).to_owned(),
                        _ => vars.as_string::<i32>().value(i).to_owned(),
                    };
                    (ids.value(i), var, (!vals.is_null(i)).then(|| vals.value(i)))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    got.sort_by(|x, y| (x.0, &x.1).cmp(&(y.0, &y.1)));
    assert_eq!(
        got,
        [
            (1, "a".to_owned(), Some(10.0)),
            (1, "b".to_owned(), Some(1.5)),
            (2, "a".to_owned(), None),
            (2, "b".to_owned(), Some(2.5)),
        ]
    );
    // Without value columns the frame fails, as Spark leaves it unresolved.
    let df = s
        .table("wide")
        .unpivot([col("id")], Vec::<(String, Column)>::new(), "var", "val");
    let e = block_on(df.count()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        e.to_string().contains("UNPIVOT without value columns"),
        "{e}"
    );
}

#[test]
fn binary_frames_carry_either_sides_deferred_error() {
    let s = session();
    let bad = block_on(s.sql("SELECT nope FROM t1")).unwrap_err();
    assert_eq!(
        bad.error_class(),
        ErrorClass::UnresolvedColumnWithSuggestion
    );
    // The SQL error above is at analysis; a deferred one rides the frame.
    let invalid = s.table("missing_table");
    let df = s.table("t1").union(&invalid);
    assert_eq!(
        class(block_on(df.collect())),
        ErrorClass::TableOrViewNotFound
    );
    let df = invalid.intersect(&s.table("t1"));
    assert_eq!(
        class(block_on(df.collect())),
        ErrorClass::TableOrViewNotFound
    );
}
