// SPDX-License-Identifier: Apache-2.0
//! `min`/`max` over structs: Spark 4.2 orders structs field by field (a
//! NULL field sorts before every value and equals another NULL; NaN is the
//! largest float), skips NULL struct rows and gives NULL for no non-NULL
//! row. The accumulator compares with the same struct comparator the
//! comparison operators use, so partial states merge like raw rows and the
//! result does not depend on the partitioning. Every test that runs a plan
//! through `exec` also runs it with the row-layout tables.

use arrow_array::types::{Decimal128Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, Decimal128Array, Float64Array, Int32Array, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Fields};

use super::*;
use AggregateFunction::{Max, Min};

/// A nullable struct field.
fn field(name: &str, t: DataType) -> Field {
    Field::new(name, t, true)
}

/// A struct column of `fields` with `children`, NULL on `valid`'s false rows.
fn structs(fields: Fields, children: Vec<ArrayRef>, valid: &[bool]) -> ArrayRef {
    let nulls = (!valid.iter().all(|v| *v)).then(|| NullBuffer::from_iter(valid.iter().copied()));
    Arc::new(StructArray::new(fields, children, nulls))
}

fn ints(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn strs(v: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringViewArray::from(v.to_vec()))
}

/// One value of any tested type, as Spark prints it in JSON (`{"a":1}`);
/// `null` for a NULL field. `srows` renders a top-level NULL as `NULL`.
fn value(a: &dyn Array, i: usize) -> String {
    if a.is_null(i) {
        return "null".to_owned();
    }
    match a.data_type() {
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Utf8View => format!("{:?}", a.as_string_view().value(i)),
        DataType::Float64 => format!("{:?}", a.as_primitive::<Float64Type>().value(i)),
        DataType::Decimal128(_, s) => plain(a.as_primitive::<Decimal128Type>().value(i), *s),
        DataType::Struct(_) => {
            let s = a.as_struct();
            let fields = s
                .columns()
                .iter()
                .enumerate()
                .map(|(k, child)| {
                    let name = s.column_names().get(k).cloned().unwrap_or_default();
                    format!("\"{name}\":{}", value(child.as_ref(), i))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{fields}}}")
        }
        other => panic!("unrendered test value of {other}"),
    }
}

/// Rows as `k|struct`, sorted (the grouping order is not the input's);
/// a top-level NULL is `NULL`.
fn srows(batches: &[Batch]) -> Vec<String> {
    let mut out: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let d = b.data();
            (0..d.num_rows()).map(move |r| {
                let cells: Vec<String> = d
                    .columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(r) {
                            "NULL".to_owned()
                        } else {
                            value(c.as_ref(), r)
                        }
                    })
                    .collect();
                cells.join("|")
            })
        })
        .collect();
    out.sort();
    out
}

async fn run_structs(plan: Arc<dyn ExecNode>) -> Vec<String> {
    let o = Opts {
        slots: 1,
        ..Opts::default()
    };
    srows(&exec(plan, o).await.unwrap().0)
}

/// Spark 4.2: `SELECT k, min(s), max(s) FROM t GROUP BY k` orders structs
/// field by field; a NULL struct row is skipped and all-NULL gives NULL.
#[tokio::test]
async fn min_and_max_of_structs_follow_field_order() {
    let fields: Fields = vec![field("a", DataType::Int32), field("b", DataType::Utf8View)].into();
    let s = schema(vec![
        ("k", DataType::Int64),
        ("s", DataType::Struct(fields.clone())),
    ]);
    let structs_col = structs(
        fields,
        vec![
            ints(&[
                Some(1),
                Some(1),
                Some(2),
                Some(1),
                Some(4),
                Some(7),
                Some(8),
            ]),
            strs(&[
                Some("y"),
                Some("x"),
                Some("a"),
                None,
                Some("q"),
                Some("u"),
                Some("v"),
            ]),
        ],
        // Group 1 orders by `a` then `b`; group 2's smaller row has a NULL
        // field (NULL sorts first) next to a NULL struct (skipped); group 3
        // holds only NULL structs (whatever their fields hold).
        &[true, true, true, true, false, false, false],
    );
    let k = i64s(&[1, 1, 1, 2, 2, 3, 3].map(Some));
    let one = node(
        scan(&s, vec![vec![k, structs_col]]),
        spec(&s, vec![0], vec![agg(Min, 1, &s), agg(Max, 1, &s)]),
    );
    assert_eq!(
        run_structs(one).await,
        [
            "1|{\"a\":1,\"b\":\"x\"}|{\"a\":2,\"b\":\"a\"}",
            "2|{\"a\":1,\"b\":null}|{\"a\":1,\"b\":null}",
            "3|NULL|NULL",
        ]
    );
    // A global aggregate over no rows is NULL, not an error.
    let empty_fields: Fields = vec![field("a", DataType::Int32)].into();
    let es = schema(vec![("s", DataType::Struct(empty_fields.clone()))]);
    let empty = structs(empty_fields, vec![ints(&[])], &[]);
    let none = node(
        scan(&es, vec![vec![empty]]),
        spec(&es, vec![], vec![agg(Min, 0, &es), agg(Max, 0, &es)]),
    );
    assert_eq!(run_structs(none).await, ["NULL|NULL"]);
}

/// Nested structs, decimals and floats with NaN: NaN is the largest value
/// and equals itself; `-0.0 = 0.0`.
#[tokio::test]
async fn min_and_max_of_nested_structs_order_like_comparisons() {
    let inner: Fields = vec![field("x", DataType::Float64)].into();
    let fields: Fields = vec![
        field("n", DataType::Struct(inner.clone())),
        field("d", DataType::Decimal128(10, 2)),
        field("t", DataType::Utf8View),
    ]
    .into();
    let s = schema(vec![("s", DataType::Struct(fields.clone()))]);
    let nan = f64::NAN;
    let col = structs(
        fields,
        vec![
            structs(
                inner,
                vec![Arc::new(Float64Array::from(vec![
                    Some(nan),
                    Some(1.0),
                    None,
                    Some(-0.0),
                ])) as ArrayRef],
                &[true, true, true, true],
            ),
            Arc::new(
                Decimal128Array::from(vec![100_i128, 200, 50, 300])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ) as ArrayRef,
            strs(&[Some("a"), Some("z"), Some("m"), Some("b")]),
        ],
        &[true, true, true, true],
    );
    let plan = node(
        scan(&s, vec![vec![col]]),
        spec(&s, vec![], vec![agg(Min, 0, &s), agg(Max, 0, &s)]),
    );
    // The minimum has the NULL float (a NULL field sorts first); the
    // maximum the NaN (NaN beats every value, `-0.0 = 0.0` loses to it).
    assert_eq!(
        run_structs(plan).await,
        ["{\"n\":{\"x\":null},\"d\":0.50,\"t\":\"m\"}|{\"n\":{\"x\":NaN},\"d\":1.00,\"t\":\"a\"}"]
    );
}

/// Partial states from any split of the input merge to the single-phase
/// result, whatever the partitioning: ties keep an equal value, so the
/// rows are the same.
#[tokio::test]
async fn struct_partial_states_merge_like_raw_rows() {
    let fields: Fields = vec![field("a", DataType::Int32), field("b", DataType::Utf8View)].into();
    let s = schema(vec![
        ("k", DataType::Int64),
        ("s", DataType::Struct(fields.clone())),
    ]);
    let col = |a: &[Option<i32>], b: &[Option<&str>], valid: &[bool]| {
        structs(fields.clone(), vec![ints(a), strs(b)], valid)
    };
    // The maximum `{a:2,b:"a"}` ties across the batches (`"a"` first);
    // the minimum `{a:1,b:"x"}` too. Group 2's one valid row sits among
    // NULL structs (skipped like any NULL input).
    let batches = vec![
        vec![
            i64s(&[Some(1), Some(2)]),
            col(&[Some(2), None], &[Some("a"), None], &[true, false]),
        ],
        vec![
            i64s(&[Some(1), Some(2)]),
            col(&[Some(1), Some(1)], &[Some("x"), Some("q")], &[true, true]),
        ],
        vec![
            i64s(&[Some(1), Some(2)]),
            col(&[Some(1), None], &[Some("x"), None], &[true, false]),
        ],
    ];
    let nulls = vec![vec![
        i64s(&[Some(2), Some(2)]),
        structs(
            fields.clone(),
            vec![ints(&[None, None]), strs(&[None, None])],
            &[false, false],
        ),
    ]];
    let aggs = || vec![agg(Min, 1, &s), agg(Max, 1, &s)];
    let expected = [
        "1|{\"a\":1,\"b\":\"x\"}|{\"a\":2,\"b\":\"a\"}".to_owned(),
        "2|{\"a\":1,\"b\":\"q\"}|{\"a\":1,\"b\":\"q\"}".to_owned(),
    ];
    let single = node(scan(&s, batches.clone()), spec(&s, vec![0], aggs()));
    assert_eq!(run_structs(single).await, expected);
    for policy in [
        BypassPolicy::default(),
        BypassPolicy {
            min_rows: 1,
            max_group_percent: 0,
            ..BypassPolicy::default()
        },
    ] {
        let mut first = spec(&s, vec![0], aggs());
        first.output = AggregateOutput::Partial;
        let second = second_phase(&first, None);
        let partial = PartialAggregate::new(first, policy, Some(7)).unwrap();
        let plan = node(PartialNode::new(scan(&s, batches.clone()), partial), second);
        assert_eq!(run_structs(plan).await, expected);
    }
    // Eight partitions fold the same rows: the values are equal.
    let o = Opts {
        slots: 8,
        ..Opts::default()
    };
    let plan = node(scan(&s, batches), spec(&s, vec![0], aggs()));
    assert_eq!(srows(&exec(plan, o).await.unwrap().0), expected);
    let plan = node(scan(&s, nulls), spec(&s, vec![0], aggs()));
    assert_eq!(run_structs(plan).await, ["2|NULL|NULL"]);
}

/// A struct holding a map, an array or another unordered field is not a
/// `min`/`max` argument (Spark's `INVALID_ORDERING_TYPE` at analysis; the
/// executor refuses it as unsupported).
#[test]
fn structs_with_unorderable_fields_are_rejected() {
    use krishiv_common::error::ErrorClass::KrishivUnsupportedFeature;

    let entries = Arc::new(Field::new(
        "entries",
        DataType::Struct(
            vec![
                Field::new("key", DataType::Utf8, true),
                Field::new("value", DataType::Int32, true),
            ]
            .into(),
        ),
        false,
    ));
    let struct_of = |t: DataType| DataType::Struct(vec![Field::new("f", t, true)].into());
    let unordered = [
        struct_of(DataType::Map(entries, false)),
        struct_of(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Int32,
            true,
        )))),
        struct_of(struct_of(DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::Utf8, true),
                        Field::new("value", DataType::Int32, true),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        ))),
    ];
    for t in unordered {
        for f in [Min, Max] {
            let err = AggregateExpr::new(f, 0, t.clone(), "m")
                .result_type()
                .unwrap_err();
            assert_eq!(err.error_class(), KrishivUnsupportedFeature, "{t:?}");
        }
    }
    // Orderable structs pass, with the struct as the result type.
    let ordered = struct_of(DataType::Int32);
    for f in [Min, Max] {
        assert_eq!(
            AggregateExpr::new(f, 0, ordered.clone(), "m")
                .result_type()
                .unwrap(),
            ordered
        );
    }
}

/// The reservation covers what the struct fold holds: the projected size
/// after an update is at least the size, with and without growth (the
/// winner's buffers count toward the reservation through the aggregate's
/// existing accounting path).
#[test]
fn struct_projected_size_covers_the_size_after_an_update() {
    let fields: Fields = vec![field("a", DataType::Int32), field("b", DataType::Utf8View)].into();
    let col = structs(
        fields.clone(),
        vec![
            ints(&[Some(1), None, Some(2), Some(1)]),
            strs(&[Some("x"), None, Some("yy"), Some("z")]),
        ],
        &[true, false, true, true],
    );
    for f in [Min, Max] {
        let expr = AggregateExpr::new(f, 0, DataType::Struct(fields.clone()), "m");
        let mut acc = Accumulator::new(&expr, true).unwrap();
        acc.reserve(2);
        acc.resize(2);
        acc.update(
            &[0, 0, 1, 1],
            std::slice::from_ref(&col),
            &Weights::Const(1),
        )
        .unwrap();
        assert!(acc.size() <= acc.projected(0, None, None));
        assert!(acc.size() <= acc.projected(8, Some(&col), None));
    }
}

/// The struct fold skips NULL structs and rejects anything but unit
/// weights, ragged inputs and non-struct columns without panicking.
#[test]
fn struct_folds_reject_bad_inputs() {
    use krishiv_common::error::ErrorClass::{InternalError, KrishivUnsupportedFeature};
    use krishiv_types::batch::Weights;

    let fields: Fields = vec![field("a", DataType::Int32)].into();
    let expr = AggregateExpr::new(Min, 0, DataType::Struct(fields.clone()), "m");
    let col = structs(fields.clone(), vec![ints(&[Some(1)])], &[true]);
    let one = Weights::Const(1);
    // Weights other than one need incremental view maintenance.
    let mut acc = Accumulator::new(&expr, true).unwrap();
    acc.resize(1);
    let err = acc
        .update(&[0], std::slice::from_ref(&col), &Weights::Const(2))
        .unwrap_err();
    assert_eq!(err.error_class(), KrishivUnsupportedFeature);
    // Group ids and columns run in lockstep.
    let err = acc
        .update(&[0, 1], std::slice::from_ref(&col), &one)
        .unwrap_err();
    assert_eq!(err.error_class(), InternalError);
    // The column is a struct.
    let err = acc.update(&[0], &[ints(&[Some(1)])], &one).unwrap_err();
    assert_eq!(err.error_class(), InternalError);
    // No groups: an empty struct column, not an error.
    let empty = acc.state(&[]).unwrap();
    assert_eq!(empty.len(), 1);
    assert_eq!(empty[0].len(), 0);
    assert_eq!(empty[0].data_type(), &DataType::Struct(fields));
}
