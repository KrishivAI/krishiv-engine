// SPDX-License-Identifier: Apache-2.0
//! Tests of the conditional expressions: IN, CASE, NULLIF and NANVL.

use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::types::{Decimal64Type, Float32Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{
    Decimal64Array, Float64Array, Int32Array, Int64Array, NullArray, StringArray, StringViewArray,
};
use arrow_schema::{Field, Schema};
use krishiv_common::config::Config;
use krishiv_common::error::ErrorClass;
use krishiv_plan::{AttrId, FunctionId, FunctionKind, Literal};
use krishiv_types::{DataType, DecimalType};

use super::*;
use crate::planner::eval::between::between;

/// [`EvalEnv`] with `spark.sql.ansi.enabled = ansi`.
fn env(ansi: bool) -> EvalEnv {
    let mut config = Config::new();
    config
        .set(
            "spark.sql.ansi.enabled",
            if ansi { "true" } else { "false" },
        )
        .unwrap();
    EvalEnv::new(&config, Arc::from("test"))
}

/// [`Input`] over `schema` with `layout` columns.
fn input<'a>(
    env: &'a EvalEnv,
    layout: &'a [AttrId],
    schema: &'a std::sync::Arc<Schema>,
) -> Input<'a> {
    Input {
        layout,
        schema,
        env,
        subqueries: &[],
    }
}

/// An analyzed call of `name` with result type `data_type`.
fn resolved(name: &str, data_type: DataType) -> ResolvedFunction {
    ResolvedFunction {
        id: FunctionId::Registry {
            name: name.to_owned(),
            id: 0,
        },
        kind: FunctionKind::Scalar,
        args: Vec::new(),
        distinct: false,
        filter: None,
        within_group: Vec::new(),
        data_type,
        nullable: true,
        deterministic: true,
    }
}

/// A literal expression.
fn lit(l: &Literal) -> Arc<dyn PhysExpr> {
    Arc::new(super::super::Lit::new(l).unwrap())
}

/// An input-column expression.
fn column(index: usize, data_type: ArrowType, nullable: bool) -> Arc<dyn PhysExpr> {
    Arc::new(super::super::Column {
        index,
        data_type,
        nullable,
    })
}

/// A `rows`-row batch with no usable columns, for literal-only plans.
fn rows_batch(rows: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("d", ArrowType::Null, true)]));
    RecordBatch::try_new(schema, vec![Arc::new(NullArray::new(rows))]).unwrap()
}

/// Evaluates `e` over `batch` as a column of `batch.num_rows()` rows.
fn values(e: &Arc<dyn PhysExpr>, batch: &RecordBatch) -> ArrayRef {
    e.eval(batch, 0)
        .unwrap()
        .into_array(batch.num_rows())
        .unwrap()
}

/// `1 / 0`: `NULL` without ANSI, `DIVIDE_BY_ZERO` with it.
fn one_over_zero(env: &EvalEnv) -> Arc<dyn PhysExpr> {
    super::super::arith::binary(
        Builtin::Divide,
        lit(&Literal::Double(1.0)),
        lit(&Literal::Double(0.0)),
        ArrowType::Float64,
        true,
        env,
    )
    .unwrap()
}

/// More than ten constant elements that are all NULL remove every key:
/// a probe type with no exact key (a float) keeps the per-element `IN`,
/// whose answer is NULL, instead of failing in the set lookup; the
/// integer-like probes still use the set and answer the same.
#[test]
fn an_all_null_long_in_list_keeps_in_list_answers_for_every_probe_type() {
    let env = env(false);
    let nulls =
        || -> Vec<Arc<dyn PhysExpr>> { (0..11).map(|_| lit(&Literal::Null(None))).collect() };
    let floats = Arc::new(Float64Array::from(vec![Some(1.5), None, Some(2.5)]));
    let ints = Arc::new(Int64Array::from(vec![Some(1), None, Some(2)]));
    let strs = Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("f", ArrowType::Float64, true),
        Field::new("i", ArrowType::Int64, true),
        Field::new("s", ArrowType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![floats, ints, strs]).unwrap();
    for (index, t, is_set) in [
        (0, ArrowType::Float64, false),
        (1, ArrowType::Int64, true),
        (2, ArrowType::Utf8, true),
    ] {
        let v = column(index, t, true);
        let mut args = vec![Arc::clone(&v)];
        args.extend(nulls());
        let got = in_list(args, &env).unwrap();
        assert_eq!(format!("{got:?}").starts_with("InSet"), is_set, "{got:?}");
        let list: Arc<dyn PhysExpr> = Arc::new(InList {
            v,
            items: nulls(),
            kernels: env.kernels,
        });
        let (got, want) = (values(&got, &batch), values(&list, &batch));
        assert_eq!(got.as_boolean(), want.as_boolean());
        // `x IN (NULL, ...)` is NULL for every row.
        assert_eq!(got.as_boolean().null_count(), 3);
    }
}

/// The word-at-a-time view compare crosses 64-row words, a sliced column and a ragged tail with
/// the answers of the per-element `IN`.
#[test]
fn a_view_in_list_agrees_with_in_list_across_words_and_slices() {
    let env = env(false);
    let long = "a value longer than twelve bytes";
    let values_of: Vec<Option<String>> = (0..203)
        .map(|i| match i % 7 {
            0 => Some("MAIL".to_owned()),
            1 => Some("SHIP".to_owned()),
            2 => None,
            3 => Some(long.to_owned()),
            4 => Some(format!("{long}{i}")),
            5 => Some(format!("RAIL{i}")),
            _ => Some("TRUCK".to_owned()),
        })
        .collect();
    let full: ArrayRef = Arc::new(StringViewArray::from(values_of));
    let text = |t: &str| lit(&Literal::String(t.to_owned()));
    for (offset, len) in [(0, 203), (1, 130), (63, 70), (64, 64), (5, 3)] {
        let strs = full.slice(offset, len);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            ArrowType::Utf8View,
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![strs]).unwrap();
        let col = || -> Arc<dyn PhysExpr> { column(0, ArrowType::Utf8View, true) };
        for items in [
            vec![text("MAIL"), text("SHIP")],
            vec![text(long), text("TRUCK"), text("")],
        ] {
            let views = InViews::try_new(&col(), &items).unwrap();
            let list = InList {
                v: col(),
                items: items.clone(),
                kernels: env.kernels,
            };
            let got = views.eval(&batch, 0).unwrap();
            let want = list.eval(&batch, 0).unwrap();
            assert_eq!(
                got.array().as_boolean(),
                want.array().as_boolean(),
                "{offset}+{len}"
            );
        }
    }
}

/// `l_shipmode IN ('MAIL', 'SHIP')` over 1 M short views (study 12 §5 saw 8 ns per row in the pipeline; in cache the closure form is ~2.6 ns, so the
/// gain shows end to end: 10 runs of the SF10 query, user CPU 0.79-0.83 s before, 0.56-0.60 s after).
/// Run: `cargo test -p krishiv-exec --release --lib -- --ignored --nocapture in_views_timing`.
#[test]
#[ignore = "a timing, run by hand in release"]
fn in_views_timing() {
    let modes = ["MAIL", "SHIP", "AIR", "TRUCK", "RAIL", "FOB", "REG AIR"];
    let strs: ArrayRef = Arc::new(StringViewArray::from_iter_values(
        (0..1_000_000).map(|i| modes[(i * 7 + i / 3) % modes.len()]),
    ));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        ArrowType::Utf8View,
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![strs]).unwrap();
    let col: Arc<dyn PhysExpr> = column(0, ArrowType::Utf8View, true);
    let items = vec![
        lit(&Literal::String("MAIL".to_owned())),
        lit(&Literal::String("SHIP".to_owned())),
    ];
    let views = InViews::try_new(&col, &items).unwrap();
    let mut best = f64::MAX;
    for _ in 0..20 {
        let t = std::time::Instant::now();
        let out = views.eval(&batch, 0).unwrap();
        std::hint::black_box(&out);
        best = best.min(t.elapsed().as_nanos() as f64 / 1e6);
    }
    println!("InViews ns per row: {best:.2}");
}

/// A short constant list over a view column compares views in place, with
/// the answers of the per-element `IN`: values on both sides of the 12-byte
/// inline limit, NULL rows, a NULL element, and a constant value.
#[test]
fn a_short_in_list_over_views_has_in_list_answers() {
    let env = env(false);
    let long = "a value longer than twelve bytes";
    let strs: ArrayRef = Arc::new(StringViewArray::from(vec![
        Some("MAIL"),
        Some("SHIP"),
        None,
        Some(""),
        Some(long),
        Some("a value longer than twelve byteZ"),
        Some("twelve bytes"),
    ]));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        ArrowType::Utf8View,
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![strs]).unwrap();
    let text = |t: &str| lit(&Literal::String(t.to_owned()));
    let null = || lit(&Literal::Null(None));
    type Items = Vec<Arc<dyn PhysExpr>>;
    let lists: Vec<Items> = vec![
        vec![text("MAIL"), text("SHIP")],
        vec![text(long), text("")],
        vec![text("twelve bytes"), text("nothing")],
        vec![text("MAIL"), null()],
        vec![text(long), null(), text("SHIP")],
        vec![null()],
    ];
    for items in lists {
        let v = column(0, ArrowType::Utf8View, true);
        let mut args = vec![Arc::clone(&v)];
        args.extend(items.iter().map(Arc::clone));
        let got = in_list(args, &env).unwrap();
        assert!(format!("{got:?}").starts_with("InViews"), "{got:?}");
        let list: Arc<dyn PhysExpr> = Arc::new(InList {
            v,
            items: items.clone(),
            kernels: env.kernels,
        });
        let (got, want) = (values(&got, &batch), values(&list, &batch));
        assert_eq!(
            got.as_boolean().iter().collect::<Vec<_>>(),
            want.as_boolean().iter().collect::<Vec<_>>(),
        );
    }
    // A constant value stays one value.
    let args = vec![text("SHIP"), text("MAIL"), text("SHIP")];
    let got = in_list(args, &env).unwrap().eval(&batch, 0).unwrap();
    assert!(got.is_scalar());
    assert!(got.array().as_boolean().value(0));
}

/// More than ten constant elements make a set lookup with the same
/// answers as the per-element `IN`: integers, strings (view and plain),
/// dates, with and without a NULL element, over a column with NULLs and
/// over a constant value; a float `IN` keeps the per-element path.
#[test]
fn a_long_constant_in_list_is_a_set_with_in_list_answers() {
    let env = env(false);
    let check = |v: Arc<dyn PhysExpr>, items: Vec<Arc<dyn PhysExpr>>, batch: &RecordBatch| {
        let mut args = vec![Arc::clone(&v)];
        args.extend(items.iter().cloned());
        let set = in_list(args, &env).unwrap();
        assert!(format!("{set:?}").starts_with("InSet"), "{set:?}");
        let list: Arc<dyn PhysExpr> = Arc::new(InList {
            v,
            items,
            kernels: env.kernels,
        });
        assert_eq!(set.nullable(), list.nullable());
        let (got, want) = (values(&set, batch), values(&list, batch));
        assert_eq!(got.as_boolean(), want.as_boolean());
    };
    let ints = Arc::new(Int64Array::from(vec![
        Some(3),
        None,
        Some(40),
        Some(7),
        Some(12),
    ]));
    let strs = Arc::new(StringViewArray::from(vec![
        Some("k3"),
        None,
        Some("a value longer than twelve bytes"),
        Some("k40"),
        Some(""),
    ]));
    let plain = Arc::new(StringArray::from(vec![
        Some("k3"),
        Some("x"),
        None,
        Some("k7"),
        Some("k12"),
    ]));
    let dates = Arc::new(arrow_array::Date32Array::from(vec![
        Some(3),
        Some(5),
        None,
        Some(11),
        Some(100),
    ]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", ArrowType::Int64, true),
        Field::new("s", ArrowType::Utf8View, true),
        Field::new("p", ArrowType::Utf8, true),
        Field::new("d", ArrowType::Date32, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![ints, strs, plain, dates]).unwrap();
    for with_null in [false, true] {
        let null = with_null.then(|| lit(&Literal::Null(None)));
        let ints: Vec<_> = (0..12)
            .map(|i| lit(&Literal::Long(i)))
            .chain(null.clone())
            .collect();
        check(column(0, ArrowType::Int64, true), ints.clone(), &batch);
        check(lit(&Literal::Long(5)), ints, &batch);
        let mut strs: Vec<_> = (0..12)
            .map(|i| lit(&Literal::String(format!("k{i}"))))
            .chain(null.clone())
            .collect();
        strs.push(lit(&Literal::String(
            "a value longer than twelve bytes".to_owned(),
        )));
        check(column(1, ArrowType::Utf8View, true), strs.clone(), &batch);
        check(column(2, ArrowType::Utf8, true), strs, &batch);
        let dates: Vec<_> = (0..12)
            .map(|i| lit(&Literal::Date(i)))
            .chain(null)
            .collect();
        check(column(3, ArrowType::Date32, true), dates, &batch);
    }
    // Decimal and zoned timestamp columns.
    let dec = DecimalType::try_new(10, 2).unwrap();
    let darr = Arc::new(
        arrow_array::Decimal64Array::from(vec![Some(150), None, Some(1_234), Some(7)])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    ) as ArrayRef;
    // A zoned timestamp column, of the literals' own Arrow type.
    let t_type = lit(&Literal::Timestamp(0)).data_type().clone();
    assert!(
        matches!(t_type, ArrowType::Timestamp(_, Some(_))),
        "{t_type}"
    );
    let tarr = arrow_array::make_array(
        arrow_array::TimestampMicrosecondArray::from(vec![Some(3), Some(4), None, Some(99)])
            .into_data()
            .into_builder()
            .data_type(t_type.clone())
            .build()
            .unwrap(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("d", darr.data_type().clone(), true),
        Field::new("t", t_type.clone(), true),
    ]));
    let dt = RecordBatch::try_new(schema, vec![darr, tarr]).unwrap();
    let decs: Vec<_> = (0..12)
        .map(|i| {
            lit(&Literal::Decimal {
                unscaled: i * 150,
                data_type: dec,
            })
        })
        .collect();
    check(column(0, dt.column(0).data_type().clone(), true), decs, &dt);
    let ts: Vec<_> = (0..12).map(|i| lit(&Literal::Timestamp(i))).collect();
    check(column(1, t_type, true), ts, &dt);
    // As the analyzer coerces them: a BIGINT value against INT
    // literals cast to BIGINT, and a DECIMAL(10,2) value against
    // DECIMAL(3,1) literals cast up, still make a set.
    let schema = Arc::new(Schema::new(vec![Field::new("d", ArrowType::Null, true)]));
    let inp = input(&env, &[], &schema);
    let cast = |e: Arc<dyn PhysExpr>, to: &DataType| {
        super::super::cast::compile(e, to, krishiv_plan::CastMode::Legacy, inp).unwrap()
    };
    let small = DecimalType::try_new(3, 1).unwrap();
    let ints: Vec<_> = (0..12)
        .map(|i| cast(lit(&Literal::Integer(i)), &DataType::Long))
        .collect();
    assert!(ints.iter().all(|e| e.is_constant()));
    check(column(0, ArrowType::Int64, true), ints, &batch);
    let decs: Vec<_> = (0..12)
        .map(|i| {
            cast(
                lit(&Literal::Decimal {
                    unscaled: i128::from(i) * 15,
                    data_type: small,
                }),
                &DataType::Decimal(dec),
            )
        })
        .collect();
    check(column(0, dt.column(0).data_type().clone(), true), decs, &dt);
    let floats: Vec<_> = std::iter::once(column(0, ArrowType::Float64, true))
        .chain((0..12).map(|i| lit(&Literal::Double(f64::from(i)))))
        .collect();
    let e = in_list(floats, &env).unwrap();
    assert!(format!("{e:?}").starts_with("InList"), "{e:?}");
    // Ten elements stay a list, as in Spark.
    let ten: Vec<_> = std::iter::once(column(0, ArrowType::Int64, true))
        .chain((0..10).map(|i| lit(&Literal::Long(i))))
        .collect();
    assert!(format!("{:?}", in_list(ten, &env).unwrap()).starts_with("InList"));
}

/// `coalesce` answers the first non-null argument of each row.
#[test]
fn coalesce_takes_the_first_non_null_argument_per_row() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, true),
        Field::new("b", ArrowType::Int32, true),
        Field::new("c", ArrowType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None, None, None])),
            Arc::new(Int32Array::from(vec![Some(2), Some(3), None, None])),
            Arc::new(Int32Array::from(vec![Some(4), Some(5), Some(6), None])),
        ],
    )
    .unwrap();
    let f = resolved("coalesce", DataType::Integer);
    let env = env(true);
    let e = coalesce(
        &f,
        vec![
            column(0, ArrowType::Int32, true),
            column(1, ArrowType::Int32, true),
        ],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_primitive::<Int32Type>(),
        &Int32Array::from(vec![Some(1), Some(3), None, None]),
    );
    // A single argument is itself.
    let one = coalesce(&f, vec![column(0, ArrowType::Int32, true)], &env).unwrap();
    assert_eq!(
        values(&one, &batch).as_primitive::<Int32Type>(),
        &Int32Array::from(vec![Some(1), None, None, None]),
    );
    // No arguments is an engine bug: the analyzer always has some.
    assert!(coalesce(&f, Vec::new(), &env).is_err());
}

/// `coalesce(1, 1 / 0)` answers without raising; `coalesce(1 / 0, 1)`
/// raises with ANSI on and answers `1` without.
#[test]
fn coalesce_is_lazy_in_both_modes() {
    for ansi in [true, false] {
        let env = env(ansi);
        let batch = rows_batch(2);
        let f = resolved("coalesce", DataType::Double);
        let e = coalesce(
            &f,
            vec![lit(&Literal::Double(1.0)), one_over_zero(&env)],
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Float64Type>(),
            &Float64Array::from(vec![Some(1.0), Some(1.0)]),
            "ansi = {ansi}"
        );
        let e = coalesce(
            &f,
            vec![one_over_zero(&env), lit(&Literal::Double(1.0))],
            &env,
        )
        .unwrap();
        if ansi {
            let err = e.eval(&batch, 0).unwrap_err();
            assert_eq!(err.error_class(), ErrorClass::DivideByZero);
        } else {
            assert_eq!(
                values(&e, &batch).as_primitive::<Float64Type>(),
                &Float64Array::from(vec![Some(1.0), Some(1.0)]),
            );
        }
    }
}

/// `nullif` of one type: `NULL` where equal, the value elsewhere.
#[test]
fn nullif_same_type() {
    let env = env(true);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        ArrowType::Utf8View,
        true,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringViewArray::from(vec![
            Some("x"),
            Some("y"),
            None,
        ]))],
    )
    .unwrap();
    let layout = [AttrId(0)];
    let f = resolved("nullif", DataType::String);
    let e = null_if(
        &f,
        vec![
            column(0, ArrowType::Utf8View, true),
            lit(&Literal::String("x".to_owned())),
        ],
        input(&env, &layout, &schema),
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch)
            .as_string_view()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some("y"), None]
    );
    // Two arguments are required: the analyzer always passes both.
    assert!(
        null_if(
            &f,
            vec![lit(&Literal::Integer(1))],
            input(&env, &layout, &schema),
        )
        .is_err()
    );
}

/// `nullif(1, 2.1)`: the `INT` value meets the `DOUBLE` other at
/// `DOUBLE` for the comparison, but the result keeps the `INT` value.
#[test]
fn nullif_compares_at_the_other_type_and_keeps_its_own() {
    let env = env(true);
    let batch = rows_batch(1);
    let schema = Arc::new(Schema::empty());
    let f = resolved("nullif", DataType::Integer);
    let e = null_if(
        &f,
        vec![lit(&Literal::Integer(1)), lit(&Literal::Double(2.1))],
        input(&env, &[], &schema),
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_primitive::<Int32Type>(),
        &Int32Array::from(vec![Some(1)]),
    );
    let e = null_if(
        &f,
        vec![lit(&Literal::Integer(1)), lit(&Literal::Double(1.0))],
        input(&env, &[], &schema),
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_primitive::<Int32Type>(),
        &Int32Array::from(vec![None]),
    );
}

/// `nullif(NULL, 1)` is `NULL` and `nullif(1, NULL)` is `1`: an untyped
/// `NULL` side never equals, so the value is kept.
#[test]
fn nullif_with_an_untyped_null_side() {
    let env = env(true);
    let batch = rows_batch(1);
    let schema = Arc::new(Schema::empty());
    let f = resolved("nullif", DataType::Null);
    let e = null_if(
        &f,
        vec![lit(&Literal::Null(None)), lit(&Literal::Integer(1))],
        input(&env, &[], &schema),
    )
    .unwrap();
    assert_eq!(values(&e, &batch).data_type(), &ArrowType::Null);
    let g = resolved("nullif", DataType::Integer);
    let e = null_if(
        &g,
        vec![lit(&Literal::Integer(1)), lit(&Literal::Null(None))],
        input(&env, &[], &schema),
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_primitive::<Int32Type>(),
        &Int32Array::from(vec![Some(1)]),
    );
}

/// `nullif` casts its value for the comparison in the session's mode:
/// `nullifzero('abc')` raises an ANSI cast error and answers the
/// argument without.
#[test]
fn nullif_casts_for_the_comparison_only() {
    for (ansi, want) in [(true, None), (false, Some("abc"))] {
        let env = env(ansi);
        let batch = rows_batch(1);
        let schema = Arc::new(Schema::empty());
        let f = resolved("nullif", DataType::String);
        let e = null_if(
            &f,
            vec![
                lit(&Literal::String("abc".to_owned())),
                lit(&Literal::Long(0)),
            ],
            input(&env, &[], &schema),
        )
        .unwrap();
        match want {
            None => {
                let err = e.eval(&batch, 0).unwrap_err();
                assert_eq!(err.error_class(), ErrorClass::CastInvalidInput);
                assert_eq!(err.sql_state(), Some("22018"));
            }
            Some(w) => assert_eq!(
                values(&e, &batch)
                    .as_string_view()
                    .iter()
                    .collect::<Vec<_>>(),
                [Some(w)],
            ),
        }
    }
    // A `NULL` value casts to nothing: the comparison is NULL.
    let env = env(true);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        ArrowType::Decimal64(1, 1),
        true,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(
            Decimal64Array::from(vec![Some(0)])
                .with_precision_and_scale(1, 1)
                .unwrap(),
        )],
    )
    .unwrap();
    let layout = [AttrId(0)];
    let f = resolved(
        "nullif",
        DataType::Decimal(DecimalType::try_new(1, 1).unwrap()),
    );
    let e = null_if(
        &f,
        vec![
            column(0, ArrowType::Decimal64(1, 1), true),
            lit(&Literal::Integer(0)),
        ],
        input(&env, &layout, &schema),
    )
    .unwrap();
    let got = values(&e, &batch);
    assert_eq!(got.data_type(), &ArrowType::Decimal64(1, 1));
    assert_eq!(
        got.as_primitive::<Decimal64Type>(),
        &Decimal64Array::from(vec![None as Option<i64>])
            .with_precision_and_scale(1, 1)
            .unwrap(),
    );
}

/// An expression counting its evaluations: `nullif` evaluates its first
/// argument once per batch, for the comparison and the value alike.
#[derive(Debug)]
struct Count {
    inner: Arc<dyn PhysExpr>,
    n: Arc<AtomicUsize>,
}

impl PhysExpr for Count {
    fn data_type(&self) -> &ArrowType {
        self.inner.data_type()
    }

    fn nullable(&self) -> bool {
        self.inner.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        self.n.fetch_add(1, Ordering::SeqCst);
        self.inner.eval(batch, partition)
    }
}

/// The value is computed once even though it is both compared and
/// returned.
#[test]
fn nullif_evaluates_its_first_argument_once() {
    let env = env(true);
    let batch = rows_batch(3);
    let schema = Arc::new(Schema::empty());
    let n = Arc::new(AtomicUsize::new(0));
    let value: Arc<dyn PhysExpr> = Arc::new(Count {
        inner: lit(&Literal::Long(1)),
        n: Arc::clone(&n),
    });
    let f = resolved("nullif", DataType::Long);
    let e = null_if(
        &f,
        vec![value, lit(&Literal::Long(2))],
        input(&env, &[], &schema),
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_primitive::<Int64Type>(),
        &Int64Array::from(vec![Some(1), Some(1), Some(1)]),
    );
    assert_eq!(n.load(Ordering::SeqCst), 1);
}

/// `nanvl(a, b)`: `b` where `a` is NaN, else `a`; NULL is not NaN.
#[test]
fn nanvl_takes_the_second_argument_for_nan_only() {
    let env = env(true);
    let batch = rows_batch(4);
    let schema = Arc::new(Schema::empty());
    for (ty, a, b) in [
        (
            DataType::Float,
            Literal::Float(f32::NAN),
            Literal::Float(1.0),
        ),
        (
            DataType::Double,
            Literal::Double(f64::NAN),
            Literal::Double(1.0),
        ),
    ] {
        let f = resolved("nanvl", ty);
        let e = nan_vl(&f, vec![lit(&a), lit(&b)], input(&env, &[], &schema)).unwrap();
        assert_eq!(values(&e, &batch).len(), 4);
    }
    let f = resolved("nanvl", DataType::Double);
    for (a, want) in [
        (f64::NAN, 1.0),
        (2.5, 2.5),
        (-0.0, -0.0),
        (f64::INFINITY, f64::INFINITY),
    ] {
        let e = nan_vl(
            &f,
            vec![lit(&Literal::Double(a)), lit(&Literal::Double(1.0))],
            input(&env, &[], &schema),
        )
        .unwrap();
        let got = values(&e, &batch).as_primitive::<Float64Type>().value(0);
        assert_eq!(got.to_bits(), want.to_bits(), "{a} -> {got}, want {want}");
    }
    let g = resolved("nanvl", DataType::Float);
    for (a, want) in [
        (f32::NAN, 1.0),
        (2.5, 2.5),
        (-0.0, -0.0),
        (f32::INFINITY, f32::INFINITY),
    ] {
        let e = nan_vl(
            &g,
            vec![lit(&Literal::Float(a)), lit(&Literal::Float(1.0))],
            input(&env, &[], &schema),
        )
        .unwrap();
        let got = values(&e, &batch).as_primitive::<Float32Type>().value(0);
        assert_eq!(got.to_bits(), want.to_bits(), "{a} -> {got}, want {want}");
    }
    // Anything but two same-typed floats is an engine bug.
    assert!(
        nan_vl(
            &f,
            vec![lit(&Literal::Double(1.0)), lit(&Literal::Float(1.0)),],
            input(&env, &[], &schema),
        )
        .is_err()
    );
    assert!(
        nan_vl(
            &f,
            vec![lit(&Literal::Double(1.0))],
            input(&env, &[], &schema),
        )
        .is_err()
    );
}

/// `nanvl` evaluates `b` only where `a` is NaN: `nanvl(1, 1 / 0)`
/// answers `1` without raising.
#[test]
fn nanvl_is_lazy_in_its_second_argument() {
    for ansi in [true, false] {
        let env = env(ansi);
        let batch = rows_batch(2);
        let schema = Arc::new(Schema::empty());
        let f = resolved("nanvl", DataType::Double);
        let e = nan_vl(
            &f,
            vec![lit(&Literal::Double(1.0)), one_over_zero(&env)],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Float64Type>(),
            &Float64Array::from(vec![Some(1.0), Some(1.0)]),
            "ansi = {ansi}"
        );
        let e = nan_vl(
            &f,
            vec![lit(&Literal::Double(f64::NAN)), one_over_zero(&env)],
            input(&env, &[], &schema),
        )
        .unwrap();
        if ansi {
            let err = e.eval(&batch, 0).unwrap_err();
            assert_eq!(err.error_class(), ErrorClass::DivideByZero);
        } else {
            assert!(values(&e, &batch).as_primitive::<Float64Type>().is_null(0));
        }
    }
}

/// `coalesce` of one Spark `STRING` in `Utf8` and `Utf8View` carriers:
/// every branch meets the `CASE` type, keeping its values.
#[test]
fn coalesce_unifies_string_encodings() {
    let env = env(true);
    let schema = Arc::new(Schema::new(vec![
        Field::new("u", ArrowType::Utf8, true),
        Field::new("v", ArrowType::Utf8View, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::StringArray::from(vec![None, Some("b")])) as ArrayRef,
            Arc::new(StringViewArray::from(vec![Some("v"), Some("w")])),
        ],
    )
    .unwrap();
    let f = resolved("coalesce", DataType::String);
    for (args, want) in [
        (
            vec![
                column(0, ArrowType::Utf8, true),
                column(1, ArrowType::Utf8View, true),
            ],
            [Some("v"), Some("b")],
        ),
        (
            vec![
                column(1, ArrowType::Utf8View, true),
                column(0, ArrowType::Utf8, true),
            ],
            [Some("v"), Some("w")],
        ),
    ] {
        let e = coalesce(&f, args, &env).unwrap();
        assert_eq!(e.data_type(), &ArrowType::Utf8View);
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            want
        );
    }
}

/// `CASE WHEN` with `Utf8` branches in a `STRING` (`Utf8View`) case:
/// the branches are cast; an `INT` branch still fails.
#[test]
fn when_unifies_string_encodings() {
    let env = env(true);
    let schema = Arc::new(Schema::new(vec![
        Field::new("u", ArrowType::Utf8, true),
        Field::new("i", ArrowType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::StringArray::from(vec![Some("a"), None])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(1), Some(2)])),
        ],
    )
    .unwrap();
    let e = when(
        vec![
            lit(&Literal::Boolean(true)),
            column(0, ArrowType::Utf8, true),
            lit(&Literal::String("d".to_owned())),
        ],
        ArrowType::Utf8View,
        &env,
    )
    .unwrap();
    assert_eq!(e.data_type(), &ArrowType::Utf8View);
    assert_eq!(
        values(&e, &batch)
            .as_string_view()
            .iter()
            .collect::<Vec<_>>(),
        [Some("a"), None]
    );
    // A branch of another Spark type still fails as before.
    let err = when(
        vec![
            lit(&Literal::Boolean(true)),
            column(1, ArrowType::Int32, true),
        ],
        ArrowType::Utf8View,
        &env,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("a CASE value of type Int32 in a CASE of type Utf8View"),
        "{err}"
    );
}

/// `nullif` of `Utf8` and `Utf8View` operands compares in the view and
/// returns the `STRING` carrier with the value's contents.
#[test]
fn nullif_unifies_string_encodings() {
    let env = env(true);
    let schema = Arc::new(Schema::new(vec![
        Field::new("u", ArrowType::Utf8, true),
        Field::new("v", ArrowType::Utf8View, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::StringArray::from(vec![
                Some("x"),
                Some("y"),
                None,
            ])) as ArrayRef,
            Arc::new(StringViewArray::from(vec![Some("x"), Some("z"), Some("q")])),
        ],
    )
    .unwrap();
    let layout = [AttrId(0), AttrId(1)];
    let input = input(&env, &layout, &schema);
    // A `Utf8` value against a `Utf8View` literal: the result is the
    // `STRING` carrier with the value's contents.
    let f = resolved("nullif", DataType::String);
    let e = null_if(
        &f,
        vec![
            column(0, ArrowType::Utf8, true),
            lit(&Literal::String("x".to_owned())),
        ],
        input,
    )
    .unwrap();
    assert_eq!(e.data_type(), &ArrowType::Utf8View);
    assert_eq!(
        values(&e, &batch)
            .as_string_view()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some("y"), None]
    );
    // A `Utf8View` value against a `Utf8` other: the comparison stays in
    // the view, the result keeps the view.
    let e = null_if(
        &f,
        vec![
            column(1, ArrowType::Utf8View, true),
            column(0, ArrowType::Utf8, true),
        ],
        input,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch)
            .as_string_view()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some("z"), Some("q")]
    );
}

/// A two-column batch of one Spark `STRING` in both carriers.
fn strings_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("u", ArrowType::Utf8, true),
        Field::new("v", ArrowType::Utf8View, true),
    ]));
    RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![Some("b"), None])) as ArrayRef,
            Arc::new(StringViewArray::from(vec![Some("b"), Some("c")])),
        ],
    )
    .unwrap()
}

/// `IN` over a `Utf8` value with `Utf8View` elements (and the reverse):
/// each side meets the canonical carrier once, keeping its values.
#[test]
fn in_list_meets_mixed_encodings_in_the_view() {
    let env = env(true);
    let batch = strings_batch();
    let e = in_list(
        vec![
            column(0, ArrowType::Utf8, true),
            lit(&Literal::String("b".to_owned())),
            lit(&Literal::String("c".to_owned())),
        ],
        &env,
    )
    .unwrap();
    assert_eq!(e.data_type(), &ArrowType::Boolean);
    assert!(e.nullable());
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [Some(true), None]
    );
    // A `Utf8View` value against a `Utf8` element: the element is cast.
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", ArrowType::Utf8View, true),
        Field::new("u", ArrowType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringViewArray::from(vec![Some("b"), Some("c")])),
            Arc::new(StringArray::from(vec![Some("b"), Some("b")])),
        ],
    )
    .unwrap();
    let e = in_list(
        vec![
            column(0, ArrowType::Utf8View, true),
            column(1, ArrowType::Utf8, true),
            lit(&Literal::String("x".to_owned())),
        ],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [Some(true), Some(false)]
    );
}

/// `IN` without a list is an engine bug; operands of different Spark
/// types fail here exactly as `=` does.
#[test]
fn in_list_rejects_an_empty_list_and_other_spark_types() {
    let env = env(true);
    let err = in_list(vec![column(0, ArrowType::Int32, true)], &env).unwrap_err();
    assert!(err.to_string().contains("IN without a list"), "{err}");
    let err = in_list(
        vec![
            column(0, ArrowType::Int32, true),
            lit(&Literal::String("x".to_owned())),
        ],
        &env,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("comparison of Int32 with Utf8View"),
        "{err}"
    );
}

/// An `IN` list of untyped `NULL`s never matches, and a `NULL` value
/// never matches either: every row is NULL.
#[test]
fn in_list_with_only_untyped_nulls_never_matches() {
    let env = env(true);
    let batch = strings_batch();
    let e = in_list(
        vec![
            column(0, ArrowType::Utf8, true),
            lit(&Literal::Null(None)),
            lit(&Literal::Null(None)),
        ],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [None, None]
    );
    let e = in_list(
        vec![
            lit(&Literal::Null(None)),
            lit(&Literal::String("b".to_owned())),
        ],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &rows_batch(2))
            .as_boolean()
            .iter()
            .collect::<Vec<_>>(),
        [None, None]
    );
}

/// `IN` over a batch without rows answers an empty column.
#[test]
fn in_list_on_an_empty_batch_answers_empty() {
    let env = env(true);
    let batch = rows_batch(0);
    let e = in_list(
        vec![lit(&Literal::Integer(1)), lit(&Literal::Integer(2))],
        &env,
    )
    .unwrap();
    let got = e.eval(&batch, 0).unwrap().into_array(0).unwrap();
    assert_eq!(got.len(), 0);
}

/// `IN` of scalars is one scalar: a match, or NULL when nothing
/// matched and an element is NULL.
#[test]
fn in_list_of_scalars_is_a_scalar() {
    let env = env(true);
    let batch = rows_batch(2);
    let e = in_list(
        vec![lit(&Literal::Integer(1)), lit(&Literal::Integer(1))],
        &env,
    )
    .unwrap();
    match e.eval(&batch, 0).unwrap() {
        Value::Scalar(a) => assert_eq!(a.as_boolean().iter().collect::<Vec<_>>(), [Some(true)]),
        v => panic!("not a scalar: {v:?}"),
    }
    let e = in_list(
        vec![lit(&Literal::Integer(1)), lit(&Literal::Integer(2))],
        &env,
    )
    .unwrap();
    match e.eval(&batch, 0).unwrap() {
        Value::Scalar(a) => {
            assert_eq!(a.as_boolean().iter().collect::<Vec<_>>(), [Some(false)]);
        }
        v => panic!("not a scalar: {v:?}"),
    }
    let e = in_list(
        vec![lit(&Literal::Integer(1)), lit(&Literal::Null(None))],
        &env,
    )
    .unwrap();
    match e.eval(&batch, 0).unwrap() {
        Value::Scalar(a) => {
            assert_eq!(a.as_boolean().iter().collect::<Vec<_>>(), [None]);
        }
        v => panic!("not a scalar: {v:?}"),
    }
}

/// The `IN` value is evaluated once per batch, however many elements
/// the list holds.
#[test]
fn in_list_evaluates_its_value_once() {
    let env = env(true);
    let batch = rows_batch(3);
    let n = Arc::new(AtomicUsize::new(0));
    let value: Arc<dyn PhysExpr> = Arc::new(Count {
        inner: lit(&Literal::Long(1)),
        n: Arc::clone(&n),
    });
    let e = in_list(
        vec![value, lit(&Literal::Long(1)), lit(&Literal::Long(2))],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [Some(true), Some(true), Some(true)]
    );
    assert_eq!(n.load(Ordering::SeqCst), 1);
}

/// Elements past the point where every row matched are never
/// evaluated: a later element cannot change a `true`.
#[test]
fn in_list_skips_elements_after_every_row_matched() {
    let env = env(true);
    let schema = Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int32, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
    )
    .unwrap();
    let counted = |l: &Literal, n: &Arc<AtomicUsize>| -> Arc<dyn PhysExpr> {
        Arc::new(Count {
            inner: lit(l),
            n: Arc::clone(n),
        })
    };
    let (n0, n1, n2) = (
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let e = in_list(
        vec![
            column(0, ArrowType::Int32, true),
            counted(&Literal::Integer(1), &n0),
            counted(&Literal::Integer(2), &n1),
            counted(&Literal::Integer(3), &n2),
        ],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [Some(true), Some(true)]
    );
    assert_eq!(n0.load(Ordering::SeqCst), 1);
    assert_eq!(n1.load(Ordering::SeqCst), 1);
    assert_eq!(n2.load(Ordering::SeqCst), 0);
}

/// `BETWEEN` over a `Utf8` value with `Utf8View` bounds (and a
/// `Utf8View` value with a `Utf8` bound).
#[test]
fn between_meets_mixed_encodings_in_the_view() {
    let env = env(true);
    let batch = strings_batch();
    let e = between(
        column(0, ArrowType::Utf8, true),
        lit(&Literal::String("a".to_owned())),
        lit(&Literal::String("c".to_owned())),
        [false; 2],
        &env,
    )
    .unwrap();
    assert_eq!(e.data_type(), &ArrowType::Boolean);
    assert!(e.nullable());
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [Some(true), None]
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("w", ArrowType::Utf8View, true),
        Field::new("u", ArrowType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringViewArray::from(vec![Some("b"), Some("c")])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
        ],
    )
    .unwrap();
    let e = between(
        column(0, ArrowType::Utf8View, true),
        column(1, ArrowType::Utf8, true),
        lit(&Literal::String("c".to_owned())),
        [false; 2],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [Some(true), Some(true)]
    );
}

/// `BETWEEN` over operands of different Spark types fails here exactly
/// as its two comparisons do.
#[test]
fn between_with_other_spark_types_fails_like_its_comparisons() {
    let env = env(true);
    let err = between(
        column(0, ArrowType::Int32, true),
        lit(&Literal::String("a".to_owned())),
        lit(&Literal::String("c".to_owned())),
        [false; 2],
        &env,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("comparison of Int32 with Utf8View"),
        "{err}"
    );
}

/// `LargeUtf8` is one more encoding of `STRING`: `IN` and `BETWEEN`
/// meet it in the canonical carrier like `Utf8`; a dictionary-encoded
/// string is no encoding of it and fails as `=` does.
#[test]
fn large_utf8_meets_the_view_and_dictionaries_do_not() {
    let env = env(true);
    let schema = Arc::new(Schema::new(vec![
        Field::new("l", ArrowType::LargeUtf8, true),
        Field::new(
            "d",
            ArrowType::Dictionary(Box::new(ArrowType::Int32), Box::new(ArrowType::Utf8)),
            true,
        ),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::LargeStringArray::from(vec![Some("b"), None])) as ArrayRef,
            Arc::new(
                arrow_array::DictionaryArray::<arrow_array::types::Int32Type>::from_iter([
                    Some("b"),
                    None,
                ]),
            ),
        ],
    )
    .unwrap();
    let lit_s = |t: &str| lit(&Literal::String(t.to_owned()));
    let e = in_list(
        vec![
            column(0, ArrowType::LargeUtf8, true),
            lit_s("b"),
            lit_s("c"),
        ],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [Some(true), None]
    );
    let e = between(
        column(0, ArrowType::LargeUtf8, true),
        lit_s("a"),
        lit_s("c"),
        [false; 2],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [Some(true), None]
    );
    let dictionary = ArrowType::Dictionary(Box::new(ArrowType::Int32), Box::new(ArrowType::Utf8));
    assert!(in_list(vec![column(1, dictionary.clone(), true), lit_s("b")], &env).is_err());
    assert!(
        between(
            column(1, dictionary, true),
            lit_s("a"),
            lit_s("c"),
            [false; 2],
            &env
        )
        .is_err()
    );
}

/// `BETWEEN` over a batch without rows answers an empty column.
#[test]
fn between_on_an_empty_batch_answers_empty() {
    let env = env(true);
    let batch = rows_batch(0);
    let e = between(
        lit(&Literal::Integer(1)),
        lit(&Literal::Integer(0)),
        lit(&Literal::Integer(2)),
        [false; 2],
        &env,
    )
    .unwrap();
    let got = e.eval(&batch, 0).unwrap().into_array(0).unwrap();
    assert_eq!(got.len(), 0);
}

/// `false AND _` is `false`: where `v >= lo` already decided every
/// row, the upper bound is never evaluated.
#[test]
fn between_skips_the_upper_bound_where_the_lower_decided_false() {
    let env = env(true);
    let n = Arc::new(AtomicUsize::new(0));
    let hi: Arc<dyn PhysExpr> = Arc::new(Count {
        inner: lit(&Literal::Long(3)),
        n: Arc::clone(&n),
    });
    let e = between(
        lit(&Literal::Long(1)),
        lit(&Literal::Long(5)),
        hi,
        [false; 2],
        &env,
    )
    .unwrap();
    match e.eval(&rows_batch(2), 0).unwrap() {
        Value::Scalar(a) => {
            assert_eq!(a.as_boolean().iter().collect::<Vec<_>>(), [Some(false)]);
        }
        v => panic!("not a scalar: {v:?}"),
    }
    assert_eq!(n.load(Ordering::SeqCst), 0);
    // One decided row does not skip the bound for the rest.
    let schema = Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int32, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 7])) as ArrayRef],
    )
    .unwrap();
    let n = Arc::new(AtomicUsize::new(0));
    let hi: Arc<dyn PhysExpr> = Arc::new(Count {
        inner: lit(&Literal::Integer(6)),
        n: Arc::clone(&n),
    });
    let e = between(
        column(0, ArrowType::Int32, true),
        lit(&Literal::Integer(5)),
        hi,
        [false; 2],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [Some(false), Some(false)]
    );
    assert_eq!(n.load(Ordering::SeqCst), 1);
}

/// `BETWEEN` is `v >= lo AND v <= hi` under Kleene logic: NULL in,
/// NULL out, unless one side already decided `false`.
#[test]
fn between_follows_kleene_logic_for_nulls() {
    let env = env(true);
    let batch = rows_batch(1);
    for (v, lo, hi, want) in [
        (
            Literal::Long(2),
            Literal::Long(1),
            Literal::Null(None),
            None,
        ),
        (
            Literal::Long(2),
            Literal::Null(None),
            Literal::Long(3),
            None,
        ),
        (
            Literal::Null(None),
            Literal::Long(1),
            Literal::Long(3),
            None,
        ),
        (
            Literal::Long(5),
            Literal::Long(1),
            Literal::Long(3),
            Some(false),
        ),
        // `false AND NULL` is `false`: the lower bound already decided.
        (
            Literal::Long(0),
            Literal::Long(1),
            Literal::Null(None),
            Some(false),
        ),
        (
            Literal::Long(2),
            Literal::Long(1),
            Literal::Long(3),
            Some(true),
        ),
    ] {
        let e = between(lit(&v), lit(&lo), lit(&hi), [false; 2], &env).unwrap();
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [want],
            "{v:?} BETWEEN {lo:?} AND {hi:?}"
        );
    }
}

/// The `BETWEEN` value is evaluated once per batch, for both bounds
/// alike.
#[test]
fn between_evaluates_its_value_once() {
    let env = env(true);
    let batch = rows_batch(1);
    let n = Arc::new(AtomicUsize::new(0));
    let value: Arc<dyn PhysExpr> = Arc::new(Count {
        inner: lit(&Literal::Long(1)),
        n: Arc::clone(&n),
    });
    let e = between(
        value,
        lit(&Literal::Long(0)),
        lit(&Literal::Long(2)),
        [false; 2],
        &env,
    )
    .unwrap();
    assert_eq!(
        values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
        [Some(true)]
    );
    assert_eq!(n.load(Ordering::SeqCst), 1);
}

/// A `nullif` value of another Spark type than the result keeps no
/// result cast: the mix fails at evaluation, as the analyzer never
/// produces it.
#[test]
fn nullif_with_a_value_outside_the_result_type_fails_evaluation() {
    let env = env(true);
    let batch = rows_batch(1);
    let schema = Arc::new(Schema::empty());
    let f = resolved("nullif", DataType::String);
    let e = null_if(
        &f,
        vec![lit(&Literal::Integer(1)), lit(&Literal::Integer(1))],
        input(&env, &[], &schema),
    )
    .unwrap();
    assert_eq!(e.data_type(), &ArrowType::Utf8View);
    assert!(e.eval(&batch, 0).is_err());
}
