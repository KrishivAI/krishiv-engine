// SPDX-License-Identifier: Apache-2.0
//! Higher-order functions over arrays end to end: `transform`, `filter`,
//! `exists` and `forall`, with the `array` constructor they need. Values,
//! output names and error classes follow Spark 4.2's
//! `higher-order-functions.sql` golden; the error texts of the
//! constant-lambda mismatches match its `messageParameters` exactly.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array, ListArray, MapArray, RecordBatch, StructArray};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;
use krishiv_types::{ArrayType, DataType};

/// `t(ys array<int>, zs array<array<int>>, n int)`, five rows: two
/// populated, `[12]`, a NULL row and an empty row.
fn session() -> Session {
    let s = Session::new().unwrap();
    let element = Field::new("element", ArrowType::Int32, false);
    let mut ys_builder =
        arrow_array::builder::ListBuilder::new(arrow_array::builder::Int32Builder::new())
            .with_field(Arc::new(element.clone()));
    ys_builder.append_value([Some(32), Some(97)]);
    ys_builder.append_value([Some(77), Some(-76)]);
    ys_builder.append_value([Some(12)]);
    ys_builder.append_null();
    ys_builder.append_value([]);
    let ys: ArrayRef = Arc::new(ys_builder.finish());
    let inner: ArrayRef = Arc::new(ListArray::new(
        Arc::new(element),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 3, 3, 4])),
        Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
        None,
    ));
    let zs = ListArray::new(
        Arc::new(Field::new(
            "element",
            ArrowType::List(Arc::new(Field::new("element", ArrowType::Int32, false))),
            true,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 2, 3, 3, 4])),
        inner,
        Some(NullBuffer::from(vec![true, false, true, false, true])),
    );
    let n: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("ys", ys.data_type().clone(), true),
        Field::new("zs", zs.data_type().clone(), true),
        Field::new("n", ArrowType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ys, Arc::new(zs), n]).unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

/// `SELECT CAST(<expr> AS STRING) [<from>]`: every row of the single
/// column, `None` for NULL.
fn strings(s: &Session, expr: &str, from: &str) -> Vec<Option<String>> {
    let sql = format!("SELECT CAST({expr} AS STRING) {from}");
    let df = block_on(s.sql(&sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    block_on(df.collect())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_string_view()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn error(s: &Session, sql: &str) -> KrishivError {
    match block_on(s.sql(sql)) {
        Ok(_) => panic!("{sql} resolved"),
        Err(e) => e,
    }
}

/// The runtime error of `sql` (analysis succeeds; execution raises).
fn error_at_runtime(s: &Session, sql: &str) -> KrishivError {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    match block_on(df.collect()) {
        Err(e) => e,
        Ok(_) => panic!("{sql} ran"),
    }
}

#[test]
fn transform_maps_each_element() {
    let s = session();
    assert_eq!(
        strings(&s, "transform(ys, y -> y * y)", "FROM t"),
        [
            Some("[1024, 9409]".to_owned()),
            Some("[5929, 5776]".to_owned()),
            Some("[144]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
    // A NULL array is NULL, whatever the body.
    assert_eq!(
        strings(&s, "transform(CAST(NULL AS ARRAY<INT>), x -> x + 1)", ""),
        [None]
    );
    // A constant lambda repeats its body.
    assert_eq!(
        strings(&s, "transform(array(1, 2, 3), 42)", ""),
        [Some("[42, 42, 42]".to_owned())]
    );
}

#[test]
fn transform_with_index_and_captures() {
    let s = session();
    // The index counts positions within each list.
    assert_eq!(
        strings(&s, "transform(ys, (y, i) -> y + i)", "FROM t"),
        [
            Some("[32, 98]".to_owned()),
            Some("[77, -75]".to_owned()),
            Some("[12]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
    // An outer column repeats per element.
    assert_eq!(
        strings(&s, "transform(ys, y -> y + n)", "FROM t"),
        [
            Some("[33, 98]".to_owned()),
            Some("[79, -74]".to_owned()),
            Some("[15]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
    // A lambda variable shadows an outer column of the same name.
    assert_eq!(
        strings(&s, "transform(ys, n -> n + 1)", "FROM t"),
        [
            Some("[33, 98]".to_owned()),
            Some("[78, -75]".to_owned()),
            Some("[13]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
}

#[test]
fn higher_order_functions_nest() {
    let s = session();
    assert_eq!(
        strings(&s, "transform(zs, z -> filter(z, zz -> zz > 1))", "FROM t"),
        [
            Some("[[2], [3]]".to_owned()),
            None,
            Some("[[]]".to_owned()),
            None,
            Some("[[4]]".to_owned()),
        ]
    );
}

#[test]
fn nested_lambdas_capture_enclosing_variables() {
    let s = session();
    // The inner body reads the outer call's element variable: every level
    // carries what the levels below need.
    assert_eq!(
        strings(
            &s,
            "transform(ys, y -> transform(array(1, 2), z -> z + y))",
            "FROM t"
        ),
        [
            Some("[[33, 34], [98, 99]]".to_owned()),
            Some("[[78, 79], [-75, -74]]".to_owned()),
            Some("[[13, 14]]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
    // The inner body reads the outer call's index variable.
    assert_eq!(
        strings(
            &s,
            "transform(ys, (y, i) -> transform(array(10, 2 * 10), z -> z + i))",
            "FROM t"
        ),
        [
            Some("[[10, 20], [11, 21]]".to_owned()),
            Some("[[10, 20], [11, 21]]".to_owned()),
            Some("[[10, 20]]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
    // An outer column reaches through two levels, beside the element.
    assert_eq!(
        strings(
            &s,
            "transform(ys, y -> transform(array(1, 2), z -> z + y + n))",
            "FROM t"
        ),
        [
            Some("[[34, 35], [99, 100]]".to_owned()),
            Some("[[80, 81], [-73, -72]]".to_owned()),
            Some("[[16, 17]]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
    // Three levels: the middle call carries both its element and the outer
    // call's element down to the innermost body.
    assert_eq!(
        strings(
            &s,
            "transform(ys, y -> transform(ys, z -> transform(array(1), w -> w + y + z)))",
            "FROM t"
        ),
        [
            Some("[[[65], [130]], [[130], [195]]]".to_owned()),
            Some("[[[155], [2]], [[2], [-151]]]".to_owned()),
            Some("[[[25]]]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
    // Shadowing still binds the innermost variable: the inner `y` is its
    // own element, not the outer call's.
    assert_eq!(
        strings(
            &s,
            "transform(ys, y -> transform(array(100), y -> y + 1))",
            "FROM t"
        ),
        [
            Some("[[101], [101]]".to_owned()),
            Some("[[101], [101]]".to_owned()),
            Some("[[101]]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
}

#[test]
fn filter_keeps_true_elements() {
    let s = session();
    assert_eq!(
        strings(&s, "filter(ys, y -> y > 30)", "FROM t"),
        [
            Some("[32, 97]".to_owned()),
            Some("[77]".to_owned()),
            Some("[]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
    // A NULL predicate drops its element; a NULL array is NULL.
    assert_eq!(
        strings(&s, "filter(array(1, NULL, 3), x -> x > 1)", ""),
        [Some("[3]".to_owned())]
    );
    assert_eq!(
        strings(&s, "filter(CAST(NULL AS ARRAY<INT>), y -> TRUE)", ""),
        [None]
    );
    assert_eq!(
        strings(&s, "filter(array(1, 2, 3), TRUE)", ""),
        [Some("[1, 2, 3]".to_owned())]
    );
}

#[test]
fn exists_and_forall_fold_the_marks() {
    let s = session();
    assert_eq!(
        strings(&s, "exists(ys, y -> y > 30)", "FROM t"),
        [
            Some("true".to_owned()),
            Some("true".to_owned()),
            Some("false".to_owned()),
            None,
            Some("false".to_owned()),
        ]
    );
    assert_eq!(
        strings(&s, "forall(ys, y -> y > 30)", "FROM t"),
        [
            Some("true".to_owned()),
            Some("false".to_owned()),
            Some("false".to_owned()),
            None,
            Some("true".to_owned()),
        ]
    );
    // A NULL array folds to NULL, whatever the body.
    assert_eq!(
        strings(&s, "exists(CAST(NULL AS ARRAY<INT>), y -> y > 30)", ""),
        [None]
    );
    assert_eq!(
        strings(&s, "forall(CAST(NULL AS ARRAY<INT>), y -> y > 30)", ""),
        [None]
    );
    // NULL marks decide nothing: no true for `exists`, no false for
    // `forall`, so each folds to NULL.
    assert_eq!(
        strings(&s, "exists(array(1, NULL), x -> x > 1)", ""),
        [None]
    );
    assert_eq!(
        strings(&s, "forall(array(1, NULL), x -> x >= 1)", ""),
        [None]
    );
}

#[test]
fn higher_order_functions_report_bad_arguments() {
    let s = session();
    // Every higher-order call takes the array and the lambda.
    for name in ["transform", "filter", "exists", "forall"] {
        let e = error(&s, &format!("SELECT {name}(ys) FROM t"));
        assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
        assert!(e.to_string().contains(&format!("`{name}`")), "{e}");
    }
    // The array argument needs `ARRAY`, quoting the call as Spark does.
    for name in ["transform", "filter", "exists", "forall"] {
        let e = error(&s, &format!("SELECT {name}(1, x -> TRUE)"));
        assert_eq!(
            e.error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{name}"
        );
        assert_eq!(
            e.to_string(),
            format!(
                "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve \
                 \"{name}(1, lambdafunction(true, x))\" due to data type mismatch: \
                 The first parameter requires the \"ARRAY\" type, \
                 however \"1\" has the type \"INT\". SQLSTATE: 42K09"
            )
        );
    }
}

#[test]
fn transform_keeps_null_elements() {
    let s = session();
    assert_eq!(
        strings(&s, "transform(array(1, NULL, 3), x -> x)", ""),
        [Some("[1, null, 3]".to_owned())]
    );
    assert_eq!(
        strings(&s, "transform(array(1, NULL, 3), (x, i) -> i)", ""),
        [Some("[0, 1, 2]".to_owned())]
    );
}

#[test]
fn lambdas_outside_argument_position_are_rejected() {
    let s = session();
    // A lambda under a cast is no higher-order call's argument (its body
    // binds on its own, so the lambda itself is what fails).
    let e = error(&s, "SELECT CAST((x -> 1) AS STRING)");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        e.to_string()
            .contains("lambda functions outside a higher-order function"),
        "{e}"
    );
}

#[test]
fn array_builds_lists() {
    let s = session();
    assert_eq!(
        strings(&s, "array(1, 2, 3)", ""),
        [Some("[1, 2, 3]".to_owned())]
    );
    assert_eq!(strings(&s, "array()", ""), [Some("[]".to_owned())]);
    assert_eq!(
        strings(&s, "array(NULL, NULL)", ""),
        [Some("[null, null]".to_owned())]
    );
    // Over columns, with the common type.
    assert_eq!(
        strings(&s, "array(n, n * 2)", "FROM t"),
        [
            Some("[1, 2]".to_owned()),
            Some("[2, 4]".to_owned()),
            Some("[3, 6]".to_owned()),
            Some("[4, 8]".to_owned()),
            Some("[5, 10]".to_owned()),
        ]
    );
}

#[test]
fn lambdas_outside_higher_order_functions_are_rejected() {
    let s = session();
    let e = error(&s, "SELECT upper(x -> x) AS v");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidLambdaFunctionCallNonHigherOrderFunction
    );
    assert_eq!(e.sql_state(), Some("42K0D"));
    assert!(e.to_string().contains(
        "org.apache.spark.sql.catalyst.expressions.Upper, which is not a higher order function"
    ));
    let e = error(&s, "SELECT ceil(x -> x) AS v");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidLambdaFunctionCallNonHigherOrderFunction
    );
    assert!(
        e.to_string()
            .contains("org.apache.spark.sql.catalyst.expressions.CeilExpressionBuilder$")
    );
    // A bare lambda binds its body without variables first (which fails),
    // so it stays unresolved rather than reaching the lambda check.
    let e = error(&s, "SELECT x -> x AS v");
    assert_eq!(
        e.error_class(),
        ErrorClass::UnresolvedColumnWithoutSuggestion
    );
}

#[test]
fn constant_lambdas_of_the_wrong_type_are_rejected() {
    let s = session();
    for name in ["filter", "exists", "forall"] {
        let e = error(&s, &format!("SELECT {name}(array(1, 2, 3), 1) AS v"));
        assert_eq!(
            e.error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{name}"
        );
        assert_eq!(e.sql_state(), Some("42K09"));
        assert_eq!(
            e.to_string(),
            format!(
                "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve \
                 \"{name}(array(1, 2, 3), lambdafunction(1, namedlambdavariable()))\" \
                 due to data type mismatch: The second parameter requires the \"BOOLEAN\" type, \
                 however lambdafunction(1, namedlambdavariable()) has the type \"INT\". SQLSTATE: 42K09"
            )
        );
    }
    // The array argument is checked first: it needs `ARRAY`.
    let e = error(&s, "SELECT transform(1, x -> x) AS v");
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert_eq!(
        e.to_string(),
        "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve \
         \"transform(1, lambdafunction(x, x))\" due to data type mismatch: \
         The first parameter requires the \"ARRAY\" type, however \"1\" has the type \"INT\". SQLSTATE: 42K09"
    );
}

#[test]
fn lambda_shapes_are_checked() {
    let s = session();
    // `transform` takes one or two variables, the others exactly one.
    let e = error(&s, "SELECT transform(ys, (a, b, c) -> a) FROM t");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidLambdaFunctionCallNumArgsMismatch
    );
    let e = error(&s, "SELECT filter(ys, (a, b) -> a) FROM t");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidLambdaFunctionCallNumArgsMismatch
    );
    // Duplicate variables.
    let e = error(&s, "SELECT transform(ys, (a, a) -> a) FROM t");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidLambdaFunctionCallDuplicateArgNames
    );
    // Arity of the call itself.
    let e = error(&s, "SELECT transform(ys) FROM t");
    assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
    // Aggregates and windows stay outside lambda bodies.
    let e = error(&s, "SELECT transform(ys, y -> sum(y)) FROM t");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let e = error(
        &s,
        "SELECT transform(ys, y -> rank() OVER (ORDER BY n)) FROM t",
    );
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn higher_order_output_names_and_types() {
    let s = session();
    let df = block_on(s.sql("SELECT transform(ys, y -> y + 1) FROM t")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let field = &schema.fields()[0];
    assert_eq!(
        DataType::from_arrow_field(field).unwrap(),
        DataType::Array(ArrayType::new(DataType::Integer, false).into())
    );
    // Spark's `toPrettySQL` names.
    let batches = block_on(df.collect()).unwrap();
    assert_eq!(
        batches[0].schema().field(0).name(),
        "transform(ys, lambdafunction((namedlambdavariable() + 1), namedlambdavariable()))"
    );
}

#[test]
fn lambda_variables_respect_case_sensitivity() {
    let s = session();
    // Case-insensitive (the default): `n` names the `N` parameter.
    assert_eq!(
        strings(&s, "transform(ys, N -> n + 1)", "FROM t"),
        [
            Some("[33, 98]".to_owned()),
            Some("[78, -75]".to_owned()),
            Some("[13]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    // Case-sensitive: `n` is the outer column, `N` the (unused) element.
    assert_eq!(
        strings(&s, "transform(ys, N -> n + 1)", "FROM t"),
        [
            Some("[2, 2]".to_owned()),
            Some("[3, 3]".to_owned()),
            Some("[4]".to_owned()),
            None,
            Some("[]".to_owned()),
        ]
    );
    s.conf().unset("spark.sql.caseSensitive").unwrap();
}

#[test]
fn scalar_subqueries_in_lambda_bodies_are_rejected() {
    let s = session();
    let sql = "SELECT transform(ys, y -> (SELECT max(n) FROM t)) FROM t";
    let err = match block_on(s.sql(sql)) {
        Err(e) => e,
        Ok(df) => match block_on(df.collect()) {
            Err(e) => e,
            Ok(_) => panic!("{sql} ran"),
        },
    };
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{err}"
    );
    assert!(
        err.to_string()
            .contains("scalar subqueries in lambda functions"),
        "{err}"
    );
}

#[test]
fn large_lists_are_rejected() {
    use arrow_array::LargeListArray;
    let s = Session::new().unwrap();
    let large = LargeListArray::from_iter_primitive::<arrow_array::types::Int32Type, _, _>(vec![
        Some(vec![Some(1), Some(2)]),
        Some(vec![Some(3)]),
    ]);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "ys",
        large.data_type().clone(),
        true,
    )]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(large) as ArrayRef]).unwrap();
    s.register_batches("u", schema, vec![batch]).unwrap();
    let sql = "SELECT transform(ys, y -> y + 1) FROM u";
    let err = match block_on(s.sql(sql)) {
        Err(e) => e,
        Ok(df) => match block_on(df.collect()) {
            Err(e) => e,
            Ok(_) => panic!("{sql} ran over a LargeList"),
        },
    };
    // A graceful engine error, never a panic or wrong rows.
    assert!(
        err.to_string().contains("no list") || err.to_string().contains("non-list"),
        "{err}"
    );
}

/// The golden `nested` view of `higher-order-functions.sql`: the array half.
fn nested_arrays() -> Session {
    let s = Session::new().unwrap();
    let element = Field::new("element", ArrowType::Int32, false);
    let mut ys = arrow_array::builder::ListBuilder::new(arrow_array::builder::Int32Builder::new())
        .with_field(Arc::new(element.clone()));
    ys.append_value([Some(32), Some(97)]);
    ys.append_value([Some(77), Some(-76)]);
    ys.append_value([Some(12)]);
    let inner = ListArray::new(
        Arc::new(element),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 4, 5, 8, 10, 11])),
        Arc::new(Int32Array::from(vec![
            12, 99, 123, 42, 1, 6, 96, 65, -1, -2, 17,
        ])),
        None,
    );
    let zs = ListArray::new(
        Arc::new(Field::new(
            "element",
            ArrowType::List(Arc::new(Field::new("element", ArrowType::Int32, false))),
            true,
        )),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 3, 5, 6])),
        Arc::new(inner),
        None,
    );
    let x: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    let ys: ArrayRef = Arc::new(ys.finish());
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowType::Int32, false),
        Field::new("ys", ys.data_type().clone(), true),
        Field::new("zs", zs.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![x, ys, Arc::new(zs)]).unwrap();
    s.register_batches("nested", schema, vec![batch]).unwrap();
    s
}

/// The golden `nested` view of `higher-order-functions.sql`: the map half.
fn nested_maps() -> Session {
    let s = Session::new().unwrap();
    let entries = || {
        Arc::new(StructArray::new(
            vec![
                Arc::new(Field::new("key", ArrowType::Int32, false)),
                Arc::new(Field::new("value", ArrowType::Int32, true)),
            ]
            .into(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])) as ArrayRef,
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])) as ArrayRef,
            ],
            None,
        ))
    };
    let field = Arc::new(Field::new(
        "entries",
        ArrowType::Struct(
            vec![
                Field::new("key", ArrowType::Int32, false),
                Field::new("value", ArrowType::Int32, true),
            ]
            .into(),
        ),
        false,
    ));
    let maps = MapArray::new(
        Arc::clone(&field),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 3, 6])),
        entries().slice(0, 6),
        None,
        false,
    );
    let x: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ArrowType::Int32, false),
        Field::new("ys", maps.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![x, Arc::new(maps)]).unwrap();
    s.register_batches("nested", schema, vec![batch]).unwrap();
    s
}

#[test]
fn aggregate_folds_per_row() {
    let s = nested_arrays();
    // The golden rows of `aggregate(ys, 0, (y, a) -> y + a + x)`.
    assert_eq!(
        strings(&s, "aggregate(ys, 0, (y, a) -> y + a + x)", "FROM nested"),
        [
            Some("131".to_owned()),
            Some("5".to_owned()),
            Some("15".to_owned()),
        ]
    );
    // A NULL array is NULL without running the merge; an empty array keeps
    // the zero, converted by the finish.
    assert_eq!(
        strings(
            &s,
            "aggregate(CAST(NULL AS ARRAY<INT>), 0, (a, y) -> a + y + 1, a -> a + 2)",
            ""
        ),
        [None]
    );
    assert_eq!(
        strings(
            &s,
            "aggregate(array(), 10, (a, y) -> a + y, a -> a * 2)",
            ""
        ),
        [Some("20".to_owned())]
    );
    // A constant merge repeats its body; without `finish` the accumulator
    // is the result.
    assert_eq!(
        strings(&s, "aggregate(array(1, 2, 3), 0, 100)", ""),
        [Some("100".to_owned())]
    );
    // Nested folds see the enclosing element and the outer query.
    assert_eq!(
        strings(
            &s,
            "transform(zs, z -> aggregate(z, 1, (acc, val) -> acc * val * size(z)))",
            "FROM nested"
        ),
        [
            Some("[4752, 20664, 1]".to_owned()),
            Some("[1010880, 8]".to_owned()),
            Some("[17]".to_owned()),
        ]
    );
}

#[test]
fn reduce_is_aggregates_alias() {
    let s = nested_arrays();
    assert_eq!(
        strings(&s, "reduce(ys, 0, (y, a) -> y + a + x)", "FROM nested"),
        [
            Some("131".to_owned()),
            Some("5".to_owned()),
            Some("15".to_owned()),
        ]
    );
    assert_eq!(
        strings(
            &s,
            "reduce(CAST(NULL AS ARRAY<INT>), 0, (a, y) -> a + y + 1, a -> a + 2)",
            ""
        ),
        [None]
    );
}

#[test]
fn zip_with_pads_the_shorter_side() {
    let s = nested_arrays();
    // The golden rows of `zip_with(ys, zs, (a, b) -> a + size(b))`: the
    // shorter side reads NULL.
    assert_eq!(
        strings(&s, "zip_with(ys, zs, (a, b) -> a + size(b))", "FROM nested"),
        [
            Some("[34, 99, null]".to_owned()),
            Some("[80, -74]".to_owned()),
            Some("[13]".to_owned()),
        ]
    );
    // A NULL on either side is a NULL row.
    assert_eq!(
        strings(
            &s,
            "zip_with(ys, CAST(NULL AS ARRAY<INT>), (a, b) -> a)",
            "FROM nested"
        ),
        [None, None, None]
    );
    // Unequal lengths pad with NULLs, as Spark does.
    assert_eq!(
        strings(
            &s,
            "zip_with(array(1), array('d', null, 'f'), (x, y) -> coalesce(y, 'z'))",
            ""
        ),
        [Some("[d, z, f]".to_owned())]
    );
}

#[test]
fn map_filter_keeps_matching_entries() {
    let s = nested_maps();
    assert_eq!(
        strings(
            &s,
            "CAST(map_filter(ys, (k, v) -> v > 1) AS STRING)",
            "FROM nested"
        ),
        [
            Some("{2 -> 2, 3 -> 3}".to_owned()),
            Some("{4 -> 4, 5 -> 5, 6 -> 6}".to_owned()),
        ]
    );
    // A NULL mark drops its entry; a NULL map is NULL.
    assert_eq!(
        strings(
            &s,
            "CAST(map_filter(map('a', 1, 'b', 2), (k, v) -> v > 1) AS STRING)",
            ""
        ),
        [Some("{b -> 2}".to_owned())]
    );
    assert_eq!(
        strings(
            &s,
            "map_filter(CAST(NULL AS MAP<STRING, INT>), (k, v) -> TRUE)",
            ""
        ),
        [None]
    );
    // Captured columns and nesting work through the flattened entries.
    assert_eq!(
        strings(
            &s,
            "CAST(map_filter(ys, (k, v) -> k + v > x * 2) AS STRING)",
            "FROM nested"
        ),
        [
            Some("{2 -> 2, 3 -> 3}".to_owned()),
            Some("{4 -> 4, 5 -> 5, 6 -> 6}".to_owned()),
        ]
    );
}

#[test]
fn transform_keys_rebuilds_keys() {
    let s = nested_maps();
    assert_eq!(
        strings(
            &s,
            "CAST(transform_keys(ys, (k, v) -> k + 1) AS STRING)",
            "FROM nested"
        ),
        [
            Some("{2 -> 1, 3 -> 2, 4 -> 3}".to_owned()),
            Some("{5 -> 4, 6 -> 5, 7 -> 6}".to_owned()),
        ]
    );
    // Duplicate keys raise, quoting the policy and the key.
    let e = error_at_runtime(&s, "SELECT transform_keys(map('a', 1, 'b', 2), 42)");
    assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey, "{e:?}");
    assert_eq!(e.sql_state(), Some("23505"));
    assert!(
        e.to_string().contains("Duplicate map key 42 was found"),
        "{e}"
    );
    assert!(
        e.to_string().contains("\"spark.sql.mapKeyDedupPolicy\""),
        "{e}"
    );
    // `LAST_WIN` keeps the last value instead.
    s.conf()
        .set("spark.sql.mapKeyDedupPolicy", "LAST_WIN")
        .unwrap();
    assert_eq!(
        strings(
            &s,
            "CAST(transform_keys(map('a', 1, 'b', 2), 42) AS STRING)",
            ""
        ),
        [Some("{42 -> 2}".to_owned())]
    );
    s.conf().unset("spark.sql.mapKeyDedupPolicy").unwrap();
    // NULL keys raise, whatever the policy.
    let e = error_at_runtime(
        &s,
        "SELECT transform_keys(map('a', 1), (k, v) -> CAST(NULL AS INT))",
    );
    assert_eq!(e.error_class(), ErrorClass::NullMapKey);
}

#[test]
fn transform_values_rebuilds_values() {
    let s = nested_maps();
    assert_eq!(
        strings(
            &s,
            "CAST(transform_values(ys, (k, v) -> k + v) AS STRING)",
            "FROM nested"
        ),
        [
            Some("{1 -> 2, 2 -> 4, 3 -> 6}".to_owned()),
            Some("{4 -> 8, 5 -> 10, 6 -> 12}".to_owned()),
        ]
    );
    assert_eq!(
        strings(&s, "CAST(transform_values(ys, 0) AS STRING)", "FROM nested"),
        [
            Some("{1 -> 0, 2 -> 0, 3 -> 0}".to_owned()),
            Some("{4 -> 0, 5 -> 0, 6 -> 0}".to_owned()),
        ]
    );
}

#[test]
fn map_zip_with_merges_key_sets() {
    let s = nested_maps();
    // First-seen order, missing values NULL.
    assert_eq!(
        strings(
            &s,
            "CAST(map_zip_with(map('a', 1, 'b', 2), map('b', 3, 'c', 4), (k, v1, v2) -> concat(coalesce(CAST(v1 AS STRING), '?'), coalesce(CAST(v2 AS STRING), '?'))) AS STRING)",
            ""
        ),
        [Some("{a -> 1?, b -> 23, c -> ?4}".to_owned())]
    );
    // Duplicate keys in one map read the first entry (a duplicated-key
    // map built through the Rust API, as `map()` itself dedups).
    let field = Arc::new(Field::new(
        "entries",
        ArrowType::Struct(
            vec![
                Field::new("key", ArrowType::Utf8View, false),
                Field::new("value", ArrowType::Int32, true),
            ]
            .into(),
        ),
        false,
    ));
    let entries = Arc::new(StructArray::new(
        vec![
            Arc::new(Field::new("key", ArrowType::Utf8View, false)),
            Arc::new(Field::new("value", ArrowType::Int32, true)),
        ]
        .into(),
        vec![
            Arc::new(arrow_array::StringViewArray::from(vec!["a", "a"])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        ],
        None,
    ));
    let maps = MapArray::new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2])),
        entries.as_ref().clone(),
        None,
        false,
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "m",
        maps.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(maps)]).unwrap();
    s.register_batches("dups", schema, vec![batch]).unwrap();
    assert_eq!(
        strings(
            &s,
            "CAST(map_zip_with(m, map('a', 10), (k, v1, v2) -> v1 + v2) AS STRING)",
            "FROM dups"
        ),
        [Some("{a -> 11}".to_owned())]
    );
    // A NULL on either side is a NULL map.
    assert_eq!(
        strings(
            &s,
            "map_zip_with(ys, CAST(NULL AS MAP<INT, INT>), (k, v1, v2) -> v1)",
            "FROM nested"
        ),
        [None, None]
    );
}

#[test]
fn array_sort_orders_elements() {
    let s = session();
    // Ascending with NULLs last, without a comparator.
    assert_eq!(
        strings(&s, "array_sort(array(3, 1, 2))", ""),
        [Some("[1, 2, 3]".to_owned())]
    );
    assert_eq!(
        strings(&s, "array_sort(array('b', 'd', null, 'c', 'a'))", ""),
        [Some("[a, b, c, d, null]".to_owned())]
    );
    assert_eq!(
        strings(&s, "array_sort(CAST(NULL AS ARRAY<INT>))", ""),
        [None]
    );
    // A comparator decides the order: descending here.
    assert_eq!(
        strings(
            &s,
            "array_sort(array(5, 6, 1), (l, r) -> CASE WHEN l < r THEN 1 WHEN l > r THEN -1 ELSE 0 END)",
            ""
        ),
        [Some("[6, 5, 1]".to_owned())]
    );
    // A constant comparator leaves every array in place, as Spark's sort
    // does for a run it reads as ordered.
    assert_eq!(
        strings(&s, "array_sort(array(3, 1, 2), 1)", ""),
        [Some("[3, 1, 2]".to_owned())]
    );
    // A NULL comparison result raises Spark's error.
    let e = error_at_runtime(
        &s,
        "SELECT array_sort(array(1, 2), (l, r) -> CAST(NULL AS INT))",
    );
    assert_eq!(e.error_class(), ErrorClass::ComparatorReturnsNull);
    assert_eq!(e.sql_state(), Some("22004"));
}

#[test]
fn depends_on_command_matches_the_golden_rows() {
    // The golden `nested` views, registered through the Rust API: the
    // statements the golden runner classifies `depends-on-command` assert
    // the golden's rows here instead.
    let s = nested_arrays();
    assert_eq!(
        strings(&s, "aggregate(ys, 0, (y, a) -> y + a + x)", "FROM nested"),
        [
            Some("131".to_owned()),
            Some("5".to_owned()),
            Some("15".to_owned())
        ]
    );
    assert_eq!(
        strings(
            &s,
            "transform(zs, z -> aggregate(z, 1, (acc, val) -> acc * val * size(z)))",
            "FROM nested"
        ),
        [
            Some("[4752, 20664, 1]".to_owned()),
            Some("[1010880, 8]".to_owned()),
            Some("[17]".to_owned()),
        ]
    );
    assert_eq!(
        strings(&s, "reduce(ys, 0, (y, a) -> y + a + x)", "FROM nested"),
        [
            Some("131".to_owned()),
            Some("5".to_owned()),
            Some("15".to_owned())
        ]
    );
    assert_eq!(
        strings(
            &s,
            "transform(zs, z -> reduce(z, 1, (acc, val) -> acc * val * size(z)))",
            "FROM nested"
        ),
        [
            Some("[4752, 20664, 1]".to_owned()),
            Some("[1010880, 8]".to_owned()),
            Some("[17]".to_owned()),
        ]
    );
    assert_eq!(
        strings(&s, "zip_with(ys, zs, (a, b) -> a + size(b))", "FROM nested"),
        [
            Some("[34, 99, null]".to_owned()),
            Some("[80, -74]".to_owned()),
            Some("[13]".to_owned()),
        ]
    );
    let s = nested_maps();
    for (sql, rows) in [
        (
            "transform_keys(ys, (k, v) -> k)",
            ["{1 -> 1, 2 -> 2, 3 -> 3}", "{4 -> 4, 5 -> 5, 6 -> 6}"],
        ),
        (
            "transform_keys(ys, (k, v) -> k + 1)",
            ["{2 -> 1, 3 -> 2, 4 -> 3}", "{5 -> 4, 6 -> 5, 7 -> 6}"],
        ),
        (
            "transform_values(ys, (k, v) -> v)",
            ["{1 -> 1, 2 -> 2, 3 -> 3}", "{4 -> 4, 5 -> 5, 6 -> 6}"],
        ),
        (
            "transform_values(ys, (k, v) -> v + 1)",
            ["{1 -> 2, 2 -> 3, 3 -> 4}", "{4 -> 5, 5 -> 6, 6 -> 7}"],
        ),
    ] {
        assert_eq!(
            strings(&s, &format!("CAST({sql} AS STRING)"), "FROM nested"),
            rows.map(|r| Some(r.to_owned())),
            "{sql}"
        );
    }
}

#[test]
fn size_counts_arrays_and_maps() {
    let s = nested_arrays();
    assert_eq!(
        strings(&s, "size(ys)", "FROM nested"),
        [
            Some("2".to_owned()),
            Some("2".to_owned()),
            Some("1".to_owned()),
        ]
    );
    // NULL is NULL with ANSI on and -1 with ANSI off.
    assert_eq!(strings(&s, "size(CAST(NULL AS ARRAY<INT>))", ""), [None]);
    // Maps count their entries.
    assert_eq!(
        strings(&nested_maps(), "size(ys)", "FROM nested"),
        [Some("3".to_owned()), Some("3".to_owned())]
    );
}

#[test]
fn split_cuts_strings() {
    let s = session();
    assert_eq!(
        strings(&s, "split('abcdefgh', '')", ""),
        [Some("[a, b, c, d, e, f, g, h]".to_owned())]
    );
    assert_eq!(
        strings(&s, "split('oneAtwoBthreeC', '[ABC]', -1)", ""),
        [Some("[one, two, three, ]".to_owned())]
    );
    assert_eq!(
        strings(&s, "split('oneAtwoBthreeC', '[ABC]', 2)", ""),
        [Some("[one, twoBthreeC]".to_owned())]
    );
    assert_eq!(strings(&s, "split(CAST(NULL AS STRING), ',')", ""), [None]);
}

#[test]
fn map_builds_maps() {
    let s = session();
    assert_eq!(
        strings(&s, "CAST(map('a', 1, 'b', 2) AS STRING)", ""),
        [Some("{a -> 1, b -> 2}".to_owned())]
    );
    assert_eq!(
        strings(&s, "CAST(map() AS STRING)", ""),
        [Some("{}".to_owned())]
    );
    // An odd count is the wrong arity.
    let e = error(&s, "SELECT map('a', 1, 'b')");
    assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
    // Keys and values each share a type.
    let e = error(&s, "SELECT map(1, 'a', 'b', 'c')");
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchCreateMapKeyDiffTypes
    );
    // NULL keys raise at run time.
    let e = error_at_runtime(&s, "SELECT map(CAST(NULL AS INT), 1)");
    assert_eq!(e.error_class(), ErrorClass::NullMapKey);
}
