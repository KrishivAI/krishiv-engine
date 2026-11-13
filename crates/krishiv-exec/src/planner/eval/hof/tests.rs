// SPDX-License-Identifier: Apache-2.0
//! Tests of the higher-order functions.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Int32Array, ListArray, RecordBatch};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::KernelCtx;
use krishiv_plan::{
    ColumnRef, Expr, ExprNode, FunctionId, FunctionKind, ResolvedExpr, ResolvedFunction,
};

use super::super::{EvalEnv, Input, PhysExpr, Value, compile};
use super::{Call, too_many};
use crate::error::Result;

/// A [`PhysExpr`] returning `array` for every batch.
#[derive(Debug)]
struct Const {
    data_type: ArrowType,
    nullable: bool,
    array: ArrayRef,
}

impl PhysExpr for Const {
    fn data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, _batch: &RecordBatch, _partition: u32) -> Result<Value> {
        Ok(Value::Array(Arc::clone(&self.array)))
    }
}

fn env() -> EvalEnv {
    EvalEnv {
        ansi: true,
        legacy_statistical_aggregate: false,
        zone: None,
        zone_name: None,
        kernels: KernelCtx::new(),
        query_start_micros: 0,
        user: Arc::from("test"),
        udfs: None,
        map_key_dedup_last_win: false,
        legacy_java_charsets: false,
        legacy_coding_error_action: false,
        legacy_time_parser: false,
        binary_style: krishiv_kernels::binary_style::BinaryStyle::default(),
        seed_salt: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        partition_shift: 0,
    }
}

fn col() -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Column(ColumnRef::new(vec![
        "a".to_owned(),
    ]))))
}

fn lambda() -> Arc<Expr> {
    Arc::new(Expr::lambda(vec!["x".to_owned()], Expr::column(["x"])))
}

fn func(name: &str, args: Vec<Arc<Expr>>) -> ResolvedFunction {
    ResolvedFunction {
        id: FunctionId::Registry {
            name: name.to_owned(),
            id: 0,
        },
        kind: FunctionKind::Scalar,
        args,
        distinct: false,
        filter: None,
        within_group: Vec::new(),
        data_type: krishiv_types::DataType::Null,
        nullable: true,
        deterministic: true,
    }
}

fn list(offsets: Vec<i32>, values: Vec<i32>) -> ArrayRef {
    Arc::new(ListArray::new(
        Arc::new(Field::new("element", ArrowType::Int32, true)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(Int32Array::from(values)),
        None,
    ))
}

fn batch(rows: usize) -> RecordBatch {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int32, false)]));
    RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![0; rows])) as ArrayRef],
    )
    .unwrap()
}

fn call(name: &str, array: ArrayRef, body: ArrayRef, out: ArrowType) -> Call {
    Call {
        name: name.to_owned(),
        array: Arc::new(Const {
            data_type: array.data_type().clone(),
            nullable: false,
            array,
        }),
        body: Arc::new(Const {
            data_type: body.data_type().clone(),
            nullable: true,
            array: body,
        }),
        sources: Vec::new(),
        out,
        nullable: true,
    }
}

#[test]
fn malformed_calls_are_internal_errors() {
    let e = env();
    let schema: SchemaRef = Arc::new(Schema::new(Vec::<Field>::new()));
    let input = Input {
        layout: &[],
        schema: &schema,
        env: &e,
        subqueries: &[],
    };
    // Without the array and the lambda.
    let f = func("transform", vec![lambda()]);
    let err = compile(
        &Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))),
        input,
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    assert!(
        err.to_string()
            .contains("a higher-order call without its array and lambda"),
        "{err}"
    );
    // Without a lambda as the second argument (the array never compiles).
    let f = func("transform", vec![col(), col()]);
    let err = compile(
        &Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))),
        input,
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    assert!(
        err.to_string()
            .contains("a higher-order call without a lambda"),
        "{err}"
    );
}

#[test]
fn too_many_names_the_row_count() {
    let err = too_many(5);
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    assert!(err.to_string().contains('5'), "{err}");
}

#[test]
fn mismatched_batches_are_internal_errors() {
    // The array holds one row where the batch holds two.
    let c = call(
        "transform",
        Arc::new(Int32Array::from(vec![1])),
        Arc::new(Int32Array::from(vec![1, 2])),
        ArrowType::Boolean,
    );
    let err = c.eval(&batch(2), 0).unwrap_err();
    assert!(
        err.to_string()
            .contains("a higher-order array of the wrong row count"),
        "{err}"
    );
    // The body holds three values for two elements.
    let c = call(
        "transform",
        list(vec![0, 2], vec![5, 6]),
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        ArrowType::Boolean,
    );
    let err = c.eval(&batch(1), 0).unwrap_err();
    assert!(
        err.to_string()
            .contains("a higher-order body of the wrong row count"),
        "{err}"
    );
}

#[test]
fn a_transform_over_a_non_list_is_an_internal_error() {
    let c = call(
        "transform",
        list(vec![0, 2], vec![5, 6]),
        Arc::new(Int32Array::from(vec![7, 8])),
        ArrowType::Int32,
    );
    let err = c.eval(&batch(1), 0).unwrap_err();
    assert!(
        err.to_string()
            .contains("a higher-order call over a non-list"),
        "{err}"
    );
}

#[test]
fn an_unknown_higher_order_function_is_unsupported() {
    let c = call(
        "bogus",
        list(vec![0, 2], vec![5, 6]),
        Arc::new(BooleanArray::from(vec![true, false])),
        ArrowType::Boolean,
    );
    let err = c.eval(&batch(1), 0).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        err.to_string().contains("higher-order function bogus"),
        "{err}"
    );
}

#[test]
fn a_sliced_list_rebases_to_zero_based_offsets() {
    use arrow_array::Array as _;
    use arrow_array::cast::AsArray as _;
    // `[[1, 2], [3]]` sliced to its second row: absolute offsets
    // `[2, 3]` with one element.
    let sliced = list(vec![0, 2, 3], vec![1, 2, 3]).slice(1, 1);
    let mut c = call(
        "transform",
        sliced,
        Arc::new(Int32Array::from(vec![9])),
        ArrowType::List(Arc::new(Field::new("element", ArrowType::Int32, true))),
    );
    c.sources = vec![super::Source::Element(0)];
    let out = c.eval(&batch(1), 0).unwrap().into_array(1).unwrap();
    let out = out.as_list::<i32>();
    assert_eq!(out.len(), 1);
    assert_eq!(out.value_offsets(), &[0, 1]);
    assert_eq!(
        out.values()
            .as_primitive::<arrow_array::types::Int32Type>()
            .values(),
        &[9]
    );
}

#[test]
fn a_body_of_an_unexpected_type_is_an_internal_error() {
    let out = ArrowType::List(Arc::new(Field::new("element", ArrowType::Utf8, true)));
    let c = call(
        "transform",
        list(vec![0, 2], vec![5, 6]),
        Arc::new(Int32Array::from(vec![7, 8])),
        out,
    );
    let err = c.eval(&batch(1), 0).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    assert!(
        err.to_string().contains("transform returned List(Int32"),
        "{err}"
    );
    assert!(
        err.to_string().contains("analyzer typed List(Utf8"),
        "{err}"
    );
}

/// String and binary storage alternatives read as the analyzer's view
/// carriers; any other mismatch is an engine error.
#[test]
fn canonical_reads_storage_alternatives_as_views() {
    use arrow_array::{BinaryArray, LargeBinaryArray, LargeStringArray, StringArray};
    let cases: Vec<(ArrayRef, ArrowType)> = vec![
        (Arc::new(StringArray::from(vec!["a"])), ArrowType::Utf8View),
        (
            Arc::new(LargeStringArray::from(vec!["a"])),
            ArrowType::Utf8View,
        ),
        (
            Arc::new(BinaryArray::from_vec(vec![b"a"])),
            ArrowType::BinaryView,
        ),
        (
            Arc::new(LargeBinaryArray::from_vec(vec![b"a"])),
            ArrowType::BinaryView,
        ),
    ];
    for (column, target) in cases {
        let out = super::super::hof::canonical(&column, &target).unwrap();
        assert_eq!(out.data_type(), &target);
        assert_eq!(out.len(), 1);
    }
    let ints: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    assert!(super::super::hof::canonical(&ints, &ArrowType::Utf8View).is_err());
    assert!(super::super::hof::canonical(&ints, &ArrowType::Int32).is_ok());
}

/// A NULL row may hold elements: `array_sort` keeps them in place and
/// still sorts the others.
#[test]
fn array_sort_keeps_the_elements_of_null_rows() {
    use arrow_array::Array as _;
    use arrow_array::cast::AsArray;
    use arrow_buffer::NullBuffer;
    let field = Arc::new(Field::new("item", ArrowType::Int32, true));
    let list: ArrayRef = Arc::new(ListArray::new(
        Arc::clone(&field),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 4, 6])),
        Arc::new(Int32Array::from(vec![2, 1, 9, 8, 4, 3])),
        Some(NullBuffer::from(vec![true, false, true])),
    ));
    let out = ArrowType::List(Arc::clone(&field));
    let array = Arc::new(Const {
        data_type: out.clone(),
        nullable: true,
        array: list,
    });
    let sort = super::Sort {
        array,
        comparator: None,
        schema: Arc::new(Schema::empty()),
        sources: Vec::new(),
        out,
        nullable: true,
    };
    let batch = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(3)),
    )
    .unwrap();
    let got = sort.eval(&batch, 0).unwrap().into_array(3).unwrap();
    let got = got.as_list::<i32>();
    assert!(got.is_null(1));
    let values = got.values().as_primitive::<arrow_array::types::Int32Type>();
    assert_eq!(values.values().as_ref(), [1, 2, 9, 8, 3, 4]);
}

/// The state machine sorts like the sequential algorithm for every
/// input shape: ascending, descending and mixed runs, ties, singles.
#[test]
fn the_sort_machine_matches_the_sequential_sort() {
    let cases: Vec<Vec<i32>> = vec![
        vec![],
        vec![1],
        vec![2, 1],
        vec![1, 2, 3],
        vec![3, 2, 1],
        vec![1, 3, 2, 4, 0, 5, 5, 1],
        vec![5, 5, 5, 4, 4, 9, 1],
        (0..40).map(|i| (i * 7919) % 31).collect(),
    ];
    for case in cases {
        let cmp = |a: usize, b: usize| Ok(case[a].cmp(&case[b]));
        let mut expected: Vec<usize> = (0..case.len()).collect();
        super::stable_sort(&mut expected, &cmp).unwrap();
        let mut machine = super::Machine::new((0..case.len()).collect());
        while let Some((a, b)) = machine.request() {
            machine.answer(case[a].cmp(&case[b]));
        }
        assert_eq!(machine.perm, expected, "{case:?}");
    }
}

/// Key columns that differ from the analyzer's type only in nested
/// nullability read as the analyzer's type.
#[test]
fn canonical_reads_nullability_variants() {
    let have = ArrowType::List(Arc::new(Field::new("item", ArrowType::Int32, true)));
    let want = ArrowType::List(Arc::new(Field::new("element", ArrowType::Int32, false)));
    let list: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new("item", ArrowType::Int32, true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1])),
        Arc::new(Int32Array::from(vec![1])),
        None,
    ));
    assert_eq!(list.data_type(), &have);
    let out = super::super::hof::canonical(&list, &want).unwrap();
    assert_eq!(out.data_type(), &want);
    assert_eq!(out.len(), 1);
}
