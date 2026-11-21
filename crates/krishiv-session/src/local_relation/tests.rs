// SPDX-License-Identifier: Apache-2.0
//! `Session::local_relation` (P3-T012a part 2a): IPC in, a carrier over the session's relation
//! store out; eager casts; the lease keeping the entry alive; a decoded carrier resolved by hash.

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, StringArray, StructArray};
use arrow_schema::{DataType as A, Field, Fields, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_memory::RelationStore;
use krishiv_plan::{LocalRelation, LogicalPlan, Plan, PlanNode, ReadSource};
use krishiv_types::batch::Batch;
use krishiv_types::{DataType, StructField, StructType};

use super::{conform, relabel, same_shape};
use crate::functions::col;
use crate::session::Session;

fn ints(name: &str, nullable: bool) -> Field {
    Field::new(name, A::Int32, nullable)
}

fn batch(schema: &SchemaRef, cols: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::clone(schema), cols).unwrap()
}

/// `batches` as the IPC bytes a client sends.
fn ipc(schema: &SchemaRef, batches: Vec<RecordBatch>) -> Vec<u8> {
    let scratch = RelationStore::new();
    let batches: Vec<Batch> = batches.into_iter().map(Batch::new).collect();
    let lease = scratch.insert(schema, &batches).unwrap();
    scratch.encode_ipc(&lease.hash()).unwrap()
}

fn two_ints() -> (SchemaRef, Vec<u8>) {
    let schema = Arc::new(Schema::new(vec![ints("x", false), ints("y", true)]));
    let b = |from: i32| {
        batch(
            &schema,
            vec![
                Arc::new(Int32Array::from(vec![from, from + 1])),
                Arc::new(Int32Array::from(vec![Some(from), None])),
            ],
        )
    };
    let bytes = ipc(&schema, vec![b(1), b(10)]);
    (schema, bytes)
}

fn dataframe(s: &Session, l: &LocalRelation) -> crate::dataframe::DataFrame {
    let plan = LogicalPlan::new(PlanNode::Read(ReadSource::Local(l.clone())));
    s.from_plan(&Plan::try_new(plan).unwrap())
}

fn rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

fn user(fields: &[(&str, DataType, bool)]) -> StructType {
    StructType::new(
        fields
            .iter()
            .map(|(n, t, nullable)| StructField::new(*n, t.clone(), *nullable)),
    )
}

#[test]
fn data_round_trips_through_collect() {
    let s = Session::new().unwrap();
    let (_, bytes) = two_ints();
    let l = block_on(s.local_relation(Some(&bytes), None)).unwrap();
    assert_eq!((l.rows(), l.batches()), (4, 2));
    let out = block_on(dataframe(&s, &l).collect()).unwrap();
    assert_eq!(rows(&out), 4);
    let names: Vec<_> = out[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["x", "y"]);
    let x: Vec<i32> = out
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    assert_eq!(x, [1, 2, 10, 11]);
}

#[test]
fn schema_only_data_and_a_schema_alone_give_zero_rows_and_neither_is_an_error() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![ints("x", true)]));
    let empty = ipc(&schema, Vec::new());
    let l = block_on(s.local_relation(Some(&empty), None)).unwrap();
    assert_eq!(l.rows(), 0);
    assert_eq!(rows(&block_on(dataframe(&s, &l).collect()).unwrap()), 0);

    let want = user(&[("a", DataType::Long, true)]);
    let l = block_on(s.local_relation(None, Some(want))).unwrap();
    assert_eq!((l.rows(), l.batches()), (0, 0));
    assert_eq!(l.schema().field(0).data_type(), &A::Int64);
    assert_eq!(rows(&block_on(dataframe(&s, &l).collect()).unwrap()), 0);

    let err = block_on(s.local_relation(None, None)).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::UnableToInferSchema);
}

#[test]
fn a_pure_rename_skips_execution_and_stores_one_copy() {
    let s = Session::new().unwrap();
    let (_, bytes) = two_ints();
    let plain = block_on(s.local_relation(Some(&bytes), None)).unwrap();
    let one_copy = s.relations().reserved();
    assert_eq!(s.relations().len(), 1);
    drop(plain);
    assert_eq!(s.relations().len(), 0);
    let want = user(&[
        ("a", DataType::Integer, false),
        ("b", DataType::Integer, true),
    ]);
    let l = block_on(s.local_relation(Some(&bytes), Some(want))).unwrap();
    assert_eq!(
        s.relations().len(),
        1,
        "only the renamed relation is stored"
    );
    assert_eq!(s.relations().reserved(), one_copy);
    let names: Vec<_> = l
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["a", "b"]);
}

#[test]
fn relabelling_shares_the_buffers_of_nested_columns() {
    let inner = Fields::from(vec![ints("p", true)]);
    let outer = Fields::from(vec![ints("q", true)]);
    let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    let have: ArrayRef = Arc::new(StructArray::new(inner, vec![Arc::clone(&values)], None));
    let want = A::Struct(outer);
    assert!(same_shape(have.data_type(), &want));
    let out = relabel(&have, &want).unwrap();
    assert_eq!(out.data_type(), &want);
    let before = have.as_struct().column(0).to_data().buffers()[0].as_ptr();
    let after = out.as_struct().column(0).to_data().buffers()[0].as_ptr();
    assert_eq!(before, after, "the values buffer is shared");
}

#[test]
fn a_null_in_a_non_nullable_field_fails_at_any_depth() {
    let s = Session::new().unwrap();
    // Top level: `y` holds a null.
    let (_, bytes) = two_ints();
    let want = user(&[
        ("a", DataType::Integer, false),
        ("b", DataType::Integer, false),
    ]);
    let err = block_on(s.local_relation(Some(&bytes), Some(want))).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::NotNullAssertViolation);
    assert_eq!(s.relations().len(), 0, "nothing is left behind");

    // Nested: a struct child declared non-nullable holds a null under a valid struct.
    let child = Fields::from(vec![ints("p", true)]);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        A::Struct(child.clone()),
        true,
    )]));
    let col: ArrayRef = Arc::new(StructArray::new(
        child,
        vec![Arc::new(Int32Array::from(vec![Some(1), None])) as ArrayRef],
        None,
    ));
    let bytes = ipc(&schema, vec![batch(&schema, vec![col])]);
    let strict = StructType::new([StructField::new("p", DataType::Integer, false)]);
    let want = user(&[("s", DataType::Struct(strict), true)]);
    let err = block_on(s.local_relation(Some(&bytes), Some(want))).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::NotNullAssertViolation);

    // The same shape with the child nullable is fine.
    let loose = StructType::new([StructField::new("q", DataType::Integer, true)]);
    let want = user(&[("t", DataType::Struct(loose), true)]);
    let l = block_on(s.local_relation(Some(&bytes), Some(want))).unwrap();
    let out = block_on(dataframe(&s, &l).collect()).unwrap();
    let col = out[0].column(0).as_struct();
    assert_eq!(col.column_names(), ["q"], "the nested field is renamed");
    let q = col.column(0).as_primitive::<Int32Type>();
    assert_eq!((q.len(), q.value(0), q.is_null(1)), (2, 1, true));
}

#[test]
fn the_cast_path_stores_only_the_result() {
    let s = Session::new().unwrap();
    let (_, bytes) = two_ints();
    let want = user(&[("a", DataType::Long, false), ("b", DataType::Long, true)]);
    let l = block_on(s.local_relation(Some(&bytes), Some(want))).unwrap();
    assert_eq!(s.relations().len(), 1, "the temporary input was released");
    assert_eq!(l.schema().field(0).data_type(), &A::Int64);
    let out = block_on(dataframe(&s, &l).collect()).unwrap();
    let a: Vec<i64> = out
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
        .collect();
    assert_eq!(a, [1, 2, 10, 11]);
    // One copy of the result: the same batches stored on their own account for the same bytes.
    let held = s.relations().reserved();
    let scratch = RelationStore::new();
    let stored: Vec<Batch> = s
        .relations()
        .resolve(&l.relation().hash, 0..l.batches())
        .unwrap();
    let _kept = scratch.insert(l.schema(), &stored).unwrap();
    assert_eq!(held, scratch.reserved());
}

#[test]
fn a_failing_cast_or_a_wrong_count_errors_from_local_relation() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("w", A::Utf8, true)]));
    let col: ArrayRef = Arc::new(StringArray::from(vec!["abc"]));
    let bytes = ipc(&schema, vec![batch(&schema, vec![col])]);
    let want = user(&[("n", DataType::Integer, false)]);
    let err = block_on(s.local_relation(Some(&bytes), Some(want))).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::CastInvalidInput);
    assert_eq!(s.relations().len(), 0, "nothing is left behind");
    // Control: a string that parses casts.
    let col: ArrayRef = Arc::new(StringArray::from(vec!["123"]));
    let ok = ipc(&schema, vec![batch(&schema, vec![col])]);
    let want = user(&[("n", DataType::Integer, false)]);
    let l = block_on(s.local_relation(Some(&ok), Some(want))).unwrap();
    let out = block_on(dataframe(&s, &l).collect()).unwrap();
    assert_eq!(out[0].column(0).as_primitive::<Int32Type>().value(0), 123);

    let (_, bytes) = two_ints();
    let want = user(&[("only", DataType::Integer, true)]);
    let err = block_on(s.local_relation(Some(&bytes), Some(want))).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::NumColumnsMismatch);
}

#[test]
fn identical_data_shares_one_entry_and_the_last_clone_releases_it() {
    let s = Session::new().unwrap();
    let (_, bytes) = two_ints();
    let a = block_on(s.local_relation(Some(&bytes), None)).unwrap();
    let b = block_on(s.local_relation(Some(&bytes), None)).unwrap();
    assert_eq!(a, b);
    assert_eq!(s.relations().len(), 1);
    let held = s.relations().reserved();
    assert!(held > 0);
    let clone = a.clone();
    drop((a, b));
    assert_eq!(s.relations().reserved(), held, "a clone still holds it");
    drop(clone);
    assert_eq!(s.relations().len(), 0);
    assert_eq!(s.relations().reserved(), 0);
}

#[test]
fn a_scan_deals_its_batches_as_splits() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let s = Session::new().unwrap();
    let (_, bytes) = two_ints();
    let l = block_on(s.local_relation(Some(&bytes), None)).unwrap();
    let df = dataframe(&s, &l);
    let analysis = rt.block_on(df.analysis()).unwrap();
    let (_, physical) = rt
        .block_on(s.physical(&analysis.analyzed, &analysis.env))
        .unwrap();
    let specs = analysis.env.catalog.task_specs(&physical, 2, 2).unwrap();
    let mut seen = 0;
    for task in &specs {
        for splits in task.scans.values() {
            if let krishiv_exec::physical::ScanSplits::Batches { relation, range } = splits {
                assert_eq!(relation, &l.relation());
                seen += range.len();
            }
        }
    }
    assert_eq!(seen, 2, "both batches are dealt: {specs:?}");
    let mut total = 0;
    let mut xs: Vec<i32> = Vec::new();
    for task in &specs {
        let out = rt
            .block_on(s.run_task(&physical, &analysis.env, task))
            .unwrap();
        total += rows(&out);
        for b in &out {
            xs.extend(b.column(0).as_primitive::<Int32Type>().values().iter());
        }
    }
    assert_eq!(total, 4);
    xs.sort_unstable();
    assert_eq!(xs, [1, 2, 10, 11], "each row is read once");
}

#[test]
fn the_same_query_returns_the_same_rows_under_plan_round_trip() {
    let (_, bytes) = two_ints();
    let run = |round_trip: &str| {
        let s = Session::builder()
            .config("krishiv.exec.debug.planRoundTrip", round_trip)
            .build()
            .unwrap();
        let l = block_on(s.local_relation(Some(&bytes), None)).unwrap();
        let df = dataframe(&s, &l)
            .filter(col("x").gt(1))
            .order_by([col("x")]);
        block_on(df.collect()).unwrap()
    };
    let (plain, through) = (run("false"), run("true"));
    assert_eq!(rows(&plain), 3);
    assert_eq!(plain, through);
}

#[test]
fn a_decoded_carrier_whose_entry_is_gone_fails_loudly() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![ints("x", true)]));
    let l = LocalRelation::from_decoded(schema, [9; 32], 1, 1, 8);
    let err = block_on(dataframe(&s, &l).collect()).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    assert!(err.to_string().contains(&"09".repeat(32)), "{err}");
}

#[test]
fn conform_relabels_names_without_copying() {
    let schema = Arc::new(Schema::new(vec![ints("x", true)]));
    let have = batch(&schema, vec![Arc::new(Int32Array::from(vec![1, 2]))]);
    let target = Arc::new(Schema::new(vec![ints("renamed", true)]));
    let out = conform(&have, &target).unwrap();
    assert_eq!(out.schema(), target);
    assert_eq!(
        have.column(0).to_data().buffers()[0].as_ptr(),
        out.column(0).to_data().buffers()[0].as_ptr()
    );
}

#[test]
fn an_empty_stream_validates_the_conversion_like_a_zero_row_batch() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![ints("x", true)]));
    let want = || {
        user(&[(
            "x",
            krishiv_types::ArrayType::new(DataType::Integer, true).into(),
            true,
        )])
    };
    let empty = ipc(&schema, Vec::new());
    let zero_col: ArrayRef = Arc::new(Int32Array::from(Vec::<i32>::new()));
    let zero = ipc(&schema, vec![batch(&schema, vec![zero_col])]);
    let a = block_on(s.local_relation(Some(&zero), Some(want()))).unwrap_err();
    let b = block_on(s.local_relation(Some(&empty), Some(want()))).unwrap_err();
    // Both refuse the conversion `INT` to `ARRAY<INT>`, as Spark's cast check does.
    for e in [&a, &b] {
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::DatatypeMismatchCastWithoutSuggestion,
            "{e}"
        );
    }
    assert_eq!(a.error_class(), b.error_class());
}

#[test]
fn utf8_and_binary_columns_meet_a_string_and_binary_schema() {
    use arrow_array::BinaryArray;
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("s", A::Utf8, true),
        Field::new("b", A::Binary, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![
            Some("a"),
            None,
            Some("a longer string value"),
        ])),
        Arc::new(BinaryArray::from_iter(vec![
            Some(b"x".as_slice()),
            None,
            Some(b"yz".as_slice()),
        ])),
    ];
    let bytes = ipc(&schema, vec![batch(&schema, cols)]);
    let want = user(&[("s", DataType::String, true), ("b", DataType::Binary, true)]);
    let l = block_on(s.local_relation(Some(&bytes), Some(want))).unwrap();
    let out = block_on(dataframe(&s, &l).collect()).unwrap();
    assert_eq!(rows(&out), 3);
    let strs = out[0].column(0).as_string_view();
    assert_eq!(strs.value(0), "a");
    assert!(strs.is_null(1));
    assert_eq!(strs.value(2), "a longer string value");
    let bins = out[0].column(1).as_binary_view();
    assert_eq!(bins.value(2), b"yz");
}

/// MLlib's `VectorUDT` storage: `struct<type: tinyint, size: int, indices: array<int>, values: array<double>>`.
fn vector_storage() -> DataType {
    use krishiv_types::ArrayType;
    DataType::Struct(StructType::new(vec![
        StructField::new("type", DataType::Byte, false),
        StructField::new("size", DataType::Integer, true),
        StructField::new(
            "indices",
            DataType::Array(Arc::new(ArrayType::new(DataType::Integer, false))),
            true,
        ),
        StructField::new(
            "values",
            DataType::Array(Arc::new(ArrayType::new(DataType::Double, false))),
            true,
        ),
    ]))
}

fn vector_udt() -> DataType {
    DataType::Udt(Arc::new(krishiv_types::UdtType::jvm(
        vector_storage(),
        "org.apache.spark.ml.linalg.VectorUDT".to_owned(),
        Some("pyspark.ml.linalg.VectorUDT".to_owned()),
    )))
}

/// P3-T079: a LocalRelation whose user schema names a UDT. The result keeps the UDT in its schema whether the data
/// already has the storage type (no cast) or needs a cast to it.
#[test]
fn a_udt_column_in_the_user_schema_is_kept_with_and_without_a_cast() {
    let s = Session::new().unwrap();
    // (a) a UDT over int, data of another width: the cast goes to the storage type.
    let long_schema = Arc::new(Schema::new(vec![Field::new("p", A::Int64, true)]));
    let data = batch(
        &long_schema,
        vec![Arc::new(arrow_array::Int64Array::from(vec![Some(7), None]))],
    );
    let point = DataType::Udt(Arc::new(krishiv_types::UdtType::jvm(
        DataType::Integer,
        "org.example.PointUDT".to_owned(),
        None,
    )));
    let l = block_on(s.local_relation(
        Some(&ipc(&long_schema, vec![data])),
        Some(user(&[("p", point.clone(), true)])),
    ))
    .unwrap();
    assert_eq!(
        DataType::from_arrow_field(l.schema().field(0)).unwrap(),
        point
    );
    let out = block_on(dataframe(&s, &l).collect()).unwrap();
    assert_eq!(out[0].column(0).as_primitive::<Int32Type>().value(0), 7);
    assert!(out[0].column(0).is_null(1));
    assert_eq!(
        DataType::from_arrow_field(out[0].schema().field_with_name("p").unwrap()).unwrap(),
        point
    );

    // (b) a VectorUDT over a struct of arrays: data that needs a cast (`size` arrives as bigint)...
    let storage = vector_storage();
    let arrow_storage = storage.to_arrow();
    let A::Struct(fields) = &arrow_storage else {
        panic!("storage is a struct")
    };
    let mut wide: Vec<Field> = fields.iter().map(|f| f.as_ref().clone()).collect();
    wide[1] = Field::new("size", A::Int64, true);
    let wide_schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        A::Struct(wide.clone().into()),
        true,
    )]));
    let v = StructArray::try_new(
        wide.into(),
        vec![
            Arc::new(arrow_array::Int8Array::from(vec![1])) as ArrayRef,
            Arc::new(arrow_array::Int64Array::from(vec![3])),
            Arc::new(list_of(arrow_array::Int32Array::from(vec![0, 2]))),
            Arc::new(list_of(arrow_array::Float64Array::from(vec![1.5, 2.5]))),
        ],
        None,
    )
    .unwrap();
    let data = batch(&wide_schema, vec![Arc::new(v)]);
    let vector = vector_udt();
    let l = block_on(s.local_relation(
        Some(&ipc(&wide_schema, vec![data])),
        Some(user(&[("v", vector.clone(), true)])),
    ))
    .unwrap();
    assert_eq!(
        DataType::from_arrow_field(l.schema().field(0)).unwrap(),
        vector
    );
    let out = block_on(dataframe(&s, &l).collect()).unwrap();
    assert_eq!(rows(&out), 1);
    let got = out[0].column(0).as_struct();
    assert_eq!(got.column(1).as_primitive::<Int32Type>().value(0), 3);

    // ... and data that already has the storage type needs none.
    let exact_schema = Arc::new(Schema::new(vec![storage.to_arrow_field("v", true)]));
    let exact = StructArray::try_new(
        fields.clone(),
        vec![
            Arc::new(arrow_array::Int8Array::from(vec![0])) as ArrayRef,
            Arc::new(Int32Array::from(vec![4])),
            Arc::new(list_of(Int32Array::from(vec![1]))),
            Arc::new(list_of(arrow_array::Float64Array::from(vec![9.0]))),
        ],
        None,
    )
    .unwrap();
    let data = batch(&exact_schema, vec![Arc::new(exact)]);
    let l = block_on(s.local_relation(
        Some(&ipc(&exact_schema, vec![data])),
        Some(user(&[("v", vector.clone(), true)])),
    ))
    .unwrap();
    assert_eq!(
        DataType::from_arrow_field(l.schema().field(0)).unwrap(),
        vector
    );
}

/// A one-row `list<element>` array holding `values`.
fn list_of<T: arrow_array::Array + 'static>(values: T) -> arrow_array::ListArray {
    let field = Arc::new(Field::new("element", values.data_type().clone(), false));
    let n = i32::try_from(values.len()).unwrap();
    arrow_array::ListArray::new(
        field,
        arrow_buffer::OffsetBuffer::new(vec![0, n].into()),
        Arc::new(values),
        None,
    )
}
