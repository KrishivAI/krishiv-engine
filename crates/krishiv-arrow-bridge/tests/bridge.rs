// SPDX-License-Identifier: Apache-2.0
//! `ffi_layout` and `bridge_roundtrip` (`design/P3-lakehouse.md` §5.2, §5.3):
//! the C Data Interface structs of both Arrow majors match the specification,
//! and what crosses arrives equal to the same value built natively on Arrow 60.
//!
//! The round-trip cases are written once as a macro over an Arrow major, so the
//! next bump (61) changes the crate names, not the cases.

use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::mem::{align_of, offset_of, size_of};
use std::sync::Arc;

use krishiv_arrow_bridge::{record_batch_58_to_60, schema_58_to_60};

/// The specification's `struct ArrowSchema`, transcribed.
#[repr(C)]
struct SpecSchema {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *mut *mut SpecSchema,
    dictionary: *mut SpecSchema,
    release: Option<unsafe extern "C" fn(*mut SpecSchema)>,
    private_data: *mut c_void,
}

/// The specification's `struct ArrowArray`, transcribed.
#[repr(C)]
struct SpecArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut *const c_void,
    children: *mut *mut SpecArray,
    dictionary: *mut SpecArray,
    release: Option<unsafe extern "C" fn(*mut SpecArray)>,
    private_data: *mut c_void,
}

#[test]
fn ffi_layout() {
    use arrow_array::ffi::{FFI_ArrowArray as A60, FFI_ArrowSchema as S60};
    use arrow_array_58::ffi::{FFI_ArrowArray as A58, FFI_ArrowSchema as S58};

    // Arrow 60's fields are private, so its offsets cannot be read here; its
    // size and alignment can, and the round trips below exercise every field.
    for (size, align) in [
        (size_of::<S58>(), align_of::<S58>()),
        (size_of::<S60>(), align_of::<S60>()),
    ] {
        assert_eq!(
            (size, align),
            (size_of::<SpecSchema>(), align_of::<SpecSchema>())
        );
    }
    for (size, align) in [
        (size_of::<A58>(), align_of::<A58>()),
        (size_of::<A60>(), align_of::<A60>()),
    ] {
        assert_eq!(
            (size, align),
            (size_of::<SpecArray>(), align_of::<SpecArray>())
        );
    }

    assert_eq!(offset_of!(S58, format), offset_of!(SpecSchema, format));
    assert_eq!(offset_of!(S58, name), offset_of!(SpecSchema, name));
    assert_eq!(offset_of!(S58, metadata), offset_of!(SpecSchema, metadata));
    assert_eq!(offset_of!(S58, flags), offset_of!(SpecSchema, flags));
    assert_eq!(
        offset_of!(S58, n_children),
        offset_of!(SpecSchema, n_children)
    );
    assert_eq!(offset_of!(S58, children), offset_of!(SpecSchema, children));
    assert_eq!(
        offset_of!(S58, dictionary),
        offset_of!(SpecSchema, dictionary)
    );
    assert_eq!(offset_of!(S58, release), offset_of!(SpecSchema, release));
    assert_eq!(
        offset_of!(S58, private_data),
        offset_of!(SpecSchema, private_data)
    );

    assert_eq!(offset_of!(A58, length), offset_of!(SpecArray, length));
    assert_eq!(
        offset_of!(A58, null_count),
        offset_of!(SpecArray, null_count)
    );
    assert_eq!(offset_of!(A58, offset), offset_of!(SpecArray, offset));
    assert_eq!(offset_of!(A58, n_buffers), offset_of!(SpecArray, n_buffers));
    assert_eq!(
        offset_of!(A58, n_children),
        offset_of!(SpecArray, n_children)
    );
    assert_eq!(offset_of!(A58, buffers), offset_of!(SpecArray, buffers));
    assert_eq!(offset_of!(A58, children), offset_of!(SpecArray, children));
    assert_eq!(
        offset_of!(A58, dictionary),
        offset_of!(SpecArray, dictionary)
    );
    assert_eq!(offset_of!(A58, release), offset_of!(SpecArray, release));
    assert_eq!(
        offset_of!(A58, private_data),
        offset_of!(SpecArray, private_data)
    );
}

fn field_id(id: i32) -> HashMap<String, String> {
    HashMap::from([("PARQUET:field_id".to_owned(), id.to_string())])
}

/// One batch with every shape the bridge carries: nested struct and list
/// with field ids on every level, a dictionary, nulls, timezone-bearing
/// timestamps, decimals, binary, an offset slice, and schema metadata.
macro_rules! sample_batch {
    ($array:ident, $schema:ident) => {{
        use $array::builder::{Int32Builder, ListBuilder};
        use $array::types::Int32Type;
        use $array::{
            ArrayRef, BinaryArray, Decimal128Array, DictionaryArray, Int32Array, Int64Array,
            RecordBatch, StringArray, StructArray, TimestampMicrosecondArray,
        };
        use $schema::{DataType, Field, Schema, TimeUnit};

        let id = Field::new("id", DataType::Int64, false).with_metadata(field_id(1));
        let name = Field::new("name", DataType::Utf8, true).with_metadata(field_id(2));
        let a = Field::new("a", DataType::Int32, true).with_metadata(field_id(4));
        let element = Field::new("element", DataType::Int32, true).with_metadata(field_id(6));
        let b = Field::new("b", DataType::List(Arc::new(element.clone())), true)
            .with_metadata(field_id(5));
        let nested = Field::new(
            "nested",
            DataType::Struct(vec![a.clone(), b.clone()].into()),
            true,
        )
        .with_metadata(field_id(3));
        let category = Field::new(
            "category",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        )
        .with_metadata(field_id(7));
        let ts = Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        )
        .with_metadata(field_id(8));
        let amount =
            Field::new("amount", DataType::Decimal128(38, 10), true).with_metadata(field_id(9));
        let payload = Field::new("payload", DataType::Binary, true).with_metadata(field_id(10));
        let schema = Schema::new(vec![id, name, nested, category, ts, amount, payload])
            .with_metadata(HashMap::from([(
                "iceberg.schema".to_owned(),
                r#"{"type":"struct","schema-id":0}"#.to_owned(),
            )]));

        let mut list = ListBuilder::new(Int32Builder::new()).with_field(Arc::new(element));
        list.values().append_value(1);
        list.values().append_null();
        list.append(true);
        list.append(false);
        list.values().append_value(3);
        list.append(true);
        list.append(true);
        let nested_array = StructArray::from(vec![
            (
                Arc::new(a),
                Arc::new(Int32Array::from(vec![Some(10), None, Some(30), Some(40)])) as ArrayRef,
            ),
            (Arc::new(b), Arc::new(list.finish()) as ArrayRef),
        ]);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                Some("x"),
                None,
                Some(""),
                Some("ünï"),
            ])),
            Arc::new(nested_array),
            Arc::new(
                vec![Some("red"), None, Some("red"), Some("blue")]
                    .into_iter()
                    .collect::<DictionaryArray<Int32Type>>(),
            ),
            Arc::new(
                TimestampMicrosecondArray::from(vec![Some(0), Some(-1), None, Some(i64::MAX)])
                    .with_timezone("UTC"),
            ),
            Arc::new(
                Decimal128Array::from(vec![Some(1), None, Some(-10_i128.pow(37)), Some(0)])
                    .with_precision_and_scale(38, 10)
                    .unwrap(),
            ),
            Arc::new(BinaryArray::from(vec![
                Some(&b""[..]),
                Some(&b"\x00\xff"[..]),
                None,
                Some(&b"abc"[..]),
            ])),
        ];
        RecordBatch::try_new(Arc::new(schema), columns).unwrap()
    }};
}

fn assert_same(bridged: &arrow_array::RecordBatch, native: &arrow_array::RecordBatch) {
    assert_eq!(
        bridged.schema(),
        native.schema(),
        "schema, field and schema metadata"
    );
    assert_eq!(bridged.num_rows(), native.num_rows());
    for (i, (b, n)) in bridged.columns().iter().zip(native.columns()).enumerate() {
        assert_eq!(b.to_data(), n.to_data(), "column {i}");
    }
}

#[test]
fn bridge_roundtrip_schema() {
    let old = sample_batch!(arrow_array_58, arrow_schema_58);
    let native = sample_batch!(arrow_array, arrow_schema);
    let bridged = schema_58_to_60(old.schema_ref()).unwrap();
    assert_eq!(&bridged, native.schema_ref().as_ref());
    // Byte-identical field ids at every nesting level.
    let nested = bridged.field_with_name("nested").unwrap();
    assert_eq!(nested.metadata()["PARQUET:field_id"], "3");
    let arrow_schema::DataType::Struct(children) = nested.data_type() else {
        panic!("nested is a struct");
    };
    let arrow_schema::DataType::List(element) = children[1].data_type() else {
        panic!("b is a list");
    };
    assert_eq!(element.metadata()["PARQUET:field_id"], "6");
}

#[test]
fn bridge_roundtrip_batch() {
    let old = sample_batch!(arrow_array_58, arrow_schema_58);
    let native = sample_batch!(arrow_array, arrow_schema);
    let bridged = record_batch_58_to_60(&old).unwrap();
    assert_same(&bridged, &native);
}

#[test]
fn bridge_roundtrip_sliced_batch() {
    let old = sample_batch!(arrow_array_58, arrow_schema_58).slice(1, 2);
    let native = sample_batch!(arrow_array, arrow_schema).slice(1, 2);
    let bridged = record_batch_58_to_60(&old).unwrap();
    assert_eq!(bridged.num_rows(), 2);
    assert_eq!(bridged.schema(), native.schema());
    for (b, n) in bridged.columns().iter().zip(native.columns()) {
        // Logical equality: a slice may cross with a different offset.
        assert_eq!(b.as_ref(), n.as_ref());
    }
}

#[test]
fn bridged_batch_outlives_its_source() {
    let old = sample_batch!(arrow_array_58, arrow_schema_58);
    let bridged = record_batch_58_to_60(&old).unwrap();
    drop(old);
    let native = sample_batch!(arrow_array, arrow_schema);
    assert_same(&bridged, &native);
}

#[test]
fn bridge_keeps_row_count_without_columns() {
    let schema = Arc::new(arrow_schema_58::Schema::empty());
    let old = arrow_array_58::RecordBatch::try_new_with_options(
        schema,
        vec![],
        &arrow_array_58::RecordBatchOptions::new().with_row_count(Some(3)),
    )
    .unwrap();
    let bridged = record_batch_58_to_60(&old).unwrap();
    assert_eq!(bridged.num_rows(), 3);
    assert_eq!(bridged.num_columns(), 0);
}

#[test]
fn bridge_carries_an_empty_batch() {
    let old = sample_batch!(arrow_array_58, arrow_schema_58).slice(0, 0);
    let bridged = record_batch_58_to_60(&old).unwrap();
    assert_eq!(bridged.num_rows(), 0);
    assert_eq!(
        bridged.schema(),
        sample_batch!(arrow_array, arrow_schema).schema()
    );
}
