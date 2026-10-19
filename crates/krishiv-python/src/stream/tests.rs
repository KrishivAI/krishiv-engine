// SPDX-License-Identifier: Apache-2.0
//! Tests of the pyo3-free Arrow C stream code (`ffi.rs`), compiled here on
//! its own because the extension module cannot link a test binary. An
//! integration test (`[[test]] stream_ffi` in Cargo.toml) kept next to the
//! module, so coverage tools see `src/stream/ffi.rs` by its own path.
//! Runs under Miri on pull requests and AddressSanitizer nightly (P2-T098a).

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test code: a failed unwrap or a panicking test producer is a failed test"
)]

#[path = "ffi.rs"]
mod ffi;
#[path = "query.rs"]
mod query;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{
    Array, ArrayRef, Int64Array, RecordBatch, RecordBatchReader, StringArray, StructArray,
};
use arrow_buffer::Buffer;
use arrow_data::ArrayData;
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};

use ffi::{MemoryReader, Requested};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn batch() -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("ccc")])),
        ],
    )
    .unwrap()
}

/// Exports `reader` and imports it back, as a Python consumer would.
fn round_trip(
    reader: Box<dyn RecordBatchReader + Send>,
    requested: Option<Requested>,
) -> Result<ffi::ValidatingReader, String> {
    let mut stream = ffi::export_stream(reader, requested);
    // SAFETY: `stream` is a valid, initialized stream owned by this frame.
    let imported = unsafe { ffi::import_stream(&raw mut stream) };
    // The import moved the stream out: dropping the husk releases nothing.
    assert!(stream.release().is_none());
    imported.map_err(|e| e.to_string())
}

fn buffer_addresses(batch: &RecordBatch) -> Vec<usize> {
    batch
        .columns()
        .iter()
        .flat_map(|c| {
            let data = c.to_data();
            data.buffers()
                .iter()
                .map(|b| b.as_ptr() as usize)
                .chain(data.nulls().map(|n| n.buffer().as_ptr() as usize))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn round_trip_is_zero_copy() {
    let original = batch();
    let addresses = buffer_addresses(&original);
    let reader = MemoryReader::new(schema(), vec![original.clone(), original.slice(1, 2)]);
    let mut imported = round_trip(Box::new(reader), None).unwrap();
    assert_eq!(imported.schema(), schema());
    let first = imported.next_batch().unwrap().unwrap();
    assert_eq!(first, original);
    assert_eq!(buffer_addresses(&first), addresses, "a buffer was copied");
    let second = imported.next_batch().unwrap().unwrap();
    assert_eq!(second, original.slice(1, 2));
    assert!(imported.next_batch().is_none());
    assert!(
        imported.next_batch().is_none(),
        "the end of the stream is sticky"
    );
}

#[test]
fn reader_trait_reports_errors_as_arrow_errors() {
    let reader = MemoryReader::new(schema(), vec![batch()]);
    let mut imported = round_trip(Box::new(reader), None).unwrap();
    assert_eq!(RecordBatchReader::schema(&imported), schema());
    assert_eq!(imported.next().unwrap().unwrap(), batch());
    assert!(imported.next().is_none());
}

#[test]
fn capsule_names_follow_the_pycapsule_interface() {
    assert_eq!(ffi::STREAM_CAPSULE, c"arrow_array_stream");
    assert_eq!(ffi::SCHEMA_CAPSULE, c"arrow_schema");
    assert_eq!(ffi::ARRAY_CAPSULE, c"arrow_array");
}

#[test]
fn consumed_stream_is_rejected() {
    let mut stream = FFI_ArrowArrayStream::empty();
    // SAFETY: a valid (released) stream.
    let err = unsafe { ffi::import_stream(&raw mut stream) }.unwrap_err();
    assert!(err.to_string().contains("already consumed"), "{err}");
    assert!(
        err.to_string().starts_with("[KRISHIV_INVALID_ARROW_DATA]"),
        "{err}"
    );
}

static RELEASED: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn count_release(stream: *mut FFI_ArrowArrayStream) {
    RELEASED.fetch_add(1, Ordering::SeqCst);
    // SAFETY: called by `Drop` with the live stream.
    unsafe {
        (*stream).set_release(None);
    }
}

#[test]
fn stream_without_callbacks_is_released_and_rejected() {
    let mut stream = FFI_ArrowArrayStream::empty();
    // SAFETY: `count_release` releases this stream (it has no private data).
    unsafe {
        stream.set_release(Some(count_release));
    }
    let before = RELEASED.load(Ordering::SeqCst);
    // SAFETY: a valid, initialized stream.
    let err = unsafe { ffi::import_stream(&raw mut stream) }.unwrap_err();
    assert!(
        err.to_string().contains("no get_schema or get_next"),
        "{err}"
    );
    assert_eq!(
        RELEASED.load(Ordering::SeqCst),
        before + 1,
        "released exactly once"
    );
    drop(stream);
    assert_eq!(RELEASED.load(Ordering::SeqCst), before + 1);
}

/// A reader yielding prepared items, for producers that misbehave.
struct Scripted {
    schema: SchemaRef,
    items: std::vec::IntoIter<Result<RecordBatch, ArrowError>>,
    panic_at_end: bool,
}

impl Iterator for Scripted {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.items.next() {
            None if self.panic_at_end => panic!("producer bug"),
            item => item,
        }
    }
}

impl RecordBatchReader for Scripted {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

fn scripted(schema: SchemaRef, items: Vec<Result<RecordBatch, ArrowError>>) -> Box<Scripted> {
    Box::new(Scripted {
        schema,
        items: items.into_iter(),
        panic_at_end: false,
    })
}

/// A one-column batch of `Utf8` whose data the producer did not validate.
/// The batch carries a nullable field; `nullable` is what the exported stream
/// declares.
fn unchecked_strings(
    offsets: &[i32],
    values: &[u8],
    nulls: Option<Buffer>,
    nullable: bool,
) -> (SchemaRef, RecordBatch) {
    let len = offsets.len() - 1;
    // SAFETY: deliberately invalid data, standing in for a foreign producer;
    // the import must reject it before reading values.
    let data = unsafe {
        ArrayData::new_unchecked(
            DataType::Utf8,
            len,
            None,
            nulls,
            0,
            vec![
                Buffer::from_slice_ref(offsets),
                Buffer::from_slice_ref(values),
            ],
            vec![],
        )
    };
    let column: ArrayRef = Arc::new(StringArray::from(data));
    let declared = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, nullable)]));
    let carrier = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
    (
        declared,
        RecordBatch::try_new(carrier, vec![column]).unwrap(),
    )
}

fn import_error((declared, batch): (SchemaRef, RecordBatch)) -> String {
    let reader = scripted(declared, vec![Ok(batch)]);
    let mut imported = round_trip(reader, None).unwrap();
    let err = imported.next_batch().unwrap().unwrap_err().to_string();
    assert!(
        imported.next_batch().is_none(),
        "the stream is released after an error"
    );
    err
}

#[test]
fn decreasing_offsets_are_rejected() {
    // (A last offset past the end of the values cannot be detected: the C
    // data interface carries no buffer sizes, so the import takes the values
    // buffer to end at the last offset. See the module documentation.)
    let err = import_error(unchecked_strings(&[0, 2, 1], b"abc", None, true));
    assert!(err.starts_with("[KRISHIV_INVALID_ARROW_DATA]"), "{err}");
}

#[test]
fn invalid_utf8_is_rejected() {
    let err = import_error(unchecked_strings(&[0, 2], &[0xff, 0xfe], None, true));
    assert!(err.to_lowercase().contains("utf"), "{err}");
}

#[test]
fn nulls_in_a_non_nullable_field_are_rejected() {
    let nulls = Buffer::from_slice_ref([0b0000_0010_u8]);
    let err = import_error(unchecked_strings(&[0, 1, 2], b"ab", Some(nulls), false));
    assert!(err.contains("non-nullable"), "{err}");
}

#[test]
fn producer_errors_are_reported_without_nul_bytes() {
    let reader = scripted(
        schema(),
        vec![
            Ok(batch()),
            Err(ArrowError::ComputeError("bad\0value".into())),
        ],
    );
    let mut imported = round_trip(reader, None).unwrap();
    assert!(imported.next_batch().unwrap().is_ok());
    let err = imported.next_batch().unwrap().unwrap_err().to_string();
    assert!(err.contains("bad\u{FFFD}value"), "{err}");
    assert!(imported.next_batch().is_none());
}

#[test]
fn a_panicking_producer_becomes_an_error() {
    let reader = Box::new(Scripted {
        schema: schema(),
        items: vec![Ok(batch())].into_iter(),
        panic_at_end: true,
    });
    let mut imported = round_trip(reader, None).unwrap();
    assert!(imported.next_batch().unwrap().is_ok());
    let err = imported.next_batch().unwrap().unwrap_err().to_string();
    assert!(err.contains("producer bug"), "{err}");
    assert!(imported.next_batch().is_none());
}

#[test]
fn requested_schema_rules() {
    let ours = schema();
    let fewer = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    assert!(
        ffi::reconcile_requested(&ours, &fewer)
            .unwrap_err()
            .contains("1 fields")
    );

    let renamed = Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("label", DataType::Utf8, true),
    ]);
    let Requested::Use(use_schema) = ffi::reconcile_requested(&ours, &renamed).unwrap() else {
        panic!("equal types must be honoured");
    };
    assert_eq!(use_schema.as_ref(), &renamed);

    let cast = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]);
    assert_eq!(
        ffi::reconcile_requested(&ours, &cast).unwrap(),
        Requested::Own
    );

    let stricter = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    assert_eq!(
        ffi::reconcile_requested(&ours, &stricter).unwrap(),
        Requested::Own
    );
}

#[test]
fn requested_schema_renames_the_stream() {
    let renamed = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("label", DataType::Utf8, true),
    ]));
    let reader = MemoryReader::new(schema(), vec![batch()]);
    let mut imported =
        round_trip(Box::new(reader), Some(Requested::Use(Arc::clone(&renamed)))).unwrap();
    assert_eq!(imported.schema(), renamed);
    let got = imported.next_batch().unwrap().unwrap();
    assert_eq!(got.schema(), renamed);
    assert_eq!(got.columns(), batch().columns());

    let reader = MemoryReader::new(schema(), vec![batch()]);
    let imported = round_trip(Box::new(reader), Some(Requested::Own)).unwrap();
    assert_eq!(imported.schema(), schema());
}

fn exported_struct(array: &StructArray) -> (FFI_ArrowSchema, FFI_ArrowArray) {
    let field = Field::new("", array.data_type().clone(), false);
    (
        FFI_ArrowSchema::try_from(&field).unwrap(),
        FFI_ArrowArray::new(&array.to_data()),
    )
}

#[test]
fn single_array_import_is_zero_copy() {
    let original = batch();
    let addresses = buffer_addresses(&original);
    let (schema_c, mut array_c) = exported_struct(&StructArray::from(original.clone()));
    // SAFETY: valid C structs owned by this frame.
    let got = unsafe { ffi::import_array(&raw const schema_c, &raw mut array_c) }.unwrap();
    assert!(array_c.is_released(), "the array was moved out");
    assert_eq!(got, original);
    assert_eq!(buffer_addresses(&got), addresses);

    // SAFETY: as above; the array is now released.
    let err = unsafe { ffi::import_array(&raw const schema_c, &raw mut array_c) }.unwrap_err();
    assert!(err.to_string().contains("already consumed"), "{err}");
}

#[test]
fn single_array_must_be_a_table() {
    let ints = Int64Array::from(vec![1, 2]);
    let schema_c = FFI_ArrowSchema::try_from(&Field::new("x", DataType::Int64, false)).unwrap();
    let mut array_c = FFI_ArrowArray::new(&ints.to_data());
    // SAFETY: valid C structs owned by this frame.
    let err = unsafe { ffi::import_array(&raw const schema_c, &raw mut array_c) }.unwrap_err();
    assert!(err.to_string().contains("expected a struct array"), "{err}");

    let with_null_row = StructArray::new(
        schema().fields().clone(),
        batch().columns().to_vec(),
        Some(vec![true, false, true].into()),
    );
    let (schema_c, mut array_c) = exported_struct(&with_null_row);
    // SAFETY: valid C structs owned by this frame.
    let err = unsafe { ffi::import_array(&raw const schema_c, &raw mut array_c) }.unwrap_err();
    assert!(err.to_string().contains("null rows"), "{err}");
}

#[test]
fn short_children_are_rejected() {
    // A struct of length 3 whose child has 2 rows: arrow-rs would panic
    // building a batch from it; the import validates first.
    let child = Int64Array::from(vec![1, 2]).into_data();
    let struct_type = DataType::Struct(vec![Field::new("id", DataType::Int64, false)].into());
    // SAFETY: deliberately inconsistent data, as a foreign producer may send.
    let data = unsafe {
        ArrayData::new_unchecked(struct_type.clone(), 3, None, None, 0, vec![], vec![child])
    };
    let schema_c = FFI_ArrowSchema::try_from(&Field::new("", struct_type, false)).unwrap();
    let mut array_c = FFI_ArrowArray::new(&data);
    // SAFETY: valid C structs owned by this frame.
    let err = unsafe { ffi::import_array(&raw const schema_c, &raw mut array_c) }.unwrap_err();
    assert!(
        err.to_string().starts_with("[KRISHIV_INVALID_ARROW_DATA]"),
        "{err}"
    );
}

/// The C layout of `struct ArrowArray`, to corrupt a header as a hostile
/// producer would.
#[repr(C)]
struct CArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *const *const std::ffi::c_void,
    children: *const *const CArray,
    dictionary: *const CArray,
    release: Option<unsafe extern "C" fn(*mut CArray)>,
    private_data: *mut std::ffi::c_void,
}

/// The C layout of `struct ArrowSchema`.
#[repr(C)]
struct CSchema {
    format: *const std::ffi::c_char,
    name: *const std::ffi::c_char,
    metadata: *const std::ffi::c_char,
    flags: i64,
    n_children: i64,
    children: *const *const CSchema,
    dictionary: *const CSchema,
    release: Option<unsafe extern "C" fn(*mut CSchema)>,
    private_data: *mut std::ffi::c_void,
}

/// Runs `check_array` on `array` (of `data_type`) after `corrupt` changed
/// its header; the header is restored before the array is released.
/// `corrupt` also gets the header's own address (for self-references).
fn array_error(
    array: &dyn Array,
    data_type: &DataType,
    corrupt: impl FnOnce(&mut CArray, *const CArray),
) -> String {
    let mut c = FFI_ArrowArray::new(&array.to_data());
    let raw = (&raw mut c).cast::<CArray>();
    // SAFETY: `CArray` is the C layout of `FFI_ArrowArray`; the header is
    // saved and restored around the corruption.
    let saved = unsafe { raw.read() };
    // SAFETY: `raw` points to the live, exclusively borrowed `c`.
    unsafe { corrupt(&mut *raw, raw) };
    // SAFETY: a C array whose pointers are valid; only header values lie.
    // Every access goes through `raw` (one pointer provenance).
    let result = unsafe { ffi::check_array(raw.cast(), data_type) };
    // SAFETY: as above.
    unsafe { raw.write(saved) };
    result.unwrap_err().to_string()
}

fn strings() -> StringArray {
    StringArray::from(vec![Some("a"), None, Some("bc")])
}

#[test]
fn valid_headers_pass_the_check() {
    let table = StructArray::from(batch());
    let c = FFI_ArrowArray::new(&table.to_data());
    // SAFETY: an exported array.
    unsafe { ffi::check_array(&raw const c, table.data_type()) }.unwrap();
    let dict: arrow_array::DictionaryArray<arrow_array::types::Int32Type> =
        vec!["x", "y", "x"].into_iter().collect();
    let c = FFI_ArrowArray::new(&dict.to_data());
    // SAFETY: an exported array.
    unsafe { ffi::check_array(&raw const c, dict.data_type()) }.unwrap();
    let views =
        arrow_array::StringViewArray::from(vec!["short", "a string longer than twelve bytes"]);
    let c = FFI_ArrowArray::new(&views.to_data());
    // SAFETY: an exported array.
    unsafe { ffi::check_array(&raw const c, views.data_type()) }.unwrap();
    let large = arrow_array::LargeStringArray::from(vec!["a", "bc"]);
    let c = FFI_ArrowArray::new(&large.to_data());
    // SAFETY: an exported array.
    unsafe { ffi::check_array(&raw const c, large.data_type()) }.unwrap();
    let list =
        arrow_array::ListArray::from_iter_primitive::<arrow_array::types::Int64Type, _, _>(vec![
            Some(vec![Some(1)]),
            None,
        ]);
    let c = FFI_ArrowArray::new(&list.to_data());
    // SAFETY: an exported array.
    unsafe { ffi::check_array(&raw const c, list.data_type()) }.unwrap();
}

#[test]
fn negative_and_overflowing_header_fields_are_rejected() {
    let s = strings();
    let t = DataType::Utf8;
    assert!(array_error(&s, &t, |a, _| a.length = -1).contains("negative array length -1"));
    assert!(array_error(&s, &t, |a, _| a.offset = -2).contains("negative array offset"));
    assert!(array_error(&s, &t, |a, _| a.null_count = -2).contains("invalid null count"));
    assert!(array_error(&s, &t, |a, _| a.length = i64::MAX).contains("overflows"));
    assert!(array_error(&s, &t, |a, _| a.n_buffers = -1).contains("negative buffer count"));
    assert!(array_error(&s, &t, |a, _| a.n_buffers = 2).contains("2 buffers"));
    assert!(
        array_error(&s, &t, |a, _| a.buffers = std::ptr::null()).contains("buffer list is null")
    );
    assert!(array_error(&s, &t, |a, _| a.n_children = -1).contains("negative child count"));
    assert!(array_error(&s, &t, |a, _| a.n_children = 1).contains("1 children"));
    assert!(array_error(&s, &t, |a, _| a.release = None).contains("already released"));
    let ints = Int64Array::from(vec![1, 2]);
    let huge = i64::MAX / 4;
    assert!(array_error(&ints, &DataType::Int64, |a, _| a.length = huge).contains("overflows"));
}

#[test]
fn negative_last_offsets_are_rejected() {
    // SAFETY: deliberately invalid offsets, as a hostile producer may send.
    let data = unsafe {
        ArrayData::new_unchecked(
            DataType::Utf8,
            1,
            None,
            None,
            0,
            vec![
                Buffer::from_slice_ref([0_i32, -8]),
                Buffer::from_slice_ref(b"ab"),
            ],
            vec![],
        )
    };
    let s = StringArray::from(data);
    assert!(array_error(&s, &DataType::Utf8, |_, _| ()).contains("negative last offset -8"));
    // SAFETY: as above, with 64-bit offsets.
    let data = unsafe {
        ArrayData::new_unchecked(
            DataType::LargeUtf8,
            1,
            None,
            None,
            0,
            vec![
                Buffer::from_slice_ref([0_i64, -8]),
                Buffer::from_slice_ref(b"ab"),
            ],
            vec![],
        )
    };
    let large = arrow_array::LargeStringArray::from(data);
    assert!(
        array_error(&large, &DataType::LargeUtf8, |_, _| ()).contains("invalid last offset -8")
    );
}

#[test]
fn child_and_dictionary_mismatches_are_rejected() {
    let table = StructArray::from(batch());
    let t = table.data_type().clone();
    assert!(
        array_error(&table, &t, |a, _| a.children = std::ptr::null())
            .contains("pointer to them is null")
    );
    // A struct whose child list points back to the struct itself: a cycle.
    let mut nested = DataType::Int64;
    for _ in 0..=ffi::MAX_DEPTH + 1 {
        nested = DataType::Struct(vec![Field::new("c", nested, true)].into());
    }
    let one = StructArray::from(vec![(
        Arc::new(Field::new("c", DataType::Int64, true)),
        Arc::new(Int64Array::from(vec![1])) as ArrayRef,
    )]);
    let mut self_ptr: *const CArray = std::ptr::null();
    let err = array_error(&one, &nested, |a, this| {
        self_ptr = this;
        a.children = &raw const self_ptr;
    });
    assert!(err.contains("deeper than"), "{err}");
    let dict_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let keys = arrow_array::Int32Array::from(vec![0]);
    assert!(array_error(&keys, &dict_type, |_, _| ()).contains("has no dictionary"));
    let s = strings();
    let other = FFI_ArrowArray::new(&strings().to_data());
    let err = array_error(&s, &DataType::Utf8, |a, _| {
        a.dictionary = (&raw const other).cast();
    });
    assert!(err.contains("has a dictionary"), "{err}");
    let views = arrow_array::StringViewArray::from(vec!["x"]);
    assert!(array_error(&views, &DataType::Utf8View, |a, _| a.n_buffers = 2).contains("2 buffers"));
}

/// Runs `check_schema` on the schema of `field` after `corrupt`.
fn schema_error(field: &Field, corrupt: impl FnOnce(&mut CSchema, *const CSchema)) -> String {
    let mut c = FFI_ArrowSchema::try_from(field).unwrap();
    let raw = (&raw mut c).cast::<CSchema>();
    // SAFETY: `CSchema` is the C layout of `FFI_ArrowSchema`; restored below.
    let saved = unsafe { raw.read() };
    // SAFETY: `raw` points to the live, exclusively borrowed `c`.
    unsafe { corrupt(&mut *raw, raw) };
    // SAFETY: a C schema whose pointers are valid; only header values lie.
    let result = unsafe { ffi::check_schema(raw.cast()) };
    // SAFETY: as above.
    unsafe { raw.write(saved) };
    result.unwrap_err().to_string()
}

#[test]
fn hostile_schemas_are_rejected() {
    let table = Field::new("", DataType::Struct(schema().fields().clone()), false);
    assert!(schema_error(&table, |s, _| s.format = std::ptr::null()).contains("no format string"));
    assert!(schema_error(&table, |s, _| s.n_children = -1).contains("negative schema child count"));
    assert!(schema_error(&table, |s, _| s.release = None).contains("already released"));
    let mut self_ptr: *const CSchema = std::ptr::null();
    let err = schema_error(&table, |s, this| {
        self_ptr = this;
        s.n_children = 1;
        s.children = &raw const self_ptr;
    });
    assert!(err.contains("deeper than"), "{err}");
    let negative_pairs = (-1_i32).to_le_bytes();
    let err = schema_error(&table, |s, _| s.metadata = negative_pairs.as_ptr().cast());
    assert!(err.contains("negative schema metadata count"), "{err}");
    let mut bad_key = Vec::new();
    bad_key.extend_from_slice(&1_i32.to_ne_bytes());
    bad_key.extend_from_slice(&(-3_i32).to_ne_bytes());
    let err = schema_error(&table, |s, _| s.metadata = bad_key.as_ptr().cast());
    assert!(err.contains("negative schema metadata key length"), "{err}");
}

#[test]
fn schemas_with_metadata_pass_and_read_schema_only_borrows() {
    let with_meta = Schema::new_with_metadata(
        schema().fields().clone(),
        std::collections::HashMap::from([("k".to_owned(), "v".to_owned())]),
    );
    let c = FFI_ArrowSchema::try_from(&with_meta).unwrap();
    // SAFETY: an exported schema, borrowed.
    let read = unsafe { ffi::read_schema(&raw const c) }.unwrap();
    assert_eq!(read, with_meta);
    // Still owned by `c` (released once, by its drop): it reads again.
    // SAFETY: as above.
    let again = unsafe { ffi::read_schema(&raw const c) }.unwrap();
    assert_eq!(again, with_meta);
    let released = FFI_ArrowSchema::empty();
    // SAFETY: a valid (released) schema.
    let err = unsafe { ffi::read_schema(&raw const released) }.unwrap_err();
    assert!(err.to_string().contains("already released"), "{err}");
}

#[test]
fn null_pointers_in_headers_are_rejected() {
    // SAFETY: null is checked first.
    let err = unsafe { ffi::check_array(std::ptr::null(), &DataType::Int64) }.unwrap_err();
    assert!(err.to_string().contains("array pointer is null"), "{err}");
    // SAFETY: as above.
    let err = unsafe { ffi::check_schema(std::ptr::null()) }.unwrap_err();
    assert!(err.to_string().contains("schema pointer is null"), "{err}");
    let table = StructArray::from(batch());
    let t = table.data_type().clone();
    let null_children: [*const CArray; 2] = [std::ptr::null(); 2];
    let err = array_error(&table, &t, |a, _| a.children = null_children.as_ptr());
    assert!(err.contains("array child 0 is null"), "{err}");
    // Offsets buffers that are null.
    let s = strings();
    let values = s.values().as_ptr().cast::<std::ffi::c_void>();
    let no_offsets: [*const std::ffi::c_void; 3] = [std::ptr::null(), std::ptr::null(), values];
    let err = array_error(&s, &DataType::Utf8, |a, _| a.buffers = no_offsets.as_ptr());
    assert!(err.contains("offsets buffer is null"), "{err}");
    let large = arrow_array::LargeStringArray::from(vec!["a"]);
    let err = array_error(&large, &DataType::LargeUtf8, |a, _| {
        a.buffers = no_offsets.as_ptr();
    });
    assert!(err.contains("offsets buffer is null"), "{err}");
}

#[test]
fn view_variadic_sizes_are_checked() {
    let views = arrow_array::StringViewArray::from(vec!["a string longer than twelve bytes"]);
    let c = FFI_ArrowArray::new(&views.to_data());
    let raw = (&raw const c).cast::<CArray>();
    // SAFETY: an exported view array: null, views, one data buffer, sizes.
    let (n, buffers) = unsafe { ((*raw).n_buffers, (*raw).buffers) };
    assert_eq!(n, 4);
    // SAFETY: as above.
    let original: Vec<_> = (0..4).map(|i| unsafe { *buffers.add(i) }).collect();
    let null_sizes = [original[0], original[1], original[2], std::ptr::null()];
    let err = array_error(&views, &DataType::Utf8View, |a, _| {
        a.buffers = null_sizes.as_ptr();
    });
    assert!(err.contains("variadic buffer sizes are null"), "{err}");
    let negative = [-1_i64];
    let bad_sizes = [
        original[0],
        original[1],
        original[2],
        negative.as_ptr().cast(),
    ];
    let err = array_error(&views, &DataType::Utf8View, |a, _| {
        a.buffers = bad_sizes.as_ptr();
    });
    assert!(err.contains("invalid variadic buffer size -1"), "{err}");
}

#[test]
fn union_run_end_and_dictionary_headers_pass() {
    let union = arrow_array::UnionArray::try_new(
        arrow_schema::UnionFields::try_new(
            [0, 1],
            [
                Field::new("i", DataType::Int64, true),
                Field::new("s", DataType::Utf8, true),
            ],
        )
        .unwrap(),
        vec![0_i8, 1].into(),
        None,
        vec![
            Arc::new(Int64Array::from(vec![Some(1), None])) as ArrayRef,
            Arc::new(StringArray::from(vec![None, Some("x")])),
        ],
    )
    .unwrap();
    let c = FFI_ArrowArray::new(&union.to_data());
    // SAFETY: an exported array.
    unsafe { ffi::check_array(&raw const c, union.data_type()) }.unwrap();
    let ree = arrow_array::RunArray::<arrow_array::types::Int32Type>::try_new(
        &arrow_array::Int32Array::from(vec![2]),
        &Int64Array::from(vec![7]),
    )
    .unwrap();
    let c = FFI_ArrowArray::new(&ree.to_data());
    // SAFETY: an exported array.
    unsafe { ffi::check_array(&raw const c, ree.data_type()) }.unwrap();
    let dict_field = Field::new(
        "d",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        true,
    );
    let c = FFI_ArrowSchema::try_from(&dict_field).unwrap();
    // SAFETY: an exported schema.
    unsafe { ffi::check_schema(&raw const c) }.unwrap();
}

/// A C schema built by hand: `format`, no children, nothing to free.
fn bare_schema(format: &'static std::ffi::CStr) -> CSchema {
    unsafe extern "C" fn release(s: *mut CSchema) {
        // SAFETY: called by the consumer with this live schema.
        unsafe { (*s).release = None };
    }
    CSchema {
        format: format.as_ptr(),
        name: std::ptr::null(),
        metadata: std::ptr::null(),
        flags: 0,
        n_children: 0,
        children: std::ptr::null(),
        dictionary: std::ptr::null(),
        release: Some(release),
        private_data: std::ptr::null_mut(),
    }
}

/// The C layout of `struct ArrowArrayStream`, for a hand-built producer.
#[repr(C)]
struct CStream {
    get_schema: Option<unsafe extern "C" fn(*mut CStream, *mut CSchema) -> std::ffi::c_int>,
    get_next: Option<unsafe extern "C" fn(*mut CStream, *mut CArray) -> std::ffi::c_int>,
    get_last_error: Option<unsafe extern "C" fn(*mut CStream) -> *const std::ffi::c_char>,
    release: Option<unsafe extern "C" fn(*mut CStream)>,
    private_data: *mut std::ffi::c_void,
}

unsafe extern "C" fn failing_schema(_: *mut CStream, _: *mut CSchema) -> std::ffi::c_int {
    5
}

unsafe extern "C" fn bad_format_schema(_: *mut CStream, out: *mut CSchema) -> std::ffi::c_int {
    // SAFETY: `out` is the consumer's schema to fill.
    unsafe { out.write(bare_schema(c"zzz")) };
    0
}

unsafe extern "C" fn never_next(_: *mut CStream, _: *mut CArray) -> std::ffi::c_int {
    5
}

unsafe extern "C" fn last_error(_: *mut CStream) -> *const std::ffi::c_char {
    c"the producer's own message".as_ptr()
}

unsafe extern "C" fn release_stream(s: *mut CStream) {
    // SAFETY: called with this live stream.
    unsafe { (*s).release = None };
}

fn hand_built_stream_error(
    get_schema: unsafe extern "C" fn(*mut CStream, *mut CSchema) -> std::ffi::c_int,
) -> String {
    let mut stream = CStream {
        get_schema: Some(get_schema),
        get_next: Some(never_next),
        get_last_error: Some(last_error),
        release: Some(release_stream),
        private_data: std::ptr::null_mut(),
    };
    // SAFETY: a valid C stream (the C layout of `FFI_ArrowArrayStream`).
    let err = unsafe { ffi::import_stream((&raw mut stream).cast()) }.unwrap_err();
    assert!(
        stream.release.is_none(),
        "the stream was moved out (and released)"
    );
    err.to_string()
}

#[test]
fn producer_schema_failures_are_reported() {
    let err = hand_built_stream_error(failing_schema);
    assert!(
        err.contains(
            "reading the stream's schema failed with error code 5: the producer's own message"
        ),
        "{err}"
    );
    let err = hand_built_stream_error(bad_format_schema);
    assert!(err.contains("the stream's schema is invalid"), "{err}");
}

#[test]
fn single_array_with_an_invalid_format_is_rejected() {
    let schema_c = bare_schema(c"zzz");
    let mut array_c = FFI_ArrowArray::new(&StructArray::from(batch()).to_data());
    // SAFETY: valid C structs owned by this frame.
    let err =
        unsafe { ffi::import_array((&raw const schema_c).cast(), &raw mut array_c) }.unwrap_err();
    assert!(
        err.to_string().contains("the array's schema is invalid"),
        "{err}"
    );
}

#[test]
fn an_exported_stream_stays_done_after_an_error() {
    use arrow_array::ffi_stream::ArrowArrayStreamReader;
    let reader = scripted(
        schema(),
        vec![Err(ArrowError::ComputeError("first".into())), Ok(batch())],
    );
    let stream = ffi::export_stream(reader, None);
    let mut consumer = ArrowArrayStreamReader::try_new(stream).unwrap();
    assert!(consumer.next().unwrap().is_err());
    assert!(consumer.next().is_none(), "no batch after an error");
}

/// A source that counts the batches it produces and records that it was dropped.
struct Probe {
    produced: Arc<AtomicUsize>,
    dropped: Arc<std::sync::atomic::AtomicBool>,
}

impl futures::Stream for Probe {
    type Item = Result<RecordBatch, krishiv_common::error::KrishivError>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.produced.fetch_add(1, Ordering::SeqCst);
        std::task::Poll::Ready(Some(Ok(batch())))
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

/// Releasing the exported C stream drops the engine stream (which cancels
/// the query) and nothing polls it afterwards.
#[test]
fn releasing_the_exported_query_stream_drops_the_engine_stream() {
    use arrow_array::ffi_stream::ArrowArrayStreamReader;
    use futures::StreamExt as _;

    let produced = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut stream: krishiv_session::RecordBatchStream = Box::pin(Probe {
        produced: Arc::clone(&produced),
        dropped: Arc::clone(&dropped),
    });
    // As `DataFrame.execute` does: one batch is read to learn the schema.
    let first = futures::executor::block_on(stream.next()).unwrap().unwrap();
    let reader = query::QueryReader::new(first.schema(), Some(first), stream);
    let exported = ffi::export_stream(Box::new(reader), None);
    let mut consumer = ArrowArrayStreamReader::try_new(exported).unwrap();
    consumer.next().unwrap().unwrap(); // the first batch
    consumer.next().unwrap().unwrap(); // one pulled from the engine
    assert_eq!(produced.load(Ordering::SeqCst), 2);
    assert!(!dropped.load(Ordering::SeqCst));
    drop(consumer);
    assert!(
        dropped.load(Ordering::SeqCst),
        "the engine stream was not dropped"
    );
    assert_eq!(produced.load(Ordering::SeqCst), 2, "polled after release");
}
