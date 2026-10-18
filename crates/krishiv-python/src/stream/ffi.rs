// SPDX-License-Identifier: Apache-2.0
//! The Arrow C stream interface, without Python (P2-T091b, FFI review P2-T098a).
//!
//! This module has no pyo3 dependency so that `tests.rs` (the `stream_ffi` test) compiles
//! it into a plain Rust test binary (the extension module itself cannot link
//! one) and Miri and AddressSanitizer can run it.
//!
//! - **Import** treats the producer as untrusted: [`import_stream`] checks the
//!   callbacks before arrow-rs calls them; every C schema and array header is
//!   checked ([`check_schema`], [`check_array`]: signs, counts, overflow,
//!   null pointers, depth) before arrow-rs does arithmetic or pointer offsets
//!   with it; [`ValidatingReader`] then runs `ArrayData::validate_full` on
//!   every batch (arrow-rs builds imported arrays without validation) and
//!   turns a panic inside arrow-rs's import into an error.
//! - **Export** goes through [`export_stream`], whose reader never unwinds into
//!   the C callbacks (a panic there would abort the process) and never reports
//!   an error message with a NUL byte (arrow-rs panics on one).
//! - Imported buffers are not copied, except buffers the producer under-aligns,
//!   which arrow-rs realigns. Only the pointers and the pointed-to sizes are
//!   trusted: the C data interface carries no buffer sizes, so arrow-rs
//!   derives them from the declared length and offsets, and a producer whose
//!   pointers are dangling or whose buffers are shorter than it declares (more
//!   rows than its buffers hold, a last offset past the end of its values)
//!   cannot be detected, as with every C data interface consumer. Every other
//!   inconsistency (negative or overflowing lengths and counts, missing
//!   buffers or children, cycles, decreasing offsets, UTF-8, dictionary keys,
//!   child lengths, null counts, nulls in a non-nullable field) is rejected.

use std::any::Any;
use std::os::raw::{c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{RecordBatch, RecordBatchOptions, RecordBatchReader};
use arrow_data::ArrayData;
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use krishiv_common::error::{KrishivError, KrishivInvalidArrowData};

/// The capsule name of an `ArrowArrayStream` (Arrow PyCapsule interface).
pub(crate) const STREAM_CAPSULE: &std::ffi::CStr = c"arrow_array_stream";
/// The capsule name of an `ArrowSchema` (Arrow PyCapsule interface).
pub(crate) const SCHEMA_CAPSULE: &std::ffi::CStr = c"arrow_schema";
/// The capsule name of an `ArrowArray` (Arrow PyCapsule interface).
pub(crate) const ARRAY_CAPSULE: &std::ffi::CStr = c"arrow_array";

/// `KRISHIV_INVALID_ARROW_DATA` with `reason`.
pub(crate) fn invalid(reason: impl Into<String>) -> KrishivError {
    KrishivError::from(KrishivInvalidArrowData {
        reason: reason.into(),
    })
}

/// The text of an arrow-rs error without its `Display` prefixes (`Invalid
/// argument error: …`), which nest when errors wrap errors.
fn arrow_message(e: &ArrowError) -> String {
    let text = e.to_string();
    let mut rest = text.as_str();
    while let Some(stripped) = rest.strip_prefix("Invalid argument error: ") {
        rest = stripped;
    }
    rest.to_owned()
}

/// Deepest nesting of schemas and arrays accepted from a producer; also stops
/// a cyclic `children`/`dictionary` graph before it overflows the stack.
pub(crate) const MAX_DEPTH: usize = 64;

/// The C ABI layout of `struct ArrowSchema` (Arrow C data interface).
#[repr(C)]
struct RawSchema {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *const *const RawSchema,
    dictionary: *const RawSchema,
    release: Option<unsafe extern "C" fn(*mut RawSchema)>,
    private_data: *mut c_void,
}

/// The C ABI layout of `struct ArrowArray` (Arrow C data interface).
#[repr(C)]
struct RawArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *const *const c_void,
    children: *const *const RawArray,
    dictionary: *const RawArray,
    release: Option<unsafe extern "C" fn(*mut RawArray)>,
    private_data: *mut c_void,
}

const _: () = assert!(size_of::<RawSchema>() == size_of::<FFI_ArrowSchema>());
const _: () = assert!(align_of::<RawSchema>() == align_of::<FFI_ArrowSchema>());
const _: () = assert!(size_of::<RawArray>() == size_of::<FFI_ArrowArray>());
const _: () = assert!(align_of::<RawArray>() == align_of::<FFI_ArrowArray>());

/// A non-negative count or length from a producer, as `usize`.
fn count(value: i64, what: &str) -> Result<usize, KrishivError> {
    usize::try_from(value).map_err(|_| invalid(format!("negative {what} {value}")))
}

/// `n` pointers at `base` (a producer's `children` array): non-null, and each
/// entry non-null.
///
/// # Safety
///
/// If `n > 0`, `base` must be null or point to `n` readable pointers (a
/// producer's promise; the C data interface carries no array sizes).
unsafe fn pointers<T>(
    base: *const *const T,
    n: usize,
    what: &str,
) -> Result<Vec<*const T>, KrishivError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if base.is_null() {
        return Err(invalid(format!(
            "{n} {what} declared, but the pointer to them is null"
        )));
    }
    (0..n)
        .map(|i| {
            // SAFETY: `i < n` and `base` points to `n` pointers (caller).
            let p = unsafe { base.add(i).read_unaligned() };
            if p.is_null() {
                Err(invalid(format!("{what} {i} is null")))
            } else {
                Ok(p)
            }
        })
        .collect()
}

/// Checks a producer's C schema before arrow-rs reads it: not released, a
/// format string, non-negative child and metadata counts, no null child, at
/// most [`MAX_DEPTH`] levels.
///
/// # Safety
///
/// `schema` must point to a C schema whose pointers are valid as the C data
/// interface describes (the producer's promise).
pub(crate) unsafe fn check_schema(schema: *const FFI_ArrowSchema) -> Result<(), KrishivError> {
    // SAFETY: forwarded from the caller; `RawSchema` is the C layout.
    unsafe { check_raw_schema(schema.cast(), 0) }
}

unsafe fn check_raw_schema(schema: *const RawSchema, depth: usize) -> Result<(), KrishivError> {
    if depth > MAX_DEPTH {
        return Err(invalid(format!(
            "the schema is nested deeper than {MAX_DEPTH} levels"
        )));
    }
    if schema.is_null() {
        return Err(invalid("a schema pointer is null"));
    }
    // SAFETY: a valid C schema (caller), read by value.
    let s = unsafe { schema.read_unaligned() };
    if s.release.is_none() {
        return Err(invalid("the schema was already released"));
    }
    if s.format.is_null() {
        return Err(invalid("the schema has no format string"));
    }
    let n = count(s.n_children, "schema child count")?;
    // SAFETY: `children` holds `n_children` pointers (caller).
    for child in unsafe { pointers(s.children, n, "schema child") }? {
        // SAFETY: a child of a valid schema is a valid schema (caller).
        unsafe { check_raw_schema(child, depth + 1) }?;
    }
    if !s.dictionary.is_null() {
        // SAFETY: as for the children.
        unsafe { check_raw_schema(s.dictionary, depth + 1) }?;
    }
    if !s.metadata.is_null() {
        // SAFETY: the metadata encoding of the C data interface (caller).
        unsafe { check_metadata(s.metadata.cast()) }?;
    }
    Ok(())
}

/// The metadata encoding: an `int32` pair count, then per pair an `int32`
/// length and the key bytes, an `int32` length and the value bytes.
///
/// # Safety
///
/// `p` must point to a metadata block in that encoding (the producer's
/// promise); only the signs of its counts are checked.
unsafe fn check_metadata(p: *const u8) -> Result<(), KrishivError> {
    /// The `int32` at `*p` as a length; advances `p` past it.
    unsafe fn next_len(p: &mut *const u8, what: &str) -> Result<usize, KrishivError> {
        // SAFETY: inside the producer's metadata block (caller).
        let v = unsafe { p.cast::<i32>().read_unaligned() };
        // SAFETY: as above; the 4 bytes just read are part of the block.
        *p = unsafe { p.add(4) };
        usize::try_from(v).map_err(|_| invalid(format!("negative schema metadata {what} {v}")))
    }
    let mut p = p;
    // SAFETY: the block starts with the pair count (caller).
    let pairs = unsafe { next_len(&mut p, "count") }?;
    for _ in 0..pairs {
        for what in ["key length", "value length"] {
            // SAFETY: each key and value starts with its length (caller).
            let len = unsafe { next_len(&mut p, what) }?;
            // SAFETY: the key or value bytes follow their length (caller).
            p = unsafe { p.add(len) };
        }
    }
    Ok(())
}

/// Checks a producer's C array of type `data_type` before arrow-rs computes
/// with it: not released, non-negative length/offset, `null_count >= -1`,
/// buffer and child counts matching the type, buffer byte sizes that fit
/// `isize`, non-negative last offsets and variadic sizes, a dictionary exactly
/// when the type has one, at most [`MAX_DEPTH`] levels.
///
/// # Safety
///
/// `array` must point to a C array whose pointers are valid as the C data
/// interface describes for `data_type` (the producer's promise).
pub(crate) unsafe fn check_array(
    array: *const FFI_ArrowArray,
    data_type: &DataType,
) -> Result<(), KrishivError> {
    // SAFETY: forwarded from the caller; `RawArray` is the C layout.
    unsafe { check_raw_array(array.cast(), data_type, 0) }
}

#[expect(
    clippy::too_many_lines,
    reason = "one check per field of the C struct, in order"
)]
unsafe fn check_raw_array(
    array: *const RawArray,
    data_type: &DataType,
    depth: usize,
) -> Result<(), KrishivError> {
    if depth > MAX_DEPTH {
        return Err(invalid(format!(
            "the array is nested deeper than {MAX_DEPTH} levels"
        )));
    }
    if array.is_null() {
        return Err(invalid("an array pointer is null"));
    }
    // SAFETY: a valid C array (caller), read by value.
    let a = unsafe { array.read_unaligned() };
    if a.release.is_none() {
        return Err(invalid("the array was already released"));
    }
    let length = count(a.length, "array length")?;
    let offset = count(a.offset, "array offset")?;
    if a.null_count < -1 {
        return Err(invalid(format!("invalid null count {}", a.null_count)));
    }
    let total = length
        .checked_add(offset)
        .filter(|t| t.checked_add(1).is_some())
        .ok_or_else(|| invalid("array length + offset overflows"))?;
    let key_type = match data_type {
        DataType::Dictionary(key, _) => key.as_ref(),
        other => other,
    };
    let layout = arrow_data::layout(key_type);
    let n_buffers = count(a.n_buffers, "buffer count")?;
    let fixed = layout.buffers.len() + usize::from(layout.can_contain_null_mask);
    let buffers_ok = if layout.variadic {
        n_buffers > fixed
    } else {
        n_buffers == fixed
    };
    if !buffers_ok {
        return Err(invalid(format!(
            "{n_buffers} buffers for an array of type {data_type}"
        )));
    }
    if n_buffers > 0 && a.buffers.is_null() {
        return Err(invalid("the array's buffer list is null"));
    }
    for spec in &layout.buffers {
        if let arrow_data::BufferSpec::FixedWidth { byte_width, .. } = spec {
            (total + 1)
                .checked_mul(*byte_width)
                .filter(|b| isize::try_from(*b).is_ok())
                .ok_or_else(|| {
                    invalid(format!(
                        "an array of {total} values of type {data_type} overflows"
                    ))
                })?;
        }
    }
    // SAFETY: `buffers` holds `n_buffers` pointers (caller); a null buffer is
    // read as such here and reported by arrow-rs.
    let buffer = |i: usize| unsafe { a.buffers.add(i).read_unaligned() };
    let non_empty = length > 0 || offset > 0;
    match key_type {
        DataType::Utf8 | DataType::Binary if non_empty => {
            let offsets = buffer(1).cast::<i32>();
            if offsets.is_null() {
                return Err(invalid("the offsets buffer is null"));
            }
            // SAFETY: the offsets buffer holds `total + 1` values (caller).
            let last = unsafe { offsets.add(total).read_unaligned() };
            if last < 0 {
                return Err(invalid(format!("negative last offset {last}")));
            }
        }
        DataType::LargeUtf8 | DataType::LargeBinary if non_empty => {
            let offsets = buffer(1).cast::<i64>();
            if offsets.is_null() {
                return Err(invalid("the offsets buffer is null"));
            }
            // SAFETY: as above, with 64-bit offsets.
            let last = unsafe { offsets.add(total).read_unaligned() };
            if usize::try_from(last)
                .ok()
                .and_then(|l| isize::try_from(l).ok())
                .is_none()
            {
                return Err(invalid(format!("invalid last offset {last}")));
            }
        }
        DataType::Utf8View | DataType::BinaryView => {
            let variadic = n_buffers - fixed - 1;
            let sizes = buffer(n_buffers - 1).cast::<i64>();
            if variadic > 0 && sizes.is_null() {
                return Err(invalid("the variadic buffer sizes are null"));
            }
            for i in 0..variadic {
                // SAFETY: the last buffer holds one size per variadic buffer.
                let size = unsafe { sizes.add(i).read_unaligned() };
                if usize::try_from(size)
                    .ok()
                    .and_then(|l| isize::try_from(l).ok())
                    .is_none()
                {
                    return Err(invalid(format!("invalid variadic buffer size {size}")));
                }
            }
        }
        _ => {}
    }
    let child_types: Vec<&DataType> = match data_type {
        DataType::Struct(fields) => fields.iter().map(|f| f.data_type()).collect(),
        DataType::Union(fields, _) => fields.iter().map(|(_, f)| f.data_type()).collect(),
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::FixedSizeList(f, _)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::Map(f, _) => vec![f.data_type()],
        DataType::RunEndEncoded(ends, values) => vec![ends.data_type(), values.data_type()],
        _ => Vec::new(),
    };
    let n_children = count(a.n_children, "child count")?;
    if n_children != child_types.len() {
        return Err(invalid(format!(
            "{n_children} children for an array of type {data_type}"
        )));
    }
    // SAFETY: `children` holds `n_children` pointers (caller).
    let children = unsafe { pointers(a.children, n_children, "array child") }?;
    for (child, child_type) in children.into_iter().zip(child_types) {
        // SAFETY: a child of a valid array is a valid array (caller).
        unsafe { check_raw_array(child, child_type, depth + 1) }?;
    }
    match (data_type, a.dictionary.is_null()) {
        // SAFETY: the dictionary of a valid array is a valid array (caller).
        (DataType::Dictionary(_, values), false) => unsafe {
            check_raw_array(a.dictionary, values, depth + 1)
        },
        (DataType::Dictionary(..), true) => Err(invalid("a dictionary array has no dictionary")),
        (_, false) => Err(invalid(format!(
            "an array of type {data_type} has a dictionary"
        ))),
        (_, true) => Ok(()),
    }
}

/// Reads a C schema the caller keeps owning (a consumer's
/// `requested_schema`): checked, then converted, never moved or released.
///
/// # Safety
///
/// `schema` must be non-null and point to a C schema whose pointers are valid
/// as the C data interface describes.
pub(crate) unsafe fn read_schema(schema: *const FFI_ArrowSchema) -> Result<Schema, KrishivError> {
    // SAFETY: forwarded from the caller.
    unsafe { check_schema(schema) }?;
    // SAFETY: checked above; borrowed only.
    let schema = unsafe { &*schema };
    catch_panic(|| Schema::try_from(schema))?
        .map_err(|e| invalid(format!("invalid schema: {}", arrow_message(&e))))
}

/// The `get_next` callback of a C stream.
type GetNext = unsafe extern "C" fn(*mut RawStream, *mut c_void) -> c_int;

/// The C ABI layout of `struct ArrowArrayStream` (Arrow C stream interface).
///
/// arrow-rs keeps the callbacks of [`FFI_ArrowArrayStream`] private and
/// unwraps `get_schema`/`get_next` when it calls them, so a producer that
/// leaves one NULL would make it panic; this mirror lets the import read them
/// first. The layout is fixed by the specification.
#[repr(C)]
struct RawStream {
    get_schema: Option<unsafe extern "C" fn(*mut RawStream, *mut c_void) -> c_int>,
    get_next: Option<GetNext>,
    get_last_error: Option<unsafe extern "C" fn(*mut RawStream) -> *const c_char>,
    release: Option<unsafe extern "C" fn(*mut RawStream)>,
    private_data: *mut c_void,
}

const _: () = assert!(size_of::<RawStream>() == size_of::<FFI_ArrowArrayStream>());
const _: () = assert!(align_of::<RawStream>() == align_of::<FFI_ArrowArrayStream>());

/// Moves the stream out of `ptr` (leaving it released, as the protocol
/// requires of a consumer), checks its callbacks and reads its schema.
///
/// A stream whose `release` is NULL was already consumed. A stream missing
/// `get_schema` or `get_next` is malformed; it is released and rejected.
///
/// # Safety
///
/// `ptr` must be non-null, aligned and point to an initialized `struct
/// ArrowArrayStream` that the caller may move out of (for a PyCapsule: the
/// pointer of a live capsule named [`STREAM_CAPSULE`]).
pub(crate) unsafe fn import_stream(
    ptr: *mut FFI_ArrowArrayStream,
) -> Result<ValidatingReader, KrishivError> {
    // SAFETY: the caller guarantees `ptr` is valid for reads and writes and
    // initialized; `from_raw` moves the struct out and leaves an empty,
    // released one, so the producer's capsule destructor does nothing.
    let stream = Box::new(unsafe { FFI_ArrowArrayStream::from_raw(ptr) });
    if stream.release().is_none() {
        return Err(invalid("the Arrow stream was already consumed"));
    }
    let raw: *const RawStream = std::ptr::from_ref::<FFI_ArrowArrayStream>(&stream).cast();
    // SAFETY: `raw` points to the live stream in `stream`, which has the C
    // layout of `RawStream` (size and alignment asserted).
    let (get_schema, get_next) = unsafe { ((*raw).get_schema, (*raw).get_next) };
    let (Some(get_schema), Some(get_next)) = (get_schema, get_next) else {
        // Dropping `stream` calls its `release`, which is set.
        return Err(invalid(
            "the Arrow stream has no get_schema or get_next callback",
        ));
    };
    let mut reader = ValidatingReader {
        schema: Arc::new(Schema::empty()),
        get_next,
        stream: Some(stream),
    };
    let mut out = FFI_ArrowSchema::empty();
    // `reader` holds the stream it was just built with.
    let Some(raw) = reader.raw() else {
        return Err(invalid("the Arrow stream was already consumed"));
    };
    // SAFETY: `get_schema` is the producer's callback for this live stream;
    // `out` is a valid, released schema it may write to.
    let code = unsafe { get_schema(raw, std::ptr::from_mut(&mut out).cast()) };
    if code != 0 {
        return Err(reader.failure("reading the stream's schema", code));
    }
    // SAFETY: the producer filled `out` with a C schema.
    unsafe { check_schema(&raw const out) }?;
    let schema = catch_panic(|| Schema::try_from(&out))?.map_err(|e| {
        invalid(format!(
            "the stream's schema is invalid: {}",
            arrow_message(&e)
        ))
    })?;
    reader.schema = Arc::new(schema);
    Ok(reader)
}

/// Reads an imported stream and validates every batch in full before it is
/// assembled into a `RecordBatch`.
///
/// After the end of the stream or an error the stream is released and the
/// reader returns `None`.
#[derive(Debug)]
pub(crate) struct ValidatingReader {
    schema: SchemaRef,
    /// The producer's `get_next`, checked non-null at import.
    get_next: GetNext,
    /// Boxed so that the address passed to the producer's callbacks is stable.
    stream: Option<Box<FFI_ArrowArrayStream>>,
}

impl ValidatingReader {
    /// The stream as the C struct, for calling its callbacks; `None` once
    /// released.
    fn raw(&mut self) -> Option<*mut RawStream> {
        self.stream
            .as_deref_mut()
            .map(|s| std::ptr::from_mut(s).cast())
    }

    /// An error for a callback that returned `code`, with the producer's
    /// message; releases the stream.
    fn failure(&mut self, what: &str, code: c_int) -> KrishivError {
        let mut message = format!("{what} failed with error code {code}");
        if let Some(stream) = self.stream.as_deref_mut() {
            let raw: *mut RawStream = std::ptr::from_mut(stream).cast();
            // SAFETY: `raw` points to the live stream held in `self.stream`
            // (non-null by construction), with the C layout of `RawStream`.
            if let Some(get_last_error) = unsafe { (*raw).get_last_error } {
                // SAFETY: the stream's last call failed, so the protocol
                // allows `get_last_error` on it.
                let text = unsafe { get_last_error(raw) };
                if !text.is_null() {
                    // SAFETY: a non-null result is a NUL-terminated string
                    // that lives until the next call on the stream; it is
                    // copied out here, before any other call.
                    let text = unsafe { std::ffi::CStr::from_ptr(text) };
                    message = format!("{message}: {}", text.to_string_lossy());
                }
            }
        }
        self.stream = None;
        invalid(message)
    }

    /// The next batch, validated; `None` at the end of the stream.
    pub(crate) fn next_batch(&mut self) -> Option<Result<RecordBatch, KrishivError>> {
        let raw = self.raw()?;
        let get_next = self.get_next;
        let mut out = FFI_ArrowArray::empty();
        // SAFETY: the producer's callback for this live stream; `out` is a
        // valid, released array it may write to.
        let code = unsafe { get_next(raw, std::ptr::from_mut(&mut out).cast()) };
        if code != 0 {
            return Some(Err(self.failure("reading a batch", code)));
        }
        if out.is_released() {
            self.stream = None;
            return None;
        }
        let result = self.assemble(out);
        if result.is_err() {
            self.stream = None;
        }
        Some(result)
    }

    /// Imports `array` as a struct of the stream's fields, validates it in
    /// full, then builds the batch.
    fn assemble(&self, array: FFI_ArrowArray) -> Result<RecordBatch, KrishivError> {
        let struct_type = DataType::Struct(self.schema.fields().clone());
        // SAFETY: the producer filled `array` with a C array of the stream's
        // struct type.
        unsafe { check_array(&raw const array, &struct_type) }?;
        let data = catch_panic(|| {
            // SAFETY: `array` is the producer's C array for this stream, its
            // header checked above (only its pointers and the sizes they
            // point to are trusted); the values are validated in full before
            // anything reads them.
            unsafe { arrow_array::ffi::from_ffi_and_data_type(array, struct_type) }
        })?
        .map_err(|e| invalid(format!("importing a batch failed: {}", arrow_message(&e))))?;
        validated_batch(Arc::clone(&self.schema), &data)
    }
}

impl Iterator for ValidatingReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        // The consumer shows the message after its own `Invalid argument
        // error:` prefix; the Spark-style text follows it unchanged.
        self.next_batch()
            .map(|r| r.map_err(|e| ArrowError::InvalidArgumentError(e.to_string())))
    }
}

impl RecordBatchReader for ValidatingReader {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Validates imported struct data in full (offsets, UTF-8, dictionary keys,
/// child lengths, null counts, nulls in non-nullable fields) and only then
/// builds a batch of `schema` from its children; a table row cannot be null.
fn validated_batch(schema: SchemaRef, data: &ArrayData) -> Result<RecordBatch, KrishivError> {
    catch_panic(|| data.validate_full())?.map_err(|e| invalid(arrow_message(&e)))?;
    if data.null_count() > 0 {
        return Err(invalid("a table's struct array has null rows"));
    }
    let columns: Vec<_> = data
        .child_data()
        .iter()
        .map(|child| arrow_array::make_array(child.slice(data.offset(), data.len())))
        .collect();
    let options = RecordBatchOptions::new().with_row_count(Some(data.len()));
    RecordBatch::try_new_with_options(schema, columns, &options)
        .map_err(|e| invalid(arrow_message(&e)))
}

/// Imports one `ArrowArray` + `ArrowSchema` pair (the `__arrow_c_array__`
/// protocol) as a batch: the array must be a struct array, whose fields become
/// the columns.
///
/// # Safety
///
/// `schema` must point to an initialized C schema, only borrowed here (the
/// schema capsule keeps ownership); `array` must point to an initialized C
/// array the caller may move out of (the pointers of live capsules named
/// [`SCHEMA_CAPSULE`] and [`ARRAY_CAPSULE`]). Both non-null and aligned.
pub(crate) unsafe fn import_array(
    schema: *const FFI_ArrowSchema,
    array: *mut FFI_ArrowArray,
) -> Result<RecordBatch, KrishivError> {
    // SAFETY: the caller guarantees `array` is a valid, initialized C array
    // we may move out of; `from_raw` leaves a released one behind.
    let array = unsafe { FFI_ArrowArray::from_raw(array) };
    if array.is_released() {
        return Err(invalid("the Arrow array was already consumed"));
    }
    // SAFETY: the caller guarantees `schema` points to a C schema.
    unsafe { check_schema(schema) }?;
    // SAFETY: checked above; it is borrowed, never moved or released.
    let schema = unsafe { &*schema };
    let field = catch_panic(|| Field::try_from(schema))?.map_err(|e| {
        invalid(format!(
            "the array's schema is invalid: {}",
            arrow_message(&e)
        ))
    })?;
    let DataType::Struct(fields) = field.data_type() else {
        return Err(invalid(format!(
            "expected a struct array for a table, got {}",
            field.data_type()
        )));
    };
    let table = Arc::new(Schema::new_with_metadata(
        fields.clone(),
        field.metadata().clone(),
    ));
    let struct_type = field.data_type().clone();
    // SAFETY: `array` is the producer's C array of `struct_type` (caller).
    unsafe { check_array(&raw const array, &struct_type) }?;
    let data = catch_panic(|| {
        // SAFETY: `array` is the producer's C array of `struct_type`, read
        // from its own schema, its header checked above (only its pointers
        // and the sizes they point to are trusted); the values are validated
        // in full before anything reads them.
        unsafe { arrow_array::ffi::from_ffi_and_data_type(array, struct_type) }
    })?
    .map_err(|e| invalid(format!("importing the array failed: {}", arrow_message(&e))))?;
    validated_batch(table, &data)
}

/// A reader over batches already in memory.
#[derive(Debug)]
pub(crate) struct MemoryReader {
    schema: SchemaRef,
    batches: std::vec::IntoIter<RecordBatch>,
}

impl MemoryReader {
    /// A reader returning `batches`, which must all have `schema`.
    pub(crate) fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        Self {
            schema,
            batches: batches.into_iter(),
        }
    }
}

impl Iterator for MemoryReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.batches.next().map(Ok)
    }
}

impl RecordBatchReader for MemoryReader {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// What a consumer's `requested_schema` (Arrow PyCapsule interface) makes the
/// producer do.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Requested {
    /// Produce the batches under this schema: same types, the requested names,
    /// nullability and metadata.
    Use(SchemaRef),
    /// The request cannot be met without casting; produce the own schema (the
    /// protocol allows this).
    Own,
}

/// Decides how to honour `requested` for a stream of `ours`.
///
/// A different number of fields is an error (the protocol asks the producer to
/// raise). Equal types are honoured, unless the request declares a field
/// non-nullable that may hold nulls. Anything else falls back to the own
/// schema; casting on request is not supported.
pub(crate) fn reconcile_requested(ours: &Schema, requested: &Schema) -> Result<Requested, String> {
    if ours.fields().len() != requested.fields().len() {
        return Err(format!(
            "the requested schema has {} fields, the data has {}",
            requested.fields().len(),
            ours.fields().len()
        ));
    }
    let compatible = ours
        .fields()
        .iter()
        .zip(requested.fields())
        .all(|(o, r)| o.data_type() == r.data_type() && (r.is_nullable() || !o.is_nullable()));
    Ok(if compatible {
        Requested::Use(Arc::new(requested.clone()))
    } else {
        Requested::Own
    })
}

/// The reader behind every exported stream: renames to the requested schema
/// and keeps panics and NUL bytes away from arrow-rs's C callbacks.
struct ExportReader {
    schema: SchemaRef,
    rename: bool,
    inner: Box<dyn RecordBatchReader + Send>,
    done: bool,
}

impl Iterator for ExportReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let item = match catch_panic(|| self.inner.next()) {
            Ok(item) => item,
            Err(e) => Some(Err(ArrowError::ExternalError(Box::new(e)))),
        };
        let item = match item {
            Some(Ok(batch)) if self.rename => Some(RecordBatch::try_new(
                Arc::clone(&self.schema),
                batch.columns().to_vec(),
            )),
            other => other,
        };
        match item {
            Some(Err(e)) => {
                self.done = true;
                Some(Err(nul_free(e)))
            }
            other => other,
        }
    }
}

impl RecordBatchReader for ExportReader {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// arrow-rs panics (inside an `extern "C"` callback, so the process aborts)
/// when an error message holds a NUL byte; replace them.
fn nul_free(e: ArrowError) -> ArrowError {
    let message = e.to_string();
    if message.contains('\0') {
        ArrowError::InvalidArgumentError(message.replace('\0', "\u{FFFD}"))
    } else {
        e
    }
}

/// Exports `reader` as a C stream, under `requested` if given.
pub(crate) fn export_stream(
    reader: Box<dyn RecordBatchReader + Send>,
    requested: Option<Requested>,
) -> FFI_ArrowArrayStream {
    let (schema, rename) = match requested {
        Some(Requested::Use(schema)) => (schema, true),
        Some(Requested::Own) | None => (reader.schema(), false),
    };
    FFI_ArrowArrayStream::new(Box::new(ExportReader {
        schema,
        rename,
        inner: reader,
        done: false,
    }))
}

/// Runs `f`, turning a panic into `KRISHIV_INVALID_ARROW_DATA`.
pub(crate) fn catch_panic<T>(f: impl FnOnce() -> T) -> Result<T, KrishivError> {
    catch_unwind(AssertUnwindSafe(f))
        .map_err(|p| invalid(format!("Arrow import failed: {}", panic_message(&*p))))
}

fn panic_message(p: &(dyn Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_owned())
}
