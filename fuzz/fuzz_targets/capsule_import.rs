// SPDX-License-Identifier: Apache-2.0
//! The Arrow PyCapsule import path (P2-T108, the bridge of P2-T091b): a producer's C schema and
//! array headers are untrusted, and `krishiv-python`'s `stream/ffi.rs` must reject every
//! inconsistency it can see before arrow-rs does arithmetic or pointer offsets with it.
//!
//! The input picks one of a few valid record batches, exports it through the Arrow C data
//! interface and then corrupts header fields the way a hostile producer could. Only fields the
//! validator can check are touched: lengths, null counts, offsets and flags take any value;
//! buffer, child and dictionary counts are only lowered or pointers nulled, and schema formats
//! swapped for other valid format strings. Raising a count or pointing a buffer at foreign
//! memory is not corruption the C data interface lets a consumer detect, so it is not generated.
//! Importing must return a batch or an error, never panic, abort or read out of bounds.
#![no_main]
#![allow(
    dead_code,
    reason = "ffi.rs is the production file; the target uses part of it"
)]

#[path = "../../crates/krishiv-python/src/stream/ffi.rs"]
mod ffi;

use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;

use arrow_array::ffi::to_ffi;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, Int8Array, Int32Array, Int64Array,
    LargeStringArray, ListArray, StringArray, StringViewArray, StructArray, types::Int8Type,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{DataType, Field};
use libfuzzer_sys::fuzz_target;

#[repr(C)]
struct CArr {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *const *const c_void,
    children: *const *const CArr,
    dictionary: *const CArr,
    release: Option<unsafe extern "C" fn(*mut CArr)>,
    private_data: *mut c_void,
}

#[repr(C)]
struct CSch {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *const *const CSch,
    dictionary: *const CSch,
    release: Option<unsafe extern "C" fn(*mut CSch)>,
    private_data: *mut c_void,
}

/// The formats a schema node may be changed to: ones whose buffers are read no wider than the
/// original's. A wider type (`i` to `l`, `u` to `U`) reads past buffers whose sizes the C data
/// interface does not carry, which no consumer can detect.
fn swaps(original: &CStr) -> &'static [&'static CStr] {
    match original.to_bytes() {
        b"l" => &[c"i", c"b", c"l"],
        b"i" => &[c"b", c"i"],
        b"u" => &[c"z", c"u"],
        b"U" => &[c"u", c"z", c"U"],
        b"vu" => &[c"vz", c"vu"],
        b"z" => &[c"u", c"z"],
        b"+s" => &[c"+l", c"+L", c"+s"],
        b"+l" => &[c"+s", c"+l"],
        b"+L" => &[c"+s", c"+l", c"+L"],
        _ => &[c"n"],
    }
}
const MAX_NODES: usize = 64;

fn batch(kind: u8) -> StructArray {
    let col = |name: &str, a: ArrayRef, nullable| {
        (
            Arc::new(Field::new(name, a.data_type().clone(), nullable)),
            a,
        )
    };
    let ids: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
    let names: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("ccc")]));
    match kind % 8 {
        0 => StructArray::from(vec![col("id", ids, true), col("name", names, true)]),
        1 => {
            let list = ListArray::new(
                Arc::new(Field::new_list_field(DataType::Int32, true)),
                OffsetBuffer::new(vec![0, 2, 2, 4].into()),
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
                None,
            );
            StructArray::from(vec![col("l", Arc::new(list), true)])
        }
        2 => {
            let d: DictionaryArray<Int8Type> = DictionaryArray::new(
                Int8Array::from(vec![0, 1, 0]),
                Arc::new(StringArray::from(vec!["x", "y"])),
            );
            let full: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
            StructArray::from(vec![col("d", Arc::new(d), true), col("id", full, false)])
        }
        4 => {
            // A view column with a long string: one variadic data buffer.
            let v: ArrayRef = Arc::new(StringViewArray::from(vec![
                Some("short"),
                None,
                Some("a string long enough to live in a variadic data buffer"),
            ]));
            StructArray::from(vec![col("v", v, true)])
        }
        5 => {
            let l: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("a"), None, Some("ccc")]));
            StructArray::from(vec![col("l", l, true), col("id", ids, true)])
        }
        6 => {
            let f: ArrayRef = Arc::new(FixedSizeListArray::new(
                Arc::new(Field::new_list_field(DataType::Int32, true)),
                2,
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
                None,
            ));
            StructArray::from(vec![col("f", f, true)])
        }
        _ => {
            let inner = StructArray::from(vec![col("n", names, true)]);
            StructArray::from(vec![col("s", Arc::new(inner), true), col("id", ids, true)])
        }
    }
}

/// Every array node of the tree under `root`, depth first, bounded.
unsafe fn array_nodes(root: *mut CArr, out: &mut Vec<*mut CArr>) {
    if out.len() >= MAX_NODES || root.is_null() {
        return;
    }
    out.push(root);
    // SAFETY: the tree was exported by arrow-rs and only scalar fields were changed so far.
    let (n, children, dict) = unsafe { ((*root).n_children, (*root).children, (*root).dictionary) };
    for i in 0..usize::try_from(n).unwrap_or(0) {
        // SAFETY: `children` holds the original `n_children` pointers; counts are only lowered.
        unsafe { array_nodes((*children.add(i)).cast_mut(), out) };
    }
    // SAFETY: a dictionary pointer is null or a node of the same tree.
    unsafe { array_nodes(dict.cast_mut(), out) };
}

unsafe fn schema_nodes(root: *mut CSch, out: &mut Vec<*mut CSch>) {
    if out.len() >= MAX_NODES || root.is_null() {
        return;
    }
    out.push(root);
    // SAFETY: as for `array_nodes`.
    let (n, children, dict) = unsafe { ((*root).n_children, (*root).children, (*root).dictionary) };
    for i in 0..usize::try_from(n).unwrap_or(0) {
        // SAFETY: as for `array_nodes`.
        unsafe { schema_nodes((*children.add(i)).cast_mut(), out) };
    }
    // SAFETY: as for `array_nodes`.
    unsafe { schema_nodes(dict.cast_mut(), out) };
}

fn lowered(v: i64, current: i64) -> i64 {
    if current <= 0 {
        current
    } else {
        i64::try_from(v.unsigned_abs() % (current as u64 + 1)).unwrap_or(0)
    }
}

fn quiet_panics() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // `ffi.rs` contains arrow-rs panics on corrupt headers and reports them as errors; print
    // them instead of letting libFuzzer's hook abort. A panic that escapes still crashes.
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            eprintln!("contained: {info}");
            if std::env::var_os("RUST_BACKTRACE").is_some() {
                eprintln!("{}", std::backtrace::Backtrace::force_capture());
            }
        }));
    });
}

fuzz_target!(|data: &[u8]| {
    quiet_panics();
    let Some((&kind, rest)) = data.split_first() else {
        return;
    };
    let original = batch(kind);
    // The same tree also goes through `import_column` (a Python UDF result, P2-T059 unit 2a),
    // declared as the struct it is; the input's length picks which import runs.
    let (declared, rows, as_column) = (original.data_type().clone(), original.len(), rest.len() % 2 == 1);
    let Ok((mut array, schema)) = to_ffi(&original.to_data()) else {
        return;
    };
    let mut schema: FFI_ArrowSchema = schema;
    let mut found = Vec::new();
    let mut snodes = Vec::new();
    // SAFETY: both structs were just exported by arrow-rs and are exclusively ours.
    unsafe {
        array_nodes((&raw mut array).cast(), &mut found);
        schema_nodes((&raw mut schema).cast(), &mut snodes);
    }
    // arrow-rs releases a schema through these very header fields (it frees `format`, walks
    // `n_children`, boxes `dictionary`), so each node is restored before the schema is dropped.
    // SAFETY: the nodes are live; the copies are put back unchanged below.
    let saved: Vec<(*mut CSch, std::mem::ManuallyDrop<CSch>)> = snodes
        .iter()
        .map(|&n| unsafe { (n, std::mem::ManuallyDrop::new(std::ptr::read(n))) })
        .collect();
    // SAFETY: the nodes are live; the lengths are read before any change.
    let anodes: Vec<(*mut CArr, i64, i64)> = found
        .iter()
        .map(|&a| unsafe { (a, (*a).length, (*a).offset) })
        .collect();
    for m in rest.chunks_exact(10).take(8) {
        let v = i64::from_le_bytes(m[2..10].try_into().unwrap_or([0; 8]));
        let ((a, len, off), s) = (
            anodes[usize::from(m[1]) % anodes.len()],
            snodes[usize::from(m[1]) % snodes.len()],
        );
        // SAFETY: the nodes are live parts of the exported trees; see the module docs for why
        // only these fields change.
        unsafe {
            match m[0] % 11 {
                // A length or offset past the buffers cannot be told from a valid one (the
                // interface carries no buffer sizes), so a positive value only slices.
                0 => {
                    // Within the original extent, given the current offset (mutations compose).
                    let room = (len + off - (*a).offset.max(0)).max(0);
                    (*a).length = if v < 0 { v } else { v % (room + 1) };
                }
                1 => (*a).null_count = v,
                2 if v < 0 => (*a).offset = v,
                2 => {
                    let k = v % (len + 1);
                    (*a).offset = off + k;
                    (*a).length = (*a).length.min(len - k);
                }
                3 => (*a).n_buffers = lowered(v, (*a).n_buffers),
                4 => (*a).n_children = lowered(v, (*a).n_children),
                5 => {
                    let n = usize::try_from((*a).n_buffers).unwrap_or(0);
                    // Sound: arrow-rs's `buffers` points into its own `Vec`, and its release
                    // frees through `private_data`, never through this list.
                    if n > 0 {
                        let slot = (*a)
                            .buffers
                            .add(usize::try_from(v.unsigned_abs()).unwrap_or(0) % n);
                        slot.cast_mut().write(std::ptr::null());
                    }
                }
                6 => (*a).dictionary = std::ptr::null(),
                7 => (*s).flags = v,
                8 => (*s).n_children = lowered(v, (*s).n_children),
                9 => {
                    // Always from the node's original format: swaps must not compose into a wider one.
                    let original = saved
                        .iter()
                        .find(|(n, _)| *n == s)
                        .map_or((*s).format, |(_, c)| c.format);
                    let options = swaps(CStr::from_ptr(original));
                    (*s).format = options
                        [usize::try_from(v.unsigned_abs()).unwrap_or(0) % options.len()]
                    .as_ptr();
                }
                _ => (*s).dictionary = std::ptr::null(),
            }
        }
    }
    // SAFETY: both are valid structs (headers possibly inconsistent, which is what is tested).
    unsafe {
        drop(ffi::read_schema(&raw const schema));
        if as_column {
            drop(ffi::import_column(&raw const schema, &raw mut array, &declared, rows));
        } else {
            drop(ffi::import_array(&raw const schema, &raw mut array));
        }
        for (n, original) in saved {
            std::ptr::write(n, std::mem::ManuallyDrop::into_inner(original));
        }
    }
});
