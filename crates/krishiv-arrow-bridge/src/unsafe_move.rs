// SPDX-License-Identifier: Apache-2.0
//! The crate's only `unsafe`: moving C Data Interface structs from Arrow 58's
//! Rust types to Arrow 60's, and importing the moved array.

use std::mem::{align_of, size_of};

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
use arrow_array_58::ffi::{FFI_ArrowArray as FfiArray58, FFI_ArrowSchema as FfiSchema58};
use arrow_data::ArrayData;

use crate::{Result, arrow_error};

// Both majors' structs are `#[repr(C)]` transcriptions of the C Data
// Interface's `ArrowSchema` / `ArrowArray`; the reads below rely on it. A
// change in either shows up here at compile time (field offsets are checked
// in the `ffi_layout` test, as far as Arrow 60's private fields allow).
const _: () = {
    assert!(size_of::<FfiSchema58>() == size_of::<FFI_ArrowSchema>());
    assert!(align_of::<FfiSchema58>() == align_of::<FFI_ArrowSchema>());
    assert!(size_of::<FfiArray58>() == size_of::<FFI_ArrowArray>());
    assert!(align_of::<FfiArray58>() == align_of::<FFI_ArrowArray>());
};

/// Moves an exported Arrow 58 schema into an Arrow 60 one. `exported` is left
/// released (release callback NULL), so dropping it frees nothing.
pub(crate) fn schema(mut exported: FfiSchema58) -> FFI_ArrowSchema {
    let ptr = (&raw mut exported).cast::<FFI_ArrowSchema>();
    // SAFETY: `ptr` points to a live, aligned, initialised `FfiSchema58`,
    // which has the same `#[repr(C)]` layout as `FFI_ArrowSchema` (both
    // transcribe the C Data Interface's `ArrowSchema`; size and alignment are
    // asserted above, offsets in `ffi_layout`). `from_raw` reads the struct and
    // writes an empty one (release NULL) in its place, so `exported`'s drop
    // runs no release and ownership of the export passes to the result.
    unsafe { FFI_ArrowSchema::from_raw(ptr) }
}

/// Moves an exported Arrow 58 array into Arrow 60 and imports it with its
/// schema. Buffers are not copied: the result owns the export.
pub(crate) fn array(mut exported: FfiArray58, c_schema: FfiSchema58) -> Result<ArrayData> {
    let c_schema = schema(c_schema);
    let ptr = (&raw mut exported).cast::<FFI_ArrowArray>();
    // SAFETY: as in `schema`: `ptr` points to a live, aligned, initialised
    // `FfiArray58` with `FFI_ArrowArray`'s layout; `from_raw` leaves it
    // released, so its drop frees nothing.
    let moved = unsafe { FFI_ArrowArray::from_raw(ptr) };
    // SAFETY: `moved` and `c_schema` are one export of one array by Arrow 58's
    // `to_ffi`, a conforming C Data Interface producer, so they describe each
    // other; the import keeps `moved` alive for as long as its buffers are
    // referenced and calls its release callback after.
    unsafe { from_ffi(moved, &c_schema) }.map_err(|e| arrow_error("import batch", e))
}
