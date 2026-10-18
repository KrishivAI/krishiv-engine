// SPDX-License-Identifier: Apache-2.0
//! The Arrow PyCapsule bridge (P2-T091b): `krishiv._native.ArrowStream`.
//!
//! An `ArrowStream` is a one-shot stream of Arrow record batches. It is built
//! from any object implementing the Arrow PyCapsule interface
//! (`__arrow_c_stream__`, or `__arrow_c_array__` for a single batch) and is
//! itself consumable through `__arrow_c_stream__`, so pyarrow, Polars and
//! DuckDB read it without copying. The C-level work is in [`ffi`], which has
//! no pyo3 dependency so that it can be tested under Miri.

mod ffi;

use std::ffi::CStr;
use std::sync::Mutex;

use arrow_array::RecordBatchReader;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::SchemaRef;
use arrow_schema::ffi::FFI_ArrowSchema;
use krishiv_common::error::KrishivError;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::critical_section::with_critical_section;
use pyo3::types::{PyCapsule, PyCapsuleMethods, PyTuple};

use self::ffi::{ARRAY_CAPSULE, MemoryReader, SCHEMA_CAPSULE, STREAM_CAPSULE};

/// `KRISHIV_INVALID_ARROW_DATA` and other bridge errors surface as
/// `ValueError` carrying the Spark-style message (`[CLASS] … SQLSTATE: …`).
fn value_error(e: &KrishivError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The pointer of `capsule`, which must be a PyCapsule named `name`.
fn capsule_pointer(capsule: &Bound<'_, PyAny>, name: &CStr) -> PyResult<*mut std::ffi::c_void> {
    let wanted = name.to_string_lossy();
    let capsule = capsule
        .cast::<PyCapsule>()
        .map_err(|_| PyTypeError::new_err(format!("expected a PyCapsule named {wanted}")))?;
    if !capsule.is_valid_checked(Some(name)) {
        return Err(PyValueError::new_err(format!(
            "expected a PyCapsule named {wanted}, got one named {}",
            capsule_name(capsule)
        )));
    }
    Ok(capsule.pointer_checked(Some(name))?.as_ptr())
}

/// The name of `capsule` for an error message.
fn capsule_name(capsule: &Bound<'_, PyCapsule>) -> String {
    match capsule.name() {
        // SAFETY: the name of a live capsule stays valid while it lives; it
        // is copied out at once.
        Ok(Some(name)) => unsafe { name.as_cstr() }.to_string_lossy().into_owned(),
        _ => "NULL".to_owned(),
    }
}

/// A one-shot stream of Arrow record batches (Arrow PyCapsule interface).
#[pyclass(module = "krishiv._native", frozen)]
pub struct ArrowStream {
    schema: SchemaRef,
    /// Taken by the first `__arrow_c_stream__` call.
    reader: Mutex<Option<Box<dyn RecordBatchReader + Send>>>,
}

impl std::fmt::Debug for ArrowStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowStream")
            .field("schema", &self.schema)
            .field("consumed", &self.consumed())
            .finish()
    }
}

impl ArrowStream {
    fn new(reader: Box<dyn RecordBatchReader + Send>) -> Self {
        Self {
            schema: reader.schema(),
            reader: Mutex::new(Some(reader)),
        }
    }
}

#[pymethods]
impl ArrowStream {
    /// Imports `data`, any object with `__arrow_c_stream__` (a pyarrow
    /// `Table`/`RecordBatchReader`, a Polars or DuckDB result, …) or, for a
    /// single batch, `__arrow_c_array__`. The data is not copied; it is
    /// validated in full as it is read.
    #[staticmethod]
    fn from_arrow(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        if data.hasattr("__arrow_c_stream__")? {
            let capsule = data.call_method0("__arrow_c_stream__")?;
            // The move out of the capsule runs in a critical section on it:
            // on the free-threaded build, two threads importing one cached
            // capsule must not both take the stream.
            let reader = with_critical_section(&capsule, || {
                let ptr = capsule_pointer(&capsule, STREAM_CAPSULE)?;
                // SAFETY: `ptr` is the pointer of a live capsule named
                // `arrow_array_stream`, which by the PyCapsule interface
                // points to an initialized `ArrowArrayStream` the consumer
                // moves out of. `capsule` stays alive (it is a local) until
                // the move is done.
                unsafe { ffi::import_stream(ptr.cast::<FFI_ArrowArrayStream>()) }
                    .map_err(|e| value_error(&e))
            })?;
            return Ok(Self::new(Box::new(reader)));
        }
        if data.hasattr("__arrow_c_array__")? {
            let pair = data.call_method0("__arrow_c_array__")?;
            let pair = pair
                .cast::<PyTuple>()
                .map_err(|_| PyTypeError::new_err("__arrow_c_array__ must return a tuple"))?;
            if pair.len() != 2 {
                return Err(PyTypeError::new_err(
                    "__arrow_c_array__ must return (schema capsule, array capsule)",
                ));
            }
            let schema_capsule = pair.get_item(0)?;
            let array_capsule = pair.get_item(1)?;
            // As for streams: the array is moved out in a critical section
            // on its capsule, so threads sharing one cached capsule take it
            // once.
            let batch = with_critical_section(&array_capsule, || {
                let schema = capsule_pointer(&schema_capsule, SCHEMA_CAPSULE)?;
                let array = capsule_pointer(&array_capsule, ARRAY_CAPSULE)?;
                // SAFETY: the pointers of live capsules named `arrow_schema`
                // and `arrow_array` (kept alive by `pair`): an initialized
                // schema, borrowed, and an initialized array the consumer
                // moves out of.
                unsafe {
                    ffi::import_array(
                        schema.cast::<FFI_ArrowSchema>(),
                        array.cast::<FFI_ArrowArray>(),
                    )
                }
                .map_err(|e| value_error(&e))
            })?;
            return Ok(Self::new(Box::new(MemoryReader::new(
                batch.schema(),
                vec![batch],
            ))));
        }
        Err(PyTypeError::new_err(format!(
            "{} does not implement the Arrow PyCapsule interface (__arrow_c_stream__ or __arrow_c_array__)",
            data.get_type().name()?
        )))
    }

    /// The schema as an `arrow_schema` capsule; does not consume the stream.
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let schema = FFI_ArrowSchema::try_from(self.schema.as_ref())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        PyCapsule::new_with_value(py, schema, SCHEMA_CAPSULE)
    }

    /// The batches as an `arrow_array_stream` capsule. A stream can be read
    /// once: a second call raises `ValueError`.
    ///
    /// The consumer must not call the stream's callbacks from two threads at
    /// once (the C stream interface requires calls to be serialized).
    ///
    /// `requested_schema` (an `arrow_schema` capsule, owned by the caller and
    /// only read here) is honoured when its types equal the stream's; a
    /// different field count raises `ValueError`; otherwise the stream keeps
    /// its own schema, as the protocol allows.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let requested = match requested_schema {
            None => None,
            Some(obj) if obj.is_none() => None,
            Some(capsule) => {
                let ptr = capsule_pointer(capsule, SCHEMA_CAPSULE)?;
                // SAFETY: the pointer of a live `arrow_schema` capsule (the
                // caller holds it for this call); `read_schema` only borrows
                // it: the schema is converted, never moved out or released.
                let requested = unsafe { ffi::read_schema(ptr.cast::<FFI_ArrowSchema>()) }
                    .map_err(|e| PyValueError::new_err(format!("invalid requested_schema: {e}")))?;
                Some(
                    ffi::reconcile_requested(&self.schema, &requested)
                        .map_err(PyValueError::new_err)?,
                )
            }
        };
        let reader = self
            .reader
            .lock()
            .map_err(|_| PyValueError::new_err("the Arrow stream is poisoned"))?
            .take()
            .ok_or_else(|| PyValueError::new_err("the Arrow stream was already consumed"))?;
        PyCapsule::new_with_value(py, ffi::export_stream(reader, requested), STREAM_CAPSULE)
    }

    /// Whether the stream was already handed out by `__arrow_c_stream__`.
    #[getter]
    fn consumed(&self) -> bool {
        self.reader.lock().map_or(true, |r| r.is_none())
    }

    fn __repr__(&self) -> String {
        let names: Vec<&str> = self
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        format!("ArrowStream(columns=[{}])", names.join(", "))
    }
}
