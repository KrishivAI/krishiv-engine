//! Minimal Arrow C Data Interface <-> Python bridge (PyCapsule protocol), built on `arrow::ffi`.
//! Export: `ArrowExport` implements `__arrow_c_array__`. Import: any object with `__arrow_c_array__`.
use arrow::array::{make_array, Array, ArrayData, ArrayRef};
use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods, PyTuple};

#[pyclass(frozen)]
pub struct ArrowExport {
    data: ArrayData,
}

impl ArrowExport {
    pub fn new(a: &ArrayRef) -> Self {
        Self { data: a.to_data() }
    }
}

#[pymethods]
impl ArrowExport {
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _ = requested_schema;
        let schema = FFI_ArrowSchema::try_from(self.data.data_type())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        // Zero-copy: FFI_ArrowArray holds Arc refs to the Rust buffers.
        let array = FFI_ArrowArray::new(&self.data);
        let s = PyCapsule::new_with_value(py, schema, c"arrow_schema")?;
        let a = PyCapsule::new_with_value(py, array, c"arrow_array")?;
        PyTuple::new(py, [s.into_any(), a.into_any()])
    }
}

/// Import any `__arrow_c_array__` object (e.g. `pyarrow.Array`) into Rust without copying buffers.
pub fn import_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    let tup = obj.call_method0("__arrow_c_array__")?;
    let (sc, ac): (Bound<'_, PyCapsule>, Bound<'_, PyCapsule>) = tup.extract()?;
    let sp = sc.pointer_checked(Some(c"arrow_schema"))?.as_ptr() as *mut FFI_ArrowSchema;
    let ap = ac.pointer_checked(Some(c"arrow_array"))?.as_ptr() as *mut FFI_ArrowArray;
    // SAFETY: capsules were validated by name; per the PyCapsule Arrow protocol the consumer
    // takes ownership of the array struct by moving it out and leaving an empty (released) struct.
    let (arr, data) = unsafe {
        let arr = std::ptr::replace(ap, FFI_ArrowArray::empty());
        let data = from_ffi(arr, &*sp).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        ((), data)
    };
    let _ = arr;
    Ok(make_array(data))
}
