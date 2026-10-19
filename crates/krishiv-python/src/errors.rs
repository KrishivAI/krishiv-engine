// SPDX-License-Identifier: Apache-2.0
//! Engine errors as Python exceptions: `krishiv.errors._from_native` picks the
//! `pyspark.errors`-shaped class for the Spark error class and SQLSTATE.

use krishiv_common::error::KrishivError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The Python exception for `error`; a `RuntimeError` carrying the message if
/// `krishiv.errors` cannot be used.
pub(crate) fn to_py(py: Python<'_>, error: &KrishivError) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let parameters = PyDict::new(py);
        for (key, value) in error.message_parameters() {
            parameters.set_item(key, value)?;
        }
        let exception = py
            .import("krishiv.errors")?
            .getattr("_from_native")?
            .call1((
                error.error_class().name(),
                error.sql_state(),
                error.to_string(),
                parameters,
            ))?;
        Ok(PyErr::from_value(exception))
    };
    build().unwrap_or_else(|_| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

/// Runs the engine future `work` to completion with the GIL released.
pub(crate) fn block_on<T: Send>(
    py: Python<'_>,
    work: impl std::future::Future<Output = Result<T, KrishivError>> + Send,
) -> PyResult<T> {
    py.detach(|| futures::executor::block_on(work))
        .map_err(|e| to_py(py, &e))
}
