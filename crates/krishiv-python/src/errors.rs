// SPDX-License-Identifier: Apache-2.0
//! Engine errors as Python exceptions: `krishiv.errors._from_native` picks the
//! `pyspark.errors`-shaped class for the Spark error class and SQLSTATE.

use krishiv_common::error::{KrishivError, render_with_context};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// One query context for [`to_py`]: `(kind, fragment, start, stop,
/// object type, object name, call site, summary)`.
type RawContext = (u8, String, usize, usize, String, String, String, String);

/// The Python exception for `error`; a `RuntimeError` carrying the message (and
/// the conversion failure as its cause) if `krishiv.errors` cannot be used.
pub(crate) fn to_py(py: Python<'_>, error: &KrishivError) -> PyErr {
    convert(py, error, None)
}

/// [`to_py`] for an error of the query `sql`: the exception's `str` also marks
/// the SQL span that raised it over the query text with its secrets redacted;
/// `getMessage()` stays the message alone. A parse error uses Spark's
/// `ParseException` layout (as `krishiv sql` does), any other error the
/// query-context summary.
pub(crate) fn to_py_for_sql(py: Python<'_>, error: &KrishivError, sql: &str) -> PyErr {
    let redacted = krishiv_sql::redact_sql(sql);
    let text = redacted.as_str();
    let suffix = krishiv_sql::parse_exception_suffix(text, 0, error);
    let rendered = if suffix.is_empty() {
        render_with_context(error, text)
    } else {
        format!("{error}{suffix}")
    };
    convert(py, error, Some(rendered))
}

fn convert(py: Python<'_>, error: &KrishivError, rendered: Option<String>) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let parameters = PyDict::new(py);
        for (key, value) in error.message_parameters() {
            parameters.set_item(key, value)?;
        }
        let contexts: Vec<RawContext> = error
            .query_contexts()
            .iter()
            .map(|c| {
                let kind = match c.context_type {
                    krishiv_common::error::ErrorContextType::Sql => 0,
                    krishiv_common::error::ErrorContextType::DataFrame => 1,
                };
                (
                    kind,
                    c.fragment.clone(),
                    c.start,
                    c.stop,
                    c.object_type.clone(),
                    c.object_name.clone(),
                    c.call_site.clone(),
                    c.summary.clone(),
                )
            })
            .collect();
        let exception = py
            .import("krishiv.errors")?
            .getattr("_from_native")?
            .call1((
                error.error_class().name(),
                error.sql_state(),
                error.to_string(),
                parameters,
                contexts,
                rendered,
            ))?;
        Ok(PyErr::from_value(exception))
    };
    build().unwrap_or_else(|cause| {
        // Keep why the conversion failed (a bug in `krishiv.errors`) as the cause.
        let err = pyo3::exceptions::PyRuntimeError::new_err(error.to_string());
        err.set_cause(py, Some(cause));
        err
    })
}

/// Runs the engine future `work` to completion with the GIL released.
pub(crate) fn block_on<T: Send>(
    py: Python<'_>,
    work: impl std::future::Future<Output = Result<T, KrishivError>> + Send,
) -> PyResult<T> {
    py.detach(|| futures::executor::block_on(work))
        .map_err(|e| to_py(py, &e))
}

/// [`block_on`] for the query `sql`, whose errors are converted by
/// [`to_py_for_sql`].
pub(crate) fn block_on_sql<T: Send>(
    py: Python<'_>,
    sql: &str,
    work: impl std::future::Future<Output = Result<T, KrishivError>> + Send,
) -> PyResult<T> {
    py.detach(|| futures::executor::block_on(work))
        .map_err(|e| to_py_for_sql(py, &e, sql))
}
