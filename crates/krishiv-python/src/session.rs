// SPDX-License-Identifier: Apache-2.0
//! `krishiv._native.Session`: a handle on a `krishiv-session` `Session`.

use std::sync::atomic::{AtomicU64, Ordering};

use krishiv_session::Session;
use pyo3::prelude::*;

use crate::dataframe::{OwnedTable, PyDataFrame};
use crate::errors::{block_on, to_py};
use crate::stream::ArrowStream;

/// Numbers the tables `create_dataframe` registers.
static ARROW_TABLES: AtomicU64 = AtomicU64::new(0);

/// A Krishiv session.
#[pyclass(name = "Session", module = "krishiv._native", frozen)]
#[derive(Debug)]
pub struct PySession {
    inner: Session,
}

#[pymethods]
impl PySession {
    /// A session with the configuration `config` (`(key, value)` pairs).
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(py: Python<'_>, config: Option<Vec<(String, String)>>) -> PyResult<Self> {
        let mut builder = Session::builder();
        for (key, value) in config.unwrap_or_default() {
            builder = builder.config(key, value);
        }
        let inner = builder.build().map_err(|e| to_py(py, &e))?;
        Ok(Self { inner })
    }

    /// Runs the SQL query `query` through analysis; its DataFrame.
    fn sql(&self, py: Python<'_>, query: &str) -> PyResult<PyDataFrame> {
        let (session, query) = (self.inner.clone(), query.to_owned());
        let inner = block_on(py, async move { session.sql(&query).await })?;
        Ok(PyDataFrame::new(inner))
    }

    fn table(&self, name: &str) -> PyDataFrame {
        PyDataFrame::new(self.inner.table(name))
    }

    /// Drops the temporary table or view `name`; whether it existed.
    fn drop_temp_view(&self, name: &str) -> bool {
        self.inner.drop_temp_view(name)
    }

    /// The DataFrame reading the files `paths` as `format`, with `options`.
    #[pyo3(signature = (format, paths, options=None))]
    fn read(
        &self,
        format: &str,
        paths: Vec<String>,
        options: Option<Vec<(String, String)>>,
    ) -> PyDataFrame {
        let mut reader = self.inner.read().format(format);
        for (key, value) in options.unwrap_or_default() {
            reader = reader.option(&key, &value);
        }
        PyDataFrame::new(reader.load(paths))
    }

    /// A DataFrame of the Arrow data `data` (any object implementing the
    /// Arrow PyCapsule interface), read now. The data lives in a temporary
    /// table of a generated name that the returned DataFrame (and those
    /// derived from it) own: it is dropped with the last of them.
    fn create_dataframe(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let (schema, batches) = ArrowStream::from_arrow(data)?.read_all()?;
        let name = format!(
            "__krishiv_arrow_{}",
            ARROW_TABLES.fetch_add(1, Ordering::Relaxed)
        );
        self.inner
            .register_batches(&name, schema, batches)
            .map_err(|e| to_py(py, &e))?;
        Ok(PyDataFrame {
            inner: self.inner.table(&name),
            owned: vec![OwnedTable::new(self.inner.clone(), name)],
        })
    }

    /// Names of the session's temporary tables and views.
    fn temp_table_names(&self) -> Vec<String> {
        self.inner.temp_table_names()
    }

    fn conf_is_modifiable(&self, key: &str) -> bool {
        self.inner.conf().is_modifiable(key)
    }

    fn conf_get(&self, key: &str) -> Option<String> {
        self.inner.conf().get(key)
    }

    fn conf_set(&self, py: Python<'_>, key: &str, value: &str) -> PyResult<()> {
        self.inner.conf().set(key, value).map_err(|e| to_py(py, &e))
    }

    fn conf_unset(&self, py: Python<'_>, key: &str) -> PyResult<()> {
        self.inner.conf().unset(key).map_err(|e| to_py(py, &e))
    }
}
