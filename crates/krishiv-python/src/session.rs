// SPDX-License-Identifier: Apache-2.0
//! `krishiv._native.Session`: a handle on a `krishiv-session` `Session`.

use std::sync::atomic::{AtomicU64, Ordering};

use krishiv_session::Session;
use pyo3::prelude::*;

use crate::dataframe::{OwnedTable, PyDataFrame};
use crate::errors::{block_on_sql, to_py};
use crate::stream::ArrowStream;

/// Numbers the tables `create_dataframe` registers.
static ARROW_TABLES: AtomicU64 = AtomicU64::new(0);

/// The name prefix of the tables `create_dataframe` registers.
const ARROW_TABLE_PREFIX: &str = "__krishiv_arrow_";

/// Whether `name` is a table `create_dataframe` registered (names match
/// case-insensitively, as the catalog's do).
fn is_arrow_table(name: &str) -> bool {
    name.get(..ARROW_TABLE_PREFIX.len())
        .is_some_and(|p| p.eq_ignore_ascii_case(ARROW_TABLE_PREFIX))
}

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
        let (session, owned) = (self.inner.clone(), query.to_owned());
        let inner = block_on_sql(py, query, async move { session.sql(&owned).await })?;
        Ok(PyDataFrame::new(inner))
    }

    fn table(&self, name: &str) -> PyDataFrame {
        PyDataFrame::new(self.inner.table(name))
    }

    /// Drops the temporary table or view `name`; whether it existed. The
    /// tables that hold `create_dataframe` data are not the user's to drop:
    /// a DataFrame still reads them.
    fn drop_temp_view(&self, name: &str) -> bool {
        !is_arrow_table(name) && self.inner.drop_temp_view(name)
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
            "{ARROW_TABLE_PREFIX}{}",
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

    /// Names of the session's temporary tables and views (with the tables
    /// that hold `create_dataframe` data).
    fn temp_table_names(&self) -> Vec<String> {
        self.inner.temp_table_names()
    }

    /// Whether `name` is a table that holds `create_dataframe` data.
    #[staticmethod]
    fn is_dataframe_table(name: &str) -> bool {
        is_arrow_table(name)
    }

    /// Drops every cached relation (Spark's `CLEAR CACHE` and
    /// `spark.catalog.clearCache`).
    fn clear_cache(&self) {
        self.inner.clear_cache();
    }

    /// Drops the cached relations reading the table `name` (Spark's
    /// `UNCACHE TABLE`).
    fn uncache_table(&self, name: &str) {
        self.inner.uncache_table(name);
    }

    /// Drops the cached relations reading the table `name` (Spark's
    /// `REFRESH TABLE`).
    fn refresh_table(&self, name: &str) {
        self.inner.refresh_table(name);
    }

    /// Drops the cached relations reading files by path (Spark's
    /// `REFRESH path`).
    fn refresh_files(&self) {
        self.inner.refresh_files();
    }

    /// Bytes the relation cache holds now.
    fn cache_reserved(&self) -> usize {
        self.inner.cache_reserved()
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

    /// The engine's metrics snapshot as plain dicts: each has `name`,
    /// `kind` (`counter`, `gauge` or `histogram`), `unit` and `labels`;
    /// counters and gauges add `value` (gauges also `worker_id`);
    /// histograms add cumulative `buckets` as `(le, count)` pairs with
    /// `count` and `sum`. Query series appear only when the session was
    /// built with `krishiv.metrics.enabled=true`.
    fn metrics(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.inner
            .metrics()
            .dicts()
            .into_iter()
            .map(|d| metric_dict(py, &d))
            .collect()
    }
}

/// One [`MetricDict`](krishiv_session::MetricDict) as a Python dict.
fn metric_dict(py: Python<'_>, d: &krishiv_session::MetricDict) -> Py<PyAny> {
    let out = pyo3::types::PyDict::new(py);
    let _ = out.set_item("name", d.name);
    let _ = out.set_item("kind", d.kind);
    let _ = out.set_item("unit", d.unit);
    let labels = pyo3::types::PyDict::new(py);
    for (name, value) in &d.labels {
        let _ = labels.set_item(name, value);
    }
    let _ = out.set_item("labels", labels);
    if let Some(value) = d.value {
        let _ = out.set_item("value", value);
    }
    if let Some(worker_id) = &d.worker_id {
        let _ = out.set_item("worker_id", worker_id);
    }
    if let Some(buckets) = &d.buckets {
        let pairs: Vec<(f64, u64)> = buckets.clone();
        let _ = out.set_item("buckets", pairs);
    }
    if let Some(count) = d.count {
        let _ = out.set_item("count", count);
    }
    if let Some(sum) = d.sum {
        let _ = out.set_item("sum", sum);
    }
    out.into_any().unbind()
}
