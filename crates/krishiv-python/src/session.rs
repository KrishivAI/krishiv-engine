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
    /// The session's Python UDF runtime, made by the first registration.
    udfs: std::sync::OnceLock<std::sync::Arc<crate::udf::runtime::PyUdfRuntime>>,
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
        Ok(Self {
            inner,
            udfs: std::sync::OnceLock::new(),
        })
    }

    /// Runs the SQL query `query` through analysis; its DataFrame.
    fn sql(&self, py: Python<'_>, query: &str) -> PyResult<PyDataFrame> {
        let (session, owned) = (self.inner.clone(), query.to_owned());
        let inner = block_on_sql(py, query, async move { session.sql(&owned).await })?;
        Ok(PyDataFrame::new(inner))
    }

    /// Registers a Python UDF (P2-T059): `wrapper` takes one
    /// `pyarrow.Array` per argument and returns one `pyarrow.Array` of
    /// `return_type` (a Spark DDL type) with as many values. The
    /// registration version, which plans name the UDF by.
    #[pyo3(signature = (name, eval_type, return_type, deterministic, thread_safe, wrapper))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the fields of one registration, as Spark's UDF has them"
    )]
    fn register_python_udf(
        &self,
        py: Python<'_>,
        name: &str,
        eval_type: i32,
        return_type: &str,
        deterministic: bool,
        thread_safe: bool,
        wrapper: Py<PyAny>,
    ) -> PyResult<u32> {
        let eval_type = krishiv_plan::PythonEvalType::from_code(eval_type).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported Python UDF eval type {eval_type}"
            ))
        })?;
        let return_type = krishiv_types::parse_data_type(return_type).map_err(|e| to_py(py, &e))?;
        let runtime = self.udf_runtime(py)?;
        let entry = self
            .inner
            .register_python_udf(krishiv_session::PythonUdfDef {
                name: name.to_owned(),
                eval_type,
                return_type,
                deterministic,
                thread_safe,
            });
        runtime.insert(
            name.to_ascii_lowercase(),
            entry.version,
            wrapper,
            thread_safe,
        );
        Ok(entry.version)
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
    /// `count` and `sum`. Query series appear only for queries that ran
    /// while `krishiv.metrics.enabled` was true.
    fn metrics(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .metrics()
            .dicts()
            .into_iter()
            .map(|d| metric_dict(py, &d))
            .collect()
    }
}

/// One [`MetricDict`](krishiv_session::MetricDict) as a Python dict.
fn metric_dict(py: Python<'_>, d: &krishiv_session::MetricDict) -> PyResult<Py<PyAny>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("name", d.name)?;
    out.set_item("kind", d.kind)?;
    out.set_item("unit", d.unit)?;
    let labels = pyo3::types::PyDict::new(py);
    for (name, value) in &d.labels {
        labels.set_item(name, value)?;
    }
    out.set_item("labels", labels)?;
    if let Some(value) = d.value {
        out.set_item("value", value)?;
    }
    if let Some(worker_id) = &d.worker_id {
        out.set_item("worker_id", worker_id)?;
    }
    if let Some(buckets) = &d.buckets {
        let pairs: Vec<(f64, u64)> = buckets.clone();
        out.set_item("buckets", pairs)?;
    }
    if let Some(count) = d.count {
        out.set_item("count", count)?;
    }
    if let Some(sum) = d.sum {
        out.set_item("sum", sum)?;
    }
    Ok(out.into_any().unbind())
}

impl PySession {
    /// The session's Python UDF runtime, made (and given to the session) on
    /// first use, its calls on the process's pool.
    fn udf_runtime(
        &self,
        py: Python<'_>,
    ) -> PyResult<std::sync::Arc<crate::udf::runtime::PyUdfRuntime>> {
        if let Some(r) = self.udfs.get() {
            return Ok(std::sync::Arc::clone(r));
        }
        let threads = self
            .inner
            .conf()
            .get("krishiv.python.udf.threads")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let size = crate::udf::pool_size(py, threads)?;
        let made = std::sync::Arc::new(crate::udf::runtime::PyUdfRuntime::new(size));
        let runtime = std::sync::Arc::clone(self.udfs.get_or_init(|| made));
        self.inner
            .set_python_udf_runtime(std::sync::Arc::clone(&runtime)
                as std::sync::Arc<dyn krishiv_exec::python_eval::UdfRuntime>);
        Ok(runtime)
    }
}
