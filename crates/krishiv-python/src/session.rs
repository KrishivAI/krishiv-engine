// SPDX-License-Identifier: Apache-2.0
//! `krishiv._native.Session`: a handle on a `krishiv-session` `Session`.

use std::sync::atomic::{AtomicU64, Ordering};

use krishiv_session::Session;
use pyo3::prelude::*;

use crate::dataframe::{OwnedTable, PyDataFrame};
use crate::errors::{block_on, block_on_sql, to_py};
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

/// `(name, catalog, description, location_uri)`.
type DatabaseRow = (String, String, Option<String>, String);
/// `(name, catalog, namespace, description, table_type, is_temporary)`.
type TableRow = (
    String,
    Option<String>,
    Option<Vec<String>>,
    Option<String>,
    String,
    bool,
);
/// `(name, description, data_type, nullable, is_partition, is_bucket, is_cluster)`.
type ColumnRow = (String, Option<String>, String, bool, bool, bool, bool);
/// `(name, catalog, namespace, description, class_name, is_temporary)`.
type FunctionRow = (
    String,
    Option<String>,
    Option<Vec<String>>,
    Option<String>,
    String,
    bool,
);

fn database_row(d: krishiv_session::spark_catalog::Database) -> DatabaseRow {
    (d.name, d.catalog, d.description, d.location_uri)
}

fn table_row(t: krishiv_session::spark_catalog::Table) -> TableRow {
    (
        t.name,
        t.catalog,
        t.namespace,
        t.description,
        t.table_type,
        t.is_temporary,
    )
}

fn function_row(f: krishiv_session::spark_catalog::Function) -> FunctionRow {
    (
        f.name,
        f.catalog,
        f.namespace,
        f.description,
        f.class_name,
        f.is_temporary,
    )
}

fn ids(ids: &[krishiv_session::QueryId]) -> Vec<String> {
    ids.iter().map(ToString::to_string).collect()
}

/// A Krishiv session.
#[pyclass(name = "Session", module = "krishiv._native", frozen)]
#[derive(Debug)]
pub struct PySession {
    inner: Session,
    /// The session's state as `errors::LIVE_SESSIONS` holds it weakly, for a Ctrl-C to cancel through.
    live: std::sync::Arc<Session>,
    /// The session's Python UDF runtime, made by the first registration.
    udfs: std::sync::Arc<std::sync::OnceLock<std::sync::Arc<crate::udf::runtime::PyUdfRuntime>>>,
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
        let live = std::sync::Arc::new(inner.clone());
        crate::errors::register_session(&live);
        let session = Self {
            inner,
            live,
            udfs: std::sync::Arc::default(),
        };
        // Every session runs unregistered UDFs (a DataFrame-side `udf(...)`
        // registers in no session), so it holds its runtime from the start;
        // the pool itself starts with the first call. After the exit hook
        // there is none (registration then raises, as before).
        let _ = session.udf_runtime(py);
        Ok(session)
    }

    /// A handle of this session whose frames carry the request tags `tags`
    /// (`SparkSession.addTag`): it shares the session's state and UDF runtime.
    fn tagged(&self, py: Python<'_>, tags: Vec<String>) -> PyResult<Self> {
        let inner = self
            .inner
            .with_request_tags(tags)
            .map_err(|e| to_py(py, &e))?;
        Ok(Self {
            inner,
            live: std::sync::Arc::clone(&self.live),
            udfs: std::sync::Arc::clone(&self.udfs),
        })
    }

    /// Cancels every running query of the session; their operation ids.
    fn interrupt_all(&self) -> Vec<String> {
        ids(&self.inner.interrupt_all())
    }

    /// Cancels the running queries that carry the request tag `tag`; their operation ids.
    fn interrupt_tag(&self, tag: &str) -> Vec<String> {
        ids(&self.inner.interrupt_tag(tag))
    }

    /// Cancels the query with operation id `id`; its id in a list, or an empty list.
    fn interrupt_operation(&self, id: &str) -> Vec<String> {
        // As Spark Connect: an id that is no operation's (malformed included) cancels nothing.
        match krishiv_session::QueryId::parse(id) {
            Ok(query) if self.inner.cancel(&query) => vec![query.to_string()],
            _ => Vec::new(),
        }
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
        runtime.insert(entry.def.name.clone(), entry.version, wrapper, thread_safe);
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

    /// The DataFrame reading the files `paths` as `format`, with `options`
    /// and the user's schema, if any: `schema_json` a struct type's JSON form
    /// (`StructType.json()`), or `schema_ddl` a DDL string (`"a INT"`), as
    /// PySpark's `DataFrameReader.schema` hands either to Spark.
    #[pyo3(signature = (format, paths, options=None, schema_json=None, schema_ddl=None))]
    fn read(
        &self,
        py: Python<'_>,
        format: &str,
        paths: Vec<String>,
        options: Option<Vec<(String, String)>>,
        schema_json: Option<&str>,
        schema_ddl: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let mut reader = self.inner.read().format(format);
        for (key, value) in options.unwrap_or_default() {
            reader = reader.option(&key, &value);
        }
        if let Some(json) = schema_json {
            match krishiv_types::DataType::from_json(json) {
                Ok(krishiv_types::DataType::Struct(s)) => reader = reader.schema(s),
                Ok(other) => {
                    let e = krishiv_common::error::InvalidSchemaNonStructType {
                        input_schema: format!("\"{json}\""),
                        data_type: format!("\"{}\"", other.sql()),
                    };
                    return Err(to_py(py, &e.into()));
                }
                Err(e) => return Err(to_py(py, &e)),
            }
        }
        if let Some(ddl) = schema_ddl {
            reader = reader.schema_ddl(ddl).map_err(|e| to_py(py, &e))?;
        }
        Ok(PyDataFrame::new(reader.load(paths)))
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

    /// A DataFrame over a Python data source (design P3-T040): `schema` is any Arrow
    /// object carrying the source's schema (read now, from an empty table), `read` a
    /// callable returning an Arrow object of that schema. The engine calls it once for
    /// every scan of the table in a query, so a source that lists its
    /// partitions in `read` sees new ones on the next action. A failure of `read` is `PYTHON_EXCEPTION`; a planning failure the
    /// Python side classed as `PYTHON_DATA_SOURCE_ERROR` keeps its action and type.
    fn register_python_source(
        &self,
        py: Python<'_>,
        label: &str,
        schema: &Bound<'_, PyAny>,
        read: Py<PyAny>,
    ) -> PyResult<PyDataFrame> {
        let (schema, _) = ArrowStream::from_arrow(schema)?.read_all()?;
        let name = format!(
            "{ARROW_TABLE_PREFIX}{}",
            ARROW_TABLES.fetch_add(1, Ordering::Relaxed)
        );
        let source: krishiv_session::BatchSource = std::sync::Arc::new(move || {
            Python::attach(|py| {
                // One wrap: a planning failure the reader's Python side already classed as
                // `PYTHON_DATA_SOURCE_ERROR` keeps its action and type (Spark's
                // `pythonDataSourceError` pairs); every other exception of user code is
                // Spark's `PYTHON_EXCEPTION`, as a task of the read raises it.
                let fail = |err: &PyErr| {
                    let value = err.value(py);
                    let class = value
                        .call_method0("getCondition")
                        .and_then(|c| c.extract::<String>())
                        .ok();
                    if matches!(
                        class.as_deref(),
                        Some("PYTHON_DATA_SOURCE_ERROR" | "KRISHIV_PYTHON_DATA_SOURCE_LIMIT")
                    ) {
                        let parameter = |key: &str| {
                            value
                                .call_method0("getMessageParameters")
                                .and_then(|p| p.get_item(key))
                                .and_then(|v| v.extract::<String>())
                                .unwrap_or_default()
                        };
                        if class.as_deref() == Some("KRISHIV_PYTHON_DATA_SOURCE_LIMIT") {
                            return krishiv_common::error::KrishivError::from(
                                krishiv_common::error::KrishivPythonDataSourceLimit {
                                    source: parameter("source"),
                                    bytes: parameter("bytes"),
                                    limit: parameter("limit"),
                                },
                            );
                        }
                        return krishiv_common::error::KrishivError::from(
                            krishiv_common::error::PythonDataSourceError {
                                action: parameter("action"),
                                r#type: parameter("type"),
                                msg: parameter("msg"),
                            },
                        );
                    }
                    let traceback = err
                        .traceback(py)
                        .and_then(|t| t.format().ok())
                        .unwrap_or_default();
                    krishiv_common::error::KrishivError::from(
                        krishiv_common::error::PythonException {
                            msg: "An exception was thrown from the Python worker".to_owned(),
                            traceback: format!("{traceback}{err}"),
                        },
                    )
                };
                let table = read.call0(py).map_err(|e| fail(&e))?;
                let (_, batches) = ArrowStream::from_arrow(table.bind(py))
                    .and_then(|s| s.read_all())
                    .map_err(|e| fail(&e))?;
                Ok(batches)
            })
        });
        self.inner
            .register_batch_source(&name, label, schema, source)
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

    // ---- spark.catalog (`krishiv_session::Catalog`); the Python `Catalog` wraps the tuples in named tuples ----

    /// The current catalog's name.
    fn catalog_current_catalog(&self) -> String {
        self.inner.catalog().current_catalog()
    }

    /// Makes `name` the current catalog (`CATALOG_NOT_FOUND` for any but `spark_catalog`).
    fn catalog_set_current_catalog(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        let (session, name) = (self.inner.clone(), name.to_owned());
        block_on(py, async move {
            session.catalog().set_current_catalog(&name).await
        })
    }

    /// The catalogs' names.
    fn catalog_list_catalogs(&self) -> Vec<String> {
        self.inner.catalog().list_catalogs()
    }

    /// The current database's name.
    fn catalog_current_database(&self) -> String {
        self.inner.catalog().current_database()
    }

    /// Makes `name` the current database.
    fn catalog_set_current_database(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        let (session, name) = (self.inner.clone(), name.to_owned());
        block_on(py, async move {
            session.catalog().set_current_database(&name).await
        })
    }

    /// `(name, catalog, description, location_uri)` of the databases matching `pattern`.
    #[pyo3(signature = (pattern=None))]
    fn catalog_list_databases(
        &self,
        py: Python<'_>,
        pattern: Option<String>,
    ) -> PyResult<Vec<DatabaseRow>> {
        let session = self.inner.clone();
        let rows = block_on(py, async move {
            session.catalog().list_databases(pattern.as_deref()).await
        })?;
        Ok(rows.into_iter().map(database_row).collect())
    }

    /// The database `name`, as [`Self::catalog_list_databases`] rows.
    fn catalog_get_database(&self, py: Python<'_>, name: &str) -> PyResult<DatabaseRow> {
        let (session, name) = (self.inner.clone(), name.to_owned());
        block_on(
            py,
            async move { session.catalog().get_database(&name).await },
        )
        .map(database_row)
    }

    /// Whether the database `name` exists.
    fn catalog_database_exists(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        let (session, name) = (self.inner.clone(), name.to_owned());
        block_on(
            py,
            async move { session.catalog().database_exists(&name).await },
        )
    }

    /// `(name, catalog, namespace, description, table_type, is_temporary)` of the tables in `database`.
    #[pyo3(signature = (database=None, pattern=None))]
    fn catalog_list_tables(
        &self,
        py: Python<'_>,
        database: Option<String>,
        pattern: Option<String>,
    ) -> PyResult<Vec<TableRow>> {
        let session = self.inner.clone();
        let rows = block_on(py, async move {
            session
                .catalog()
                .list_tables(database.as_deref(), pattern.as_deref())
                .await
        })?;
        Ok(rows.into_iter().map(table_row).collect())
    }

    /// The table or view `name`, as [`Self::catalog_list_tables`] rows.
    fn catalog_get_table(&self, py: Python<'_>, name: &str) -> PyResult<TableRow> {
        let (session, name) = (self.inner.clone(), name.to_owned());
        block_on(py, async move { session.catalog().get_table(&name).await }).map(table_row)
    }

    /// Whether the table or view `name` exists.
    fn catalog_table_exists(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        let (session, name) = (self.inner.clone(), name.to_owned());
        block_on(
            py,
            async move { session.catalog().table_exists(&name).await },
        )
    }

    /// `(name, description, data_type, nullable, is_partition, is_bucket, is_cluster)` of the columns of `name`.
    fn catalog_list_columns(&self, py: Python<'_>, name: &str) -> PyResult<Vec<ColumnRow>> {
        let (session, name) = (self.inner.clone(), name.to_owned());
        let rows = block_on(
            py,
            async move { session.catalog().list_columns(&name).await },
        )?;
        Ok(rows
            .into_iter()
            .map(|c| {
                (
                    c.name,
                    c.description,
                    c.data_type,
                    c.nullable,
                    c.is_partition,
                    c.is_bucket,
                    c.is_cluster,
                )
            })
            .collect())
    }

    /// `(name, catalog, namespace, description, class_name, is_temporary)` of the functions matching `pattern`.
    #[pyo3(signature = (pattern=None))]
    fn catalog_list_functions(
        &self,
        py: Python<'_>,
        pattern: Option<String>,
    ) -> PyResult<Vec<FunctionRow>> {
        let session = self.inner.clone();
        let rows = block_on(py, async move {
            session.catalog().list_functions(pattern.as_deref()).await
        })?;
        Ok(rows.into_iter().map(function_row).collect())
    }

    /// The function `name`, as [`Self::catalog_list_functions`] rows.
    fn catalog_get_function(&self, py: Python<'_>, name: &str) -> PyResult<FunctionRow> {
        let (session, name) = (self.inner.clone(), name.to_owned());
        block_on(
            py,
            async move { session.catalog().get_function(&name).await },
        )
        .map(function_row)
    }

    /// Whether the function `name` exists.
    fn catalog_function_exists(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        let (session, name) = (self.inner.clone(), name.to_owned());
        block_on(
            py,
            async move { session.catalog().function_exists(&name).await },
        )
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
        let executor_threads =
            krishiv_exec::Executor::parallelism(self.inner.runtime().executor().as_ref());
        let size = crate::udf::pool_size(py, threads, executor_threads)?;
        let made = std::sync::Arc::new(crate::udf::runtime::PyUdfRuntime::new(
            size,
            self.inner.conf(),
        ));
        let runtime = std::sync::Arc::clone(self.udfs.get_or_init(|| made));
        self.inner
            .set_python_udf_runtime(std::sync::Arc::clone(&runtime)
                as std::sync::Arc<dyn krishiv_exec::python_eval::UdfRuntime>);
        Ok(runtime)
    }
}
