// SPDX-License-Identifier: Apache-2.0
//! `krishiv._native.DataFrame` and `GroupedData`: immutable handles on
//! `krishiv-session`'s builders. Transformations only build plans; `schema`
//! analyzes (no execution).

use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::StreamExt as _;
use krishiv_session::{DataFrame, GroupedData, JoinType, Session};
use krishiv_types::DataType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::column::PyColumn;
use crate::errors::{block_on, to_py};
use crate::stream::{ArrowStream, QueryReader};

/// A lazily evaluated query.
#[pyclass(name = "DataFrame", module = "krishiv._native", frozen)]
#[derive(Debug)]
pub struct PyDataFrame {
    pub(crate) inner: DataFrame,
    /// Session tables this DataFrame's plan reads that it owns: dropped when
    /// the last DataFrame derived from them is.
    pub(crate) owned: Vec<Arc<OwnedTable>>,
}

/// A temporary table dropped from its session when the last holder goes.
#[derive(Debug)]
pub(crate) struct OwnedTable {
    session: Session,
    name: String,
}

impl OwnedTable {
    pub(crate) fn new(session: Session, name: String) -> Arc<Self> {
        Arc::new(Self { session, name })
    }
}

impl Drop for OwnedTable {
    fn drop(&mut self) {
        // Whether it still existed does not matter.
        let _ = self.session.drop_temp_view(&self.name);
    }
}

impl PyDataFrame {
    /// A DataFrame that owns no tables.
    pub(crate) fn new(inner: DataFrame) -> Self {
        Self {
            inner,
            owned: Vec::new(),
        }
    }

    /// `inner`, derived from this DataFrame: it keeps the tables alive.
    fn derive_with(&self, inner: DataFrame, other: &Self) -> Self {
        let mut owned = self.owned.clone();
        owned.extend(other.owned.iter().cloned());
        Self { inner, owned }
    }

    fn derive(&self, inner: DataFrame) -> Self {
        Self {
            inner,
            owned: self.owned.clone(),
        }
    }
}

fn columns(cols: Vec<PyColumn>) -> impl Iterator<Item = krishiv_session::Column> {
    cols.into_iter().map(|c| c.inner)
}

fn join_type(how: &str) -> PyResult<JoinType> {
    Ok(match how {
        "inner" => JoinType::Inner,
        "left" | "leftouter" | "left_outer" => JoinType::LeftOuter,
        "right" | "rightouter" | "right_outer" => JoinType::RightOuter,
        "full" | "outer" | "fullouter" | "full_outer" => JoinType::FullOuter,
        "semi" | "leftsemi" | "left_semi" => JoinType::LeftSemi,
        "anti" | "leftanti" | "left_anti" => JoinType::LeftAnti,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown join type {other:?}"
            )));
        }
    })
}

#[pymethods]
impl PyDataFrame {
    fn select(&self, cols: Vec<PyColumn>) -> Self {
        self.derive(self.inner.select(columns(cols)))
    }

    /// `SELECT` of SQL expression strings (PySpark's `selectExpr`): each
    /// string holds one expression with an optional alias, parsed with
    /// the session's parser options. Parse errors fail here; resolution
    /// errors fail at the first action.
    fn select_expr(&self, py: Python<'_>, exprs: Vec<String>) -> PyResult<Self> {
        let inner = self.inner.select_expr(exprs).map_err(|e| to_py(py, &e))?;
        Ok(self.derive(inner))
    }

    fn filter(&self, condition: &PyColumn) -> Self {
        self.derive(self.inner.filter(condition.inner.clone()))
    }

    /// Rows where the SQL expression `condition` is true (PySpark's
    /// `filter(str)` and `where(str)`), parsed with the session's parser
    /// options.
    fn filter_expr(&self, py: Python<'_>, condition: &str) -> PyResult<Self> {
        let inner = self
            .inner
            .filter_expr(condition)
            .map_err(|e| to_py(py, &e))?;
        Ok(self.derive(inner))
    }

    fn with_column(&self, name: &str, value: &PyColumn) -> Self {
        self.derive(self.inner.with_column(name, value.inner.clone()))
    }

    fn group_by(&self, cols: Vec<PyColumn>) -> PyGroupedData {
        PyGroupedData {
            inner: self.inner.group_by(columns(cols)),
            owned: self.owned.clone(),
        }
    }

    fn agg(&self, cols: Vec<PyColumn>) -> Self {
        self.derive(self.inner.agg(columns(cols)))
    }

    /// Joins `other` on the condition `on`, or on the shared columns `using`;
    /// `how` is a Spark join type, `cross` (no condition) a cross join.
    #[pyo3(signature = (other, on=None, using=None, how="inner"))]
    fn join(
        &self,
        other: &Self,
        on: Option<&PyColumn>,
        using: Option<Vec<String>>,
        how: &str,
    ) -> PyResult<Self> {
        if how == "cross" {
            if on.is_some() || using.is_some() {
                return Err(PyValueError::new_err("a cross join takes no condition"));
            }
            return Ok(self.derive_with(self.inner.cross_join(&other.inner), other));
        }
        let how = join_type(how)?;
        let inner = match (on, using) {
            (Some(c), None) => self.inner.join(&other.inner, c.inner.clone(), how),
            (None, Some(names)) => {
                self.inner
                    .join_using(&other.inner, names.iter().map(String::as_str), how)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "a join takes either a condition or column names",
                ));
            }
        };
        Ok(self.derive_with(inner, other))
    }

    fn order_by(&self, cols: Vec<PyColumn>) -> Self {
        self.derive(self.inner.order_by(columns(cols)))
    }

    fn limit(&self, n: i32) -> Self {
        self.derive(self.inner.limit(n))
    }

    fn offset(&self, n: i32) -> Self {
        self.derive(self.inner.offset(n))
    }

    fn distinct(&self) -> Self {
        self.derive(self.inner.distinct())
    }

    fn union(&self, other: &Self) -> Self {
        self.derive_with(self.inner.union(&other.inner), other)
    }

    fn union_by_name(&self, other: &Self, allow_missing_columns: bool) -> Self {
        self.derive_with(
            self.inner
                .union_by_name(&other.inner, allow_missing_columns),
            other,
        )
    }

    fn intersect(&self, other: &Self) -> Self {
        self.derive_with(self.inner.intersect(&other.inner), other)
    }

    fn intersect_all(&self, other: &Self) -> Self {
        self.derive_with(self.inner.intersect_all(&other.inner), other)
    }

    fn except_all(&self, other: &Self) -> Self {
        self.derive_with(self.inner.except_all(&other.inner), other)
    }

    fn subtract(&self, other: &Self) -> Self {
        self.derive_with(self.inner.subtract(&other.inner), other)
    }

    fn hint(&self, name: &str, params: Vec<PyColumn>) -> Self {
        self.derive(self.inner.hint(name, params.into_iter().map(|c| c.inner)))
    }

    fn to_df(&self, names: Vec<String>) -> Self {
        self.derive(self.inner.to_df(names))
    }

    fn sample(&self, with_replacement: bool, fraction: f64, seed: i64) -> Self {
        self.derive(self.inner.sample(with_replacement, fraction, seed))
    }

    fn unpivot(
        &self,
        ids: Vec<PyColumn>,
        values: Vec<(String, PyColumn)>,
        variable: &str,
        value: &str,
    ) -> Self {
        self.derive(self.inner.unpivot(
            ids.into_iter().map(|c| c.inner),
            values.into_iter().map(|(label, c)| (label, c.inner)),
            variable,
            value,
        ))
    }

    fn alias(&self, alias: &str) -> Self {
        self.derive(self.inner.alias(alias))
    }

    /// The column `name` of this DataFrame (`df["name"]`).
    fn col(&self, name: &str) -> PyColumn {
        PyColumn {
            inner: self.inner.col(name),
        }
    }

    /// Runs the query (the GIL released) and returns its result as a one-shot
    /// `ArrowStream`; the batches are computed as the consumer reads them.
    /// Releasing the stream cancels the query.
    fn execute(&self, py: Python<'_>) -> PyResult<ArrowStream> {
        let df = self.inner.clone();
        let (schema, first, stream) = block_on(py, async move {
            let analyzed = df.schema().await?;
            let mut stream = df.execute_stream().await?;
            // The operators' output types (not the analysis') are the data's
            // schema; an empty result has no batch and takes the analysis'.
            let first = stream.next().await.transpose()?;
            let schema = first.as_ref().map_or(analyzed, RecordBatch::schema);
            Ok((schema, first, stream))
        })?;
        let reader = QueryReader::new(schema, first, stream);
        let slot = reader.error_slot();
        Ok(ArrowStream::new(Box::new(reader)).with_error_slot(slot))
    }

    /// The number of result rows (executes the query, the GIL released).
    fn count(&self, py: Python<'_>) -> PyResult<u64> {
        let df = self.inner.clone();
        block_on(py, async move { df.count().await })
    }

    /// Writes the rows to `path` in `format` with `mode` (PySpark's
    /// `DataFrameWriter.save`): `options` pairs and the `partition_by`
    /// columns are those the facade collected. Runs with the GIL released;
    /// a partitioned write fails with `KRISHIV_UNSUPPORTED_FEATURE`.
    #[pyo3(signature = (format, mode, path, options=None, partition_by=None))]
    fn write(
        &self,
        py: Python<'_>,
        format: &str,
        mode: &str,
        path: &str,
        options: Option<Vec<(String, String)>>,
        partition_by: Option<Vec<String>>,
    ) -> PyResult<()> {
        let mut writer = self.inner.write().format(format).mode(mode);
        for (key, value) in options.unwrap_or_default() {
            writer = writer.option(&key, &value);
        }
        let partition_by = partition_by.unwrap_or_default();
        if !partition_by.is_empty() {
            writer = writer.partition_by(partition_by.iter().map(String::as_str));
        }
        let path = path.to_owned();
        block_on(py, async move { writer.save(&path).await })
    }

    /// The text (or, with `json`, the JSON document) of explain `mode`:
    /// Spark's `simple`, `extended`, `codegen`, `cost`, `formatted`, or
    /// Krishiv's `analyze` (runs the query) and `verbose`. JSON exists for
    /// `analyze` and `verbose` only.
    #[pyo3(signature = (mode, json=false))]
    fn explain(&self, py: Python<'_>, mode: &str, json: bool) -> PyResult<String> {
        use krishiv_session::{ExplainMode, TextOptions};
        let spark = match mode {
            "simple" => Some(ExplainMode::Simple),
            "extended" => Some(ExplainMode::Extended),
            "codegen" => Some(ExplainMode::Codegen),
            "cost" => Some(ExplainMode::Cost),
            "formatted" => Some(ExplainMode::Formatted),
            "analyze" | "verbose" => None,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown explain mode: {other}"
                )));
            }
        };
        if json && spark.is_some() {
            return Err(PyValueError::new_err(format!(
                "explain mode {mode} has no JSON form (analyze and verbose have)"
            )));
        }
        let df = self.inner.clone();
        match (spark, mode) {
            (Some(m), _) => block_on(py, async move { df.explain(m).await }),
            (None, "analyze") => {
                let p = block_on(py, async move { df.explain_analyze().await })?;
                Ok(if json {
                    p.to_json()
                } else {
                    p.to_text(TextOptions::default())
                })
            }
            _ => {
                let v = block_on(py, async move { df.explain_verbose().await })?;
                Ok(if json { v.to_json() } else { v.to_text() })
            }
        }
    }

    /// Registers this DataFrame as the temporary view `name`.
    fn create_or_replace_temp_view(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        let (df, name) = (self.inner.clone(), name.to_owned());
        block_on(
            py,
            async move { df.create_or_replace_temp_view(&name).await },
        )
    }

    /// `(name, Spark simple type string, nullable)` per column, by analysis.
    fn schema(&self, py: Python<'_>) -> PyResult<Vec<(String, String, bool)>> {
        let df = self.inner.clone();
        let schema = block_on(py, async move { df.schema().await })?;
        schema
            .fields()
            .iter()
            .map(|f| {
                let data_type = DataType::from_arrow_field(f).map_err(|e| to_py(py, &e))?;
                Ok((f.name().clone(), data_type.simple_string(), f.is_nullable()))
            })
            .collect()
    }
}

/// A DataFrame grouped by key columns.
#[pyclass(name = "GroupedData", module = "krishiv._native", frozen)]
#[derive(Debug)]
pub struct PyGroupedData {
    inner: GroupedData,
    owned: Vec<Arc<OwnedTable>>,
}

#[pymethods]
impl PyGroupedData {
    fn agg(&self, cols: Vec<PyColumn>) -> PyDataFrame {
        PyDataFrame {
            inner: self.inner.agg(columns(cols)),
            owned: self.owned.clone(),
        }
    }
}
