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

    fn filter(&self, condition: &PyColumn) -> Self {
        self.derive(self.inner.filter(condition.inner.clone()))
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
        Ok(ArrowStream::new(Box::new(QueryReader::new(
            schema, first, stream,
        ))))
    }

    /// The number of result rows (executes the query, the GIL released).
    fn count(&self, py: Python<'_>) -> PyResult<u64> {
        let df = self.inner.clone();
        block_on(py, async move { df.count().await })
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
