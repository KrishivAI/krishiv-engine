// SPDX-License-Identifier: Apache-2.0
//! `krishiv._native.DataFrame` and `GroupedData`: immutable handles on
//! `krishiv-session`'s builders. Transformations only build plans; `schema`
//! analyzes (no execution).

use krishiv_session::{DataFrame, GroupedData, JoinType};
use krishiv_types::DataType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::column::PyColumn;
use crate::errors::{block_on, to_py};

/// A lazily evaluated query.
#[pyclass(name = "DataFrame", module = "krishiv._native", frozen)]
#[derive(Debug)]
pub struct PyDataFrame {
    pub(crate) inner: DataFrame,
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
        Self {
            inner: self.inner.select(columns(cols)),
        }
    }

    fn filter(&self, condition: &PyColumn) -> Self {
        Self {
            inner: self.inner.filter(condition.inner.clone()),
        }
    }

    fn with_column(&self, name: &str, value: &PyColumn) -> Self {
        Self {
            inner: self.inner.with_column(name, value.inner.clone()),
        }
    }

    fn group_by(&self, cols: Vec<PyColumn>) -> PyGroupedData {
        PyGroupedData {
            inner: self.inner.group_by(columns(cols)),
        }
    }

    fn agg(&self, cols: Vec<PyColumn>) -> Self {
        Self {
            inner: self.inner.agg(columns(cols)),
        }
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
            return Ok(Self {
                inner: self.inner.cross_join(&other.inner),
            });
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
        Ok(Self { inner })
    }

    fn order_by(&self, cols: Vec<PyColumn>) -> Self {
        Self {
            inner: self.inner.order_by(columns(cols)),
        }
    }

    fn limit(&self, n: i32) -> Self {
        Self {
            inner: self.inner.limit(n),
        }
    }

    fn alias(&self, alias: &str) -> Self {
        Self {
            inner: self.inner.alias(alias),
        }
    }

    /// The column `name` of this DataFrame (`df["name"]`).
    fn col(&self, name: &str) -> PyColumn {
        PyColumn {
            inner: self.inner.col(name),
        }
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
}

#[pymethods]
impl PyGroupedData {
    fn agg(&self, cols: Vec<PyColumn>) -> PyDataFrame {
        PyDataFrame {
            inner: self.inner.agg(columns(cols)),
        }
    }
}
