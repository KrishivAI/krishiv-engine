// SPDX-License-Identifier: Apache-2.0
//! `krishiv._native.Column`: an immutable handle on a `krishiv-session`
//! column expression. All builder semantics live in `krishiv-session`.

use krishiv_session::Column;
use krishiv_session::functions::{call_function, col, count_distinct, count_star, lit, lit_null};
use krishiv_types::DataType;
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

use crate::errors::to_py;

/// A column expression.
#[pyclass(name = "Column", module = "krishiv._native", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyColumn {
    pub(crate) inner: Column,
}

impl PyColumn {
    fn wrap(inner: Column) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyColumn {
    fn alias(&self, name: &str) -> Self {
        Self::wrap(self.inner.clone().alias(name))
    }

    /// `CAST(self AS to)`, `to` a Spark type string (`int`, `decimal(10,2)`).
    fn cast(&self, py: Python<'_>, to: &str) -> PyResult<Self> {
        let data_type = to.parse::<DataType>().map_err(|e| to_py(py, &e))?;
        Ok(Self::wrap(self.inner.clone().cast(data_type)))
    }

    fn add(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone() + other.inner.clone())
    }

    fn sub(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone() - other.inner.clone())
    }

    fn mul(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone() * other.inner.clone())
    }

    fn div(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone() / other.inner.clone())
    }

    fn rem(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone() % other.inner.clone())
    }

    fn neg(&self) -> Self {
        Self::wrap(-self.inner.clone())
    }

    fn invert(&self) -> Self {
        Self::wrap(!self.inner.clone())
    }

    #[pyo3(name = "eq")]
    fn equal(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone().eq(other.inner.clone()))
    }

    fn ne(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone().not_eq(other.inner.clone()))
    }

    fn lt(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone().lt(other.inner.clone()))
    }

    fn le(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone().lt_eq(other.inner.clone()))
    }

    fn gt(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone().gt(other.inner.clone()))
    }

    fn ge(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone().gt_eq(other.inner.clone()))
    }

    fn eq_null_safe(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone().eq_null_safe(other.inner.clone()))
    }

    #[pyo3(name = "and_")]
    fn and(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone().and(other.inner.clone()))
    }

    #[pyo3(name = "or_")]
    fn or(&self, other: &Self) -> Self {
        Self::wrap(self.inner.clone().or(other.inner.clone()))
    }

    fn is_null(&self) -> Self {
        Self::wrap(self.inner.clone().is_null())
    }

    fn is_not_null(&self) -> Self {
        Self::wrap(self.inner.clone().is_not_null())
    }

    fn like(&self, pattern: &str) -> Self {
        Self::wrap(self.inner.clone().like(pattern))
    }

    fn isin(&self, values: Vec<Self>) -> Self {
        Self::wrap(
            self.inner
                .clone()
                .is_in(values.into_iter().map(|c| c.inner)),
        )
    }

    fn between(&self, low: &Self, high: &Self) -> Self {
        Self::wrap(
            self.inner
                .clone()
                .between(low.inner.clone(), high.inner.clone()),
        )
    }

    /// The sort order: `ascending`, nulls first by default (Spark's), or
    /// last when `nulls_first` is false; the same for descending.
    fn sort_order(&self, ascending: bool, nulls_first: bool) -> Self {
        let c = self.inner.clone();
        Self::wrap(match (ascending, nulls_first) {
            (true, true) => c.asc(),
            (true, false) => c.asc_nulls_last(),
            (false, true) => c.desc_nulls_first(),
            (false, false) => c.desc(),
        })
    }
}

/// The column named `name` (`a`, `t.a`, `` `a.b` ``, `*`, `t.*`).
#[pyfunction(name = "col")]
pub(crate) fn column(name: &str) -> PyColumn {
    PyColumn::wrap(col(name))
}

/// A literal: `None` (an untyped null), `bool`, `int` (Spark `int`, or `bigint` beyond 32 bits),
/// `float` (`double`) or `str`.
#[pyfunction(name = "lit")]
pub(crate) fn literal(value: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
    // `bool` is an `int` subclass: test it first.
    let column = if value.is_none() {
        lit_null()
    } else if let Ok(b) = value.cast::<PyBool>() {
        lit(b.is_true())
    } else if value.is_instance_of::<PyInt>() {
        let n: i64 = value
            .extract()
            .map_err(|_| PyOverflowError::new_err("integer literal out of the 64-bit range"))?;
        i32::try_from(n).map_or_else(|_| lit(n), lit)
    } else if value.is_instance_of::<PyFloat>() {
        lit(value.extract::<f64>()?)
    } else if value.is_instance_of::<PyString>() {
        lit(value.extract::<String>()?)
    } else {
        return Err(PyTypeError::new_err(format!(
            "unsupported literal type {}",
            value.get_type().name()?
        )));
    };
    Ok(PyColumn::wrap(column))
}

/// A call of the function `name` with `args` (resolved when queried).
#[pyfunction(name = "call_function")]
pub(crate) fn call(name: &str, args: Vec<PyColumn>) -> PyColumn {
    PyColumn::wrap(call_function(name, args.into_iter().map(|c| c.inner)))
}

/// `count(*)`.
#[pyfunction(name = "count_star")]
pub(crate) fn count_all() -> PyColumn {
    PyColumn::wrap(count_star())
}

/// `count(DISTINCT cols)`.
#[pyfunction(name = "count_distinct")]
pub(crate) fn count_distinct_of(cols: Vec<PyColumn>) -> PyResult<PyColumn> {
    if cols.is_empty() {
        return Err(PyValueError::new_err("count_distinct needs a column"));
    }
    Ok(PyColumn::wrap(count_distinct(
        cols.into_iter().map(|c| c.inner),
    )))
}

/// Parses `text` as a SQL expression (`F.expr`) with the default parser
/// options; parse and lowering errors become the engine's exceptions.
#[pyfunction(name = "expr")]
pub(crate) fn expression(py: Python<'_>, text: &str) -> PyResult<PyColumn> {
    let column = krishiv_session::functions::expr(text).map_err(|e| to_py(py, &e))?;
    Ok(PyColumn::wrap(column))
}
