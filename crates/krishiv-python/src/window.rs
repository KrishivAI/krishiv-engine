// SPDX-License-Identifier: Apache-2.0
//! `krishiv._native.WindowSpec`: an immutable handle on a
//! `krishiv-session` window specification. Builder semantics live in
//! `krishiv-session`; the PySpark-shaped surface is in
//! `python/krishiv/sql/window.py`.

use krishiv_session::WindowSpec;
use pyo3::prelude::*;

use crate::column::PyColumn;

/// A window specification: partitioning, ordering and an optional frame.
#[pyclass(
    name = "WindowSpec",
    module = "krishiv._native",
    frozen,
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyWindowSpec {
    pub(crate) inner: WindowSpec,
}

#[pymethods]
impl PyWindowSpec {
    #[new]
    fn new() -> Self {
        Self {
            inner: WindowSpec::new(),
        }
    }

    /// This window partitioning by `cols`, replacing any partitioning.
    fn partition_by(&self, cols: Vec<PyColumn>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .partition_by(cols.into_iter().map(|c| c.inner)),
        }
    }

    /// This window ordering by `cols`, replacing any ordering. A column
    /// built by `asc`/`desc` orders that way.
    fn order_by(&self, cols: Vec<PyColumn>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .order_by(cols.into_iter().map(|c| c.inner)),
        }
    }

    /// This window with the `ROWS` frame from `start` to `end`
    /// (`i64::MIN` is `UNBOUNDED PRECEDING`, `i64::MAX` is `UNBOUNDED
    /// FOLLOWING`, `0` is `CURRENT ROW`).
    fn rows_between(&self, start: i64, end: i64) -> Self {
        Self {
            inner: self.inner.clone().rows_between(start, end),
        }
    }

    /// This window with the `RANGE` frame from `start` to `end`, with the
    /// same bounds as [`Self::rows_between`].
    fn range_between(&self, start: i64, end: i64) -> Self {
        Self {
            inner: self.inner.clone().range_between(start, end),
        }
    }
}
