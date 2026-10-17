// SPDX-License-Identifier: Apache-2.0
//! Python bindings (layer L6). Builds the `krishiv._native` extension module.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

/// The `krishiv._native` module.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
