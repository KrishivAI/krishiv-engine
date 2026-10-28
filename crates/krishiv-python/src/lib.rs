// SPDX-License-Identifier: Apache-2.0
//! Python bindings (layer L6). Builds the `krishiv._native` extension module.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

mod column;
mod dataframe;
mod errors;
mod session;
mod stream;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Name of the global allocator this build links.
#[cfg(feature = "mimalloc")]
const ALLOCATOR: &str = "mimalloc";
/// Name of the global allocator this build links.
#[cfg(not(feature = "mimalloc"))]
const ALLOCATOR: &str = "system";

/// The `krishiv._native` module.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // No CPU check here: raising from this module on a CPU below the build level would run
    // v3-compiled code on it. `krishiv/_cpu.py` checks first, in pure Python.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__allocator__", ALLOCATOR)?;
    m.add_class::<stream::ArrowStream>()?;
    m.add_class::<session::PySession>()?;
    m.add_class::<dataframe::PyDataFrame>()?;
    m.add_class::<dataframe::PyGroupedData>()?;
    m.add_class::<column::PyColumn>()?;
    m.add_function(wrap_pyfunction!(column::column, m)?)?;
    m.add_function(wrap_pyfunction!(column::literal, m)?)?;
    m.add_function(wrap_pyfunction!(column::map_literal, m)?)?;
    m.add_function(wrap_pyfunction!(column::call, m)?)?;
    m.add_function(wrap_pyfunction!(column::count_all, m)?)?;
    m.add_function(wrap_pyfunction!(column::count_distinct_of, m)?)?;
    m.add_function(wrap_pyfunction!(column::expression, m)?)?;
    Ok(())
}
