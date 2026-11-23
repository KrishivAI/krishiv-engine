// SPDX-License-Identifier: Apache-2.0
//! Python bindings (layer L6). Builds the `krishiv._native` extension module.

#![deny(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

mod column;
mod console;
mod dataframe;
mod errors;
mod local;
mod session;
mod stream;
mod udf;
mod window;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Name of the global allocator this build links.
#[cfg(feature = "mimalloc")]
const ALLOCATOR: &str = "mimalloc";
/// Name of the global allocator this build links.
#[cfg(not(feature = "mimalloc"))]
const ALLOCATOR: &str = "system";

/// The commit this build was made from: `KRISHIV_GIT_SHA` at build time (the release
/// workflows set it; a build from a source archive or a plain `cargo build` has none).
const COMMIT: &str = match option_env!("KRISHIV_GIT_SHA") {
    Some(sha) if !sha.is_empty() => sha,
    _ => "unknown",
};

/// Returns the calling thread's freed memory to the OS: the memory pools' pressure release hook
/// (`krishiv_memory::set_release_hook`), as `krishiv_alloc::release` is for the CLI.
#[cfg(feature = "mimalloc")]
fn release_freed_memory() {
    // SAFETY: `mi_collect` has no preconditions; it only touches mimalloc's state for the calling
    // thread and the arenas, and this module's global allocator is mimalloc.
    unsafe { libmimalloc_sys::mi_collect(true) };
}

/// The extension module owns its global allocator, so it installs that allocator's policy, as the
/// CLI does (acceptance SQL-33, SQL-34): under pressure, threads return freed memory; and a limited
/// root pool (the default limit follows the container) keeps the resident set within its
/// reservations plus an allowance (`krishiv_memory` resident-set guard), so a process near its
/// cgroup spills instead of being killed. mimalloc keeps its defaults here, on purpose: huge-page
/// options take effect only before the first allocation, which the interpreter makes long before
/// this import, and the 1 s purge delay (which a setting here would still change) is kept because
/// the module usually runs at a generous limit, where prompt purging costs page faults; under
/// pressure the release hook purges anyway.
#[cfg(feature = "mimalloc")]
fn install_allocator_policy() {
    krishiv_memory::set_release_hook(release_freed_memory);
    krishiv_memory::set_resident_probe(krishiv_memory::process_resident_bytes);
}

/// What of the allocator policy is installed, comma-separated: `release-hook` (threads return
/// freed memory under pressure) and `resident-probe` (the resident-set guard samples); empty
/// without either.
fn allocator_policy() -> String {
    let parts = [
        (krishiv_memory::release_hook_installed(), "release-hook"),
        (krishiv_memory::resident_probe_installed(), "resident-probe"),
    ];
    parts
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, name)| *name)
        .collect::<Vec<_>>()
        .join(",")
}

/// All pressure epochs since the process started (`krishiv_memory::pressure_epoch`): how often
/// threads were asked to return freed memory. Both a limited pool's refused grant (at most one a
/// millisecond, no back-off) and the resident-set guard's excess (with its back-off) start one, so
/// the guard's own count shows only in a run whose pools are never refused. For measurements.
#[pyfunction]
fn _pressure_epoch() -> u64 {
    krishiv_memory::pressure_epoch()
}

/// The `krishiv._native` module.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // No CPU check here: raising from this module on a CPU below the build level would run
    // v3-compiled code on it. `krishiv/_cpu.py` checks first, in pure Python.
    #[cfg(feature = "mimalloc")]
    install_allocator_policy();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__allocator__", ALLOCATOR)?;
    m.add("__allocator_policy__", allocator_policy())?;
    m.add("__commit__", COMMIT)?;
    m.add("__ui__", console::AVAILABLE)?;
    m.add_class::<console::PyConsole>()?;
    m.add_class::<stream::ArrowStream>()?;
    m.add_function(wrap_pyfunction!(udf::_udf_pool_shutdown, m)?)?;
    m.add_function(wrap_pyfunction!(_pressure_epoch, m)?)?;
    m.add_function(wrap_pyfunction!(udf::runtime::_udf_runner_conf, m)?)?;
    m.add_class::<udf::PyAnonymousUdf>()?;
    m.add_class::<session::PySession>()?;
    m.add_class::<local::PyLocalConnectServer>()?;
    m.add_class::<dataframe::PyDataFrame>()?;
    m.add_class::<dataframe::PyGroupedData>()?;
    m.add_class::<dataframe::PyObservation>()?;
    m.add_class::<column::PyColumn>()?;
    m.add_class::<window::PyWindowSpec>()?;
    m.add_function(wrap_pyfunction!(column::column, m)?)?;
    m.add_function(wrap_pyfunction!(column::literal, m)?)?;
    m.add_function(wrap_pyfunction!(column::map_literal, m)?)?;
    m.add_function(wrap_pyfunction!(column::lambda_function, m)?)?;
    m.add_function(wrap_pyfunction!(column::call, m)?)?;
    m.add_function(wrap_pyfunction!(column::count_all, m)?)?;
    m.add_function(wrap_pyfunction!(column::count_distinct_of, m)?)?;
    m.add_function(wrap_pyfunction!(column::expression, m)?)?;
    Ok(())
}
