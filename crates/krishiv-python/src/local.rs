// SPDX-License-Identifier: Apache-2.0
//! `krishiv._native.LocalConnectServer`: the in-process Spark Connect server behind
//! `krishiv.start_local()` (P3-T016). The server is `krishiv_connect_server::LocalServer`: a
//! loopback listener on a free port, a token generated for this start, one thread of its own.

use std::sync::Mutex;

use krishiv_common::secret::SecretString;
use krishiv_connect_server::LocalServer;
use krishiv_session::Session;
use pyo3::prelude::*;

use crate::errors::to_py;

/// A running local Connect server.
#[pyclass(name = "LocalConnectServer", module = "krishiv._native", frozen)]
pub struct PyLocalConnectServer {
    server: Mutex<LocalServer>,
    port: u16,
    token: SecretString,
    connection_string: SecretString,
}

impl std::fmt::Debug for PyLocalConnectServer {
    /// Never the token or the connection string, which holds it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyLocalConnectServer")
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

#[pymethods]
impl PyLocalConnectServer {
    /// Starts a server over a new session with the configuration `config` (`(key, value)` pairs).
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(py: Python<'_>, config: Option<Vec<(String, String)>>) -> PyResult<Self> {
        let pairs = config.unwrap_or_default();
        let mut builder = Session::builder();
        for (key, value) in pairs.clone() {
            builder = builder.config(key, value);
        }
        let base = builder.build().map_err(|e| to_py(py, &e))?;
        let pool_pairs = pairs_for_pool(&pairs);
        let server = py
            .detach(|| start_server(base, &pool_pairs))
            .map_err(|e| to_py(py, &e))?;
        Ok(Self {
            port: server.addr().port(),
            token: SecretString::new(server.token()),
            connection_string: SecretString::new(server.connection_string()),
            server: Mutex::new(server),
        })
    }

    /// The loopback port the server listens on.
    #[getter]
    fn port(&self) -> u16 {
        self.port
    }

    /// The token every RPC must carry.
    #[getter]
    fn token(&self) -> &str {
        self.token.expose_secret()
    }

    /// `sc://localhost:<port>/;token=<token>`.
    #[getter]
    fn connection_string(&self) -> &str {
        self.connection_string.expose_secret()
    }

    /// Whether the server is serving.
    fn is_running(&self) -> bool {
        self.server
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_running()
    }

    /// Stops the server (idempotent): the listener closes and the server thread is joined.
    fn stop(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.server
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .stop()
        })
        .map_err(|e| to_py(py, &e))
    }
}

/// The settings the UDF worker pool reads (`krishiv.python.worker.*`, `krishiv.udf.*`).
fn pairs_for_pool(pairs: &[(String, String)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .filter(|(k, _)| k.starts_with("krishiv.python.worker.") || k.starts_with("krishiv.udf."))
        .cloned()
        .collect()
}

/// Starts the server with a Python UDF worker pool (Linux): workers are this interpreter's own
/// `python -c` running the sandbox trampoline of this module, so the wheel needs no `krishiv` binary.
/// A pool that cannot start (no kernel sandbox, no interpreter) leaves the server without one: Python
/// UDFs are then refused with the server's usual message, everything else works.
#[cfg(target_os = "linux")]
fn start_server(
    base: Session,
    pool_pairs: &[(String, String)],
) -> Result<LocalServer, krishiv_common::error::KrishivError> {
    use std::sync::Arc;

    use krishiv_common::config::Config;
    use krishiv_udf_pool::{Launcher, Pool, PoolConfig};

    let (mut config, _warnings) = Config::from_pairs(pool_pairs.iter().cloned())?;
    let python = Python::attach(|py| {
        py.import("sys")
            .and_then(|sys| sys.getattr("executable"))
            .and_then(|e| e.extract::<String>())
            .ok()
    });
    if let Some(python) = &python
        && !pool_pairs
            .iter()
            .any(|(k, _)| k == "krishiv.python.worker.executables")
    {
        let _ = config.set("krishiv.python.worker.executables", python);
    }
    let mut pool_config = PoolConfig::from_config(&config);
    if let Some(python) = python {
        pool_config.launcher = Launcher {
            program: python.into(),
            // `-I` (isolated mode, implies `-P`): no working directory or script directory on `sys.path`, so a
            // file a Connect client uploaded into the worker directory (`krishiv.py`) is not imported before the
            // sandbox is up.
            args: vec![
                "-I".into(),
                "-c".into(),
                "import sys; from krishiv._native import _udf_trampoline; sys.exit(_udf_trampoline(sys.argv[1:]))"
                    .into(),
            ],
        };
    }
    match Pool::new(pool_config) {
        Ok(pool) => LocalServer::start_with_udf_host(base, Arc::new(pool)),
        Err(_) => LocalServer::start(base),
    }
}

#[cfg(not(target_os = "linux"))]
fn start_server(
    base: Session,
    _pool_pairs: &[(String, String)],
) -> Result<LocalServer, krishiv_common::error::KrishivError> {
    LocalServer::start(base)
}

/// The worker launcher's entry (`python -c ...`): runs the sandbox trampoline, which replaces the process
/// on success; returns its exit code otherwise.
#[cfg(target_os = "linux")]
#[pyfunction]
pub(crate) fn _udf_trampoline(args: Vec<std::ffi::OsString>) -> i32 {
    use std::process::ExitCode;
    let code: ExitCode = krishiv_udf_sandbox::trampoline_main(args);
    // `ExitCode` has no accessor: its only failure value here is the sandbox's own code.
    if code == ExitCode::SUCCESS {
        0
    } else {
        i32::from(krishiv_udf_sandbox::EXIT_SANDBOX_FAILED)
    }
}
