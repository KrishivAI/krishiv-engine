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
        let mut builder = Session::builder();
        for (key, value) in config.unwrap_or_default() {
            builder = builder.config(key, value);
        }
        let base = builder.build().map_err(|e| to_py(py, &e))?;
        let server = py
            .detach(|| LocalServer::start(base))
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
