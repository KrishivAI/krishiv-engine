// SPDX-License-Identifier: Apache-2.0
//! `krishiv._native.Console`: the web console of one session, served in this process behind
//! `krishiv.ui()` (ADR 0024). The server is `krishiv_server`'s embedded console: the loopback
//! interface only, a free port unless one is asked for, one token generated for this start, one
//! thread of its own.
//!
//! The class exists in every build, so the module's surface does not depend on features; without
//! the `ui` feature starting one fails and `krishiv._native.__ui__` is `False`.

use std::sync::Mutex;

use krishiv_common::secret::SecretString;
use krishiv_session::Session;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::session::PySession;

/// Whether this build can serve the console.
pub(crate) const AVAILABLE: bool = cfg!(feature = "ui");

/// A serving console: dropping it stops the server and joins its thread.
trait Serving: Send {
    /// Whether the server has stopped by itself (the session was closed, or serving failed).
    fn is_stopped(&self) -> bool;
}

/// What a start gives: the server and where it listens.
struct Started {
    server: Box<dyn Serving>,
    port: u16,
    token: String,
    url: String,
}

#[cfg(feature = "ui")]
impl Serving for krishiv_server::session_source::EmbeddedUi {
    fn is_stopped(&self) -> bool {
        Self::is_stopped(self)
    }
}

/// Why a start failed. The class and SQLSTATE are the engine's, when the engine raised the failure (the
/// console's own session could not be built); a set-up failure (the port is taken, the session is closed, a
/// build without the console) has none.
struct StartError {
    message: String,
    error_class: Option<String>,
    sql_state: Option<String>,
}

impl StartError {
    /// The Python exception: the engine's error as `krishiv.errors` shapes it, a `RuntimeError` otherwise.
    fn into_py(self, py: Python<'_>) -> PyErr {
        let Some(class) = self.error_class else {
            return PyRuntimeError::new_err(self.message);
        };
        let built = py
            .import("krishiv.errors")
            .and_then(|m| m.getattr("_from_native"))
            .and_then(|f| {
                f.call1((
                    class,
                    self.sql_state,
                    self.message.as_str(),
                    PyDict::new(py),
                ))
            });
        match built {
            Ok(exception) => PyErr::from_value(exception),
            Err(cause) => {
                // Keep why the conversion failed (a bug in `krishiv.errors`) as the cause.
                let err = PyRuntimeError::new_err(self.message);
                err.set_cause(py, Some(cause));
                err
            }
        }
    }
}

#[cfg(feature = "ui")]
fn start(session: &Session, port: u16) -> Result<Started, StartError> {
    let ui =
        krishiv_server::session_source::EmbeddedUi::for_session(session, port).map_err(|e| {
            StartError {
                message: e.message,
                error_class: e.error_class,
                sql_state: e.sql_state,
            }
        })?;
    Ok(Started {
        port: ui.addr().port(),
        token: ui.token().to_owned(),
        url: ui.url(),
        server: Box::new(ui),
    })
}

#[cfg(not(feature = "ui"))]
fn start(_session: &Session, _port: u16) -> Result<Started, StartError> {
    Err(StartError {
        message: "this build has no web console (it was built without the `ui` feature)".to_owned(),
        error_class: None,
        sql_state: None,
    })
}

/// The running console of one session.
#[pyclass(name = "Console", module = "krishiv._native", frozen)]
pub struct PyConsole {
    server: Mutex<Option<Box<dyn Serving>>>,
    port: u16,
    token: SecretString,
    url: SecretString,
    session_id: String,
}

impl std::fmt::Debug for PyConsole {
    /// Never the token or the URL, which holds it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyConsole")
            .field("port", &self.port)
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}

#[pymethods]
impl PyConsole {
    /// Starts the console of `session` on `127.0.0.1`; `port` 0 picks a free port.
    #[new]
    #[pyo3(signature = (session, port=0))]
    fn new(py: Python<'_>, session: &PySession, port: u16) -> PyResult<Self> {
        let of = session.session().clone();
        let started = py.detach(|| start(&of, port)).map_err(|e| e.into_py(py))?;
        Ok(Self {
            server: Mutex::new(Some(started.server)),
            port: started.port,
            token: SecretString::new(started.token),
            url: SecretString::new(started.url),
            session_id: of.id().to_owned(),
        })
    }

    /// The loopback port the console listens on.
    #[getter]
    fn port(&self) -> u16 {
        self.port
    }

    /// The token the console's requests must carry.
    #[getter]
    fn token(&self) -> &str {
        self.token.expose_secret()
    }

    /// `http://127.0.0.1:<port>/?token=<token>`: the address to open.
    #[getter]
    fn url(&self) -> &str {
        self.url.expose_secret()
    }

    /// The id of the session the console shows.
    #[getter]
    fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Whether the console is serving: not stopped, and its session not closed.
    fn is_running(&self) -> bool {
        self.server
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .is_some_and(|s| !s.is_stopped())
    }

    /// Stops the console (idempotent): the listener closes and the server thread is joined.
    fn stop(&self, py: Python<'_>) {
        let server = self
            .server
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        py.detach(|| drop(server));
    }
}
