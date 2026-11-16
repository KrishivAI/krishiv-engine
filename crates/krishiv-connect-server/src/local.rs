// SPDX-License-Identifier: Apache-2.0
//! The in-process Connect server of `krishiv.start_local()` (P3-T016): one Connect listener on a
//! free loopback port, owned by one thread of its own, that a stock `pyspark-client` connects to.
//!
//! * The listener binds `127.0.0.1:0` and nothing else.
//! * Every RPC authenticates with a token generated for this start (32 random bytes, hex) through
//!   the server's one authenticator (`spark.connect.authenticate.token`, compared in constant time
//!   as a SHA-256 digest); a connection without it is `UNAUTHENTICATED`. The token is the caller's
//!   `Session` identity-free capability: it gives the session of the process that started the
//!   server, no more (the sessions are the base session's, as in the embedded API).
//! * [`LocalServer::stop`] (and `Drop`) drains the server, joins its thread and drops the engine
//!   runtime: no port stays bound and no thread stays alive.

use std::net::{Ipv4Addr, SocketAddr, TcpListener as StdListener};
use std::sync::Arc;
use std::thread::JoinHandle;

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_common::secret::SecretString;
use krishiv_session::Session;
use krishiv_session::auth::{AuthConfig, ServerAuth};
use tokio::sync::oneshot;

use crate::auth::ConnectAuth;
use crate::manager::{SessionLimits, SessionManager};
use crate::server::{ConnectServer, ConnectServerConfig};

/// The host name clients connect to. pyspark-client sends a token over a plain channel only to
/// the host name `localhost` (gRPC local credentials); any other host needs TLS.
const CLIENT_HOST: &str = "localhost";

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// A Connect server running in this process on a loopback port.
#[derive(Debug)]
pub struct LocalServer {
    addr: SocketAddr,
    token: SecretString,
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<Result<(), KrishivError>>>,
}

impl LocalServer {
    /// Starts a server whose sessions are made from `base`, on a free port of `127.0.0.1`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when no random source, port or thread is available.
    pub fn start(base: Session) -> Result<Self, KrishivError> {
        let token = new_token()?;
        let config = AuthConfig::from_settings([(
            "spark.connect.authenticate.token",
            token.expose_secret(),
        )])?;
        let auth = Arc::new(ServerAuth::new(&config)?);
        let std_listener = StdListener::bind((Ipv4Addr::LOCALHOST, 0))
            .map_err(|e| internal(format!("cannot bind a loopback port: {e}")))?;
        std_listener
            .set_nonblocking(true)
            .map_err(|e| internal(format!("cannot configure the listener: {e}")))?;
        let addr = std_listener
            .local_addr()
            .map_err(|e| internal(format!("cannot read the listener address: {e}")))?;
        let (stop, stopped) = oneshot::channel::<()>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), KrishivError>>();
        let thread = std::thread::Builder::new()
            .name("krishiv-local-connect".to_owned())
            .spawn(move || run(base, auth, std_listener, stopped, &ready_tx))
            .map_err(|e| internal(format!("cannot start the server thread: {e}")))?;
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = thread.join();
                return Err(e);
            }
            Err(_) => {
                let _ = thread.join();
                return Err(internal(
                    "the server thread ended before it served".to_owned(),
                ));
            }
        }
        Ok(Self {
            addr,
            token,
            stop: Some(stop),
            thread: Some(thread),
        })
    }

    /// The bound address (always loopback).
    #[must_use]
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The token this server accepts.
    #[must_use]
    pub fn token(&self) -> &str {
        self.token.expose_secret()
    }

    /// The connection string a Spark Connect client takes: `sc://localhost:<port>/;token=<token>`.
    #[must_use]
    pub fn connection_string(&self) -> String {
        format!(
            "sc://{CLIENT_HOST}:{}/;token={}",
            self.addr.port(),
            self.token.expose_secret()
        )
    }

    /// Whether the server has not been stopped.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.thread.is_some()
    }

    /// Stops the server: drains it, joins its thread (the listener closes and the engine runtime
    /// is dropped). Stopping a stopped server is a no-op.
    ///
    /// # Errors
    /// The server's transport error, or `INTERNAL_ERROR` when its thread panicked.
    pub fn stop(&mut self) -> Result<(), KrishivError> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        match self.thread.take() {
            Some(thread) => thread
                .join()
                .map_err(|_| internal("the server thread panicked".to_owned()))?,
            None => Ok(()),
        }
    }
}

impl Drop for LocalServer {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// A fresh token: 32 random bytes as 64 hex digits.
fn new_token() -> Result<SecretString, KrishivError> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes).map_err(|e| internal(format!("no random source: {e}")))?;
    let mut hex = String::with_capacity(64);
    for b in bytes {
        hex.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
        hex.push(char::from_digit(u32::from(b & 0xf), 16).unwrap_or('0'));
    }
    Ok(SecretString::new(hex))
}

/// The server thread: a runtime of its own, serving until `stopped`.
#[expect(
    clippy::disallowed_methods,
    reason = "the server's own thread owns this runtime and does nothing else: no worker of the engine blocks"
)]
fn run(
    base: Session,
    auth: Arc<ServerAuth>,
    listener: StdListener,
    stopped: oneshot::Receiver<()>,
    ready: &std::sync::mpsc::Sender<Result<(), KrishivError>>,
) -> Result<(), KrishivError> {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            let err = internal(format!("cannot start the server runtime: {e}"));
            let _ = ready.send(Err(internal(err.to_string())));
            return Err(err);
        }
    };
    let result = runtime.block_on(async {
        let listener = tokio::net::TcpListener::from_std(listener)
            .map_err(|e| internal(format!("cannot adopt the listener: {e}")))?;
        let manager = Arc::new(SessionManager::from_base(base, SessionLimits::default()));
        let config = ConnectServerConfig {
            // Nothing in flight outlives a stop for long: this is the user's own process.
            shutdown_timeout: std::time::Duration::from_secs(5),
            ..ConnectServerConfig::default()
        };
        let server = ConnectServer::new(manager, config).with_auth(ConnectAuth::new(auth));
        let _ = ready.send(Ok(()));
        server
            .serve(listener, async {
                let _ = stopped.await;
            })
            .await
    });
    // Connection tasks that still hold on end with the runtime.
    drop(runtime);
    result
}
