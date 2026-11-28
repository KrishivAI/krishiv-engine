// SPDX-License-Identifier: Apache-2.0
//! The in-process Connect server of `krishiv.start_local()` (P3-T016): one Connect listener on a
//! free loopback port, owned by one thread of its own, that a stock `pyspark-client` connects to.
//!
//! * The listener binds `127.0.0.1:0` and, on the same port, `[::1]` (`localhost` may resolve to
//!   `::1` first, and a client sends the token to the first address that answers: no other process
//!   may be able to hold that address). `[::1]` is skipped only when the host has no IPv6 loopback.
//!   Nothing else is bound.
//! * Every RPC authenticates with a token generated for this start (32 random bytes, hex) through
//!   the server's one authenticator (`spark.connect.authenticate.token`, compared in constant time
//!   as a SHA-256 digest); a connection without it is `UNAUTHENTICATED`. The token is the caller's
//!   `Session` identity-free capability: it gives the session of the process that started the
//!   server, no more (the sessions are the base session's, as in the embedded API).
//! * [`LocalServer::stop`] (and `Drop`) drains the server, joins its thread and drops the engine
//!   runtime: no port stays bound and no thread stays alive.

use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener as StdListener};
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
    addr6: Option<SocketAddr>,
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
        Self::start_with(base, true, None)
    }

    /// [`Self::start`] with `host` running the Python UDFs of the server's sessions (the embedding
    /// process's worker pool). Without a host a Python UDF is refused with `KRISHIV_UNSUPPORTED_FEATURE`.
    ///
    /// # Errors
    /// As [`Self::start`].
    pub fn start_with_udf_host(
        base: Session,
        host: Arc<dyn crate::UdfHost>,
    ) -> Result<Self, KrishivError> {
        Self::start_with(base, true, Some(host))
    }

    fn start_with(
        base: Session,
        want_v6: bool,
        host: Option<Arc<dyn crate::UdfHost>>,
    ) -> Result<Self, KrishivError> {
        let token = new_token()?;
        // No failure throttle: every local client shares the peer address 127.0.0.1, so another
        // process sending wrong tokens would lock the owner out. The token is 256 random bits, so
        // guessing it is not a risk the throttle has to cover.
        let config = AuthConfig::from_settings([
            ("spark.connect.authenticate.token", token.expose_secret()),
            ("krishiv.auth.failureThrottle.maxFailures", "0"),
        ])?;
        let auth = Arc::new(ServerAuth::new(&config)?);
        let (std_listener, v6_listener) = bind_loopback(want_v6)?;
        let addr = std_listener
            .local_addr()
            .map_err(|e| internal(format!("cannot read the listener address: {e}")))?;
        let addr6 = v6_listener.as_ref().and_then(|l| l.local_addr().ok());
        let (stop, stopped) = oneshot::channel::<()>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), KrishivError>>();
        let thread = std::thread::Builder::new()
            .name("krishiv-local-connect".to_owned())
            .spawn(move || {
                run(
                    base,
                    host,
                    &auth,
                    std_listener,
                    v6_listener,
                    stopped,
                    &ready_tx,
                )
            })
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
            addr6,
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

    /// Every address the server listens on: `127.0.0.1:<port>` and, when the host has one,
    /// `[::1]:<port>`.
    #[must_use]
    pub fn addrs(&self) -> Vec<SocketAddr> {
        std::iter::once(self.addr).chain(self.addr6).collect()
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
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
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

/// Binds `127.0.0.1:0`, then `[::1]` on the same port (another port is tried when `::1` has that
/// one taken). A host with no IPv6 loopback gets the IPv4 listener only. Both are non-blocking.
fn bind_loopback(want_v6: bool) -> Result<(StdListener, Option<StdListener>), KrishivError> {
    let fail = |what: &str, e: std::io::Error| internal(format!("{what}: {e}"));
    for _ in 0..32 {
        let v4 = StdListener::bind((Ipv4Addr::LOCALHOST, 0))
            .map_err(|e| fail("cannot bind a loopback port", e))?;
        let port = v4
            .local_addr()
            .map_err(|e| fail("cannot read the listener address", e))?
            .port();
        let v6 = match if want_v6 {
            StdListener::bind((Ipv6Addr::LOCALHOST, port))
        } else {
            Err(std::io::ErrorKind::Unsupported.into())
        } {
            Ok(l) => Some(l),
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => continue,
            // No IPv6 loopback on this host: `localhost` cannot reach `::1` either.
            Err(e) if ipv6_unavailable(&e) => None,
            // Anything else (permissions, exhausted descriptors) must not leave `[::1]` open to
            // another process: fail closed.
            Err(e) => return Err(fail("cannot bind the IPv6 loopback", e)),
        };
        v4.set_nonblocking(true)
            .map_err(|e| fail("cannot configure the listener", e))?;
        if let Some(l) = &v6 {
            l.set_nonblocking(true)
                .map_err(|e| fail("cannot configure the listener", e))?;
        }
        return Ok((v4, v6));
    }
    Err(internal(
        "no loopback port is free on both 127.0.0.1 and ::1".to_owned(),
    ))
}

/// Whether `e` says this host has no IPv6 loopback (as opposed to a real bind failure).
fn ipv6_unavailable(e: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    matches!(
        e.kind(),
        ErrorKind::AddrNotAvailable | ErrorKind::Unsupported
    ) || e.raw_os_error() == Some(EAFNOSUPPORT)
}

/// `EAFNOSUPPORT`: 47 on macOS and the BSDs, 97 on Linux.
#[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
const EAFNOSUPPORT: i32 = 47;
/// `EAFNOSUPPORT`: 47 on macOS and the BSDs, 97 on Linux.
#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "freebsd")))]
const EAFNOSUPPORT: i32 = 97;

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
    host: Option<Arc<dyn crate::UdfHost>>,
    auth: &Arc<ServerAuth>,
    listener: StdListener,
    listener6: Option<StdListener>,
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
        let mut manager = SessionManager::from_base(base, SessionLimits::default());
        if let Some(host) = host {
            manager = manager.with_udf_host(host);
        }
        let manager = Arc::new(manager);
        let config = ConnectServerConfig {
            // Nothing in flight outlives a stop for long: this is the user's own process.
            shutdown_timeout: std::time::Duration::from_secs(5),
            ..ConnectServerConfig::default()
        };
        let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
        // Both sockets are registered with the runtime before the start is reported: a failure
        // here is the start's error, not a handle that advertises an address nobody serves.
        let adopt = |l: StdListener| {
            tokio::net::TcpListener::from_std(l)
                .map_err(|e| internal(format!("cannot adopt the listener: {e}")))
        };
        let adopted =
            adopt(listener).and_then(|a| listener6.map(adopt).transpose().map(|b| (a, b)));
        let (listener, listener6) = match adopted {
            Ok(pair) => pair,
            Err(e) => {
                let _ = ready.send(Err(internal(e.to_string())));
                return Err(e);
            }
        };
        // Each listener has its own `ConnectServer`, so the connection cap and the response
        // budget apply per listener (twice over for one local server): the owner's own process,
        // every client authenticated by the same token.
        let serve_one = |listener: tokio::net::TcpListener| {
            let manager = Arc::clone(&manager);
            let auth = Arc::clone(auth);
            let config = config.clone();
            let mut stop = stop_rx.clone();
            async move {
                ConnectServer::new(manager, config)
                    .with_auth(ConnectAuth::new(auth))
                    .serve(listener, async move {
                        let _ = stop.wait_for(|stopped| *stopped).await;
                    })
                    .await
            }
        };
        let main = serve_one(listener);
        let second = async {
            match listener6 {
                Some(l) => serve_one(l).await,
                None => Ok(()),
            }
        };
        let signal = async {
            let _ = stopped.await;
            let _ = stop_tx.send(true);
        };
        let _ = ready.send(Ok(()));
        let ((first, second), ()) = tokio::join!(async { tokio::join!(main, second) }, signal);
        first.and(second)
    });
    // Connection tasks that still hold on end with the runtime.
    drop(runtime);
    result
}

#[cfg(test)]
mod tests {
    use std::io::{Error, ErrorKind};

    use super::*;

    #[test]
    fn a_host_without_ipv6_loopback_is_told_apart_from_a_real_bind_failure() {
        assert!(ipv6_unavailable(&Error::from(ErrorKind::AddrNotAvailable)));
        assert!(ipv6_unavailable(&Error::from(ErrorKind::Unsupported)));
        assert!(ipv6_unavailable(&Error::from_raw_os_error(EAFNOSUPPORT)));
        assert!(!ipv6_unavailable(&Error::from(ErrorKind::PermissionDenied)));
        assert!(!ipv6_unavailable(&Error::from(ErrorKind::AddrInUse)));
    }

    #[test]
    fn internal_errors_carry_their_message() {
        assert!(internal("boom".to_owned()).to_string().contains("boom"));
    }

    #[test]
    fn tokens_are_64_hex_digits_and_differ() {
        let (a, b) = (new_token().expect("a"), new_token().expect("b"));
        assert_eq!(a.expose_secret().len(), 64);
        assert!(a.expose_secret().bytes().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a.expose_secret(), b.expose_secret());
    }

    #[test]
    fn a_server_without_the_ipv6_listener_serves_on_ipv4_only_and_stops() {
        let mut server =
            LocalServer::start_with(Session::new().expect("session"), false, None).expect("start");
        assert_eq!(server.addrs(), vec![server.addr()]);
        assert!(server.is_running());
        server.stop().expect("stop");
        assert!(!server.is_running());
        server.stop().expect("a second stop is a no-op");
    }

    #[test]
    fn the_listeners_share_one_port_and_are_non_blocking() {
        let (v4, v6) = bind_loopback(true).expect("bind");
        if let Some(v6) = v6 {
            assert_eq!(
                v4.local_addr().expect("v4").port(),
                v6.local_addr().expect("v6").port()
            );
        }
    }
}
