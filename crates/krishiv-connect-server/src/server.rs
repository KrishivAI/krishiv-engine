// SPDX-License-Identifier: Apache-2.0
//! The gRPC listener (design P3-T001 §3.3, §3.4): HTTP/2 limits set before
//! any request is read, a connection cap on the accept loop, the idle
//! sweeper, and graceful shutdown.
//!
//! Shutdown: once the shutdown future resolves the server is draining (new
//! RPCs get `UNAVAILABLE`, no new connection is accepted), running queries
//! get up to [`ConnectServerConfig::shutdown_timeout`] to finish, then every
//! session is closed, which cancels what is left.

use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use krishiv_common::config::{
    CONNECT_HANDSHAKE_TIMEOUT, CONNECT_HTTP2_MAX_CONCURRENT_STREAMS,
    CONNECT_HTTP2_MAX_HEADER_LIST_BYTES, CONNECT_HTTP2_MAX_PENDING_ACCEPT_RESETS,
    CONNECT_KEEPALIVE_ENABLED, CONNECT_KEEPALIVE_TIME, CONNECT_KEEPALIVE_TIMEOUT,
    CONNECT_MAX_CONNECTIONS, CONNECT_MAX_INBOUND_MESSAGE_SIZE,
    CONNECT_SESSION_MAINTENANCE_INTERVAL, Config, ConfigValue, SERVER_SHUTDOWN_TIMEOUT,
};
use krishiv_common::error::KrishivError;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::transport::Server;
use tonic::transport::server::{Connected, TcpConnectInfo};

use crate::manager::SessionManager;
use crate::service::{ConnectService, SparkConnectServiceServer};

/// The listener's settings. The defaults are Spark 4.3.0's where Spark has
/// the setting, the threat model's otherwise (design P3-T001 §3.3).
#[derive(Debug, Clone)]
pub struct ConnectServerConfig {
    /// The largest request message (`spark.connect.grpc.maxInboundMessageSize`,
    /// 128 MiB; tonic's own default is 4 MiB).
    pub max_inbound_message_size: usize,
    /// Server keepalive pings: interval and timeout
    /// (`spark.connect.grpc.keepAlive.{time,timeout}`, 60 s and 20 s);
    /// `None` when `spark.connect.grpc.keepAlive.enabled` is false.
    pub keepalive: Option<(Duration, Duration)>,
    /// Concurrent HTTP/2 streams per connection
    /// (`krishiv.connect.http2.maxConcurrentStreams`, 100; tonic's default
    /// is unlimited).
    pub max_concurrent_streams: u32,
    /// The largest request header list
    /// (`krishiv.connect.http2.maxHeaderListBytes`, 16 KiB).
    pub max_header_list_bytes: u32,
    /// Reset streams not yet accepted before the connection is closed with
    /// `GOAWAY` (`krishiv.connect.http2.maxPendingAcceptResets`, 20: the
    /// HTTP/2 rapid-reset bound).
    pub max_pending_accept_resets: usize,
    /// Streams the server reset because of an error before the connection
    /// is closed (1024, hyper's default, set explicitly).
    pub max_local_error_resets: usize,
    /// Open connections (`krishiv.connect.maxConnections`, 1024); past it
    /// the accept loop waits for one to close.
    pub max_connections: usize,
    /// A new connection that sends no first byte for this long is closed
    /// (`krishiv.connect.handshakeTimeout`, 10 s; zero: no deadline).
    pub handshake_timeout: Duration,
    /// How often idle sessions are swept
    /// (`spark.connect.session.manager.maintenanceInterval`, 30 s).
    pub maintenance_interval: Duration,
    /// How long running queries may finish after shutdown starts
    /// (`krishiv.server.shutdownTimeout`, 30 s).
    pub shutdown_timeout: Duration,
}

impl Default for ConnectServerConfig {
    fn default() -> Self {
        Self {
            max_inbound_message_size: 128 * 1024 * 1024,
            keepalive: Some((Duration::from_secs(60), Duration::from_secs(20))),
            max_concurrent_streams: 100,
            max_header_list_bytes: 16 * 1024,
            max_pending_accept_resets: 20,
            max_local_error_resets: 1024,
            max_connections: 1024,
            handshake_timeout: Duration::from_secs(10),
            maintenance_interval: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(30),
        }
    }
}

impl ConnectServerConfig {
    /// The settings `config` gives (the `krishiv.connect.*` and
    /// `krishiv.server.*` keys, or their Spark names); defaults otherwise.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let d = Self::default();
        let duration = |e| config.get(e).and_then(ConfigValue::as_duration);
        let count = |e| {
            config
                .get(e)
                .and_then(ConfigValue::as_int)
                .and_then(|v| usize::try_from(v).ok())
        };
        let keepalive = config
            .get(&CONNECT_KEEPALIVE_ENABLED)
            .and_then(ConfigValue::as_bool)
            .unwrap_or(true)
            .then(|| {
                (
                    duration(&CONNECT_KEEPALIVE_TIME).unwrap_or(Duration::from_secs(60)),
                    duration(&CONNECT_KEEPALIVE_TIMEOUT).unwrap_or(Duration::from_secs(20)),
                )
            });
        Self {
            max_inbound_message_size: config
                .get(&CONNECT_MAX_INBOUND_MESSAGE_SIZE)
                .and_then(ConfigValue::as_size)
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(d.max_inbound_message_size),
            keepalive,
            max_concurrent_streams: count(&CONNECT_HTTP2_MAX_CONCURRENT_STREAMS)
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(d.max_concurrent_streams),
            max_header_list_bytes: config
                .get(&CONNECT_HTTP2_MAX_HEADER_LIST_BYTES)
                .and_then(ConfigValue::as_size)
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(d.max_header_list_bytes),
            max_pending_accept_resets: count(&CONNECT_HTTP2_MAX_PENDING_ACCEPT_RESETS)
                .unwrap_or(d.max_pending_accept_resets),
            max_local_error_resets: d.max_local_error_resets,
            max_connections: count(&CONNECT_MAX_CONNECTIONS).unwrap_or(d.max_connections),
            handshake_timeout: duration(&CONNECT_HANDSHAKE_TIMEOUT).unwrap_or(d.handshake_timeout),
            maintenance_interval: duration(&CONNECT_SESSION_MAINTENANCE_INTERVAL)
                .unwrap_or(d.maintenance_interval),
            shutdown_timeout: duration(&SERVER_SHUTDOWN_TIMEOUT).unwrap_or(d.shutdown_timeout),
        }
    }
}

/// How long the listener may take to stop after every session was closed.
pub const CLOSE_WAIT: Duration = Duration::from_secs(5);

/// The grace period when `krishiv.server.shutdownTimeout` overflows a clock.
const MAX_GRACE: Duration = Duration::from_secs(24 * 60 * 60);

/// Aborts its task when dropped.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The Spark Connect gRPC server of one process.
#[derive(Debug)]
pub struct ConnectServer {
    manager: Arc<SessionManager>,
    draining: Arc<AtomicBool>,
    config: ConnectServerConfig,
}

impl ConnectServer {
    /// A server over `manager`.
    #[must_use]
    pub fn new(manager: Arc<SessionManager>, config: ConnectServerConfig) -> Self {
        Self {
            manager,
            draining: Arc::new(AtomicBool::new(false)),
            config,
        }
    }

    /// The flag that is set once shutdown starts (`/readyz` answers 503
    /// from then on).
    #[must_use]
    pub fn draining(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.draining)
    }

    /// The session manager.
    #[must_use]
    pub fn manager(&self) -> &Arc<SessionManager> {
        &self.manager
    }

    /// Serves on `listener` until `shutdown` resolves, then shuts down
    /// gracefully (module docs): at most `shutdown_timeout` plus
    /// [`CLOSE_WAIT`] after the signal. Connection tasks may outlive it
    /// until the tokio runtime is dropped. Returns once every session is
    /// closed and
    /// the listener is stopped.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a transport failure.
    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send,
    ) -> Result<(), KrishivError> {
        let Self {
            manager,
            draining,
            config,
        } = self;
        let service = ConnectService::new(Arc::clone(&manager), Arc::clone(&draining));
        let service = SparkConnectServiceServer::from_arc(Arc::new(service))
            .max_decoding_message_size(config.max_inbound_message_size)
            .max_encoding_message_size(config.max_inbound_message_size);
        let (keepalive_interval, keepalive_timeout) = match config.keepalive {
            Some((i, t)) => (Some(i), Some(t)),
            None => (None, None),
        };
        let router = Server::builder()
            .max_concurrent_streams(config.max_concurrent_streams)
            .http2_max_header_list_size(config.max_header_list_bytes)
            .http2_max_pending_accept_reset_streams(Some(config.max_pending_accept_resets))
            .http2_max_local_error_reset_streams(Some(config.max_local_error_resets))
            .http2_keepalive_interval(keepalive_interval)
            .http2_keepalive_timeout(keepalive_timeout)
            .tcp_nodelay(true)
            .add_service(service);

        // Aborted on drop, so a caller that cancels `serve` leaves no task
        // holding the manager (and through it the engine runtime).
        let _sweeper = AbortOnDrop(tokio::spawn(sweep_loop(
            Arc::clone(&manager),
            config.maintenance_interval,
        )));
        let incoming = capped(listener, config.max_connections, config.handshake_timeout);
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let server = router.serve_with_incoming_shutdown(incoming, async {
            let _ = stop_rx.await;
        });
        tokio::pin!(server);
        tokio::pin!(shutdown);

        let early = tokio::select! {
            result = &mut server => Some(result),
            () = &mut shutdown => None,
        };
        let result = match early {
            Some(result) => result,
            None => {
                draining.store(true, Ordering::Release);
                let _ = stop_tx.send(());
                tracing::info!(target: "krishiv::connect", "shutting down: draining");
                let grace = config.shutdown_timeout;
                let now = tokio::time::Instant::now();
                let deadline = now.checked_add(grace).unwrap_or(now + MAX_GRACE);
                // The listener may stop first (no stream open): queries
                // still get their grace period.
                let stopped = tokio::select! {
                    result = &mut server => Some(result),
                    () = queries_done(&manager, deadline) => None,
                };
                if stopped.is_some() {
                    queries_done(&manager, deadline).await;
                }
                let closed = manager.close_all();
                if closed > 0 {
                    tracing::info!(
                        target: "krishiv::connect",
                        sessions = closed,
                        "closed the remaining sessions"
                    );
                }
                match stopped {
                    Some(result) => result,
                    // The cancelled streams end at once. tonic runs each
                    // connection as a task of its own, which outlives this
                    // future: the caller drops the tokio runtime to end a
                    // connection that still holds on.
                    None => tokio::time::timeout(CLOSE_WAIT, &mut server)
                        .await
                        .unwrap_or(Ok(())),
                }
            }
        };
        manager.close_all();
        result.map_err(|e| transport_error(&e))
    }
}

/// The local address a listener bound, for logs.
///
/// # Errors
/// When the socket has no local address.
pub fn local_addr(listener: &TcpListener) -> Result<SocketAddr, KrishivError> {
    listener.local_addr().map_err(|e| transport_error(&e))
}

fn transport_error(e: &dyn std::fmt::Display) -> KrishivError {
    krishiv_common::error::InternalError {
        message: format!("Spark Connect listener: {e}"),
    }
    .into()
}

/// Waits until no session runs a query, or `deadline` passed.
async fn queries_done(manager: &SessionManager, deadline: tokio::time::Instant) {
    while manager.running_queries() > 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn sweep_loop(manager: Arc<SessionManager>, every: Duration) {
    let mut tick = tokio::time::interval(every.max(Duration::from_millis(10)));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        manager.sweep();
    }
}

/// The accepted connections of `listener`, at most `max` open at once. A
/// connection that sends nothing within `handshake` (zero: no deadline) is
/// closed before it is handed to the server, so a silent socket holds a
/// slot for at most that long (threat model T-D6). The deadline covers the
/// first byte only: a client that sends one byte and stalls keeps its slot
/// until the established-connection idle timeout exists (T011a). Accepting runs on a task of its own, aborted when the
/// stream is dropped; each new connection waits for its first byte on a
/// task of its own, so one slow client delays no other.
fn capped(
    listener: TcpListener,
    max: usize,
    handshake: Duration,
) -> impl futures::Stream<Item = io::Result<CappedStream>> + Send + 'static {
    let permits = Arc::new(Semaphore::new(max.max(1)));
    let (tx, rx) = tokio::sync::mpsc::channel::<io::Result<CappedStream>>(64);
    let acceptor = AbortOnDrop(tokio::spawn(accept_loop(listener, permits, tx, handshake)));
    futures::stream::unfold((rx, acceptor), |(mut rx, acceptor)| async move {
        rx.recv().await.map(|item| (item, (rx, acceptor)))
    })
}

async fn accept_loop(
    listener: TcpListener,
    permits: Arc<Semaphore>,
    tx: tokio::sync::mpsc::Sender<io::Result<CappedStream>>,
    handshake: Duration,
) {
    let mut backoff = Duration::from_millis(10);
    loop {
        let Ok(permit) = Arc::clone(&permits).acquire_owned().await else {
            return;
        };
        let stream = loop {
            match listener.accept().await {
                Ok((stream, _)) => break stream,
                // A failed accept (the peer reset before it was taken, too
                // many open files) does not stop the server; back off, as
                // every retry fails while file descriptors are out.
                Err(e) => {
                    tracing::warn!(target: "krishiv::connect", error = %e, "accept failed");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(1));
                }
            }
        };
        backoff = Duration::from_millis(10);
        let _ = stream.set_nodelay(true);
        let sender = tx.clone();
        tokio::spawn(async move {
            let mut first = [0_u8; 1];
            // A zero deadline means none (as `0s` for the session idle
            // timeout).
            let peeked = if handshake.is_zero() {
                Ok(stream.peek(&mut first).await)
            } else {
                tokio::time::timeout(handshake, stream.peek(&mut first)).await
            };
            match peeked {
                Ok(Ok(n)) if n > 0 => {
                    let _ = sender
                        .send(Ok(CappedStream {
                            inner: stream,
                            _permit: permit,
                        }))
                        .await;
                }
                // Silent, closed or failed: drop it and its permit.
                _ => {}
            }
        });
        if tx.is_closed() {
            return;
        }
    }
}

/// A connection that holds one of the listener's permits until it closes.
#[derive(Debug)]
pub struct CappedStream {
    inner: TcpStream,
    _permit: OwnedSemaphorePermit,
}

impl Connected for CappedStream {
    type ConnectInfo = TcpConnectInfo;

    fn connect_info(&self) -> TcpConnectInfo {
        self.inner.connect_info()
    }
}

impl AsyncRead for CappedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for CappedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}
