// SPDX-License-Identifier: Apache-2.0
//! The admin listener over TLS (P3-T002 design §3.5; feature `tls`): [`serve_capped_tls`] is
//! [`crate::admin::serve_capped`] with each connection's TLS handshake run on a task of its own,
//! within [`HANDSHAKE_TIMEOUT`], so a slow or silent client delays no other and holds its slot for
//! at most that long. The client certificate rustls verified rides on the connection
//! ([`crate::guard::PeerAddr`]) for the authenticator.

use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Router;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::task::JoinHandle;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls::ServerConfig;
use tokio_rustls::server::TlsStream;

/// The longest a connection may take from accept to a finished TLS handshake.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Serves `app` on `listener` over TLS with `config` (the ALPN list is set to `http/1.1`: the
/// admin server speaks HTTP/1.1 only; the certificate resolver is shared, so a reload of the
/// server's TLS material reaches new handshakes here too), at most `max_connections` open at
/// once (a connection in its handshake holds a slot), until `shutdown` resolves.
///
/// # Errors
/// An I/O error of the server.
pub async fn serve_capped_tls(
    listener: TcpListener,
    app: Router,
    max_connections: usize,
    config: &ServerConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> io::Result<()> {
    let mut config = config.clone();
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    let local = listener.local_addr()?;
    let permits = Arc::new(Semaphore::new(
        max_connections.clamp(1, Semaphore::MAX_PERMITS),
    ));
    let (tx, rx) = mpsc::channel(64);
    let acceptor = tokio::spawn(accept_loop(
        listener,
        TlsAcceptor::from(Arc::new(config)),
        permits,
        tx,
    ));
    let listener = TlsCappedListener {
        rx,
        local,
        _acceptor: AbortOnDrop(acceptor),
    };
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<crate::guard::PeerAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await
}

async fn accept_loop(
    listener: TcpListener,
    acceptor: TlsAcceptor,
    permits: Arc<Semaphore>,
    tx: mpsc::Sender<(TlsCappedStream, SocketAddr)>,
) {
    let mut backoff = Duration::from_millis(10);
    loop {
        let Ok(permit) = Arc::clone(&permits).acquire_owned().await else {
            return;
        };
        let (stream, addr) = match listener.accept().await {
            Ok(accepted) => accepted,
            // Out of file descriptors every retry fails: log and back off.
            Err(e) => {
                tracing::warn!(target: "krishiv::server", error = %e, "admin accept failed");
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(1));
                continue;
            }
        };
        backoff = Duration::from_millis(10);
        let (acceptor, sender) = (acceptor.clone(), tx.clone());
        tokio::spawn(async move {
            match tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(stream)).await {
                Ok(Ok(tls)) => {
                    let leaf = tls
                        .get_ref()
                        .1
                        .peer_certificates()
                        .and_then(<[_]>::first)
                        .map(|c| Arc::from(c.as_ref()));
                    let stream = TlsCappedStream {
                        inner: Box::new(tls),
                        leaf,
                        _permit: permit,
                    };
                    let _ = sender.send((stream, addr)).await;
                }
                Ok(Err(e)) => {
                    tracing::debug!(
                        target: "krishiv::server",
                        peer = %addr,
                        error = %e,
                        "admin TLS handshake failed"
                    );
                }
                // Too slow: drop it and its permit.
                Err(_) => {}
            }
        });
        if tx.is_closed() {
            return;
        }
    }
}

/// Aborts its task when dropped, so a stopped server leaves no accept loop behind.
#[derive(Debug)]
struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The TLS admin listener: the connections whose handshake finished.
#[derive(Debug)]
pub struct TlsCappedListener {
    rx: mpsc::Receiver<(TlsCappedStream, SocketAddr)>,
    local: SocketAddr,
    _acceptor: AbortOnDrop,
}

impl axum::serve::Listener for TlsCappedListener {
    type Io = TlsCappedStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (TlsCappedStream, SocketAddr) {
        match self.rx.recv().await {
            Some(accepted) => accepted,
            // The accept loop ended (it never does while the listener lives): wait as a
            // closed listener would.
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.local)
    }
}

/// A TLS admin connection holding one of the listener's permits.
#[derive(Debug)]
pub struct TlsCappedStream {
    inner: Box<TlsStream<TcpStream>>,
    leaf: Option<Arc<[u8]>>,
    _permit: OwnedSemaphorePermit,
}

impl TlsCappedStream {
    /// The client's leaf certificate (DER) rustls verified at the handshake, if it sent one.
    #[must_use]
    pub fn peer_leaf(&self) -> Option<Arc<[u8]>> {
        self.leaf.clone()
    }
}

impl AsyncRead for TlsCappedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(self.inner.as_mut()).poll_read(cx, buf)
    }
}

impl AsyncWrite for TlsCappedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(self.inner.as_mut()).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(self.inner.as_mut()).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(self.inner.as_mut()).poll_shutdown(cx)
    }
}
