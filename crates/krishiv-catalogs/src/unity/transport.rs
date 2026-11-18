// SPDX-License-Identifier: Apache-2.0
//! The HTTP side of the client (design P3-lakehouse §5.6): one thread per client runs a current-thread Tokio runtime
//! with an async `reqwest` client, and a call blocks the caller until its answer or its deadline. The caller waits on
//! a plain `std` channel, so the deadline is exact (a server that trickles bytes is cut at the deadline) and dropping
//! the client from any thread, inside a runtime or not, only closes the channel.
//!
//! Every request is a GET: redirects are not followed (a 3xx is an error), no proxy from the environment, `https` only
//! unless the declaration allows `http`, `Accept-Encoding: identity` and a refusal of any other `Content-Encoding`
//! (no body is ever decompressed, so the byte limit counts what the parser sees), and a body capped at the response
//! limit while it is read.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use krishiv_common::secret::{SecretString, wipe};
use krishiv_store::PolicyConnector;
use reqwest::header::{ACCEPT, ACCEPT_ENCODING, AUTHORIZATION, CONTENT_ENCODING, HeaderValue};
use url::Url;

use super::error::{UnityError, UnityLimit};

/// One GET.
#[derive(Debug)]
pub(crate) struct Request {
    pub(crate) url: Url,
    pub(crate) bearer: Option<SecretString>,
}

/// A response that arrived whole, within the limits.
#[derive(Debug)]
pub(crate) struct Response {
    pub(crate) status: u16,
    pub(crate) body: Vec<u8>,
}

struct Job {
    request: Request,
    deadline: Instant,
    reply: mpsc::SyncSender<Result<Response, UnityError>>,
}

/// The HTTP client of one catalog.
#[derive(Debug)]
pub(crate) struct Transport {
    jobs: tokio::sync::mpsc::UnboundedSender<Job>,
}

/// Extra time the caller waits past the deadline for the job's own timeout to answer, before it gives up itself.
const REPLY_GRACE: Duration = Duration::from_millis(500);

impl Transport {
    /// Starts the client thread.
    pub(crate) fn start(
        connector: &PolicyConnector,
        allow_http: bool,
        timeout: Duration,
        max_bytes: usize,
    ) -> Result<Self, UnityError> {
        // Built here, outside any runtime: the policy client needs none to be built, and an error is the caller's.
        let client = connector
            .catalog_client(
                timeout,
                allow_http,
                concat!("krishiv/", env!("CARGO_PKG_VERSION")),
            )
            .map_err(|e| UnityError::Transport(format!("HTTP client: {e}")))?;
        let connector = connector.clone();
        let (jobs, mut rx) = tokio::sync::mpsc::unbounded_channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), UnityError>>(1);
        thread::Builder::new()
            .name("krishiv-unity".to_owned())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = ready_tx.send(Err(UnityError::Transport(format!("runtime: {e}"))));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(()));
                runtime.block_on(async move {
                    while let Some(job) = rx.recv().await {
                        let client = client.clone();
                        let connector = connector.clone();
                        tokio::spawn(async move {
                            let deadline = tokio::time::Instant::from_std(job.deadline);
                            let result = tokio::time::timeout_at(
                                deadline,
                                send(&client, &connector, job.request, max_bytes),
                            )
                            .await
                            .unwrap_or(Err(UnityError::Timeout));
                            // The caller may have stopped waiting: nothing to do then.
                            let _ = job.reply.send(result);
                        });
                    }
                });
            })
            .map_err(|e| UnityError::Transport(format!("client thread: {e}")))?;
        ready_rx.recv().map_err(|_| {
            UnityError::Transport("the client thread stopped while starting".to_owned())
        })??;
        Ok(Self { jobs })
    }

    /// Sends `request` and waits for the whole response until `deadline`.
    pub(crate) fn call(&self, request: Request, deadline: Instant) -> Result<Response, UnityError> {
        let now = Instant::now();
        if deadline <= now {
            return Err(UnityError::Timeout);
        }
        let (reply, answer) = mpsc::sync_channel(1);
        self.jobs
            .send(Job {
                request,
                deadline,
                reply,
            })
            .map_err(|_| UnityError::Transport("the client thread has stopped".to_owned()))?;
        match answer.recv_timeout(
            deadline
                .saturating_duration_since(now)
                .saturating_add(REPLY_GRACE),
        ) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(UnityError::Timeout),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(UnityError::Transport(
                "the client thread has stopped".to_owned(),
            )),
        }
    }
}

fn transport_error(e: reqwest::Error) -> UnityError {
    if e.is_timeout() {
        return UnityError::Timeout;
    }
    // The detail goes to the log only; the URL is left out all the same.
    let e = e.without_url();
    UnityError::Transport(
        std::error::Error::source(&e).map_or_else(|| e.to_string(), |s| format!("{e}: {s}")),
    )
}

async fn send(
    client: &reqwest::Client,
    connector: &PolicyConnector,
    request: Request,
    max_bytes: usize,
) -> Result<Response, UnityError> {
    let Request { url, bearer } = request;
    // An address literal never reaches the egress resolver: checked here, before anything is sent.
    connector
        .check_request_host(url.host_str().unwrap_or_default())
        .map_err(|refusal| UnityError::Transport(format!("egress policy: {refusal}")))?;
    let mut builder = client
        .get(url)
        .header(ACCEPT, "application/json")
        .header(ACCEPT_ENCODING, "identity");
    if let Some(token) = bearer {
        let mut text = format!("Bearer {}", token.expose_secret());
        let value = HeaderValue::from_str(&text);
        wipe(&mut text);
        let mut value = value.map_err(|_| {
            UnityError::Protocol("the token is not a valid header value".to_owned())
        })?;
        value.set_sensitive(true);
        builder = builder.header(AUTHORIZATION, value);
    }
    let mut response = builder.send().await.map_err(transport_error)?;
    let status = response.status().as_u16();
    if (300..400).contains(&status) {
        return Err(UnityError::Redirect(status));
    }
    if let Some(encoding) = response.headers().get(CONTENT_ENCODING) {
        let identity = encoding
            .to_str()
            .is_ok_and(|e| e.trim().eq_ignore_ascii_case("identity"));
        if !identity {
            return Err(UnityError::Protocol(
                "a compressed response (Content-Encoding) although identity was asked for"
                    .to_owned(),
            ));
        }
    }
    if response
        .content_length()
        .is_some_and(|n| usize::try_from(n).map_or(true, |n| n > max_bytes))
    {
        return Err(UnityError::Limit(UnityLimit::Bytes));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err(UnityError::Limit(UnityLimit::Bytes));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Response { status, body })
}
