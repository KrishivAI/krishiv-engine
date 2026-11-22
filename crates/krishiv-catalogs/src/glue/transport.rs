// SPDX-License-Identifier: Apache-2.0
//! The HTTP side of the client (design P3-lakehouse §5.6): one thread per client runs a current-thread Tokio runtime
//! with an async `reqwest` client, and a call blocks the caller until its answer or its deadline. The caller waits on
//! a plain `std` channel, so the deadline is exact (a server that trickles bytes is cut at the deadline) and dropping
//! the client from any thread, inside a runtime or not, only closes the channel.
//!
//! Every request is a signed POST: redirects are not followed (a 3xx is an error), no proxy from the environment, `https` only
//! unless the declaration allows `http`, `Accept-Encoding: identity` and a refusal of any other `Content-Encoding`
//! (no body is ever decompressed, so the byte limit counts what the parser sees), and a body capped at the response
//! limit while it is read.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use krishiv_common::secret::SecretString;
use krishiv_store::PolicyConnector;
use reqwest::header::{ACCEPT_ENCODING, CONTENT_ENCODING, HeaderValue};
use url::Url;

use super::error::{GlueError, GlueLimit};

/// One POST of a Glue JSON-1.1 call. `headers` are the signed headers (the `Authorization` and the session token are
/// marked sensitive); the body is already signed.
pub(crate) struct Request {
    pub(crate) url: Url,
    pub(crate) headers: Vec<(&'static str, SecretString)>,
    pub(crate) body: Vec<u8>,
}

impl std::fmt::Debug for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Request")
            .field("url", &self.url)
            .finish_non_exhaustive()
    }
}

/// A response that arrived whole, within the limits.
#[derive(Debug)]
pub(crate) struct Response {
    pub(crate) status: u16,
    pub(crate) body: Vec<u8>,
    /// The `X-Amzn-Errortype` header, which AWS JSON 1.1 allows as the only error discriminator.
    pub(crate) error_type: Option<String>,
}

struct Job {
    request: Request,
    deadline: Instant,
    reply: mpsc::SyncSender<Result<Response, GlueError>>,
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
    ) -> Result<Self, GlueError> {
        // Built here, outside any runtime: the policy client needs none to be built, and an error is the caller's.
        let client = connector
            .catalog_client(
                timeout,
                allow_http,
                concat!("krishiv/", env!("CARGO_PKG_VERSION")),
            )
            .map_err(|e| GlueError::Transport(format!("HTTP client: {e}")))?;
        let connector = connector.clone();
        let (jobs, mut rx) = tokio::sync::mpsc::unbounded_channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), GlueError>>(1);
        thread::Builder::new()
            .name("krishiv-glue".to_owned())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = ready_tx.send(Err(GlueError::Transport(format!("runtime: {e}"))));
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
                            .unwrap_or(Err(GlueError::Timeout));
                            // The caller may have stopped waiting: nothing to do then.
                            let _ = job.reply.send(result);
                        });
                    }
                });
            })
            .map_err(|e| GlueError::Transport(format!("client thread: {e}")))?;
        ready_rx.recv().map_err(|_| {
            GlueError::Transport("the client thread stopped while starting".to_owned())
        })??;
        Ok(Self { jobs })
    }

    /// Sends `request` and waits for the whole response until `deadline`.
    pub(crate) fn call(&self, request: Request, deadline: Instant) -> Result<Response, GlueError> {
        let now = Instant::now();
        if deadline <= now {
            return Err(GlueError::Timeout);
        }
        let (reply, answer) = mpsc::sync_channel(1);
        self.jobs
            .send(Job {
                request,
                deadline,
                reply,
            })
            .map_err(|_| GlueError::Transport("the client thread has stopped".to_owned()))?;
        match answer.recv_timeout(
            deadline
                .saturating_duration_since(now)
                .saturating_add(REPLY_GRACE),
        ) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(GlueError::Timeout),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(GlueError::Transport(
                "the client thread has stopped".to_owned(),
            )),
        }
    }
}

fn transport_error(e: reqwest::Error) -> GlueError {
    if e.is_timeout() {
        return GlueError::Timeout;
    }
    // The detail goes to the log only; the URL is left out all the same.
    let e = e.without_url();
    GlueError::Transport(
        std::error::Error::source(&e).map_or_else(|| e.to_string(), |s| format!("{e}: {s}")),
    )
}

async fn send(
    client: &reqwest::Client,
    connector: &PolicyConnector,
    request: Request,
    max_bytes: usize,
) -> Result<Response, GlueError> {
    let Request { url, headers, body } = request;
    // An address literal never reaches the egress resolver: checked here, before anything is sent.
    connector
        .check_request_host(url.host_str().unwrap_or_default())
        .map_err(|refusal| GlueError::Transport(format!("egress policy: {refusal}")))?;
    let mut builder = client
        .post(url)
        .header(ACCEPT_ENCODING, "identity")
        .body(body);
    for (name, value) in headers {
        let mut value = HeaderValue::from_str(value.expose_secret())
            .map_err(|_| GlueError::Protocol(format!("the {name} header is not valid")))?;
        value.set_sensitive(matches!(name, "authorization" | "x-amz-security-token"));
        builder = builder.header(name, value);
    }
    let mut response = builder.send().await.map_err(transport_error)?;
    let status = response.status().as_u16();
    if (300..400).contains(&status) {
        return Err(GlueError::Redirect(status));
    }
    if let Some(encoding) = response.headers().get(CONTENT_ENCODING) {
        let identity = encoding
            .to_str()
            .is_ok_and(|e| e.trim().eq_ignore_ascii_case("identity"));
        if !identity {
            return Err(GlueError::Protocol(
                "a compressed response (Content-Encoding) although identity was asked for"
                    .to_owned(),
            ));
        }
    }
    if response
        .content_length()
        .is_some_and(|n| usize::try_from(n).map_or(true, |n| n > max_bytes))
    {
        return Err(GlueError::Limit(GlueLimit::Bytes));
    }
    let error_type = response
        .headers()
        .get("x-amzn-errortype")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.chars().take(256).collect::<String>());
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err(GlueError::Limit(GlueLimit::Bytes));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Response {
        status,
        body,
        error_type,
    })
}
