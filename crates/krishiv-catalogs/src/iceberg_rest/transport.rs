// SPDX-License-Identifier: Apache-2.0
//! The HTTP side of the client (design P3-lakehouse §5.6): one thread per client runs a current-thread Tokio
//! runtime with an async `reqwest` client, and a call blocks the caller until its answer or its deadline.
//!
//! Why a thread of its own: the `Catalog` trait is synchronous, `reqwest::blocking` panics when it is built or used
//! inside a runtime, and a `Runtime` panics when it is dropped inside one. Here nothing runtime-shaped is ever
//! touched by the caller: it sends a job over a channel and waits on a plain `std` channel with a timeout, so the
//! deadline is exact (a server that trickles bytes is cut at the deadline, not at a per-read timeout) and dropping
//! the client from any thread only closes the channel.
//!
//! Every request goes through the egress policy client (design P3-T067 §5.2, krishiv-store's
//! `PolicyConnector::catalog_client`; no `reqwest::Client` is built here): the egress resolver, no redirect
//! followed (a 3xx is an error naming the catalog), no proxy from the environment (only the admin's egress proxy),
//! `https` only unless the declaration allows `http`, and the request's host checked before it is sent (an address
//! literal never reaches the resolver). On top: `Accept-Encoding: identity` and a refusal of any other
//! `Content-Encoding` (so no body is ever decompressed: the byte limit counts what the parser sees), and a body
//! capped at the response limit while it is read.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use krishiv_common::secret::{SecretString, wipe};
use krishiv_store::PolicyConnector;
use reqwest::header::{
    ACCEPT, ACCEPT_ENCODING, AUTHORIZATION, CONTENT_ENCODING, CONTENT_TYPE, HeaderValue,
};
use url::Url;

use super::error::{RestError, RestLimit};

/// One request.
#[derive(Debug)]
pub(crate) struct Request {
    pub(crate) post: bool,
    pub(crate) url: Url,
    pub(crate) bearer: Option<SecretString>,
    /// An `application/x-www-form-urlencoded` body (a token request: it carries the client secret).
    pub(crate) form: Option<SecretString>,
    /// An `application/json` body (a commit request); `post` must be set.
    pub(crate) json: Option<Vec<u8>>,
    pub(crate) headers: Vec<(&'static str, String)>,
}

impl Request {
    pub(crate) fn get(url: Url) -> Self {
        Self {
            post: false,
            url,
            bearer: None,
            form: None,
            json: None,
            headers: Vec::new(),
        }
    }
}

/// A response that arrived whole, within the limits.
#[derive(Debug)]
pub(crate) struct Response {
    pub(crate) status: u16,
    /// `Retry-After` in delta-seconds (an HTTP date is ignored).
    pub(crate) retry_after: Option<Duration>,
    pub(crate) body: Vec<u8>,
}

/// A request that got no whole answer, and whether any of it may have reached the server.
#[derive(Debug)]
pub(crate) struct SendFailure {
    pub(crate) error: RestError,
    /// `false` only when the request provably never left: refused by the egress policy, a
    /// request that could not be built, a connection that was never made, a deadline passed
    /// before sending. Everything else (a timeout, a reset, an answer refused while read) may
    /// have been delivered.
    pub(crate) sent: bool,
}

impl SendFailure {
    fn unsent(error: RestError) -> Self {
        Self { error, sent: false }
    }

    fn sent(error: RestError) -> Self {
        Self { error, sent: true }
    }
}

struct Job {
    request: Request,
    deadline: Instant,
    reply: mpsc::SyncSender<Result<Response, SendFailure>>,
}

/// The HTTP client of one catalog.
#[derive(Debug)]
pub(crate) struct Transport {
    jobs: tokio::sync::mpsc::UnboundedSender<Job>,
}

/// Extra time the caller waits past the deadline for the job's own timeout to answer, before it gives up itself.
const REPLY_GRACE: Duration = Duration::from_millis(500);

impl Transport {
    /// Starts the client thread; requests go out on `connector`'s catalog client.
    pub(crate) fn start(
        connector: &PolicyConnector,
        allow_http: bool,
        timeout: Duration,
        max_bytes: usize,
    ) -> Result<Self, RestError> {
        // Built here, outside any runtime: the policy client needs none to be built, and an error is
        // the caller's at once.
        let client = connector
            .catalog_client(
                timeout,
                allow_http,
                concat!("krishiv/", env!("CARGO_PKG_VERSION")),
            )
            .map_err(|e| RestError::Client {
                no_tls_roots: krishiv_store::is_missing_tls_roots(&e),
                detail: e.to_string(),
            })?;
        let connector = connector.clone();
        let (jobs, mut rx) = tokio::sync::mpsc::unbounded_channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), RestError>>(1);
        thread::Builder::new()
            .name("krishiv-iceberg-rest".to_owned())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = ready_tx.send(Err(RestError::Transport(format!("runtime: {e}"))));
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
                            .unwrap_or(Err(SendFailure::sent(RestError::Timeout)));
                            // The caller may have stopped waiting: nothing to do then.
                            let _ = job.reply.send(result);
                        });
                    }
                });
            })
            .map_err(|e| RestError::Transport(format!("client thread: {e}")))?;
        ready_rx.recv().map_err(|_| {
            RestError::Transport("the client thread stopped while starting".to_owned())
        })??;
        Ok(Self { jobs })
    }

    /// Sends `request` and waits for the whole response until `deadline`.
    pub(crate) fn call(&self, request: Request, deadline: Instant) -> Result<Response, RestError> {
        self.call_tracked(request, deadline).map_err(|f| f.error)
    }

    /// [`Self::call`], saying on a failure whether the request may have reached the server (a
    /// commit is never sent twice after one that may have).
    pub(crate) fn call_tracked(
        &self,
        request: Request,
        deadline: Instant,
    ) -> Result<Response, SendFailure> {
        let now = Instant::now();
        if deadline <= now {
            return Err(SendFailure::unsent(RestError::Timeout));
        }
        let (reply, answer) = mpsc::sync_channel(1);
        self.jobs
            .send(Job {
                request,
                deadline,
                reply,
            })
            .map_err(|_| {
                SendFailure::unsent(RestError::Transport(
                    "the client thread has stopped".to_owned(),
                ))
            })?;
        match answer.recv_timeout(
            deadline
                .saturating_duration_since(now)
                .saturating_add(REPLY_GRACE),
        ) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(SendFailure::sent(RestError::Timeout)),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(SendFailure::sent(
                RestError::Transport("the client thread has stopped".to_owned()),
            )),
        }
    }
}

fn transport_error(e: reqwest::Error) -> RestError {
    if e.is_timeout() {
        return RestError::Timeout;
    }
    // The egress resolver's refusal (a name that resolves to no allowed address) is the policy's, not
    // a connection failure.
    let mut cause = std::error::Error::source(&e);
    while let Some(c) = cause {
        if let Some(refusal) = c.downcast_ref::<krishiv_store::egress::EgressRefusal>() {
            return RestError::Egress(*refusal);
        }
        cause = c.source();
    }
    // The detail goes to the log only; the URL is left out all the same (its query names the warehouse).
    let e = e.without_url();
    RestError::Transport(
        std::error::Error::source(&e).map_or_else(|| e.to_string(), |s| format!("{e}: {s}")),
    )
}

/// `Retry-After` as delta-seconds; an HTTP date (or anything else) is ignored.
fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

async fn send(
    client: &reqwest::Client,
    connector: &PolicyConnector,
    request: Request,
    max_bytes: usize,
) -> Result<Response, SendFailure> {
    let Request {
        post,
        url,
        bearer,
        form,
        json,
        headers,
    } = request;
    // An address literal never reaches the egress resolver: checked here, before anything is sent. The
    // message never names the address.
    connector
        .check_request_host(url.host_str().unwrap_or_default())
        .map_err(|refusal| SendFailure::unsent(RestError::Egress(refusal)))?;
    let mut builder = if post {
        client.post(url)
    } else {
        client.get(url)
    }
    .header(ACCEPT, "application/json")
    .header(ACCEPT_ENCODING, "identity");
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    if let Some(token) = bearer {
        let mut text = format!("Bearer {}", token.expose_secret());
        let value = HeaderValue::from_str(&text);
        wipe(&mut text);
        let mut value = value.map_err(|_| {
            SendFailure::unsent(RestError::Protocol(
                "the token is not a valid header value".to_owned(),
            ))
        })?;
        value.set_sensitive(true);
        builder = builder.header(AUTHORIZATION, value);
    }
    if let Some(form) = form {
        builder = builder
            .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(form.expose_secret().to_owned());
    }
    if let Some(json) = json {
        builder = builder.header(CONTENT_TYPE, "application/json").body(json);
    }
    let mut response = builder.send().await.map_err(|e| {
        // Only a request that was never built or never connected provably did not leave.
        if e.is_builder() || e.is_connect() {
            SendFailure::unsent(transport_error(e))
        } else {
            SendFailure::sent(transport_error(e))
        }
    })?;
    let status = response.status().as_u16();
    if (300..400).contains(&status) {
        return Err(SendFailure::sent(RestError::Redirect(status)));
    }
    let retry_after = retry_after(&response);
    if let Some(encoding) = response.headers().get(CONTENT_ENCODING) {
        let identity = encoding
            .to_str()
            .is_ok_and(|e| e.trim().eq_ignore_ascii_case("identity"));
        if !identity {
            return Err(SendFailure::sent(RestError::Protocol(
                "a compressed response (Content-Encoding) although identity was asked for"
                    .to_owned(),
            )));
        }
    }
    if response
        .content_length()
        .is_some_and(|n| usize::try_from(n).map_or(true, |n| n > max_bytes))
    {
        return Err(SendFailure::sent(RestError::Limit(RestLimit::Bytes)));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| SendFailure::sent(transport_error(e)))?
    {
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err(SendFailure::sent(RestError::Limit(RestLimit::Bytes)));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Response {
        status,
        retry_after,
        body,
    })
}
