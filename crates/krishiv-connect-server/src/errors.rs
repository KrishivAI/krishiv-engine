// SPDX-License-Identifier: Apache-2.0
//! Errors a session sent, kept for `FetchErrorDetails` (design P3-T011a
//! §3.10; Spark's `SessionHolder.errorIdToError`, `ErrorUtils` and
//! `SparkConnectFetchErrorDetailsHandler`, v4.3.0).
//!
//! - Every error status sent on a session ([`SessionErrors::status`]) carries
//!   an `errorId` (a random UUID) in its `ErrorInfo` metadata, and the error
//!   is kept under it (`ErrorUtils.buildStatusFromThrowable`, lines
//!   250-260). `spark.sql.connect.enrichError.enabled=false` (a session
//!   setting) turns both off. An error before a session exists (a malformed
//!   or unknown session id) carries none, as in Spark (`getIsolatedSessionIfPresent`).
//! - A session keeps the [`CACHE_SIZE`] most recent errors, each until
//!   [`CACHE_TTL`] after its last access (`SessionHolder`, lines 117-124
//!   and 654-657), and gives each out **once** (`FetchErrorDetailsHandler`,
//!   lines 43-53: read, then invalidated).
//! - The response (`details`, `ErrorUtils.throwableToProtoErrors`, lines
//!   78-187): the error and its causes, at most [`MAX_CHAIN`], root index
//!   0, each with its message, the Java class hierarchy Spark would raise
//!   (`status::exception_classes`), the class, SQLSTATE, message parameters
//!   and query contexts. The response sets neither session id (Spark
//!   leaves both unset).
//! - **No stack trace** (Spark sends the JVM's): a Rust backtrace would
//!   leak internals (threat model T-C8), and there is no JVM. Each error of
//!   the chain is sent in its client-safe form
//!   ([`KrishivError::for_client_with_id`]): an operator-only class has its
//!   parameters redacted, as its own status would. Query contexts carry the
//!   redacted SQL fragments the session attached (ADR 0038).

use std::collections::VecDeque;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use krishiv_common::error::{ErrorContextType, ErrorQueryContext, KrishivError};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::fetch_error_details_response::{
    self as fed, query_context::ContextType,
};
use krishiv_session::RuntimeConf;
use tonic::Status;

use crate::operation::random_uuid;
use crate::operation::replay::uuid_text;
use crate::status::{exception_classes, status_of, status_with_id};

/// Errors a session keeps (`SessionHolder.ERROR_CACHE_SIZE`).
pub const CACHE_SIZE: usize = 20;
/// How long an error is kept after its last access
/// (`SessionHolder.ERROR_CACHE_TIMEOUT_SEC`).
pub const CACHE_TTL: Duration = Duration::from_secs(60);
/// The longest cause chain sent (`ErrorUtils.MAX_ERROR_CHAIN_LENGTH`).
pub const MAX_CHAIN: usize = 5;

/// The session setting that turns `errorId` off
/// (`Connect.CONNECT_ENRICH_ERROR_ENABLED`, default true).
pub const ENRICH_ERROR_ENABLED: &str = "spark.sql.connect.enrichError.enabled";

/// A session's sent errors by id: the most recently accessed last.
#[derive(Debug, Default)]
pub(crate) struct ErrorCache {
    entries: VecDeque<(String, KrishivError, Instant)>,
}

impl ErrorCache {
    /// Keeps `error` under `id` at time `now`, dropping what expired and,
    /// past [`CACHE_SIZE`], the least recently accessed.
    pub(crate) fn put_at(&mut self, id: String, error: KrishivError, now: Instant) {
        self.entries
            .retain(|(_, _, at)| now.saturating_duration_since(*at) <= CACHE_TTL);
        self.entries.push_back((id, error, now));
        while self.entries.len() > CACHE_SIZE {
            self.entries.pop_front();
        }
    }

    /// Removes and returns the error kept under `id`, unless it expired.
    pub(crate) fn take_at(&mut self, id: &str, now: Instant) -> Option<KrishivError> {
        let at = self.entries.iter().position(|(k, _, _)| k == id)?;
        let (_, error, accessed) = self.entries.remove(at)?;
        (now.saturating_duration_since(accessed) <= CACHE_TTL).then_some(error)
    }

    /// Errors kept (expired ones included until the next put).
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

/// The errors one session sent, and its settings (`errorId` on or off).
#[derive(Debug)]
pub struct SessionErrors {
    cache: Mutex<ErrorCache>,
    conf: RuntimeConf,
}

impl SessionErrors {
    /// The error cache of a session with the configuration `conf`.
    #[must_use]
    pub fn new(conf: RuntimeConf) -> Self {
        Self {
            cache: Mutex::new(ErrorCache::default()),
            conf,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ErrorCache> {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether errors get an `errorId`
    /// (`spark.sql.connect.enrichError.enabled`, true unless `false`).
    fn enriched(&self) -> bool {
        self.conf
            .get(ENRICH_ERROR_ENABLED)
            .is_none_or(|v| !v.trim().eq_ignore_ascii_case("false"))
    }

    /// `error` as the gRPC status sent on this session: [`status_of`] with
    /// an `errorId` under which the error is kept for `FetchErrorDetails`.
    #[must_use]
    pub fn status(&self, error: &KrishivError) -> Status {
        if !self.enriched() {
            return status_of(error);
        }
        // Without randomness the error goes without an id.
        let Ok(id) = random_uuid() else {
            return status_of(error);
        };
        let id = uuid_text(id);
        let status = status_with_id(error, Some(&id));
        tracing::debug!(
            target: "krishiv::connect",
            error_id = %id,
            error = %error,
            "error sent to a Spark Connect client"
        );
        self.lock().put_at(id, error.clone(), Instant::now());
        status
    }

    /// The error sent under `id`, removed (it is given out once).
    #[must_use]
    pub fn take(&self, id: &str) -> Option<KrishivError> {
        self.lock().take_at(id, Instant::now())
    }

    /// Errors kept now.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no error is kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The `FetchErrorDetails` response for `error` (module docs).
#[must_use]
pub(crate) fn details(error: &KrishivError) -> pb::FetchErrorDetailsResponse {
    let errors = std::iter::once(error)
        .chain(error.causes())
        .take(MAX_CHAIN)
        .enumerate()
        .map(|(i, e)| one(e, i))
        .collect();
    pb::FetchErrorDetailsResponse {
        root_error_idx: Some(0),
        errors,
        ..Default::default()
    }
}

/// Error `i` of a chain, in its client-safe form.
fn one(error: &KrishivError, i: usize) -> fed::Error {
    let has_cause = error.cause().is_some();
    let client = error.for_client_with_id(String::new());
    let e = client.error();
    let class = e.error_class();
    let cause_idx = (has_cause && i + 1 < MAX_CHAIN)
        .then(|| i32::try_from(i + 1).ok())
        .flatten();
    fed::Error {
        error_type_hierarchy: exception_classes(class)
            .iter()
            .map(|c| (*c).to_owned())
            .collect(),
        message: e.to_string(),
        stack_trace: Vec::new(),
        cause_idx,
        spark_throwable: Some(fed::SparkThrowable {
            error_class: Some(class.name().to_owned()),
            message_parameters: e
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
            query_contexts: e.query_contexts().iter().map(context).collect(),
            sql_state: e.sql_state().map(str::to_owned),
            breaking_change_info: None,
        }),
    }
}

/// A query context as `ErrorUtils` sends it (lines 119-141).
fn context(c: &ErrorQueryContext) -> fed::QueryContext {
    let index = |n: usize| i32::try_from(n).unwrap_or(i32::MAX);
    match c.context_type {
        ErrorContextType::Sql => fed::QueryContext {
            context_type: ContextType::Sql as i32,
            object_type: c.object_type.clone(),
            object_name: c.object_name.clone(),
            start_index: index(c.start),
            stop_index: index(c.stop),
            fragment: c.fragment.clone(),
            call_site: String::new(),
            summary: c.summary.clone(),
        },
        ErrorContextType::DataFrame => fed::QueryContext {
            context_type: ContextType::Dataframe as i32,
            fragment: c.fragment.clone(),
            call_site: c.call_site.clone(),
            summary: c.summary.clone(),
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests;
