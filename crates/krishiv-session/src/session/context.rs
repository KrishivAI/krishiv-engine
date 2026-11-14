// SPDX-License-Identifier: Apache-2.0
//! The session's trace context and tags (P2-T126 `system.queries`): the W3C
//! `traceparent` queries run under and the key/value tags they carry. A child of
//! `session` so that it reads the session's private state and keeps `session.rs`
//! from growing (`cargo xtask file-size`).

use krishiv_common::error::KrishivError;

use super::Session;

impl Session {
    /// Makes the queries this session starts next belong to the W3C trace of
    /// `traceparent` (`version-trace_id-parent_id-flags`): `system.queries`
    /// shows the trace id. An inbound header from a protocol front end is
    /// passed here.
    ///
    /// ```
    /// let s = krishiv_session::Session::new()?;
    /// s.set_trace_context("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")?;
    /// assert_eq!(s.trace_id().as_deref(), Some("4bf92f3577b34da6a3ce929d0e0e4736"));
    /// assert!(s.set_trace_context("not a header").is_err());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `INVALID_PARAMETER_VALUE.STRING` for a header that is not a valid
    /// `traceparent`; the previous trace stays.
    pub fn set_trace_context(&self, traceparent: &str) -> Result<(), KrishivError> {
        let Some(trace) = crate::system::parse_remote_parent(traceparent) else {
            return Err(krishiv_common::error::InvalidParameterValueString {
                parameter: "traceparent".to_owned(),
                function_name: "Session::set_trace_context".to_owned(),
                invalid_value: format!(
                    "{traceparent} (expected version-trace_id-parent_id-flags as W3C Trace Context writes it)"
                ),
            }
            .into());
        };
        self.state.queries.set_trace(Some(trace));
        Ok(())
    }

    /// Takes the trace context of an inbound request, as a protocol front end
    /// (HTTP, Flight SQL or Connect metadata) reads it: the `traceparent` and
    /// `baggage` header values, either absent. The headers are untrusted input
    /// and never fail the request: a malformed `traceparent` starts no trace
    /// and ends the one an earlier request of this session set (`false`), and
    /// the baggage keeps at most 32 well-formed entries within 4 KiB. A request
    /// without a `traceparent` clears both.
    ///
    /// ```
    /// let s = krishiv_session::Session::new()?;
    /// let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    /// assert!(s.set_inbound_trace(Some(tp), Some("tenant=acme;p=1, bad, k=v")));
    /// assert_eq!(s.trace_id().as_deref(), Some("4bf92f3577b34da6a3ce929d0e0e4736"));
    /// assert_eq!(s.baggage().len(), 2);
    /// assert!(!s.set_inbound_trace(Some("junk"), None));
    /// assert!(s.trace_id().is_none());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn set_inbound_trace(&self, traceparent: Option<&str>, baggage: Option<&str>) -> bool {
        let trace = traceparent.and_then(crate::system::parse_remote_parent);
        let valid = trace.is_some() || traceparent.is_none();
        self.state.queries.set_baggage(
            baggage
                .filter(|_| trace.is_some())
                .map(crate::system::parse_baggage)
                .unwrap_or_default(),
        );
        self.state.queries.set_trace(trace);
        valid
    }

    /// The `baggage` entries [`Session::set_inbound_trace`] kept.
    #[must_use]
    pub fn baggage(&self) -> Vec<(String, String)> {
        if self.inbound_trace.is_some() {
            // A statement with its own request trace has its own baggage.
            return self.inbound_baggage.to_vec();
        }
        self.state.queries.baggage()
    }

    /// Tags the queries this session starts next (`system.queries.tags`):
    /// a key of 1 to 128 characters, a value of up to 256, at most 32 tags.
    /// Setting a key again replaces its value.
    ///
    /// ```
    /// let s = krishiv_session::Session::new()?;
    /// s.set_tag("team", "etl")?;
    /// assert_eq!(s.tags(), [("team".to_owned(), "etl".to_owned())]);
    /// s.remove_tag("team");
    /// assert!(s.tags().is_empty());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `INVALID_PARAMETER_VALUE.STRING` for an empty or over-long key, an
    /// over-long value or a 33rd tag; the tags stay as they were.
    pub fn set_tag(&self, key: &str, value: &str) -> Result<(), KrishivError> {
        use crate::system::{MAX_TAG_KEY, MAX_TAG_VALUE};
        let refuse = |why: &str| {
            Err(krishiv_common::error::InvalidParameterValueString {
                parameter: "tag".to_owned(),
                function_name: "Session::set_tag".to_owned(),
                invalid_value: format!("{key} ({why})"),
            }
            .into())
        };
        if key.is_empty() || key.chars().count() > MAX_TAG_KEY {
            return refuse("a key has 1 to 128 characters");
        }
        if value.chars().count() > MAX_TAG_VALUE {
            return refuse("a value has at most 256 characters");
        }
        if !self.state.queries.try_set_tag(key, value) {
            return refuse("a session carries at most 32 tags");
        }
        Ok(())
    }

    /// Removes the tag `key`, if the session has it.
    pub fn remove_tag(&self, key: &str) {
        self.state.queries.remove_tag(key);
    }

    /// The session's tags, by key.
    #[must_use]
    pub fn tags(&self) -> Vec<(String, String)> {
        self.state.queries.tags()
    }

    /// Ends the trace [`Session::set_trace_context`] set.
    pub fn clear_trace_context(&self) {
        self.state.queries.set_trace(None);
        self.state.queries.set_baggage(Vec::new());
    }

    /// The trace id in effect, if any.
    #[must_use]
    pub fn trace_id(&self) -> Option<String> {
        self.state.queries.trace_id()
    }
}
