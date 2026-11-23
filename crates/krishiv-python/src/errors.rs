// SPDX-License-Identifier: Apache-2.0
//! Engine errors as Python exceptions: `krishiv.errors._from_native` picks the
//! `pyspark.errors`-shaped class for the Spark error class and SQLSTATE.

use krishiv_common::error::{KrishivError, render_with_context};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// One query context for [`to_py`]: `(kind, fragment, start, stop,
/// object type, object name, call site, summary)`.
type RawContext = (u8, String, usize, usize, String, String, String, String);

/// The Python exception for `error`; a `RuntimeError` carrying the message (and
/// the conversion failure as its cause) if `krishiv.errors` cannot be used.
pub(crate) fn to_py(py: Python<'_>, error: &KrishivError) -> PyErr {
    convert(py, error, rendered_from_summaries(error))
}

/// The message followed by the `== SQL ==` summaries the error's contexts
/// carry (a run-time error's, filled by the session from the statement's
/// text), for an error raised where the query text is not at hand (a
/// DataFrame action); `None` when no context has one.
fn rendered_from_summaries(error: &KrishivError) -> Option<String> {
    let summaries: Vec<&str> = error
        .query_contexts()
        .iter()
        .map(|c| c.summary.as_str())
        .filter(|s| !s.is_empty())
        // Spark's summary ends with a newline; the rendered text, as
        // `render_with_context`'s, does not.
        .map(|s| s.strip_suffix('\n').unwrap_or(s))
        .collect();
    (!summaries.is_empty()).then(|| format!("{error}\n{}", summaries.join("\n")))
}

/// [`to_py`] for an error of the query `sql`: the exception's `str` also marks
/// the SQL span that raised it over the query text with its secrets redacted;
/// `getMessage()` stays the message alone. A parse error uses Spark's
/// `ParseException` layout (as `krishiv sql` does), any other error the
/// query-context summary.
pub(crate) fn to_py_for_sql(py: Python<'_>, error: &KrishivError, sql: &str) -> PyErr {
    let redacted = krishiv_sql::redact_sql(sql);
    let text = redacted.as_str();
    let suffix = krishiv_sql::parse_exception_suffix(text, 0, error);
    let rendered = if suffix.is_empty() {
        render_with_context(error, text)
    } else {
        format!("{error}{suffix}")
    };
    convert(py, error, Some(rendered))
}

fn convert(py: Python<'_>, error: &KrishivError, rendered: Option<String>) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let parameters = PyDict::new(py);
        for (key, value) in error.message_parameters() {
            parameters.set_item(key, value)?;
        }
        let contexts: Vec<RawContext> = error
            .query_contexts()
            .iter()
            .map(|c| {
                let kind = match c.context_type {
                    krishiv_common::error::ErrorContextType::Sql => 0,
                    krishiv_common::error::ErrorContextType::DataFrame => 1,
                };
                (
                    kind,
                    c.fragment.clone(),
                    c.start,
                    c.stop,
                    c.object_type.clone(),
                    c.object_name.clone(),
                    c.call_site.clone(),
                    c.summary.clone(),
                )
            })
            .collect();
        let exception = py
            .import("krishiv.errors")?
            .getattr("_from_native")?
            .call1((
                error.error_class().name(),
                error.sql_state(),
                error.to_string(),
                parameters,
                contexts,
                rendered,
            ))?;
        let err = PyErr::from_value(exception);
        // The causes become the `__cause__` chain, so the traceback shows Spark's `Caused by:` lines.
        let texts: Vec<String> = error.causes().map(ToString::to_string).collect();
        let mut cause: Option<PyErr> = None;
        for text in texts.into_iter().rev() {
            let next = pyo3::exceptions::PyRuntimeError::new_err(text);
            next.set_cause(py, cause.take());
            cause = Some(next);
        }
        err.set_cause(py, cause);
        Ok(err)
    };
    build().unwrap_or_else(|cause| {
        // Keep why the conversion failed (a bug in `krishiv.errors`) as the cause.
        let err = pyo3::exceptions::PyRuntimeError::new_err(error.to_string());
        err.set_cause(py, Some(cause));
        err
    })
}

/// The sessions made in this process, weakly: a Ctrl-C cancels through them ([`wait_for_signals`]).
static LIVE_SESSIONS: std::sync::Mutex<Vec<std::sync::Weak<krishiv_session::Session>>> =
    std::sync::Mutex::new(Vec::new());

/// Registers `session` for [`interrupt_live_sessions`]; sessions that have since been dropped are forgotten.
pub(crate) fn register_session(session: &std::sync::Arc<krishiv_session::Session>) {
    let mut live = LIVE_SESSIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    live.retain(|s| s.strong_count() > 0);
    live.push(std::sync::Arc::downgrade(session));
}

/// Interrupts every action of every live session in the process, other threads' sessions included, through
/// the session (`Session::interrupt_all`, as `interruptAll` does); PySpark classic's SIGINT handler does the
/// same (`cancelAllJobs` on the shared context). Whether there was a live session to ask.
fn interrupt_live_sessions() -> bool {
    let sessions: Vec<_> = LIVE_SESSIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .collect();
    for session in &sessions {
        session.interrupt_all();
    }
    !sessions.is_empty()
}

/// Waits for `work` with the GIL released; every [`SIGNAL_POLL`] the thread attaches again and asks
/// Python for its pending signals (`Python::check_signals`: a Ctrl-C, a Jupyter kernel interrupt,
/// `_thread.interrupt_main`). On a signal that raises (`KeyboardInterrupt`) `work` is not dropped: a
/// drop would skip the session's commit guard (`krishiv-session` `action::commit_point`), so an
/// overwrite could be left with the old table gone and the new one not written. The sessions
/// cancel their actions instead (`interrupt_all`: an action before its commit point ends at once, one
/// past it finishes), `work` is polled to its end, and the signal's error is returned in place of
/// its output. A second Ctrl-C while a past-commit action finishes waits too: Python raises it after the
/// first. With no live session to ask, `work` simply runs to its end. Off the main thread
/// `check_signals` never raises, so such a call waits as `stream::wait_for` does.
fn wait_for_signals<F: std::future::Future + Send>(py: Python<'_>, work: F) -> PyResult<F::Output>
where
    F::Output: Send,
{
    let mut waiter = crate::stream::Waiter::new();
    let mut work = std::pin::pin!(work);
    let mut raised: Option<PyErr> = None;
    loop {
        if let Some(out) =
            py.detach(|| waiter.step(work.as_mut(), Some(crate::stream::SIGNAL_POLL)))
        {
            return raised.map_or(Ok(out), Err);
        }
        if raised.is_none()
            && let Err(e) = py.check_signals()
        {
            interrupt_live_sessions();
            raised = Some(e);
        }
    }
}

/// Runs the engine future `work` to completion with the GIL released, looking at Python's signals
/// meanwhile: a Ctrl-C or kernel interrupt cancels the session's actions and raises
/// `KeyboardInterrupt` once `work` has ended (NB-1).
pub(crate) fn block_on<T: Send>(
    py: Python<'_>,
    work: impl std::future::Future<Output = Result<T, KrishivError>> + Send,
) -> PyResult<T> {
    crate::udf::runtime::refuse_inside_udf(py)?;
    wait_for_signals(py, work)?.map_err(|e| to_py(py, &e))
}

/// [`block_on`] for the query `sql`, whose errors are converted by
/// [`to_py_for_sql`].
pub(crate) fn block_on_sql<T: Send>(
    py: Python<'_>,
    sql: &str,
    work: impl std::future::Future<Output = Result<T, KrishivError>> + Send,
) -> PyResult<T> {
    crate::udf::runtime::refuse_inside_udf(py)?;
    wait_for_signals(py, work)?.map_err(|e| to_py_for_sql(py, &e, sql))
}
