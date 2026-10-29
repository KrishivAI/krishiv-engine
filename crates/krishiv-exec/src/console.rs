// SPDX-License-Identifier: Apache-2.0
//! Live async-task diagnostics over `tokio-console` (P2-T129, off by default).
//!
//! With the `tokio-console` cargo feature and `RUSTFLAGS="--cfg
//! tokio_unstable"`, every pipeline driver and every exec-side I/O task
//! (the file-write writers, the output-spill hooks) spawns with a name the
//! console shows in its task list: `q<query id>/<pipeline>/driver<n>` for
//! drivers, `io/<op>` for I/O tasks. Without the feature, or without the
//! cfg, [`spawn_named`] spawns unnamed tasks exactly as before: zero
//! overhead and zero console symbols in default builds (guarded by `cargo
//! xtask tokio-console-check`).
//!
//! [`maybe_init_tokio_console`] installs the console subscriber when the
//! feature is on and `KRISHIV_TOKIO_CONSOLE=1`. The subscriber runs on its
//! own Tokio runtime in a background thread and accounts none of its memory
//! to the query [`MemoryPool`](krishiv_memory::MemoryPool): it is
//! diagnostics, not query state, and it must never compete with queries for
//! memory. The only process-wide state is the subscriber itself; nothing
//! here blocks a compute worker.

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::context::{QueryContext, QueryId};
use crate::executor::{Executor, TaskHandle, WorkloadClass};

#[cfg(all(feature = "tokio-console", tokio_unstable))]
use crate::error::internal;
#[cfg(feature = "tokio-console")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// The console subscriber's event-buffer capacity: fixed, not the crate
/// default (`console_subscriber` buffers 102400 events by default).
pub const EVENT_BUFFER_CAPACITY: usize = 10_000;

/// The tokio-console name of one pipeline driver: the query, the pipeline
/// index and the driver's scheduler-assigned partition id.
#[must_use]
pub fn driver_task_name(query: QueryId, pipeline: u32, driver: u32) -> String {
    format!("q{}/{pipeline}/driver{driver}", query.0)
}

/// The tokio-console name of an exec-side I/O task (`file-write` for a file
/// writer, `spill` for an output-spill hook).
#[must_use]
pub fn io_task_name(op: &str) -> String {
    format!("io/{op}")
}

/// Spawns `fut` on `executor`'s compute runtime, named `name` for
/// tokio-console when the `tokio-console` feature is on and the crate is
/// built with `--cfg tokio_unstable`; an unnamed task otherwise. `query`
/// fails if the spawn itself is refused (a dead runtime), so a driver below
/// never hangs on a task that never runs.
pub fn spawn_named(
    executor: &Arc<dyn Executor>,
    query: &Arc<QueryContext>,
    class: WorkloadClass,
    name: &str,
    fut: BoxFuture<'static, ()>,
) -> TaskHandle {
    #[cfg(all(feature = "tokio-console", tokio_unstable))]
    {
        let _ = class;
        let handle = executor.compute_handle();
        match tokio::task::Builder::new().name(name).spawn_on(fut, handle) {
            Ok(join) => TaskHandle::new(join),
            Err(e) => {
                query.fail(internal(format!("cannot spawn task {name}: {e}")));
                TaskHandle::new(handle.spawn(std::future::ready(())))
            }
        }
    }
    #[cfg(not(all(feature = "tokio-console", tokio_unstable)))]
    {
        let _ = (query, name);
        executor.spawn(class, fut)
    }
}

/// Starts the tokio-console subscriber if the `tokio-console` feature is on
/// and `KRISHIV_TOKIO_CONSOLE=1`, once per process; a no-op otherwise. The
/// retention is 60 seconds and the event buffer is
/// [`EVENT_BUFFER_CAPACITY`], not the crate defaults. The subscriber is
/// installed with `try_init`: when the host already owns the global
/// tracing subscriber (an embedder's logger), diagnostics stay off instead
/// of panicking, and host logging is left alone. The CLI and the
/// session builder call this at startup, so embedders only set the variable.
pub fn maybe_init_tokio_console() {
    #[cfg(feature = "tokio-console")]
    {
        static INIT: std::sync::Once = std::sync::Once::new();
        if std::env::var("KRISHIV_TOKIO_CONSOLE").as_deref() != Ok("1") {
            return;
        }
        INIT.call_once(|| {
            let layer = console_subscriber::ConsoleLayer::builder()
                .retention(std::time::Duration::from_secs(60))
                .event_buffer_capacity(EVENT_BUFFER_CAPACITY)
                .spawn();
            // A set global subscriber (or a racing init) is not an error:
            // the console just stays off.
            let _ = tracing_subscriber::registry().with(layer).try_init();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ExecConfig;
    use crate::executor::{ExecutorConfig, TokioExecutor};
    use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};

    #[cfg(tokio_unstable)]
    use std::fmt;
    #[cfg(tokio_unstable)]
    use std::sync::atomic::{AtomicU64, Ordering};
    #[cfg(tokio_unstable)]
    use std::sync::{Mutex, PoisonError};
    #[cfg(tokio_unstable)]
    use tracing::field::Visit;
    #[cfg(tokio_unstable)]
    use tracing::span::{Attributes, Id};
    #[cfg(tokio_unstable)]
    use tracing::{Metadata, Subscriber};

    fn query() -> Arc<QueryContext> {
        let pool = TreePool::new(&MemoryConfig::default());
        QueryContext::new(
            QueryId(1),
            WorkloadClass::Batch,
            ExecConfig::default(),
            pool.child("q1", None),
        )
    }

    #[test]
    fn task_names_carry_query_pipeline_driver_and_op() {
        assert_eq!(driver_task_name(QueryId(7), 2, 3), "q7/2/driver3");
        assert_eq!(driver_task_name(QueryId(1), 0, 0), "q1/0/driver0");
        assert_eq!(io_task_name("file-write"), "io/file-write");
        assert_eq!(io_task_name("spill"), "io/spill");
    }

    /// The subscriber's buffers are fixed, not the crate defaults.
    #[cfg(feature = "tokio-console")]
    #[test]
    fn the_subscriber_uses_fixed_non_default_buffers() {
        assert_eq!(EVENT_BUFFER_CAPACITY, 10_000);
        assert_ne!(
            EVENT_BUFFER_CAPACITY,
            console_subscriber::ConsoleLayer::DEFAULT_EVENT_BUFFER_CAPACITY
        );
    }

    #[test]
    fn spawn_named_runs_the_task() {
        let executor: Arc<dyn Executor> = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
        let query = query();
        let (tx, rx) = std::sync::mpsc::channel::<u32>();
        drop(spawn_named(
            &executor,
            &query,
            WorkloadClass::Batch,
            "q1/0/driver0",
            Box::pin(async move {
                let _ = tx.send(41);
            }),
        ));
        assert_eq!(rx.recv().unwrap(), 41);
    }

    /// With `--cfg tokio_unstable`, the spawned task registers under its
    /// name: tokio wraps it in a `runtime.spawn` span carrying `task.name`.
    #[cfg(tokio_unstable)]
    #[test]
    fn named_tasks_register_with_their_names() {
        #[derive(Debug, Clone, Default)]
        struct Capture {
            fields: Arc<Mutex<Vec<String>>>,
            next: Arc<AtomicU64>,
        }

        struct FieldStrings(Vec<String>);

        impl Visit for FieldStrings {
            fn record_debug(&mut self, _field: &tracing::field::Field, value: &dyn fmt::Debug) {
                self.0.push(format!("{value:?}"));
            }
        }

        impl Subscriber for Capture {
            fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
                true
            }

            fn new_span(&self, span: &Attributes<'_>) -> Id {
                let mut strings = FieldStrings(Vec::new());
                span.values().record(&mut strings);
                self.fields
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .extend(strings.0);
                Id::from_u64(self.next.fetch_add(1, Ordering::Relaxed) + 1)
            }

            fn record(&self, _span: &Id, _values: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
            fn event(&self, _event: &tracing::Event<'_>) {}
            fn enter(&self, _span: &Id) {}
            fn exit(&self, _span: &Id) {}
        }

        let executor: Arc<dyn Executor> = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
        let query = query();
        let capture = Capture::default();
        drop(tracing::subscriber::with_default(capture.clone(), || {
            spawn_named(
                &executor,
                &query,
                WorkloadClass::Batch,
                "q9/4/driver6",
                Box::pin(async {}),
            )
        }));
        let fields = capture
            .fields
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        assert!(
            fields.iter().any(|f| f == "q9/4/driver6"),
            "a driver task registers under its name, got {fields:?}"
        );
    }
}
