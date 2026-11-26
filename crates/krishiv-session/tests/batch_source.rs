// SPDX-License-Identifier: Apache-2.0
//! `Session::register_batch_source`: a table whose rows are asked for again for every plan
//! (the Python Data Source's reader, design P3-T040 §3.3), also through a view over it.
#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::{BatchSource, Session};

fn count(session: &Session, sql: &str) -> i64 {
    let rows = block_on(async { session.sql(sql).await?.collect().await }).unwrap();
    let batch = &rows[0];
    arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(batch.column(0))
        .value(0)
}

#[test]
fn rows_are_asked_for_again_on_every_plan() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let calls = Arc::new(AtomicI32::new(0));
    let (s, c) = (Arc::clone(&schema), Arc::clone(&calls));
    let source: BatchSource = Arc::new(move || {
        let n = c.fetch_add(1, Ordering::SeqCst) + 1;
        let batch = RecordBatch::try_new(
            Arc::clone(&s),
            vec![Arc::new(Int32Array::from((0..n).collect::<Vec<_>>()))],
        )
        .map_err(|e| {
            KrishivError::from(krishiv_common::error::InternalError {
                message: e.to_string(),
            })
        })?;
        Ok(vec![batch])
    });
    session
        .register_batch_source("grows", "grows", schema, source)
        .unwrap();
    let first = count(&session, "SELECT count(*) FROM grows");
    let second = count(&session, "SELECT count(*) FROM grows");
    assert!(
        second > first,
        "a second action saw {second} rows after {first}"
    );
    // a view over it keeps reading it
    block_on(async {
        session
            .sql("CREATE TEMPORARY VIEW v AS SELECT * FROM grows")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert!(count(&session, "SELECT count(*) FROM v") > 0);
}

#[test]
fn a_failing_source_fails_the_query() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let source: BatchSource = Arc::new(|| {
        Err(krishiv_common::error::InternalError {
            message: "no rows today".to_owned(),
        }
        .into())
    });
    session
        .register_batch_source("boom", "boom", schema, source)
        .unwrap();
    let err =
        block_on(async { session.sql("SELECT * FROM boom").await?.collect().await }).unwrap_err();
    assert!(err.to_string().contains("no rows today"), "{err}");
}

#[test]
fn a_batch_of_another_schema_is_refused() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let other = Arc::new(Schema::new(vec![Field::new("z", DataType::Int32, true)]));
    let source: BatchSource = Arc::new(move || {
        Ok(vec![
            RecordBatch::try_new(
                Arc::clone(&other),
                vec![Arc::new(Int32Array::from(vec![1]))],
            )
            .unwrap(),
        ])
    });
    session
        .register_batch_source("odd", "odd", schema, source)
        .unwrap();
    assert!(block_on(async { session.sql("SELECT * FROM odd").await?.collect().await }).is_err());
}

#[test]
fn a_reserved_name_is_refused() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let source: BatchSource = Arc::new(|| Ok(Vec::new()));
    assert!(
        session
            .register_batch_source(
                "__krishiv_snapshot_x",
                "__krishiv_snapshot_x",
                schema,
                source
            )
            .is_err()
    );
}

#[test]
fn the_batches_of_a_plan_count_against_the_memory_limit() {
    let session = Session::builder()
        .config("krishiv.memory.limit", "1m")
        .build()
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let s = Arc::clone(&schema);
    // 1,500,000 ints: about 6 MB, over the 1 MiB limit
    let source: BatchSource = Arc::new(move || {
        Ok(vec![
            RecordBatch::try_new(
                Arc::clone(&s),
                vec![Arc::new(Int32Array::from(vec![7; 1_500_000]))],
            )
            .unwrap(),
        ])
    });
    session
        .register_batch_source("big", "big", schema, source)
        .unwrap();
    let error = block_on(async {
        session
            .sql("SELECT count(*) FROM big")
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert_ne!(
        error.error_class().name(),
        "KRISHIV_PYTHON_DATA_SOURCE_LIMIT",
        "the pool, not the cap, refused it: {error}"
    );
    assert!(
        error.to_string().to_lowercase().contains("memory"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_second_plan_of_the_table_needs_one_copy_not_two() {
    // 600,000 ints (about 2.4 MB) against a 4 MiB limit: one copy fits, two do not
    let session = Session::builder()
        .config("krishiv.memory.limit", "4m")
        .build()
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let s = Arc::clone(&schema);
    let source: BatchSource = Arc::new(move || {
        Ok(vec![
            RecordBatch::try_new(
                Arc::clone(&s),
                vec![Arc::new(Int32Array::from(vec![7; 600_000]))],
            )
            .unwrap(),
        ])
    });
    session
        .register_batch_source("twice", "twice", schema, source)
        .unwrap();
    assert_eq!(count(&session, "SELECT count(*) FROM twice"), 600_000);
    assert_eq!(count(&session, "SELECT count(*) FROM twice"), 600_000);
    assert_eq!(count(&session, "SELECT count(*) FROM twice"), 600_000);
}

#[test]
fn a_narrow_decimal_source_and_a_self_join_read() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "d",
        DataType::Decimal128(10, 2),
        true,
    )]));
    let s = Arc::clone(&schema);
    let source: BatchSource = Arc::new(move || {
        let values = arrow_array::Decimal128Array::from(vec![125, 250])
            .with_precision_and_scale(10, 2)
            .unwrap();
        Ok(vec![
            RecordBatch::try_new(Arc::clone(&s), vec![Arc::new(values)]).unwrap(),
        ])
    });
    session
        .register_batch_source("dec", "dec", schema, source)
        .unwrap();
    assert_eq!(count(&session, "SELECT count(*) FROM dec"), 2);
    // two scans of one source are alive together
    assert_eq!(
        count(
            &session,
            "SELECT count(*) FROM dec a JOIN dec b ON a.d = b.d"
        ),
        2
    );
}

mod planned {
    use std::sync::Mutex;

    use krishiv_exec::SourceFilter;
    use krishiv_session::{PlannedRead, SourcePlanner};

    use super::*;

    /// Claims every filter it is offered and returns all three rows regardless.
    struct Claims {
        seen: Arc<Mutex<Vec<Vec<SourceFilter>>>>,
        schema: Arc<Schema>,
    }

    impl SourcePlanner for Claims {
        #[expect(
            clippy::unwrap_in_result,
            reason = "a test fake; a poisoned lock is a test failure"
        )]
        fn plan(&self, filters: &[SourceFilter]) -> Result<PlannedRead, KrishivError> {
            self.seen.lock().unwrap().push(filters.to_vec());
            let schema = Arc::clone(&self.schema);
            Ok(PlannedRead {
                pushed: vec![true; filters.len()],
                read: Arc::new(move || {
                    Ok(vec![
                        RecordBatch::try_new(
                            Arc::clone(&schema),
                            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
                        )
                        .unwrap(),
                    ])
                }),
            })
        }
    }

    fn register(session: &Session) -> Arc<Mutex<Vec<Vec<SourceFilter>>>> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let plain_schema = Arc::clone(&schema);
        let plain: BatchSource = Arc::new(move || {
            Ok(vec![
                RecordBatch::try_new(
                    Arc::clone(&plain_schema),
                    vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
                )
                .unwrap(),
            ])
        });
        session
            .register_planned_source(
                "p",
                "p",
                Arc::clone(&schema),
                plain,
                Some(Arc::new(Claims {
                    seen: Arc::clone(&seen),
                    schema,
                })),
            )
            .unwrap();
        seen
    }

    #[test]
    fn a_claimed_filter_is_trusted_and_planned_once() {
        let session = Session::new().unwrap();
        let seen = register(&session);
        // The source claims k > 1 and returns every row: the engine does not apply it again.
        assert_eq!(count(&session, "SELECT count(*) FROM p WHERE k > 1"), 3);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert!(
            seen[0]
                .iter()
                .any(|f| matches!(f, SourceFilter::GreaterThan(..))),
            "{seen:?}"
        );
    }

    #[test]
    fn a_scan_with_no_filter_does_not_plan() {
        let session = Session::new().unwrap();
        let seen = register(&session);
        assert_eq!(count(&session, "SELECT count(*) FROM p"), 3);
        assert!(
            seen.lock().unwrap().is_empty(),
            "no filter: nothing planned"
        );
        // `k + 1 > 2` does not serialize, but Spark infers `IsNotNull(k)` from it and offers that alone.
        assert_eq!(count(&session, "SELECT count(*) FROM p WHERE k + 1 > 2"), 2);
        assert_eq!(
            *seen.lock().unwrap(),
            vec![vec![SourceFilter::IsNotNull(krishiv_io::ColumnPath::new(
                "k"
            ))]]
        );
    }

    #[test]
    fn plans_that_are_only_looked_at_run_no_user_code() {
        let session = Session::new().unwrap();
        let seen = register(&session);
        block_on(async {
            let df = session.sql("SELECT k FROM p WHERE k > 1").await?;
            df.warnings().await?;
            df.estimate().await?;
            Ok::<(), KrishivError>(())
        })
        .unwrap();
        assert!(seen.lock().unwrap().is_empty());
        // EXPLAIN is the planning call Spark runs: it asks the source once.
        block_on(async {
            session
                .sql("SELECT k FROM p WHERE k > 1")
                .await?
                .explain(krishiv_session::ExplainMode::Simple)
                .await
        })
        .unwrap();
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn round_trip_mode_refuses_a_decided_plan_with_the_named_class() {
        let session = Session::new().unwrap();
        let _seen = register(&session);
        session
            .conf()
            .set("krishiv.exec.debug.planRoundTrip", "true")
            .unwrap();
        let err = block_on(async {
            session
                .sql("SELECT k FROM p WHERE k > 1")
                .await?
                .collect()
                .await
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("KRISHIV_UNSUPPORTED_FEATURE"),
            "{err}"
        );
    }

    /// A planner whose user code blocks does not hold the executor: with one compute thread, another
    /// query of the session finishes while the first is still waiting for the source's `plan`.
    struct Blocks {
        started: std::sync::mpsc::Sender<()>,
        release: Arc<Mutex<()>>,
        schema: Arc<Schema>,
    }

    impl SourcePlanner for Blocks {
        fn plan(&self, filters: &[SourceFilter]) -> Result<PlannedRead, KrishivError> {
            let started = self.started.clone();
            let release = Arc::clone(&self.release);
            krishiv_session::run_python_source_call(move || {
                let _ = started.send(());
                drop(release.lock());
                Ok(())
            })?;
            let schema = Arc::clone(&self.schema);
            Ok(PlannedRead {
                pushed: vec![false; filters.len()],
                read: Arc::new(move || {
                    Ok(vec![
                        RecordBatch::try_new(
                            Arc::clone(&schema),
                            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
                        )
                        .unwrap(),
                    ])
                }),
            })
        }
    }

    #[test]
    fn deciding_plan_leaves_the_executor_free() {
        use krishiv_exec::{ExecutorConfig, TokioExecutor};
        let executor = TokioExecutor::new(&ExecutorConfig {
            compute_threads: 1,
            io_threads: 1,
            blocking_threads: 2,
            io_blocking_threads: 2,
            release_when_idle: true,
        })
        .unwrap();
        let session = Session::builder().executor(executor).build().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
        let plain_schema = Arc::clone(&schema);
        let plain: BatchSource = Arc::new(move || {
            Ok(vec![
                RecordBatch::try_new(
                    Arc::clone(&plain_schema),
                    vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
                )
                .unwrap(),
            ])
        });
        let release = Arc::new(Mutex::new(()));
        let held = release.lock().unwrap();
        let (started, started_rx) = std::sync::mpsc::channel();
        session
            .register_planned_source(
                "p",
                "p",
                Arc::clone(&schema),
                plain,
                Some(Arc::new(Blocks {
                    started,
                    release: Arc::clone(&release),
                    schema,
                })),
            )
            .unwrap();
        let blocked = {
            let session = session.clone_session().unwrap();
            std::thread::spawn(move || count(&session, "SELECT count(*) FROM p WHERE k > 1"))
        };
        started_rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("the planner's user code started");
        // The first query is parked in user code; a second one still gets the one compute thread.
        assert_eq!(count(&session, "SELECT count(*) FROM range(3)"), 3);
        drop(held);
        assert_eq!(blocked.join().unwrap(), 2);
    }
}
