// SPDX-License-Identifier: Apache-2.0
//! A running script as one query of the engine: its `system.queries` row, its context in the
//! runtime's registry and its memory pool (P2b row 15, R3 and R4).
//!
//! The script's statements are ordinary queries, each with its own row. The script itself is
//! registered too, under the caller's operation id when it gave one, so that
//! [`Session::cancel`](crate::Session::cancel), `interrupt_all`, `EngineRuntime::cancel` and the
//! limit `krishiv.query.maxExecutionTime` / [`QueryOptions::deadline`](crate::QueryOptions) apply to
//! the script as a whole: a cancel between two statements ends it, and the limit counts from its
//! start (Spark 4.2.0 ends a script with `OPERATION_CANCELED` on an `Interrupt` between statements,
//! which no handler catches; record `tests/spark-golden/v4.2.0/scripting/interrupt`). The rows a
//! script holds (a `FOR` query's, the last result) are reserved in the script's pool, so they show
//! under its row in `system.memory` and the only limit is memory.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivError;
use krishiv_exec::{ExecConfig, QueryContext, QueryId, WorkloadClass};
use krishiv_memory::ConsumerId;

use crate::dataframe::DataFrame;
use crate::session::{Session, SessionState};

/// The registration of one running script. Dropping it ends the script's entry: its row is closed
/// (`cancelled` unless [`Self::settle`] recorded another outcome) and its memory released.
pub(super) struct ScriptRun {
    ctx: Arc<QueryContext>,
    state: Arc<SessionState>,
    row: String,
    consumer: ConsumerId,
    held: usize,
    settled: bool,
    started: std::time::Instant,
}

impl Session {
    /// Registers the script of `text` as a query: a `system.queries` row (named by the caller's
    /// operation id when there is one), a context in the runtime's registry with the statement's
    /// time limit, and a memory pool of its own.
    ///
    /// # Errors
    /// `INVALID_HANDLE.SESSION_CLOSED`, or `INVALID_HANDLE.OPERATION_ALREADY_EXISTS` while another
    /// execution of the session runs under the same operation id.
    pub(super) fn begin_script(&self) -> Result<ScriptRun, KrishivError> {
        self.check_open()?;
        let env = self.query_env();
        let row = self.state.queries.begin_as(
            crate::system::TextPolicy::from_config(&env.config),
            env.text.as_deref(),
            env.statement_class,
            self.operation_id(),
        )?;
        let index =
            self.state
                .runtime
                .reserve(self.state.number, self.state.queries.session_id(), &row);
        let pool = self
            .state
            .memory
            .child(&crate::system::pool_name(&row), None);
        let ctx = QueryContext::new(
            QueryId(index),
            WorkloadClass::Interactive,
            ExecConfig::from_config(&env.config),
            pool,
        );
        let consumer = ctx.pool().register("script", None);
        let run = ScriptRun {
            ctx: Arc::clone(&ctx),
            state: Arc::clone(&self.state),
            row: row.clone(),
            consumer,
            held: 0,
            settled: false,
            started: std::time::Instant::now(),
        };
        // A failed attach drops `run`, which closes the row.
        self.attach(&ctx, &row, &env.config)?;
        self.state.queries.mark_started(&row, None);
        Ok(run)
    }
}

impl Session {
    /// The result of a script: `batches` as a hidden in-memory table (held once, memory-accounted
    /// in the session's relation store) and a frame that reads it. The table goes when nothing but
    /// the catalog holds it, so the frame, and any frame derived from it, keeps it.
    ///
    /// # Errors
    /// `UNABLE_TO_ACQUIRE_MEMORY` when the relation store cannot hold the rows.
    pub(crate) async fn script_result(
        &self,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<DataFrame, KrishivError> {
        // Two columns of one name (`SELECT sv, session.sv`) cannot share a table: the table holds
        // numbered names and a projection gives the frame the names the statement produced.
        let original: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        let mut seen = std::collections::HashSet::new();
        let repeated = original.iter().any(|n| !seen.insert(n.to_lowercase()));
        // The batches carry the types of the statement's schema without its field metadata (the
        // Spark type of an interval); the table is made of the analysed schema.
        let batches = batches
            .into_iter()
            .map(|b| {
                RecordBatch::try_new_with_options(
                    Arc::clone(&schema),
                    b.columns().to_vec(),
                    &arrow_array::RecordBatchOptions::new().with_row_count(Some(b.num_rows())),
                )
                .map_err(|e| crate::session::internal(e.to_string()))
            })
            .collect::<Result<Vec<_>, KrishivError>>()?;
        let (schema, batches) = if repeated {
            let fields: Vec<_> = (0..original.len())
                .map(|i| Arc::new(schema.field(i).clone().with_name(format!("__krishiv_c{i}"))))
                .collect();
            let renamed = Arc::new(arrow_schema::Schema::new(fields));
            let batches = batches
                .into_iter()
                .map(|b| {
                    RecordBatch::try_new_with_options(
                        Arc::clone(&renamed),
                        b.columns().to_vec(),
                        &arrow_array::RecordBatchOptions::new().with_row_count(Some(b.num_rows())),
                    )
                    .map_err(|e| crate::session::internal(e.to_string()))
                })
                .collect::<Result<Vec<_>, KrishivError>>()?;
            (renamed, batches)
        } else {
            (schema, batches)
        };
        let (columns, table) = self.memory_table(schema, batches)?;
        let output = columns
            .iter()
            .zip(0_u64..)
            .map(|(c, i)| krishiv_plan::BoundColumn {
                id: krishiv_plan::AttrId(i),
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
            })
            .collect();
        let (name, entry) = self
            .state
            .catalog
            .put_hidden_memory(columns, Arc::new(table));
        // A resolved scan, as a view's snapshot pins read: the analyzer checks it against the table.
        let plan =
            krishiv_plan::LogicalPlan::new(krishiv_plan::PlanNode::Scan(krishiv_plan::Scan {
                source: krishiv_plan::ReadSource::Table {
                    name: vec![name],
                    options: std::collections::BTreeMap::new(),
                },
                output,
                version: None,
            }));
        let mut df = DataFrame::new(self.clone(), Arc::new(plan));
        if repeated {
            df =
                df.select(original.iter().enumerate().map(|(i, name)| {
                    crate::functions::col(&format!("__krishiv_c{i}")).alias(name)
                }));
        }
        // Analysed now, so the frame holds the table before the catalog can let it go.
        df.analyzed_in(self.query_env()).await?;
        // The frame, and every frame derived from it, owns the table.
        df.keep.push(entry);
        Ok(df)
    }
}

impl ScriptRun {
    /// A future that completes with the error that ended the script when something cancelled it (a
    /// cancel of its operation or session, the engine, or its time limit): `OPERATION_CANCELED`,
    /// `KRISHIV_QUERY_TIMEOUT` or `KRISHIV_QUERY_KILLED`. It holds no borrow of the registration.
    pub(super) fn watcher(&self) -> impl std::future::Future<Output = KrishivError> + use<> {
        let ctx = Arc::clone(&self.ctx);
        async move {
            ctx.cancelled().await;
            ctx.check_cancelled()
                .err()
                .unwrap_or_else(|| krishiv_common::error::OperationCanceled {}.into())
        }
    }

    /// Reserves `batches`' bytes in the script's pool for as long as they are held.
    ///
    /// # Errors
    /// `UNABLE_TO_ACQUIRE_MEMORY` when the pool refuses them.
    pub(super) fn hold(&mut self, batches: &[RecordBatch]) -> Result<usize, KrishivError> {
        let bytes = batches.iter().map(RecordBatch::get_array_memory_size).sum();
        self.ctx.pool().try_reserve(self.consumer, bytes)?;
        self.held += bytes;
        Ok(bytes)
    }

    /// Gives back `bytes` that [`Self::hold`] reserved.
    pub(super) fn release(&mut self, bytes: usize) {
        let bytes = bytes.min(self.held);
        self.ctx.pool().release(self.consumer, bytes);
        self.held -= bytes;
    }

    /// Records how the script ended: `finished`, or by the error's class `failed`, `denied` or
    /// `cancelled`.
    pub(super) fn settle(&mut self, outcome: Result<(), &KrishivError>) {
        self.settled = true;
        match outcome {
            Ok(()) => self.close("finished", None),
            Err(error) => {
                let class = error.error_class().name();
                let state = crate::system::outcome_state(class);
                self.close(state, (state != "cancelled").then_some(class));
            }
        }
    }

    fn close(&self, state: &'static str, error_class: Option<&str>) {
        self.state.queries.finish(
            &self.row,
            &crate::system::Finished {
                state,
                wall_ns: i64::try_from(self.started.elapsed().as_nanos()).unwrap_or(i64::MAX),
                peak_memory_bytes: i64::try_from(self.ctx.pool().peak_reserved())
                    .unwrap_or(i64::MAX),
                error_class,
                operators: Vec::new(),
                totals: None,
            },
        );
    }
}

impl Drop for ScriptRun {
    fn drop(&mut self) {
        self.ctx.pool().unregister(self.consumer);
        if !self.settled {
            // Dropped before an outcome: the caller went away.
            self.close("cancelled", None);
        }
    }
}
