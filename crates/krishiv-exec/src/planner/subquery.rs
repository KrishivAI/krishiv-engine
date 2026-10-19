// SPDX-License-Identifier: Apache-2.0
//! Uncorrelated scalar and `EXISTS` subqueries in expressions.
//!
//! Each subquery is planned as its own pipeline ending in a
//! [`SubquerySink`], a dependency of the pipeline that evaluates the
//! expression: its `finalize` stores the value in a [`Slot`] the
//! expression reads, so the value is there before the first batch is
//! evaluated (Spark computes such subqueries before the main plan too).
//! Spark's semantics: a scalar subquery with no row is NULL, with more
//! than one row it fails with `SCALAR_SUBQUERY_TOO_MANY_ROWS`.

use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, new_null_array};
use arrow_schema::DataType as ArrowType;
use futures::future::BoxFuture;
use krishiv_common::error::ScalarSubqueryTooManyRows;
use krishiv_types::batch::Batch;

use super::build::internal;
use super::eval::{PhysExpr, Value};
use crate::context::TaskContext;
use crate::error::Result;
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus};
use crate::pipeline::{ExecNode, PipelineBuilder};

/// The value of one subquery, set once its pipeline finalized.
#[derive(Debug, Default)]
pub(crate) struct Slot(OnceLock<ArrayRef>);

/// Scalar (the first column of at most one row) or `EXISTS`.
pub(crate) use crate::physical::SubqueryValueKind as Kind;

/// Collects a subquery's rows and publishes its value.
#[derive(Debug)]
pub(crate) struct SubquerySink {
    kind: Kind,
    data_type: ArrowType,
    slot: Arc<Slot>,
}

impl SubquerySink {
    pub(crate) fn new(kind: Kind, data_type: ArrowType, slot: Arc<Slot>) -> Arc<Self> {
        Arc::new(Self {
            kind,
            data_type,
            slot,
        })
    }
}

type Rows = Mutex<Option<ArrayRef>>;

impl Sink for SubquerySink {
    fn name(&self) -> &'static str {
        "Subquery"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(Rows::default()))
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn sink(
        &self,
        input: Batch,
        _local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<SinkStatus> {
        if input.num_rows() == 0 {
            return Ok(SinkStatus::NeedMoreInput);
        }
        let rows = global
            .downcast_ref::<Rows>()
            .ok_or_else(|| internal("subquery state"))?;
        let mut seen = rows.lock().unwrap_or_else(PoisonError::into_inner);
        if self.kind == Kind::Exists {
            *seen = Some(Arc::new(BooleanArray::from(vec![true])));
            return Ok(SinkStatus::Finished);
        }
        if seen.is_some() || input.num_rows() > 1 {
            return Err(ScalarSubqueryTooManyRows {}.into());
        }
        let column = input
            .data()
            .columns()
            .first()
            .ok_or_else(|| internal("a scalar subquery without a column"))?;
        *seen = Some(Arc::clone(column));
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, _local: LocalState, _global: &GlobalState) -> Result<()> {
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let rows = global
                .downcast_ref::<Rows>()
                .ok_or_else(|| internal("subquery state"))?;
            let value = rows
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
                .unwrap_or_else(|| match self.kind {
                    Kind::Exists => Arc::new(BooleanArray::from(vec![false])),
                    Kind::Scalar => new_null_array(&self.data_type, 1),
                });
            self.slot
                .0
                .set(value)
                .map_err(|_| internal("a subquery ran twice"))
        })
    }
}

/// An expression node reading a subquery's value.
#[derive(Debug)]
pub(crate) struct SubqueryValue {
    pub(crate) kind: Kind,
    pub(crate) slot: Arc<Slot>,
    pub(crate) data_type: ArrowType,
}

impl PhysExpr for SubqueryValue {
    fn data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn nullable(&self) -> bool {
        // Spark: `EXISTS` is never null; a scalar subquery is NULL when it
        // returns no row or a NULL, whatever its type.
        self.kind == Kind::Scalar
    }

    fn eval(&self, _batch: &RecordBatch) -> Result<Value> {
        let v = self
            .slot
            .0
            .get()
            .ok_or_else(|| internal("a subquery read before it ran"))?;
        if v.data_type() != &self.data_type {
            return Err(internal("a subquery value of another type"));
        }
        Ok(Value::Scalar(Arc::clone(v)))
    }
}

/// `node` evaluated after the subqueries' pipelines: each subquery plan
/// ends in its sink, a dependency of the pipeline `node` adds its operator
/// to.
pub(crate) struct WithSubqueries {
    pub(crate) subqueries: Vec<(Arc<dyn ExecNode>, Arc<SubquerySink>)>,
    /// The input of the operator that evaluates the expressions.
    pub(crate) input: Arc<dyn ExecNode>,
    pub(crate) op: Arc<dyn crate::operator::Operator>,
}

impl ExecNode for WithSubqueries {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        Arc::clone(&self.input).build_pipelines(b)?;
        for (plan, sink) in &self.subqueries {
            b.child_pipeline(Arc::clone(sink) as Arc<dyn Sink>, Arc::clone(plan))?;
        }
        b.push_operator(Arc::clone(&self.op));
        Ok(())
    }
}
