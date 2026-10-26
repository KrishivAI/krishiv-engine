// SPDX-License-Identifier: Apache-2.0
//! The expression-evaluator interface the physical planner (P2-T074) plugs
//! into [`Filter`](super::Filter), [`Project`](super::Project) and the scan
//! pushdown. The kernels crate exposes array kernels, not expression
//! evaluation, so the operators only drive these traits.

use std::fmt;

use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_schema::DataType;

use crate::error::Result;

/// A boolean row expression. A null result does not select the row (SQL
/// `WHERE`).
pub trait Predicate: Send + Sync + fmt::Debug {
    /// One value per row of `batch`.
    ///
    /// # Errors
    /// Any error fails the query (for example a division by zero under
    /// ANSI mode).
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray>;
    /// As [`Self::evaluate`], in the executing partition `partition`
    /// (Spark's partition index). The default runs [`Self::evaluate`], for
    /// the stateless predicates.
    ///
    /// # Errors
    /// As [`Self::evaluate`].
    fn evaluate_in(&self, batch: &RecordBatch, partition: u32) -> Result<BooleanArray> {
        let _ = partition;
        self.evaluate(batch)
    }
}

/// A scalar expression evaluated over a batch, for [`Project`](super::Project).
pub trait ScalarExpr: Send + Sync + fmt::Debug {
    /// The type of the result column.
    fn data_type(&self) -> DataType;
    /// Whether the result can contain nulls.
    fn nullable(&self) -> bool;
    /// One value per row of `batch`.
    ///
    /// # Errors
    /// Any error fails the query.
    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef>;
    /// As [`Self::evaluate`], in the executing partition `partition`
    /// (Spark's partition index). The default runs [`Self::evaluate`], for
    /// the stateless expressions.
    ///
    /// # Errors
    /// As [`Self::evaluate`].
    fn evaluate_in(&self, batch: &RecordBatch, partition: u32) -> Result<ArrayRef> {
        let _ = partition;
        self.evaluate(batch)
    }
}

/// A [`Predicate`] from a closure.
pub struct FnPredicate<F>(pub F);

impl<F> fmt::Debug for FnPredicate<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FnPredicate")
    }
}

impl<F> Predicate for FnPredicate<F>
where
    F: Fn(&RecordBatch) -> Result<BooleanArray> + Send + Sync,
{
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray> {
        (self.0)(batch)
    }
}
