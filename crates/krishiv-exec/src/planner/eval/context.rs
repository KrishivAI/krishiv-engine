// SPDX-License-Identifier: Apache-2.0
//! The SQL context of a run-time error (Spark's `QueryContext`): the error of a cast, an
//! arithmetic operation or a function carries the span of the expression that raised it.
//!
//! The planner wraps such an expression in [`Contextual`] when its origin names a SQL span and
//! the plan carries that text ([`EvalEnv::sql_texts`](super::EvalEnv)). The wrapper changes
//! nothing but the error: when the inner expression fails with one of the classes Spark throws
//! with a context and the error has none yet, the span is attached. The innermost wrapper runs
//! first, so the context is the failing expression's own, as in Spark.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::ErrorQueryContext;
use krishiv_common::redact::RedactedSql;
use krishiv_plan::{Expr, Origin};

use super::{EvalEnv, PhysExpr, Value};
use crate::error::Result;

/// The error classes Spark raises with the query context of the failing expression
/// (`QueryExecutionErrors` methods that take a `QueryContext`). Other run-time errors carry none.
const WITH_CONTEXT: &[&str] = &[
    "ARITHMETIC_OVERFLOW",
    "CAST_INVALID_INPUT",
    "CAST_OVERFLOW",
    "CAST_OVERFLOW_IN_TABLE_INSERT",
    "DIVIDE_BY_ZERO",
    "INTERVAL_ARITHMETIC_OVERFLOW",
    "INTERVAL_DIVIDED_BY_ZERO",
    "INVALID_ARRAY_INDEX",
    "INVALID_ARRAY_INDEX_IN_ELEMENT_AT",
    "INVALID_INDEX_OF_ZERO",
    "MAP_KEY_DOES_NOT_EXIST",
    "NUMERIC_VALUE_OUT_OF_RANGE",
];

/// An expression whose run-time errors carry its SQL span.
#[derive(Debug)]
pub(super) struct Contextual {
    inner: Arc<dyn PhysExpr>,
    text: RedactedSql,
    start: usize,
    end: usize,
}

/// `inner`, wrapped to attach the SQL context of `e` to its run-time errors when the plan
/// carries the text `e`'s origin points into; else `inner` itself.
pub(super) fn wrap(e: &Expr, inner: Arc<dyn PhysExpr>, env: &EvalEnv) -> Arc<dyn PhysExpr> {
    let Origin {
        sql: Some((source, span)),
        ..
    } = e.origin
    else {
        return inner;
    };
    let Some(text) = env.sql_texts.iter().find(|t| t.id == source.0) else {
        return inner;
    };
    let (start, end) = (span.start as usize, span.end as usize);
    // An empty span has nothing to quote (`fragment` clamps a span past its text).
    if start >= end {
        return inner;
    }
    Arc::new(Contextual {
        inner,
        text: text.text.clone(),
        start,
        end,
    })
}

impl PhysExpr for Contextual {
    fn data_type(&self) -> &ArrowType {
        self.inner.data_type()
    }

    fn nullable(&self) -> bool {
        self.inner.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        self.inner.eval(batch, partition).map_err(|e| {
            if e.query_contexts().is_empty() && WITH_CONTEXT.contains(&e.error_class().name()) {
                e.with_query_context(ErrorQueryContext::sql(
                    self.text.fragment(self.start..self.end),
                    self.start,
                    self.end.saturating_sub(1),
                ))
            } else {
                e
            }
        })
    }

    fn is_constant(&self) -> bool {
        self.inner.is_constant()
    }

    fn null_on_overflow(&self) -> Option<Arc<dyn PhysExpr>> {
        self.inner.null_on_overflow()
    }
}
