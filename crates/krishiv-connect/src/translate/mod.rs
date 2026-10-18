// SPDX-License-Identifier: Apache-2.0
//! Translation of a Spark Connect plan into the logical-plan IR
//! ([`krishiv_plan`]), at the API boundary: the protos never reach the
//! analyzer (ADR 0003).
//!
//! The semantic reference is Spark's `SparkConnectPlanner` (4.2.0). The
//! supported subset is what `read.parquet`, `sql`, `select`, `filter`,
//! `groupBy().agg`, `join`, `orderBy`, `limit`, `withColumn`, `alias`,
//! `toArrow` and `toPandas` send:
//!
//! | Connect | IR |
//! |---|---|
//! | `Read.named_table` | `Read` (table, options) |
//! | `Read.data_source` | `Read` (files: format, paths, options) |
//! | `SQL` | `Sql` (query, parameters) |
//! | `Project`, `Filter`, `Sort`, `Limit` | `Project`, `Filter`, `Sort`, `Limit` |
//! | `Join` (all seven types, `using_columns`) | `Join` |
//! | `Aggregate` (group by, rollup, cube, grouping sets) | `Aggregate` |
//! | `SubqueryAlias` | `SubqueryAlias` |
//! | `WithColumns` | `WithColumns` |
//! | `Expression.literal`, `unresolved_attribute`, `unresolved_function`, `unresolved_star`, `alias`, `cast` | `Literal`, `Column`, `Function`, `Star`, `Alias`, `Cast` |
//!
//! `toArrow` and `toPandas` are on the result side: PySpark sends the
//! DataFrame's own plan, `Plan { root: <the DataFrame's relation> }`, and
//! asks for Arrow batches in the request's options, so the plan root they
//! produce is the one of the DataFrame they were called on.
//!
//! Anything else fails with a clear error naming the proto message and
//! field: `KRISHIV_UNSUPPORTED_FEATURE` for a valid plan Krishiv cannot run
//! yet (other relations, SQL expression strings, nested-type literals), and
//! Spark's own `CONNECT_INVALID_PLAN.*` classes for an invalid plan.
//!
//! ```
//! use krishiv_connect::proto::v4_2::{
//!     Plan, Read, Relation, plan::OpType, read::{DataSource, ReadType}, relation::RelType,
//! };
//! use krishiv_connect::translate::{TranslateConf, translate_plan};
//!
//! // spark.read.parquet("/data/t.parquet"), as the Connect client sends it.
//! let read = Read {
//!     is_streaming: false,
//!     read_type: Some(ReadType::DataSource(DataSource {
//!         format: Some("parquet".into()),
//!         paths: vec!["/data/t.parquet".into()],
//!         ..DataSource::default()
//!     })),
//! };
//! let plan = Plan {
//!     op_type: Some(OpType::Root(Box::new(Relation {
//!         common: None,
//!         rel_type: Some(RelType::Read(read)),
//!     }))),
//! };
//! let ir = translate_plan(&plan, &TranslateConf::default())?;
//! assert_eq!(ir.to_string(), "Read parquet [/data/t.parquet]\n");
//! # Ok::<(), krishiv_common::error::KrishivError>(())
//! ```
//!
//! # Limits
//!
//! A plan from [`plan_from_bytes`] has been bounded by [`crate::wire`]. A
//! proto handed to [`translate_plan`] directly is bounded here too: the
//! translator refuses to nest deeper than the IR does
//! ([`krishiv_plan::MAX_DEPTH`]) and to translate more than
//! [`crate::wire::MAX_MESSAGES`] nodes, and it continues on a thread with a
//! big stack past a few levels, so no input overflows the caller's stack.
//!
//! # Secrets
//!
//! Option values and paths of a read go through
//! [`krishiv_common::secret::OptionValue::from_user`], so a credential never
//! enters the IR as plain text. The `Debug` of the proto types prints no
//! string, byte string or map (`crate::proto`), and a decoded plan wipes its
//! option values and paths when it is dropped ([`crate::proto::Scrub`]).

mod errors;
mod expression;
mod literal;
mod names;
mod relation;
mod types;

use krishiv_common::error::{FailedToParseTooComplex, KrishivError};
use krishiv_plan::{Expr, MAX_DEPTH, Plan};
use std::sync::Arc;

use crate::proto::v4_2 as pb;
use crate::proto::v4_2::plan::OpType;
use crate::stack::{INLINE_LEVELS, on_deep_stack};
use crate::wire;
use errors::{not_set, unsupported_field};

type Result<T> = std::result::Result<T, KrishivError>;

/// The session settings translation depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslateConf {
    /// The format of a read that names none: `spark.sql.sources.default`.
    pub default_data_source: String,
}

impl Default for TranslateConf {
    /// Spark 4.2's defaults.
    fn default() -> Self {
        Self {
            default_data_source: "parquet".to_owned(),
        }
    }
}

/// Decodes a `Plan` from bytes ([`wire::decode_plan`]) and translates it.
///
/// # Errors
/// The errors of [`wire::decode_plan`] and [`translate_plan`].
pub fn plan_from_bytes(bytes: &[u8], conf: &TranslateConf) -> Result<Plan> {
    translate_plan(&*wire::decode_plan(bytes)?, conf)
}

/// Decodes a `Relation` from bytes and translates it.
///
/// # Errors
/// The errors of [`wire::decode_relation`] and [`translate_relation`].
pub fn relation_from_bytes(bytes: &[u8], conf: &TranslateConf) -> Result<Plan> {
    translate_relation(&*wire::decode_relation(bytes)?, conf)
}

/// Decodes an `Expression` from bytes and translates it.
///
/// # Errors
/// The errors of [`wire::decode_expression`] and [`translate_expression`].
pub fn expression_from_bytes(bytes: &[u8], conf: &TranslateConf) -> Result<Arc<Expr>> {
    translate_expression(&*wire::decode_expression(bytes)?, conf)
}

/// Translates a `Plan` whose root is a relation.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a command or a compressed plan (the
/// server's job, P3), plus the errors of [`translate_relation`].
pub fn translate_plan(plan: &pb::Plan, conf: &TranslateConf) -> Result<Plan> {
    match &plan.op_type {
        Some(OpType::Root(relation)) => translate_relation(relation, conf),
        Some(OpType::Command(_)) => Err(unsupported_field(
            "Plan",
            "command",
            "commands (writes, SQL commands, catalog operations)",
        )),
        Some(OpType::CompressedOperation(_)) => Err(unsupported_field(
            "Plan",
            "compressed_operation",
            "compressed plans",
        )),
        None => Err(not_set("spark.connect.Plan", "OP_TYPE_NOT_SET")),
    }
}

/// Translates a relation and everything below it.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a relation or field outside the
/// supported subset, `CONNECT_INVALID_PLAN.*` for an invalid plan,
/// `FAILED_TO_PARSE_TOO_COMPLEX` past the depth cap, and
/// `CONNECT_INVALID_PLAN.PLAN_SIZE_LARGER_THAN_MAX` past the node cap.
pub fn translate_relation(relation: &pb::Relation, conf: &TranslateConf) -> Result<Plan> {
    let root = Translator::new(conf).relation(relation)?;
    Plan::try_new(root)
}

/// Translates an expression.
///
/// # Errors
/// As [`translate_relation`].
pub fn translate_expression(
    expression: &pb::Expression,
    conf: &TranslateConf,
) -> Result<Arc<Expr>> {
    Translator::new(conf).expression(expression)
}

/// The state of one translation.
pub(crate) struct Translator<'c> {
    conf: &'c TranslateConf,
    /// Relations and expressions being translated above the current one.
    depth: usize,
    /// Relations and expressions translated so far.
    nodes: usize,
    /// Threads started so far for subtrees past [`INLINE_LEVELS`].
    hops: usize,
}

/// The most threads one translation starts for deep subtrees. A plan needs
/// one per subtree that reaches [`INLINE_LEVELS`] levels and is itself
/// composite (a plan with a thousand such columns is unusual), and the cap
/// keeps a hostile plan of millions of them from spending its time on
/// thread creation.
const MAX_HOPS: usize = 10_000;

impl<'c> Translator<'c> {
    const fn new(conf: &'c TranslateConf) -> Self {
        Self {
            conf,
            depth: 0,
            nodes: 0,
            hops: 0,
        }
    }

    /// Counts a node against the budget.
    fn count(&mut self) -> Result<()> {
        self.nodes += 1;
        if self.nodes > wire::MAX_MESSAGES {
            return Err(wire::too_many_messages());
        }
        Ok(())
    }

    /// Runs `f` one level deeper. Past [`MAX_DEPTH`] levels the plan is too
    /// complex; at [`INLINE_LEVELS`] the subtree continues on a thread with
    /// a big stack (the levels below it continue there).
    fn nested<T: Send>(&mut self, f: impl FnOnce(&mut Self) -> Result<T> + Send) -> Result<T> {
        if self.depth >= MAX_DEPTH {
            return Err(FailedToParseTooComplex {}.into());
        }
        self.depth += 1;
        let result = if self.depth == INLINE_LEVELS {
            self.hops += 1;
            if self.hops > MAX_HOPS {
                self.depth -= 1;
                return Err(FailedToParseTooComplex {}.into());
            }
            on_deep_stack(|| f(self)).and_then(|r| r)
        } else {
            f(self)
        };
        self.depth -= 1;
        result
    }
}

#[cfg(test)]
mod testing;
#[cfg(test)]
mod tests;
