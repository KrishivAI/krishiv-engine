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
//! | `Project`, `Filter`, `Sort`, `Limit`, `Offset` | `Project`, `Filter`, `Sort`, `Limit`, `Offset` |
//! | `Tail` | `Tail` |
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
//! use krishiv_connect::proto::v4_3::{
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

mod cached;
mod catalog;
mod errors;
mod expression;
mod literal;
mod local;
mod names;
mod relation;
mod relation_udf;
mod schema;
mod types;
mod udf;
mod udtf;
mod write;

pub use udf::{
    CheckpointSpec, Command, PythonUdf, SQL_BATCHED_UDF, UdfCallName, UdfHost, python_udf,
    translate_command,
};
pub use udtf::{PythonUdtf, python_udtf};

pub use write::{WriteSpec, WriteTarget, WriteV2Spec, write_spec, write_v2_spec};

pub use cached::{CACHED_REMOTE_NAMESPACE, CachedRemoteIds, cached_remote_table};
pub use catalog::{CatalogCall, CreateTable, catalog_call};
pub use local::{LocalRelations, LocalRequest, cached_request};
pub use schema::{parse_datatype_string, parse_local_schema, parse_table_schema};

use krishiv_common::error::{FailedToParseTooComplex, KrishivError};
use krishiv_plan::{Expr, MAX_DEPTH, Plan};
use std::sync::Arc;

use crate::proto::v4_3 as pb;
use crate::proto::v4_3::plan::OpType;
use crate::stack::{INLINE_LEVELS, on_deep_stack};
use crate::wire;
use errors::{not_set, unsupported_field, unsupported_variant};

type Result<T> = std::result::Result<T, KrishivError>;

/// The session settings translation depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslateConf {
    /// The format of a read that names none: `spark.sql.sources.default`.
    pub default_data_source: String,
    /// The session settings that change how SQL text parses; an
    /// `ExpressionString` (`expr()`, `selectExpr`) is parsed with them.
    pub parser: krishiv_sql::ParserOptions,
    /// The local relations of the plan (`createDataFrame`), built by the host.
    pub local_relations: LocalRelations,
    /// Where the ids of the plan's cached remote relations are recorded (the host resolves them).
    pub cached_remote: CachedRemoteIds,
}

impl Default for TranslateConf {
    /// Spark 4.2's defaults.
    fn default() -> Self {
        Self {
            default_data_source: "parquet".to_owned(),
            parser: krishiv_sql::ParserOptions::default(),
            local_relations: LocalRelations::default(),
            cached_remote: CachedRemoteIds::default(),
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

/// Translates a `Plan` whose root is a relation. A `sql_command` translates as the `SQL` relation
/// it answers for a query ([`sql_command_relation`]); running it as a command, or not, is the
/// server's job (it needs the session).
///
/// # Errors
/// `INVALID_ONE_OF_FIELD_NOT_SUPPORTED` for any other command, `KRISHIV_UNSUPPORTED_FEATURE` for a compressed plan (the
/// server's job, P3), plus the errors of [`translate_relation`] and [`sql_command_relation`].
pub fn translate_plan(plan: &pb::Plan, conf: &TranslateConf) -> Result<Plan> {
    match &plan.op_type {
        Some(OpType::Root(relation)) => translate_relation(relation, conf),
        Some(OpType::Command(command)) => match &command.command_type {
            Some(pb::command::CommandType::SqlCommand(c)) => {
                translate_relation(&sql_command_relation(c)?, conf)
            }
            Some(arm) => Err(unsupported_variant("spark.connect.Command", arm.variant())),
            None => Err(not_set("spark.connect.Command", "COMMAND_TYPE_NOT_SET")),
        },
        Some(OpType::CompressedOperation(_)) => Err(unsupported_field(
            "Plan",
            "compressed_operation",
            "compressed plans",
        )),
        None => Err(not_set("spark.connect.Plan", "OP_TYPE_NOT_SET")),
    }
}

/// The `SQL` relation of a `SqlCommand`: what `spark.sql()`'s client re-sends when the statement
/// was a query. When `input` is set (every `spark.sql()` of a PySpark 4.x client sends it, the
/// other fields being deprecated) it is the relation, unchanged and with its `common` (Spark's
/// `handleSqlCommand` answers `command.getInput`); otherwise the deprecated `sql`, `args` and
/// `pos_args` fields are the relation's `query`, `args` and `pos_args`, the named and positional
/// expressions carrying over as they are.
///
/// The library form of a `sql_command` exists for the coverage tie and for the server, which
/// runs the command; `AnalyzePlan` of a command plan stays refused by the server.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for an `input` of `WithRelations` (SQL over DataFrame
/// arguments, Connect group 4); `CONNECT_INVALID_PLAN.SQL_COMMAND_EXPECTS_SQL_OR_WITH_RELATIONS`
/// for any other `input` relation.
pub fn sql_command_relation(command: &pb::SqlCommand) -> Result<pb::Relation> {
    if let Some(input) = &command.input {
        return match &input.rel_type {
            Some(pb::relation::RelType::Sql(_)) => Ok(input.clone()),
            Some(pb::relation::RelType::WithRelations(_)) => Err(unsupported_field(
                "SqlCommand",
                "input",
                "SQL over DataFrame arguments",
            )),
            other => Err(
                krishiv_common::error::ConnectInvalidPlanSqlCommandExpectsSqlOrWithRelations {
                    other: other.as_ref().map_or_else(
                        || "REL_TYPE_NOT_SET".to_owned(),
                        |r| r.variant().field.to_ascii_uppercase(),
                    ),
                }
                .into(),
            ),
        };
    }
    #[expect(
        deprecated,
        reason = "Spark still accepts the deprecated fields of older clients"
    )]
    let sql = pb::Sql {
        query: command.sql.clone(),
        args: command.args.clone(),
        pos_args: command.pos_args.clone(),
        named_arguments: command.named_arguments.clone(),
        pos_arguments: command.pos_arguments.clone(),
    };
    Ok(pb::Relation {
        common: None,
        rel_type: Some(pb::relation::RelType::Sql(sql)),
    })
}

/// Whether a `SQL` relation carries parameters (`args`, `pos_args`, named or positional
/// expressions).
#[must_use]
pub fn sql_has_parameters(sql: &pb::Sql) -> bool {
    #[expect(deprecated, reason = "older clients send the literal fields")]
    let literals = !sql.args.is_empty() || !sql.pos_args.is_empty();
    literals || !sql.named_arguments.is_empty() || !sql.pos_arguments.is_empty()
}

/// The refusal of a command (`Plan.command`) until the server translates
/// commands (P3-T012): Spark's `CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SUPPORTED`
/// naming the command's case (`SQL_COMMAND`, 56K00), or
/// `INVALID_ONE_OF_FIELD_NOT_SET` (`COMMANDTYPE_NOT_SET`) for an empty one.
#[must_use]
pub fn command_not_supported(command: &pb::Command) -> KrishivError {
    match &command.command_type {
        None => not_set("spark.connect.Command", "COMMANDTYPE_NOT_SET"),
        Some(c) => {
            let v = c.variant();
            krishiv_common::error::ConnectInvalidPlanInvalidOneOfFieldNotSupported {
                full_name: "spark.connect.Command".to_owned(),
                name: v.field.to_ascii_uppercase(),
                number: v.number.to_string(),
            }
            .into()
        }
    }
}

/// Translates a relation and everything below it.
///
/// # Errors
/// `INVALID_ONE_OF_FIELD_NOT_SUPPORTED` for an arm outside the supported subset, `KRISHIV_UNSUPPORTED_FEATURE` for a field outside the
/// supported subset, `CONNECT_INVALID_PLAN.*` for an invalid plan,
/// `FAILED_TO_PARSE_TOO_COMPLEX` past the depth cap, and
/// `CONNECT_INVALID_PLAN.PLAN_SIZE_LARGER_THAN_MAX` past the node cap.
pub fn translate_relation(relation: &pb::Relation, conf: &TranslateConf) -> Result<Plan> {
    let root = Translator::new(conf).relation(relation)?;
    Plan::try_new(root)
}

/// [`translate_relation`] with the Python UDFs of the plan, inline calls and relations (`MapPartitions`,
/// `GroupMap`, `CoGroupMap`), sent to `udfs` (a server's session), which names the function each call
/// resolves to and the reference each relation runs.
///
/// # Errors
/// As [`translate_relation`], plus what `udfs` refuses, the errors of [`python_udf`] and the refusals of the
/// relation arms (`relation_udf`).
pub fn translate_relation_with(
    relation: &pb::Relation,
    conf: &TranslateConf,
    udfs: &mut dyn UdfHost,
) -> Result<Plan> {
    let root = Translator::with_udfs(conf, udfs).relation(relation)?;
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
    /// Where Python UDFs (inline calls and relations) go; `None` refuses them.
    udfs: Option<&'c mut dyn UdfHost>,
    /// Relations and expressions being translated above the current one.
    depth: usize,
    /// Relations and expressions translated so far.
    nodes: usize,
    /// Threads started so far for subtrees past [`INLINE_LEVELS`].
    hops: usize,
    /// Local relations met so far: the ordinal of the next one's request.
    locals: usize,
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
            udfs: None,
            depth: 0,
            nodes: 0,
            hops: 0,
            locals: 0,
        }
    }

    fn with_udfs(conf: &'c TranslateConf, udfs: &'c mut dyn UdfHost) -> Self {
        Self {
            udfs: Some(udfs),
            ..Self::new(conf)
        }
    }

    /// The function a call of the inline Python UDF `udf` resolves to.
    fn inline_python_udf(&mut self, udf: &PythonUdf) -> Result<UdfCallName> {
        match self.udfs.as_deref_mut() {
            Some(host) => host.inline_python_udf(udf),
            None => Err(udf::no_host()),
        }
    }

    /// The reference the Python UDF `udf` of a relation runs as.
    fn relation_python_udf(&mut self, udf: &PythonUdf) -> Result<krishiv_plan::PythonUdfRef> {
        match self.udfs.as_deref_mut() {
            Some(host) => host.relation_python_udf(udf),
            None => Err(udf::no_host()),
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
