// SPDX-License-Identifier: Apache-2.0
//! `AnalyzePlan` (design P3-T011a §3.5; Spark's
//! `SparkConnectAnalyzeHandler`, v4.3.0 lines 60-257): the plan is
//! translated as for `ExecutePlan` (`execute::request_plan`) and analyzed,
//! never run. One answer per arm of the request's `analyze` oneof:
//!
//! - `schema`: the analyzed schema, the same `DataType` `ExecutePlan` sends;
//! - `explain`: `DataFrame::explain` in the requested mode;
//! - `tree_string`: Spark's `StructType.treeString` (`analyze::types`);
//! - `is_local`: whether the translated root is a local relation (Spark's
//!   `LocalRelation`); `is_streaming`: always false (no streaming sources);
//! - `input_files`: the paths of the file scans, as written;
//! - `spark_version`: the session profile's release (`4.2.0` for 4.2);
//! - `ddl_parse`, `json_to_ddl`: Spark's type parsers and `toDDL`;
//! - `same_semantics`, `semantic_hash`: the semantic hash of the analyzed
//!   plans (`krishiv_plan::semantic_plan_hash`: attribute ids renumbered,
//!   origins, column and alias names and subquery aliases left out), Spark's
//!   canonicalized `sameResult` and `semanticHash`; the hash is Krishiv's own
//!   value, and Spark's other rewrites (commutative operands, renaming
//!   projects) are not made;
//! - `persist`, `unpersist`, `get_storage_level`: the frame's `persist` (or `cache` without a
//!   level), `unpersist` and `storage_level` (design P3-T083 §3.4; the level converter is
//!   `crate::cache`); `blocking` is accepted and ignored (removal is synchronous).
//!
//! Access (ADR 0046 §3b): `schema`, `tree_string`, `is_local`, `is_streaming`,
//! `semantic_hash` and `same_semantics` analyze for metadata
//! (`DataFrame::schema_for_metadata` / `analyzed_for_metadata`): every catalog
//! table and view the plan reads needs `Browse`, so a viewer may ask them. A
//! read by path keeps `Select`: its schema is inferred from file contents (CSV
//! `header=true` column names are the file's first line).
//! `input_files` lists paths and keeps `Select`. `explain`
//! (`DataFrame::explain_for_metadata`, P3 PR-14d) is the ordinary text for a
//! principal whose `Select` check passes; otherwise the plan chosen with no
//! table statistics and no distinct-value samples, with a note saying so.
//! Its `cost` mode, a read by path and a permanent view keep `Select` (they
//! show statistics, contents or a view's text). `input_files`,
//! `semantic_hash` and `same_semantics` list or hash the analyzed plan, so a
//! `SQL SECURITY DEFINER` view's inlined query in it is asked of the reader
//! as `EXPLAIN` asks it (`DataFrame::analyzed`, ADR 0048 A1 §3, review r3
//! X2); `schema`, `is_local`, `is_streaming` and `explain` validate with
//! `schema_for_metadata`, which shows no plan.

pub mod types;

use std::collections::BTreeSet;
use std::sync::Arc;

use krishiv_common::error::{
    ConnectInvalidPlanInvalidEnum, ConnectInvalidPlanUnknownAnalyzeMethod, KrishivError,
};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::analyze_plan_request::Analyze;
use krishiv_connect::proto::v4_3::analyze_plan_request::explain::ExplainMode as ProtoMode;
use krishiv_connect::proto::v4_3::analyze_plan_response as res;
use krishiv_plan::{LogicalPlan, PlanNode, ReadSource};
use krishiv_session::{DataFrame, ExplainMode, QueryOptions};

use crate::cache::{proto_of, storage_level_of};
use crate::execute::{request_plan, request_relation, schema};
use crate::manager::SessionEntry;
use crate::operation::ExecuteLimits;

/// Answers `request` on the session `entry`; `options` carry the RPC's own authentication, so
/// a delegated catalog's metadata is read with this request's token (P3-T067 §3.3).
///
/// # Errors
/// The plan's decoding, translation and analysis errors;
/// `CONNECT_INVALID_PLAN.UNKNOWN_ANALYZE_METHOD` for an unset arm;
/// `CONNECT_INVALID_PLAN.INVALID_ENUM` for an unspecified explain mode
/// (Spark throws a classless `UnsupportedOperationException`: a recorded
/// difference).
pub async fn analyze_plan(
    entry: &Arc<SessionEntry>,
    request: &pb::AnalyzePlanRequest,
    limits: &ExecuteLimits,
    options: &QueryOptions,
) -> Result<pb::AnalyzePlanResponse, KrishivError> {
    let result = analyze(entry, options, request.analyze.as_ref(), limits).await?;
    Ok(pb::AnalyzePlanResponse {
        session_id: request.session_id.clone(),
        server_side_session_id: entry.server_side_id().to_owned(),
        result: Some(result),
    })
}

/// The request's relation (`persist`, `unpersist`, `get_storage_level`) as a `DataFrame`.
async fn frame_of_relation(
    entry: &SessionEntry,
    options: &QueryOptions,
    relation: Option<&pb::Relation>,
) -> Result<DataFrame, KrishivError> {
    let empty = pb::Relation::default();
    request_relation(entry, relation.unwrap_or(&empty))
        .await?
        .frame(entry.session(), options.clone())
        .await
}

/// The request's plan as a `DataFrame` (translated, not analyzed). The
/// plan is borrowed, never cloned: a deep plan's copy (and its drop) would
/// recurse on the async worker's stack (translation runs a deep plan on a
/// big stack of its own).
async fn frame(
    entry: &SessionEntry,
    options: &QueryOptions,
    plan: Option<&pb::Plan>,
    limits: &ExecuteLimits,
) -> Result<DataFrame, KrishivError> {
    let empty = pb::Plan::default();
    request_plan(entry, plan.unwrap_or(&empty), limits)
        .await?
        .frame(entry.session(), options.clone())
        .await
}

async fn analyze(
    entry: &SessionEntry,
    options: &QueryOptions,
    analyze: Option<&Analyze>,
    limits: &ExecuteLimits,
) -> Result<res::Result, KrishivError> {
    let session = entry.session();
    Ok(match analyze {
        None => {
            return Err(ConnectInvalidPlanUnknownAnalyzeMethod {
                other: "ANALYZE_NOT_SET".to_owned(),
            }
            .into());
        }
        Some(Analyze::Schema(s)) => {
            let df = frame(entry, options, s.plan.as_ref(), limits).await?;
            res::Result::Schema(res::Schema {
                schema: Some(schema::proto_schema(&*df.schema_for_metadata().await?)?),
            })
        }
        Some(Analyze::Explain(e)) => {
            // Spark builds the Dataset (translation, analysis) before it
            // reads the mode: a metadata analysis here (`Browse`), then the
            // explain a principal may see (see the module docs).
            let df = frame(entry, options, e.plan.as_ref(), limits).await?;
            df.schema_for_metadata().await?;
            let mode = explain_mode(e.explain_mode)?;
            res::Result::Explain(res::Explain {
                explain_string: df.explain_for_metadata(mode).await?,
            })
        }
        Some(Analyze::TreeString(t)) => {
            let df = frame(entry, options, t.plan.as_ref(), limits).await?;
            let spark = schema::spark_schema(&*df.schema_for_metadata().await?)?;
            res::Result::TreeString(res::TreeString {
                tree_string: types::tree_string(&spark, t.level),
            })
        }
        Some(Analyze::IsLocal(l)) => {
            // Spark: the analyzed plan is a `LocalRelation`. A `local_relation` translates to
            // a `Read` of a local relation (its user schema already applied eagerly, so the
            // analyzed plan is that bare relation either way); Krishiv's analyzer lowers inline
            // rows to `OneRowRelation` (as for `SELECT 1`), so the question is asked of the
            // translated plan; SQL `VALUES` is SQL text until analysis and answers false. The
            // plan is still analyzed so that an invalid one fails as in Spark.
            let empty = pb::Plan::default();
            let requested = request_plan(entry, l.plan.as_ref().unwrap_or(&empty), limits).await?;
            let is_local = matches!(
                requested.plan.root().node,
                PlanNode::InlineTable { .. } | PlanNode::Read(ReadSource::Local(_))
            );
            requested
                .frame(session, options.clone())
                .await?
                .schema_for_metadata()
                .await?;
            res::Result::IsLocal(res::IsLocal { is_local })
        }
        Some(Analyze::IsStreaming(s)) => {
            // Analyzed so that an invalid plan fails as in Spark.
            frame(entry, options, s.plan.as_ref(), limits)
                .await?
                .schema_for_metadata()
                .await?;
            res::Result::IsStreaming(res::IsStreaming {
                is_streaming: false,
            })
        }
        Some(Analyze::InputFiles(f)) => {
            // The paths themselves are shown: `Select` on them (ADR 0046 §3b).
            let analyzed = frame(entry, options, f.plan.as_ref(), limits)
                .await?
                .analyzed()
                .await?;
            let mut files = BTreeSet::new();
            input_files(analyzed.plan.root(), &mut files);
            res::Result::InputFiles(res::InputFiles {
                files: files.into_iter().collect(),
            })
        }
        Some(Analyze::SparkVersion(_)) => res::Result::SparkVersion(res::SparkVersion {
            version: format!("{}.0", session.spark_profile().line()),
        }),
        Some(Analyze::DdlParse(d)) => res::Result::DdlParse(res::DdlParse {
            parsed: Some(schema::proto_type(&types::parse_datatype_string(
                &d.ddl_string,
            )?)?),
        }),
        Some(Analyze::SameSemantics(s)) => {
            let target =
                semantic_hash(frame(entry, options, s.target_plan.as_ref(), limits).await?).await?;
            let other =
                semantic_hash(frame(entry, options, s.other_plan.as_ref(), limits).await?).await?;
            res::Result::SameSemantics(res::SameSemantics {
                result: target == other,
            })
        }
        Some(Analyze::SemanticHash(s)) => {
            let hash = semantic_hash(frame(entry, options, s.plan.as_ref(), limits).await?).await?;
            res::Result::SemanticHash(res::SemanticHash {
                result: fold_hash(hash),
            })
        }
        Some(Analyze::Persist(p)) => {
            crate::auth::enforce(&crate::auth::Check::SideEffect, session).await?;
            // `persist(level)`, or `persist()` (the session's default level) without one.
            let df = frame_of_relation(entry, options, p.relation.as_ref()).await?;
            let df = match &p.storage_level {
                Some(level) => df.persist(storage_level_of(level)),
                None => df.cache(),
            };
            // Spark registers at once; the engine does at the first read of the registry: this
            // analysis (an invalid plan fails here as in Spark).
            df.is_cached().await?;
            res::Result::Persist(res::Persist {})
        }
        Some(Analyze::Unpersist(u)) => {
            crate::auth::enforce(&crate::auth::Check::SideEffect, session).await?;
            // `blocking` is accepted: removal is synchronous either way.
            frame_of_relation(entry, options, u.relation.as_ref())
                .await?
                .unpersist()
                .await?;
            res::Result::Unpersist(res::Unpersist {})
        }
        Some(Analyze::GetStorageLevel(g)) => {
            let level = frame_of_relation(entry, options, g.relation.as_ref())
                .await?
                .storage_level()
                .await?;
            res::Result::GetStorageLevel(res::GetStorageLevel {
                storage_level: Some(proto_of(level)),
            })
        }
        Some(Analyze::JsonToDdl(j)) => res::Result::JsonToDdl(res::JsonToDdl {
            ddl_string: types::json_to_ddl(&j.json_string)?,
        }),
    })
}

/// The engine's mode for a request's explain mode.
fn explain_mode(mode: i32) -> Result<ExplainMode, KrishivError> {
    match ProtoMode::try_from(mode) {
        Ok(ProtoMode::Simple) => Ok(ExplainMode::Simple),
        Ok(ProtoMode::Extended) => Ok(ExplainMode::Extended),
        Ok(ProtoMode::Codegen) => Ok(ExplainMode::Codegen),
        Ok(ProtoMode::Cost) => Ok(ExplainMode::Cost),
        Ok(ProtoMode::Formatted) => Ok(ExplainMode::Formatted),
        Ok(ProtoMode::Unspecified) | Err(_) => Err(ConnectInvalidPlanInvalidEnum {
            full_name: "spark.connect.AnalyzePlanRequest.Explain.ExplainMode".to_owned(),
            name: ProtoMode::try_from(mode)
                .map_or("UNRECOGNIZED", |m| m.as_str_name())
                .to_owned(),
            number: mode.to_string(),
        }
        .into()),
    }
}

/// The semantic hash of `df`'s analyzed plan (a metadata analysis: `Browse`).
async fn semantic_hash(df: DataFrame) -> Result<u128, KrishivError> {
    Ok(krishiv_plan::semantic_plan_hash(
        &df.analyzed_for_metadata().await?.plan,
    ))
}

/// A 128-bit hash folded into the `int32` the response carries.
fn fold_hash(hash: u128) -> i32 {
    let folded = (0..4).fold(0_u32, |acc, i| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "takes one 32-bit quarter of the hash"
        )]
        let quarter = (hash >> (32 * i)) as u32;
        acc ^ quarter
    });
    i32::from_ne_bytes(folded.to_ne_bytes())
}

/// The paths of the file scans under `plan` (Spark's `inputFiles`
/// collects the files of its file relations; Krishiv lists the paths as
/// written, not the files under them: a recorded difference). A path with
/// credentials prints them redacted.
fn input_files(plan: &Arc<LogicalPlan>, out: &mut BTreeSet<String>) {
    if let PlanNode::Scan(scan) = &plan.node
        && let ReadSource::Files { paths, .. } = &scan.source
    {
        out.extend(paths.iter().map(ToString::to_string));
    }
    plan.for_each_child(&mut |child| input_files(child, out));
}

#[cfg(test)]
mod tests {
    use super::fold_hash;

    #[test]
    fn the_hash_folds_every_quarter() {
        assert_eq!(fold_hash(0), 0);
        assert_eq!(fold_hash(1 | (1 << 32)), 0);
        assert_eq!(fold_hash(u128::from(u32::MAX)), -1);
        assert_eq!(fold_hash(5 << 96), 5);
    }
}
