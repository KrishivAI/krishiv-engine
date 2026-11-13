// SPDX-License-Identifier: Apache-2.0
//! `AnalyzePlan` (design P3-T011a §3.5; Spark's
//! `SparkConnectAnalyzeHandler`, v4.3.0 lines 60-257): the plan is
//! translated as for `ExecutePlan` (`execute::request_plan`) and analyzed,
//! never run. One answer per arm of the request's `analyze` oneof:
//!
//! - `schema`: the analyzed schema, the same `DataType` `ExecutePlan` sends;
//! - `explain`: `DataFrame::explain` in the requested mode;
//! - `tree_string`: Spark's `StructType.treeString` (`analyze::types`);
//! - `is_local`: whether the analyzed root is an inline table (Spark's
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
//! - `persist`, `unpersist`, `get_storage_level`: refused (56K00) until the
//!   cache arms land (design P3-T083, PR-47).

pub mod types;

use std::collections::BTreeSet;
use std::sync::Arc;

use krishiv_common::error::{
    ConnectInvalidPlanInvalidEnum, ConnectInvalidPlanInvalidOneOfFieldNotSupported,
    ConnectInvalidPlanUnknownAnalyzeMethod, KrishivError,
};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::analyze_plan_request::Analyze;
use krishiv_connect::proto::v4_3::analyze_plan_request::explain::ExplainMode as ProtoMode;
use krishiv_connect::proto::v4_3::analyze_plan_response as res;
use krishiv_plan::{LogicalPlan, PlanNode, ReadSource};
use krishiv_session::{DataFrame, ExplainMode};

use crate::execute::{request_plan, schema};
use crate::manager::SessionEntry;
use crate::operation::ExecuteLimits;

/// Answers `request` on the session `entry`.
///
/// # Errors
/// The plan's decoding, translation and analysis errors;
/// `CONNECT_INVALID_PLAN.UNKNOWN_ANALYZE_METHOD` for an unset arm;
/// `CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SUPPORTED` for the cache
/// arms; `CONNECT_INVALID_PLAN.INVALID_ENUM` for an unspecified explain mode
/// (Spark throws a classless `UnsupportedOperationException`: a recorded
/// difference).
pub async fn analyze_plan(
    entry: &Arc<SessionEntry>,
    request: &pb::AnalyzePlanRequest,
    limits: &ExecuteLimits,
) -> Result<pb::AnalyzePlanResponse, KrishivError> {
    let result = analyze(entry, request.analyze.as_ref(), limits).await?;
    Ok(pb::AnalyzePlanResponse {
        session_id: request.session_id.clone(),
        server_side_session_id: entry.server_side_id().to_owned(),
        result: Some(result),
    })
}

/// The request's plan as a `DataFrame` (translated, not analyzed). The
/// plan is borrowed, never cloned: a deep plan's copy (and its drop) would
/// recurse on the async worker's stack (translation runs a deep plan on a
/// big stack of its own).
async fn frame(
    session: &krishiv_session::Session,
    plan: Option<&pb::Plan>,
    limits: &ExecuteLimits,
) -> Result<DataFrame, KrishivError> {
    let empty = pb::Plan::default();
    let logical = request_plan(session, plan.unwrap_or(&empty), limits).await?;
    Ok(session.from_plan(&logical))
}

async fn analyze(
    entry: &SessionEntry,
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
            let df = frame(session, s.plan.as_ref(), limits).await?;
            res::Result::Schema(res::Schema {
                schema: Some(schema::proto_schema(&*df.schema().await?)?),
            })
        }
        Some(Analyze::Explain(e)) => {
            // Spark builds the Dataset (translation, analysis) before it
            // reads the mode.
            let df = frame(session, e.plan.as_ref(), limits).await?;
            df.analyzed().await?;
            let mode = explain_mode(e.explain_mode)?;
            res::Result::Explain(res::Explain {
                explain_string: df.explain(mode).await?,
            })
        }
        Some(Analyze::TreeString(t)) => {
            let df = frame(session, t.plan.as_ref(), limits).await?;
            let spark = schema::spark_schema(&*df.schema().await?)?;
            res::Result::TreeString(res::TreeString {
                tree_string: types::tree_string(&spark, t.level),
            })
        }
        Some(Analyze::IsLocal(l)) => {
            // Spark: the analyzed plan is a `LocalRelation`. Krishiv's
            // analyzer lowers inline rows to `OneRowRelation` (as for
            // `SELECT 1`), so the question is asked of the translated plan
            // (an inline table, as `LocalRelation` will translate in T012);
            // SQL `VALUES` is SQL text until analysis and answers false (a
            // known difference). The plan is still analyzed so that an
            // invalid one fails as in Spark.
            let empty = pb::Plan::default();
            let logical = request_plan(session, l.plan.as_ref().unwrap_or(&empty), limits).await?;
            let is_local = matches!(logical.root().node, PlanNode::InlineTable { .. });
            session.from_plan(&logical).analyzed().await?;
            res::Result::IsLocal(res::IsLocal { is_local })
        }
        Some(Analyze::IsStreaming(s)) => {
            // Analyzed so that an invalid plan fails as in Spark.
            frame(session, s.plan.as_ref(), limits)
                .await?
                .analyzed()
                .await?;
            res::Result::IsStreaming(res::IsStreaming {
                is_streaming: false,
            })
        }
        Some(Analyze::InputFiles(f)) => {
            let analyzed = frame(session, f.plan.as_ref(), limits)
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
                semantic_hash(frame(session, s.target_plan.as_ref(), limits).await?).await?;
            let other = semantic_hash(frame(session, s.other_plan.as_ref(), limits).await?).await?;
            res::Result::SameSemantics(res::SameSemantics {
                result: target == other,
            })
        }
        Some(Analyze::SemanticHash(s)) => {
            let hash = semantic_hash(frame(session, s.plan.as_ref(), limits).await?).await?;
            res::Result::SemanticHash(res::SemanticHash {
                result: fold_hash(hash),
            })
        }
        Some(Analyze::Persist(_)) => return Err(cache_arm("PERSIST", 14)),
        Some(Analyze::Unpersist(_)) => return Err(cache_arm("UNPERSIST", 15)),
        Some(Analyze::GetStorageLevel(_)) => return Err(cache_arm("GET_STORAGE_LEVEL", 16)),
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

/// The refusal of a cache arm until design P3-T083 (PR-47).
fn cache_arm(name: &str, number: u32) -> KrishivError {
    ConnectInvalidPlanInvalidOneOfFieldNotSupported {
        full_name: "spark.connect.AnalyzePlanRequest".to_owned(),
        name: name.to_owned(),
        number: number.to_string(),
    }
    .into()
}

/// The semantic hash of `df`'s analyzed plan.
async fn semantic_hash(df: DataFrame) -> Result<u128, KrishivError> {
    Ok(krishiv_plan::semantic_plan_hash(&df.analyzed().await?.plan))
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
