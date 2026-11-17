// SPDX-License-Identifier: Apache-2.0
//! `input_file_name()` and the block functions: the source count of Spark's `PreReadCheck` at analysis and
//! [`bind_input_files`], which reads them from the `_metadata` column of the file scan in the call's pipeline
//! (end-to-end goldens: `krishiv-session/tests/input_file_goldens.rs`).

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::OptionValue;
use krishiv_types::DataType;

use super::dsl::{c, f, s};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisError, AnalyzedPlan, Analyzer as _, BoxFuture, DefaultAnalyzer, FileMetadata,
    FileSchemaResolver, MemoryCatalog, ResolutionConf, SourceKind, TableColumn, bind_input_files,
};
use crate::{
    Builtin, Expr, ExprNode, JoinCondition, JoinKind, Literal, LogicalPlan, Origin, Plan, PlanNode,
    ReadSource, ResolvedExpr, SubqueryKind,
};

/// Every path holds one `id INT` column.
struct Files;

impl FileSchemaResolver for Files {
    fn source_kind(&self, _: &str) -> SourceKind {
        SourceKind::File
    }

    fn infer<'a>(
        &'a self,
        _: &'a str,
        _: &'a [OptionValue],
        _: &'a BTreeMap<String, OptionValue>,
    ) -> BoxFuture<'a, Result<Vec<TableColumn>, KrishivError>> {
        Box::pin(std::future::ready(Ok(vec![TableColumn {
            name: "id".to_owned(),
            data_type: DataType::Integer,
            nullable: true,
            char_varchar: None,
        }])))
    }
}

fn read(path: &str) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: "parquet".to_owned(),
        paths: vec![OptionValue::from_user("path", path)],
        options: BTreeMap::new(),
        schema: None,
    }))
}

/// `VALUES (1) AS v(id)`: no source.
fn values() -> LogicalPlan {
    LogicalPlan::new(PlanNode::InlineTable {
        rows: vec![vec![Arc::new(Literal::Integer(1).into())]],
        columns: vec!["id".to_owned()],
    })
}

fn ifn() -> Expr {
    f("input_file_name", vec![])
}

fn analyze(plan: LogicalPlan) -> Result<AnalyzedPlan, AnalysisError> {
    let mut ctx = context(
        MemoryCatalog::new("spark_catalog"),
        ResolutionConf::default(),
    );
    ctx.files = Some(Arc::new(Files));
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, &ctx))
}

fn class(plan: LogicalPlan) -> Option<ErrorClass> {
    analyze(plan).err().map(|e| e.error.error_class())
}

fn on_id() -> JoinCondition {
    JoinCondition::Using(vec!["id".to_owned()])
}

#[test]
fn an_operator_over_two_sources_is_refused() {
    // Spark 4.2.0 cases 19, 20: the call's operator reads both sides of a join.
    let join = || read("/a").join(read("/b"), JoinKind::Inner, on_id());
    let err = analyze(join().project(vec![ifn()])).unwrap_err();
    assert_eq!(
        err.error.error_class(),
        ErrorClass::MultiSourcesUnsupportedForExpression
    );
    assert!(
        err.error
            .to_string()
            .contains("The expression \"input_file_name()\" does not support more than one source"),
        "{}",
        err.error
    );
    let like = Expr::builtin(Builtin::Like, vec![ifn(), s("%a%")]);
    assert_eq!(
        class(join().filter(like).project(vec![c("id")])),
        Some(ErrorClass::MultiSourcesUnsupportedForExpression)
    );
    // INTERSECT counts both sides (case 28), though the analyzer rewrites it into a join.
    let intersect = LogicalPlan::new(PlanNode::SetOp {
        left: Arc::new(read("/a")),
        right: Arc::new(read("/b")),
        op: crate::SetOperation::Intersect,
        all: false,
    });
    assert_eq!(
        class(intersect.project(vec![ifn()])),
        Some(ErrorClass::MultiSourcesUnsupportedForExpression)
    );
    // A union of sources is one (case 26), but a join with it two (case 29).
    let union = || {
        LogicalPlan::new(PlanNode::Union {
            inputs: vec![Arc::new(read("/b")), Arc::new(read("/c"))],
            by_name: false,
            allow_missing_columns: false,
        })
    };
    assert_eq!(class(union().project(vec![ifn()])), None);
    assert_eq!(
        class(
            read("/a")
                .join(union(), JoinKind::Inner, on_id())
                .project(vec![ifn()])
        ),
        Some(ErrorClass::MultiSourcesUnsupportedForExpression)
    );
}

#[test]
fn leaves_without_files_and_subqueries_do_not_count() {
    // A join with an inline table (case 24) reads one source.
    let join = read("/a").join(values(), JoinKind::Cross, JoinCondition::None);
    assert_eq!(class(join.project(vec![ifn()])), None);
    // An `IN` subquery's plan is not the outer operator's input (case 43), though it is rewritten into a join.
    let sub = Expr::new(ExprNode::Subquery {
        kind: SubqueryKind::In(vec![Arc::new(c("id"))]),
        plan: Arc::new(read("/b").project(vec![c("id")])),
    });
    assert_eq!(
        class(read("/a").filter(sub).project(vec![ifn(), c("id")])),
        None
    );
    // ...but it is checked on its own (case 64).
    let inner = read("/a")
        .join(read("/b"), JoinKind::Inner, on_id())
        .filter(Expr::builtin(Builtin::Like, vec![ifn(), s("%a%")]))
        .project(vec![c("id")]);
    let sub = Expr::new(ExprNode::Subquery {
        kind: SubqueryKind::In(vec![Arc::new(c("id"))]),
        plan: Arc::new(inner),
    });
    assert_eq!(
        class(read("/c").filter(sub).project(vec![c("id")])),
        Some(ErrorClass::MultiSourcesUnsupportedForExpression)
    );
}

fn bound(plan: LogicalPlan) -> Plan {
    let a = analyze(plan).unwrap();
    bind_input_files(&a.plan, &|_| FileMetadata::WithRowIndex)
        .unwrap()
        .unwrap()
}

/// Whether the plan's main query (no subquery plans) reads a scan's `_metadata` column, and holds a call.
fn shape(p: &Plan) -> (bool, bool) {
    let mut metadata = false;
    let mut calls = false;
    let mut stack = vec![Arc::clone(p.root())];
    while let Some(n) = stack.pop() {
        if let PlanNode::Scan(s) = &n.node {
            metadata |= s.metadata_column().is_some();
        }
        n.for_each_expr(&mut |e| {
            let mut es = vec![Arc::clone(e)];
            while let Some(e) = es.pop() {
                if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node {
                    calls |= f.id.name().starts_with("input_file");
                }
                e.for_each_child(&mut |c| es.push(Arc::clone(c)));
            }
        });
        n.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    (metadata, calls)
}

#[test]
fn a_call_in_the_scan_pipeline_reads_the_metadata_column() {
    let a = analyze(read("/a").project(vec![ifn(), c("id")])).unwrap();
    let p = bind_input_files(&a.plan, &|_| FileMetadata::WithRowIndex)
        .unwrap()
        .unwrap();
    assert_eq!(shape(&p), (true, false));
    // Names and types of the output stay.
    assert_eq!(p.root().resolved_output().unwrap(), a.output);
    // A filter passes its input's columns on: a projection keeps its output as it was.
    let like = Expr::builtin(Builtin::Like, vec![ifn(), s("%a%")]);
    let p = bound(read("/a").filter(like));
    assert_eq!(shape(&p), (true, false));
    assert!(matches!(p.root().node, PlanNode::Project { .. }));
    assert_eq!(p.root().resolved_output().unwrap().len(), 1);
    // `GROUP BY input_file_name()` evaluates on the input (case 3).
    let agg = read("/a").aggregate(vec![ifn()], vec![ifn(), f("count", vec![c("id")])]);
    assert_eq!(shape(&bound(agg)), (true, false));
}

#[test]
fn a_call_past_a_pipeline_break_is_the_no_file_value() {
    // Over an aggregate's output (case 5) and over a source with no file (case 33), no metadata is read.
    let agg = read("/a")
        .aggregate(vec![], vec![f("count", vec![c("id")]).alias("n")])
        .alias("t")
        .project(vec![ifn(), c("n")]);
    let p = bound(agg);
    assert_eq!(shape(&p), (false, false));
    let p = bound(values().project(vec![ifn()]));
    assert_eq!(shape(&p), (false, false));
    // A call next to the group keys but not one (case 69) is evaluated over the groups.
    let p = bound(read("/a").aggregate(vec![c("id")], vec![ifn()]));
    assert_eq!(shape(&p), (false, false));
    // A scan whose source has no file (a cached snapshot) gives the constants too.
    let a = analyze(read("/a").project(vec![ifn()])).unwrap();
    let p = bind_input_files(&a.plan, &|_| FileMetadata::None)
        .unwrap()
        .unwrap();
    assert_eq!(shape(&p), (false, false));
    // A plan without calls is left alone.
    let a = analyze(read("/a").project(vec![c("id")])).unwrap();
    assert!(
        bind_input_files(&a.plan, &|_| FileMetadata::WithRowIndex)
            .unwrap()
            .is_none()
    );
}

#[test]
fn union_branches_read_their_own_files_or_the_constants() {
    // Case 30: one branch has the file, the other gives the constants; the output is as it was.
    let union = LogicalPlan::new(PlanNode::Union {
        inputs: vec![Arc::new(read("/a")), Arc::new(values())],
        by_name: false,
        allow_missing_columns: false,
    });
    let a = analyze(union.alias("u").project(vec![ifn(), c("id")])).unwrap();
    let p = bind_input_files(&a.plan, &|_| FileMetadata::Basic)
        .unwrap()
        .unwrap();
    assert_eq!(shape(&p), (true, false));
    assert_eq!(p.root().resolved_output().unwrap(), a.output);
}

fn plan_with(catalog: MemoryCatalog, plan: LogicalPlan) -> Result<AnalyzedPlan, AnalysisError> {
    let mut ctx = context(catalog, ResolutionConf::default());
    ctx.files = Some(Arc::new(Files));
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, &ctx))
}

/// Recorded divergence (module docs): an Iceberg or Delta scan (a versioned table) counts as a source, as Spark's
/// V2 file scans do, but has no `_metadata` column, so a call over it gives the constants where Spark gives the
/// data file.
#[test]
fn versioned_table_scans_count_but_give_the_constants() {
    let catalog = || {
        let col = TableColumn {
            name: "id".to_owned(),
            data_type: DataType::Integer,
            nullable: true,
            char_varchar: None,
        };
        MemoryCatalog::new("spark_catalog").with_table(
            &["default"],
            "v",
            vec![col],
            Some(crate::TableVersion(7)),
        )
    };
    let v = || LogicalPlan::table(["v"]);
    let r = plan_with(
        catalog(),
        v().join(read("/a"), JoinKind::Inner, on_id())
            .project(vec![ifn()]),
    );
    assert_eq!(
        r.err().map(|e| e.error.error_class()),
        Some(ErrorClass::MultiSourcesUnsupportedForExpression)
    );
    let a = plan_with(catalog(), v().project(vec![ifn()])).unwrap();
    // The session's lookup gives versioned tables no `_metadata` (`QueryCatalog::scan_file_metadata`).
    let p = bind_input_files(&a.plan, &|s| match &s.source {
        ReadSource::Files { .. } => FileMetadata::WithRowIndex,
        _ => FileMetadata::None,
    })
    .unwrap()
    .unwrap();
    assert_eq!(shape(&p), (false, false));
}

#[test]
fn outer_joins_read_the_file_only_from_a_side_kept_whole() {
    let file_left = |kind| {
        bound(
            read("/a")
                .join(values(), kind, on_id())
                .project(vec![ifn()]),
        )
    };
    assert_eq!(shape(&file_left(JoinKind::LeftOuter)), (true, false));
    assert_eq!(shape(&file_left(JoinKind::Inner)), (true, false));
    assert_eq!(shape(&file_left(JoinKind::RightOuter)), (false, false));
    assert_eq!(shape(&file_left(JoinKind::FullOuter)), (false, false));
    let file_right = |kind| {
        bound(
            values()
                .join(read("/a"), kind, on_id())
                .project(vec![ifn()]),
        )
    };
    assert_eq!(shape(&file_right(JoinKind::RightOuter)), (true, false));
    assert_eq!(shape(&file_right(JoinKind::LeftOuter)), (false, false));
    // The provided values stay non-nullable.
    let p = file_right(JoinKind::RightOuter);
    assert!(
        !p.root().resolved_output().unwrap()[0].nullable,
        "{}",
        p.root()
    );
}

/// A CTE reference counts its definition's sources (Spark checks after inlining).
#[test]
fn a_cte_reference_counts_its_definition() {
    let with = |input: LogicalPlan| {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(input),
            ctes: vec![crate::Cte {
                name: "x".to_owned(),
                plan: Arc::new(read("/a")),
                generated: false,
            }],
        })
    };
    let x = || LogicalPlan::table(["x"]);
    assert_eq!(
        class(with(
            x().join(read("/b"), JoinKind::Inner, on_id())
                .project(vec![ifn()])
        )),
        Some(ErrorClass::MultiSourcesUnsupportedForExpression)
    );
    assert_eq!(class(with(x().project(vec![ifn()]))), None);
    let p = bound(with(x().project(vec![ifn()])));
    assert_eq!(shape(&p), (true, false));
}
