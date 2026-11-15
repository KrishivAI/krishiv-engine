// SPDX-License-Identifier: Apache-2.0
//! The read privilege of one analysis (ADR 0046 §3b, P3 PR-14c): an analysis that runs the plan asks
//! `Select` on every table, view and path it reads; a metadata analysis (`AnalysisMode::Metadata`:
//! schema, tree string, semantic hash) asks `Browse`. Other privileges (`Usage` on a secret scope,
//! `Manage` for a server-scoped store option) are asked as before; a read by path asks `Select` in
//! both modes (inference reads file contents: CSV header names, JSON keys); and nothing resolved in one mode
//! is reused by the other (the context fingerprint differs, so the prefix cache misses).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::OptionValue;
use krishiv_types::DataType;

use super::dsl::c;
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AccessDecision, AccessPolicy, AllowAll, AnalysisContext, AnalysisError, AnalysisMode,
    AnalyzedPlan, Analyzer, BoxFuture, CachingAnalyzer, DefaultAnalyzer, FileSchemaResolver,
    MemoryCatalog, PolicyVersion, Principal, Privilege, ResolutionConf, Securable, SourceKind,
    TableColumn, fingerprint,
};
use crate::{LogicalPlan, Plan, PlanNode, ReadSource};

/// A viewer: `Browse` on everything, every other privilege denied (but `Select` on paths when
/// `paths`). Records every question.
#[derive(Default)]
struct Viewer {
    paths: bool,
    asked: Mutex<Vec<(String, Privilege)>>,
}

impl Viewer {
    fn asked(&self) -> Vec<(String, Privilege)> {
        self.asked.lock().unwrap().clone()
    }
}

impl AccessPolicy for Viewer {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(3)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        self.asked
            .lock()
            .unwrap()
            .push((securable.to_string(), privilege));
        let path_read =
            self.paths && privilege == Privilege::Select && matches!(securable, Securable::Path(_));
        let decision = if privilege == Privilege::Browse || path_read {
            AccessDecision::Allow
        } else {
            AccessDecision::Deny {
                reason: "viewer".to_owned(),
            }
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

/// Files: every path has one column `a INT`.
struct Files;

impl FileSchemaResolver for Files {
    fn qualify(&self, path: &OptionValue) -> Result<OptionValue, KrishivError> {
        Ok(path.clone())
    }

    fn source_kind(&self, _: &str) -> SourceKind {
        SourceKind::File
    }

    fn infer<'a>(
        &'a self,
        _: &'a str,
        _: &'a [OptionValue],
        _: &'a BTreeMap<String, OptionValue>,
    ) -> BoxFuture<'a, Result<Vec<TableColumn>, KrishivError>> {
        Box::pin(std::future::ready(Ok(vec![TableColumn::new(
            "a".to_owned(),
            DataType::Integer,
            true,
        )])))
    }
}

fn ctx(policy: &Arc<Viewer>, mode: AnalysisMode) -> AnalysisContext {
    let catalog = MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "t",
        vec![TableColumn::new("a".to_owned(), DataType::Integer, true)],
        None,
    );
    let mut c = context(catalog, ResolutionConf::default());
    c.policy = Arc::clone(policy) as Arc<dyn AccessPolicy>;
    c.files = Some(Arc::new(Files));
    c.mode = mode;
    c
}

fn files(options: &[(&str, OptionValue)]) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: "parquet".to_owned(),
        paths: vec![OptionValue::from_user("path", "/data/x")],
        options: options
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
    }))
}

fn run(
    analyzer: &dyn Analyzer,
    plan: LogicalPlan,
    c: &AnalysisContext,
) -> Result<AnalyzedPlan, AnalysisError> {
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: crate::Origin::UNKNOWN,
    })?;
    block_on(analyzer.analyze(&plan, c))
}

fn denied(r: &Result<AnalyzedPlan, AnalysisError>) -> bool {
    matches!(r, Err(e) if e.error.error_class() == ErrorClass::KrishivInsufficientPermissions)
}

#[test]
fn the_default_mode_reads_with_select() {
    assert_eq!(AnalysisMode::default(), AnalysisMode::Execute);
    assert_eq!(AnalysisMode::Execute.read_privilege(), Privilege::Select);
    assert_eq!(AnalysisMode::Metadata.read_privilege(), Privilege::Browse);
    let viewer = Arc::new(Viewer::default());
    let c = ctx(&viewer, AnalysisMode::default());
    for plan in [LogicalPlan::table(["t"]), files(&[])] {
        let r = run(&DefaultAnalyzer, plan, &c);
        assert!(denied(&r), "{r:?}");
    }
    assert!(viewer.asked().iter().all(|(_, p)| *p == Privilege::Select));
}

#[test]
fn a_metadata_analysis_reads_with_browse() {
    let viewer = Arc::new(Viewer::default());
    let c = ctx(&viewer, AnalysisMode::Metadata);
    let table = run(&DefaultAnalyzer, LogicalPlan::table(["t"]), &c).unwrap();
    assert_eq!(table.output[0].name, "a");
    assert!(
        table
            .securables
            .iter()
            .all(|s| s.privilege == Privilege::Browse),
        "{:?}",
        table.securables
    );
    // Only `Browse` was asked: never `Select`.
    let asked = viewer.asked();
    assert!(!asked.is_empty());
    assert!(
        asked.iter().all(|(_, p)| *p == Privilege::Browse),
        "{asked:?}"
    );
}

#[test]
fn a_read_by_path_asks_select_even_for_metadata() {
    // A path has no stored schema: inferring one reads the files (CSV header names, JSON keys), so
    // `Browse` would show a viewer file contents.
    let viewer = Arc::new(Viewer::default());
    let c = ctx(&viewer, AnalysisMode::Metadata);
    let r = run(&DefaultAnalyzer, files(&[]), &c);
    assert!(denied(&r), "{r:?}");
    assert!(
        viewer
            .asked()
            .iter()
            .all(|(s, p)| s.starts_with("path ") && *p == Privilege::Select),
        "{:?}",
        viewer.asked()
    );
    // With `Select` on the path it analyzes, and reports the path as `Select`.
    let reader = Arc::new(Viewer {
        paths: true,
        ..Viewer::default()
    });
    let c = ctx(&reader, AnalysisMode::Metadata);
    let a = run(&DefaultAnalyzer, files(&[]), &c).unwrap();
    assert!(
        a.securables
            .iter()
            .any(|s| matches!(s.securable, Securable::Path(_)) && s.privilege == Privilege::Select)
    );
}

#[test]
fn other_privileges_are_asked_as_before() {
    // A secret reference in the options still needs `Usage` on its scope, which a viewer lacks.
    let viewer = Arc::new(Viewer {
        paths: true,
        ..Viewer::default()
    });
    let c = ctx(&viewer, AnalysisMode::Metadata);
    let secret = OptionValue::reference("prod", "key");
    let r = run(&DefaultAnalyzer, files(&[("token", secret)]), &c);
    assert!(denied(&r), "{r:?}");
    assert!(
        viewer
            .asked()
            .iter()
            .any(|(s, p)| s.contains("prod") && *p == Privilege::Usage)
    );
}

#[test]
fn the_mode_is_part_of_the_context_fingerprint() {
    let viewer = Arc::new(Viewer::default());
    let execute = ctx(&viewer, AnalysisMode::Execute);
    let metadata = ctx(&viewer, AnalysisMode::Metadata);
    let a = fingerprint(&execute, BTreeMap::new());
    let b = fingerprint(&metadata, BTreeMap::new());
    assert_ne!(a, b);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn the_prefix_cache_never_serves_another_mode() {
    let viewer = Arc::new(Viewer::default());
    let cache = CachingAnalyzer::new();
    let metadata = ctx(&viewer, AnalysisMode::Metadata);
    let execute = ctx(&viewer, AnalysisMode::Execute);
    let base = LogicalPlan::table(["t"]);
    run(&cache, base.clone(), &metadata).unwrap();
    // The same root, run: still asked as `Select`, still denied, and not a warm start.
    let before = cache.warm_starts();
    let r = run(&cache, base, &execute);
    assert!(denied(&r), "{r:?}");
    assert_eq!(cache.warm_starts(), before);

    // Allowed in both modes: a grown plan starts warm from its own mode's prefix only.
    let allowed = |mode| {
        let mut c = ctx(&viewer, mode);
        c.policy = Arc::new(AllowAll);
        c
    };
    let (metadata, execute) = (
        allowed(AnalysisMode::Metadata),
        allowed(AnalysisMode::Execute),
    );
    let step = |p: LogicalPlan| p.project(vec![c("a")]);
    let cache = CachingAnalyzer::new();
    run(&cache, LogicalPlan::table(["t"]), &metadata).unwrap();
    run(&cache, step(LogicalPlan::table(["t"])), &metadata).unwrap();
    assert_eq!(cache.warm_starts(), 1, "the same mode starts warm");
    let ran = run(&cache, step(step(LogicalPlan::table(["t"]))), &execute).unwrap();
    assert_eq!(cache.warm_starts(), 1, "another mode never does");
    assert!(
        ran.securables
            .iter()
            .all(|s| s.privilege == Privilege::Select)
    );
}
