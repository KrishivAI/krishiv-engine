// SPDX-License-Identifier: Apache-2.0
//! `AnalysisContext::definer_texts` (P3-T062 §2.4, review B1): the tables a `SQL SECURITY
//! DEFINER` view's inlined query reads were authorized for the view's owner, so Bind asks the
//! principal nothing about a catalog table read under one of those very nodes, the column pass
//! skips it and the analysis does not report it. The principal's own read of the same table is
//! still asked, and no other node is ever trusted: not one equal to a trusted node, not one with
//! its origin, not the text a caller embeds (`PlanNode::Sql`, lowered with source numbers from 0),
//! and not a DataFrame's.

use std::sync::{Arc, Mutex};

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_types::DataType;

use crate::analyzer::testing::{block_on, context};
use crate::analyzer::views::DefinerTexts;
use crate::analyzer::{
    AccessDecision, AccessPolicy, AnalysisContext, AnalysisError, AnalyzedPlan, Analyzer,
    BoxFuture, DefaultAnalyzer, MemoryCatalog, PolicyVersion, Principal, Privilege, ResolutionConf,
    Securable, SqlFrontend, TableColumn,
};
use crate::{LogicalPlan, Origin, Plan, PlanNode, ReadSource, SourceId, Span, SqlArgs};

/// Denies every catalog table, view and column; records what it is asked.
#[derive(Default)]
struct NoTables {
    asked: Mutex<Vec<String>>,
    column_level: bool,
}

impl AccessPolicy for NoTables {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        self.asked.lock().unwrap().push(securable.to_string());
        let decision = match securable {
            Securable::Table(_) | Securable::View(_) | Securable::Column { .. } => {
                AccessDecision::Deny {
                    reason: "no tables".to_owned(),
                }
            }
            _ => AccessDecision::Allow,
        };
        Box::pin(std::future::ready(Ok(decision)))
    }

    fn column_level(&self) -> bool {
        self.column_level
    }
}

const VIEW_TEXT: SourceId = SourceId(7);

/// Lowers any text to a read of `t` at `source`, as a view text read of `t` is lowered.
struct ReadsT;

impl SqlFrontend for ReadsT {
    fn lower(&self, _: &str, source: SourceId) -> Result<Plan, AnalysisError> {
        Plan::try_new(read(source, 0)).map_err(|error| AnalysisError {
            error,
            origin: Origin::UNKNOWN,
        })
    }

    fn redact(&self, query: &str) -> krishiv_common::redact::RedactedSql {
        krishiv_common::redact::RedactedSql::opaque(query)
    }

    fn fingerprint(&self) -> u64 {
        1
    }

    fn keywords(&self) -> Vec<(String, bool)> {
        Vec::new()
    }
}

fn ctx(policy: &Arc<NoTables>, trusted: &[&Arc<LogicalPlan>]) -> AnalysisContext {
    let catalog = MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "t",
        vec![TableColumn::new("a".to_owned(), DataType::Integer, true)],
        None,
    );
    let mut c = context(catalog, ResolutionConf::default());
    c.policy = Arc::clone(policy) as Arc<dyn AccessPolicy>;
    c.definer_texts = DefinerTexts::new(trusted.iter().map(|p| Arc::clone(p)).collect());
    c.sql = Some(Arc::new(ReadsT));
    c
}

/// A read of `t` lowered from `source` (a view's text) at `start`.
fn read(source: SourceId, start: u32) -> Arc<LogicalPlan> {
    Arc::new(
        LogicalPlan::new(PlanNode::Read(ReadSource::Table {
            name: vec!["t".to_owned()],
            options: std::collections::BTreeMap::new(),
        }))
        .with_origin(Origin::sql(source, Span::new(start, start + 1))),
    )
}

fn union(inputs: Vec<Arc<LogicalPlan>>) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Union {
        inputs,
        by_name: false,
        allow_missing_columns: false,
    }))
}

fn run(plan: Arc<LogicalPlan>, c: &AnalysisContext) -> Result<AnalyzedPlan, KrishivError> {
    let plan = Plan::try_new(plan)?;
    block_on(DefaultAnalyzer.analyze(&plan, c)).map_err(|e| e.error)
}

fn denied(r: &Result<AnalyzedPlan, KrishivError>) -> bool {
    matches!(r, Err(e) if e.error_class() == ErrorClass::KrishivInsufficientPermissions)
}

#[test]
fn a_definer_texts_reads_are_not_asked_of_the_principal() {
    for column_level in [false, true] {
        let policy = Arc::new(NoTables {
            column_level,
            ..NoTables::default()
        });
        let text = read(VIEW_TEXT, 0);
        // Untrusted, the read is denied.
        assert!(denied(&run(Arc::clone(&text), &ctx(&policy, &[]))));
        policy.asked.lock().unwrap().clear();
        // The very node the session inlined is the owner's: not asked, not reported.
        let a = run(Arc::clone(&text), &ctx(&policy, &[&text])).unwrap();
        assert!(a.securables.is_empty(), "{:?}", a.securables);
        assert!(policy.asked.lock().unwrap().is_empty(), "{column_level}");
        // Under a parent of it too.
        let a = run(union(vec![Arc::clone(&text)]), &ctx(&policy, &[&text])).unwrap();
        assert!(a.securables.is_empty(), "{:?}", a.securables);
        // A read from anything else is the principal's, even beside a trusted one, in
        // either order.
        for inputs in [
            vec![Arc::clone(&text), read(SourceId(8), 0)],
            vec![read(SourceId(8), 0), Arc::clone(&text)],
        ] {
            assert!(
                denied(&run(union(inputs), &ctx(&policy, &[&text]))),
                "{column_level}"
            );
        }
        // A DataFrame node (no SQL origin) is never trusted.
        let df = Arc::new(
            LogicalPlan::new(PlanNode::Read(ReadSource::Table {
                name: vec!["t".to_owned()],
                options: std::collections::BTreeMap::new(),
            }))
            .with_origin(Origin::plan(crate::PlanId(3))),
        );
        assert!(denied(&run(df, &ctx(&policy, &[&text]))));
    }
}

/// Review B1: trust is the node, never its number. A node equal to the trusted one (same read,
/// same origin) and the text a caller embeds, whose lowering numbers its sources from 0 and so
/// gives its reads the trusted node's very origin, are both the principal's.
#[test]
fn equal_origins_and_embedded_sql_are_never_trusted() {
    for column_level in [false, true] {
        let policy = Arc::new(NoTables {
            column_level,
            ..NoTables::default()
        });
        // The session's view text numbered 0 (as Bind numbers the first embedded text).
        let text = read(SourceId(0), 0);
        let twin = read(SourceId(0), 0);
        assert!(denied(&run(
            union(vec![Arc::clone(&text), twin]),
            &ctx(&policy, &[&text])
        )));
        for n in 1..4 {
            let mut inputs = vec![Arc::clone(&text)];
            inputs.extend((0..n).map(|_| {
                Arc::new(LogicalPlan::new(PlanNode::Sql {
                    query: "SELECT a FROM t".to_owned(),
                    args: SqlArgs::None,
                }))
            }));
            assert!(
                denied(&run(union(inputs), &ctx(&policy, &[&text]))),
                "{n} {column_level}"
            );
        }
    }
}
