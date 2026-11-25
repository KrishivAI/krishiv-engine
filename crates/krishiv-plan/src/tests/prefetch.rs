// SPDX-License-Identifier: Apache-2.0
//! `AccessPolicy::prefetch` (ADR 0049 §3): before asking a stage's pairs one by one, analysis tells
//! the policy what it is about to ask, so a remote provider decides them in one round trip. Bind
//! sends its catalog reads' names as located (as written and folded), the column pass its column
//! pairs. A hint only: every pair is still asked through `check`.

use std::sync::{Arc, Mutex};

use krishiv_common::error::KrishivError;
use krishiv_types::DataType;

use super::dsl::c;
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AccessDecision, AccessPolicy, Analyzer, BoxFuture, DefaultAnalyzer, MemoryCatalog,
    PolicyVersion, Principal, Privilege, ResolutionConf, Securable, TableColumn,
};
use crate::{LogicalPlan, Plan};

/// Allows everything; records the prefetches and the checks in order.
#[derive(Default)]
struct Recording {
    column_level: bool,
    log: Mutex<Vec<String>>,
    prefetched: Mutex<Vec<Vec<(Securable, Privilege)>>>,
}

impl AccessPolicy for Recording {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        self.log
            .lock()
            .unwrap()
            .push(format!("check {securable} {privilege:?}"));
        Box::pin(std::future::ready(Ok(AccessDecision::Allow)))
    }

    fn column_level(&self) -> bool {
        self.column_level
    }

    fn prefetch<'a>(
        &'a self,
        _: &'a Principal,
        items: &'a [(Securable, Privilege)],
    ) -> BoxFuture<'a, ()> {
        self.log
            .lock()
            .unwrap()
            .push(format!("prefetch {}", items.len()));
        self.prefetched.lock().unwrap().push(items.to_vec());
        Box::pin(std::future::ready(()))
    }
}

fn analyze(policy: &Arc<Recording>, plan: LogicalPlan) {
    let catalog = MemoryCatalog::new("spark_catalog")
        .with_table(
            &["default"],
            "t",
            vec![
                TableColumn::new("a".to_owned(), DataType::Integer, true),
                TableColumn::new("b".to_owned(), DataType::Integer, true),
            ],
            None,
        )
        .with_table(
            &["default"],
            "u",
            vec![TableColumn::new("a".to_owned(), DataType::Integer, true)],
            None,
        );
    let mut ctx = context(catalog, ResolutionConf::default());
    ctx.policy = Arc::clone(policy) as Arc<dyn AccessPolicy>;
    let plan = Plan::try_new(plan).unwrap();
    block_on(DefaultAnalyzer.analyze(&plan, &ctx)).unwrap();
}

fn table(name: &str) -> Securable {
    Securable::Table(crate::analyzer::TableIdent {
        catalog: "spark_catalog".to_owned(),
        namespace: vec!["default".to_owned()],
        name: name.to_owned(),
    })
}

#[test]
fn bind_prefetches_the_table_reads_before_asking_them() {
    let policy = Arc::new(Recording::default());
    let plan = LogicalPlan::table(["default", "t"]).join(
        LogicalPlan::table(["default", "u"]),
        crate::JoinKind::Cross,
        crate::JoinCondition::None,
    );
    analyze(&policy, plan);
    let prefetched = policy.prefetched.lock().unwrap().clone();
    assert_eq!(
        prefetched,
        vec![vec![
            (table("t"), Privilege::Select),
            (table("u"), Privilege::Select),
        ]]
    );
    let log = policy.log.lock().unwrap().clone();
    assert_eq!(log[0], "prefetch 2", "{log:?}");
    // Every pair is still asked.
    for name in ["t", "u"] {
        assert!(
            log.iter()
                .any(|l| l.starts_with("check") && l.contains(&format!("`{name}`"))),
            "{log:?}"
        );
    }
}

#[test]
fn the_column_pass_prefetches_its_columns_in_one_hint() {
    let policy = Arc::new(Recording {
        column_level: true,
        ..Recording::default()
    });
    analyze(
        &policy,
        LogicalPlan::table(["default", "t"]).project(vec![c("a"), c("b")]),
    );
    let prefetched = policy.prefetched.lock().unwrap().clone();
    assert_eq!(prefetched.len(), 2, "{prefetched:?}");
    // Bind asks `Browse` on the table under a column-level policy.
    assert_eq!(prefetched[0], vec![(table("t"), Privilege::Browse)]);
    let columns: Vec<String> = prefetched[1]
        .iter()
        .map(|(s, p)| format!("{s} {p:?}"))
        .collect();
    assert_eq!(
        columns,
        vec![
            "column `a` of table `spark_catalog`.`default`.`t` Select",
            "column `b` of table `spark_catalog`.`default`.`t` Select",
        ]
    );
    let log = policy.log.lock().unwrap().clone();
    let hint = log.iter().position(|l| l == "prefetch 2").unwrap();
    let first_column = log.iter().position(|l| l.contains("column `a`")).unwrap();
    assert!(hint < first_column, "{log:?}");
}

#[test]
fn a_plan_with_no_catalog_read_prefetches_nothing() {
    let policy = Arc::new(Recording::default());
    analyze(&policy, LogicalPlan::new(crate::PlanNode::OneRowRelation));
    assert!(policy.prefetched.lock().unwrap().is_empty());
}

#[test]
fn the_default_prefetch_does_nothing_and_every_policy_decides_everything() {
    let principal = Principal::new("ana", crate::analyzer::PrincipalKind::User);
    let items = [(table("t"), Privilege::Select)];
    block_on(crate::analyzer::AllowAll.prefetch(&principal, &items));
    assert!(crate::analyzer::AllowAll.decides(&table("t")));
    assert!(crate::analyzer::DenyAll.decides(&Securable::System("cluster".to_owned())));
}
