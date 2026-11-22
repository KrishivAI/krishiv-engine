// SPDX-License-Identifier: Apache-2.0
use std::sync::Mutex;

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_plan::analyzer::{PrincipalKind, RowColumnPolicy};

use super::*;

/// A member that answers from a rule and records what it was asked, told and prefetched.
struct Member {
    version: u64,
    column_level: bool,
    decides: fn(&Securable) -> bool,
    answer: fn(&Securable, Privilege) -> Result<AccessDecision, KrishivError>,
    asked: Mutex<Vec<(Securable, Privilege)>>,
    told: Mutex<Vec<(Securable, Privilege)>>,
    prefetched: Mutex<Vec<Vec<(Securable, Privilege)>>>,
}

impl Member {
    fn new(answer: fn(&Securable, Privilege) -> Result<AccessDecision, KrishivError>) -> Self {
        Self {
            version: 1,
            column_level: false,
            decides: |_| true,
            answer,
            asked: Mutex::new(Vec::new()),
            told: Mutex::new(Vec::new()),
            prefetched: Mutex::new(Vec::new()),
        }
    }

    fn asked(&self) -> Vec<(Securable, Privilege)> {
        self.asked.lock().unwrap().clone()
    }

    fn told(&self) -> Vec<(Securable, Privilege)> {
        self.told.lock().unwrap().clone()
    }
}

impl AccessPolicy for Member {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(self.version)
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
            .push((securable.clone(), privilege));
        Box::pin(std::future::ready((self.answer)(securable, privilege)))
    }

    fn column_level(&self) -> bool {
        self.column_level
    }

    fn denied(&self, _: &Principal, securable: &Securable, privilege: Privilege) {
        self.told
            .lock()
            .unwrap()
            .push((securable.clone(), privilege));
    }

    fn prefetch<'a>(
        &'a self,
        _: &'a Principal,
        items: &'a [(Securable, Privilege)],
    ) -> BoxFuture<'a, ()> {
        self.prefetched.lock().unwrap().push(items.to_vec());
        Box::pin(std::future::ready(()))
    }

    fn decides(&self, securable: &Securable) -> bool {
        (self.decides)(securable)
    }
}

fn allow(_: &Securable, _: Privilege) -> Result<AccessDecision, KrishivError> {
    Ok(AccessDecision::Allow)
}

fn deny(_: &Securable, _: Privilege) -> Result<AccessDecision, KrishivError> {
    Ok(AccessDecision::Deny {
        reason: "no".to_owned(),
    })
}

fn fail(_: &Securable, _: Privilege) -> Result<AccessDecision, KrishivError> {
    Err(InternalError {
        message: "down".to_owned(),
    }
    .into())
}

fn ident(name: &str) -> TableIdent {
    TableIdent {
        catalog: "spark_catalog".to_owned(),
        namespace: vec!["default".to_owned()],
        name: name.to_owned(),
    }
}

fn table(name: &str) -> Securable {
    Securable::Table(ident(name))
}

fn column(name: &str, column: &str) -> Securable {
    Securable::Column {
        table: ident(name),
        column: column.to_owned(),
    }
}

fn principal() -> Principal {
    Principal::new("ana", PrincipalKind::User)
}

fn list(members: Vec<Arc<Member>>) -> IntersectionPolicy {
    IntersectionPolicy::new(
        members
            .into_iter()
            .map(|m| m as Arc<dyn AccessPolicy>)
            .collect(),
    )
}

fn run<T>(f: impl std::future::Future<Output = T>) -> T {
    futures::executor::block_on(f)
}

fn is_allow(d: &AccessDecision) -> bool {
    matches!(d, AccessDecision::Allow)
}

#[test]
fn every_member_must_allow() {
    let a = Arc::new(Member::new(allow));
    let b = Arc::new(Member::new(allow));
    let p = list(vec![Arc::clone(&a), Arc::clone(&b)]);
    let d = run(p.check(&principal(), &table("t"), Privilege::Select)).unwrap();
    assert!(is_allow(&d));
    assert_eq!(a.asked().len(), 1);
    assert_eq!(b.asked().len(), 1);
}

#[test]
fn one_denial_or_error_refuses_and_stops() {
    for second in [deny as fn(&Securable, Privilege) -> _, fail] {
        let a = Arc::new(Member::new(allow));
        let b = Arc::new(Member::new(second));
        let c = Arc::new(Member::new(allow));
        let p = list(vec![Arc::clone(&a), Arc::clone(&b), Arc::clone(&c)]);
        let d = run(p.check(&principal(), &table("t"), Privilege::Modify));
        assert!(!matches!(d, Ok(AccessDecision::Allow)), "{d:?}");
        assert!(c.asked().is_empty(), "asking stops at the refusal");
    }
}

#[test]
fn adding_a_member_never_widens() {
    // A deny-all list member keeps every answer a denial, whatever the others say.
    for others in [allow as fn(&Securable, Privilege) -> _, deny, fail] {
        let p = list(vec![
            Arc::new(Member::new(others)),
            Arc::new(Member::new(deny)),
        ]);
        let d = run(p.check(&principal(), &table("t"), Privilege::Select));
        assert!(!matches!(d, Ok(AccessDecision::Allow)));
    }
}

#[test]
fn a_securable_no_member_decides_is_refused() {
    let mut a = Member::new(allow);
    a.decides = |s| matches!(s, Securable::System(_));
    let a = Arc::new(a);
    let p = list(vec![Arc::clone(&a)]);
    let d = run(p.check(&principal(), &table("t"), Privilege::Select)).unwrap();
    assert!(matches!(d, AccessDecision::Deny { .. }));
    assert!(a.asked().is_empty());
    assert!(!p.decides(&table("t")));
    assert!(p.decides(&Securable::System("cluster".to_owned())));
    let d = run(p.check(
        &principal(),
        &Securable::System("cluster".to_owned()),
        Privilege::Manage,
    ))
    .unwrap();
    assert!(is_allow(&d));
    // An empty list refuses everything.
    let none = IntersectionPolicy::new(Vec::new());
    let d = run(none.check(&principal(), &table("t"), Privilege::Select)).unwrap();
    assert!(matches!(d, AccessDecision::Deny { .. }));
}

#[test]
fn allow_with_is_passed_on() {
    fn with(_: &Securable, _: Privilege) -> Result<AccessDecision, KrishivError> {
        Ok(AccessDecision::AllowWith(RowColumnPolicy::default()))
    }
    let p = list(vec![
        Arc::new(Member::new(allow)),
        Arc::new(Member::new(with)),
    ]);
    let d = run(p.check(&principal(), &table("t"), Privilege::Select)).unwrap();
    assert!(matches!(d, AccessDecision::AllowWith(_)));
}

#[test]
fn the_version_changes_with_any_member() {
    let mut a = Member::new(allow);
    a.version = 7;
    let b = Member::new(allow);
    let v1 = list(vec![Arc::new(a), Arc::new(b)]).version();
    let mut a = Member::new(allow);
    a.version = 7;
    let mut b = Member::new(allow);
    b.version = 8;
    let v2 = list(vec![Arc::new(a), Arc::new(b)]).version();
    assert_ne!(v1, v2);
}

#[test]
fn a_table_level_member_is_asked_about_the_relation_of_a_column() {
    let mut grants = Member::new(allow);
    grants.column_level = true;
    let grants = Arc::new(grants);
    let external = Arc::new(Member::new(allow));
    let p = list(vec![Arc::clone(&grants), Arc::clone(&external)]);
    assert!(p.column_level());
    // Bind asks `Browse` on the view first, so its kind is known.
    let view = Securable::View(ident("v"));
    run(p.check(&principal(), &view, Privilege::Browse)).unwrap();
    run(p.check(&principal(), &column("v", "a"), Privilege::Select)).unwrap();
    run(p.check(&principal(), &column("t", "a"), Privilege::Select)).unwrap();
    assert_eq!(
        grants.asked(),
        vec![
            (view.clone(), Privilege::Browse),
            (column("v", "a"), Privilege::Select),
            (column("t", "a"), Privilege::Select),
        ]
    );
    assert_eq!(
        external.asked(),
        vec![
            (view.clone(), Privilege::Browse),
            (view, Privilege::Select),
            (table("t"), Privilege::Select),
        ]
    );
}

#[test]
fn a_refusal_is_told_only_to_the_member_that_refused() {
    let a = Arc::new(Member::new(allow));
    let b = Arc::new(Member::new(deny));
    let p = list(vec![Arc::clone(&a), Arc::clone(&b)]);
    run(p.check(&principal(), &table("t"), Privilege::Select)).unwrap();
    p.denied(&principal(), &table("t"), Privilege::Select);
    assert!(a.told().is_empty());
    assert_eq!(b.told(), vec![(table("t"), Privilege::Select)]);
    // A refusal the list did not record is told to every member that decides it.
    p.denied(&principal(), &table("u"), Privilege::Select);
    assert_eq!(a.told(), vec![(table("u"), Privilege::Select)]);
}

#[test]
fn prefetch_reaches_each_member_with_its_own_questions() {
    let mut grants = Member::new(allow);
    grants.column_level = true;
    let grants = Arc::new(grants);
    let mut external = Member::new(allow);
    external.decides = |s| !matches!(s, Securable::System(_));
    let external = Arc::new(external);
    let p = list(vec![Arc::clone(&grants), Arc::clone(&external)]);
    let items = vec![
        (column("t", "a"), Privilege::Select),
        (column("t", "b"), Privilege::Select),
        (Securable::System("cluster".to_owned()), Privilege::Manage),
    ];
    run(p.prefetch(&principal(), &items));
    assert_eq!(*grants.prefetched.lock().unwrap(), vec![items]);
    assert_eq!(
        *external.prefetched.lock().unwrap(),
        vec![vec![(table("t"), Privilege::Select)]]
    );
}

#[test]
fn system_only_decides_the_server_objects_alone() {
    let roles = Arc::new(Member::new(allow));
    let only = SystemOnlyPolicy::new(Arc::clone(&roles) as Arc<dyn AccessPolicy>);
    let system = Securable::System("cluster".to_owned());
    assert!(only.decides(&system));
    assert!(!only.decides(&table("t")));
    let d = run(only.check(&principal(), &table("t"), Privilege::Select)).unwrap();
    assert!(matches!(d, AccessDecision::Deny { .. }));
    assert!(roles.asked().is_empty());
    assert!(is_allow(
        &run(only.check(&principal(), &system, Privilege::Manage)).unwrap()
    ));
    only.denied(&principal(), &system, Privilege::Manage);
    assert_eq!(roles.told(), vec![(system, Privilege::Manage)]);
    // In a list, the data securables go to the other member alone.
    let external = Arc::new(Member::new(deny));
    let p = IntersectionPolicy::new(vec![
        Arc::new(only) as Arc<dyn AccessPolicy>,
        Arc::clone(&external) as Arc<dyn AccessPolicy>,
    ]);
    let d = run(p.check(&principal(), &table("t"), Privilege::Select)).unwrap();
    assert!(matches!(d, AccessDecision::Deny { .. }));
    assert_eq!(external.asked().len(), 1);
}
