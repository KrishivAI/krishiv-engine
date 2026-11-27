// SPDX-License-Identifier: Apache-2.0
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_plan::analyzer::{PrincipalKind, RowColumnPolicy};

use super::*;

/// A member that answers from a rule and records what it was asked, told and prefetched.
struct Member {
    version: u64,
    /// Added to `version`, for a version that changes after the member is shared.
    bump: AtomicU64,
    /// Added to `bump` by every `check`: a policy that changes while it answers.
    bump_per_check: u64,
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
            bump: AtomicU64::new(0),
            bump_per_check: 0,
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
        PolicyVersion(self.version + self.bump.load(Ordering::Relaxed))
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
        self.bump.fetch_add(self.bump_per_check, Ordering::Relaxed);
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
        match d {
            Ok(AccessDecision::Deny { reason }) => assert_eq!(reason, "no"),
            Err(e) => assert!(e.to_string().contains("down"), "{e}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(a.asked(), vec![(table("t"), Privilege::Modify)]);
        assert_eq!(b.asked(), vec![(table("t"), Privilege::Modify)]);
        assert!(c.asked().is_empty(), "asking stops at the refusal");
    }
}

#[test]
fn adding_a_member_never_widens() {
    // A deny-all list member keeps every answer a denial, whatever the others say.
    for (name, others) in [
        ("allow", allow as fn(&Securable, Privilege) -> _),
        ("deny", deny),
        ("fail", fail),
    ] {
        let first = Arc::new(Member::new(others));
        let last = Arc::new(Member::new(deny));
        let p = list(vec![Arc::clone(&first), Arc::clone(&last)]);
        let d = run(p.check(&principal(), &table("t"), Privilege::Select));
        let asked = vec![(table("t"), Privilege::Select)];
        assert_eq!(first.asked(), asked);
        match d {
            // The first allowed: the deny-all member was asked and refused.
            Ok(AccessDecision::Deny { reason }) if name == "allow" => {
                assert_eq!(reason, "no");
                assert_eq!(last.asked(), asked);
            }
            // The first refused: the refusal is its own and the deny-all member stays unasked.
            Ok(AccessDecision::Deny { reason }) => {
                assert_eq!(reason, "no");
                assert!(last.asked().is_empty());
            }
            Err(e) => {
                assert!(e.to_string().contains("down"), "{e}");
                assert!(last.asked().is_empty());
            }
            other => panic!("{other:?}"),
        }
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
    run(p.check(&principal(), &table("t"), Privilege::Browse)).unwrap();
    for c in [column("v", "a"), column("t", "a")] {
        let d = run(p.check(&principal(), &c, Privilege::Select)).unwrap();
        assert_eq!(d, AccessDecision::Allow, "{c:?}");
    }
    assert_eq!(
        grants.asked(),
        vec![
            (view.clone(), Privilege::Browse),
            (table("t"), Privilege::Browse),
            (column("v", "a"), Privilege::Select),
            (column("t", "a"), Privilege::Select),
        ]
    );
    assert_eq!(
        external.asked(),
        vec![
            (view.clone(), Privilege::Browse),
            (table("t"), Privilege::Browse),
            (view, Privilege::Select),
            (table("t"), Privilege::Select),
        ]
    );
}

fn grants_and_external() -> (Arc<Member>, Arc<Member>, IntersectionPolicy) {
    let mut grants = Member::new(allow);
    grants.column_level = true;
    let grants = Arc::new(grants);
    let external = Arc::new(Member::new(allow));
    let p = list(vec![Arc::clone(&grants), Arc::clone(&external)]);
    (grants, external, p)
}

#[test]
fn a_column_of_a_relation_never_asked_about_is_refused() {
    // Its kind (table or view) is unknown, e.g. forgotten past the memory's bound: the
    // table-level member cannot be asked the right question, so the read is refused.
    let (grants, external, p) = grants_and_external();
    let d = run(p.check(&principal(), &column("v", "a"), Privilege::Select)).unwrap();
    assert_eq!(
        d,
        AccessDecision::Deny {
            reason: "the column's relation was not checked before its columns".to_owned()
        }
    );
    assert_eq!(grants.asked(), vec![(column("v", "a"), Privilege::Select)]);
    assert!(external.asked().is_empty());
    // Past the bound the memory is cleared: a relation noted before is unknown again.
    let view = Securable::View(ident("v"));
    run(p.check(&principal(), &view, Privilege::Browse)).unwrap();
    for i in 0..REMEMBERED {
        p.note_kind(&table(&format!("t{i}")));
    }
    let d = run(p.check(&principal(), &column("v", "a"), Privilege::Select)).unwrap();
    assert!(matches!(d, AccessDecision::Deny { .. }), "{d:?}");
}

#[test]
fn a_later_table_hint_never_turns_a_view_column_into_a_table_question() {
    // One analysis browses view `v`; another session stage's prefetch names `v` as a table; the
    // first analysis's column read is still asked of the view (and, as both kinds were seen, of
    // the table too: every question must be allowed).
    let (_grants, external, p) = grants_and_external();
    let view = Securable::View(ident("v"));
    run(p.check(&principal(), &view, Privilege::Browse)).unwrap();
    run(p.prefetch(&principal(), &[(table("v"), Privilege::Select)]));
    let d = run(p.check(&principal(), &column("v", "a"), Privilege::Select)).unwrap();
    assert_eq!(d, AccessDecision::Allow);
    assert_eq!(
        external.asked(),
        vec![
            (view.clone(), Privilege::Browse),
            (table("v"), Privilege::Select),
            (view, Privilege::Select),
        ]
    );
    // A member that refuses the view refuses the column, whatever the table answer is.
    fn deny_views(s: &Securable, _: Privilege) -> Result<AccessDecision, KrishivError> {
        if matches!(s, Securable::View(_)) {
            deny(s, Privilege::Select)
        } else {
            allow(s, Privilege::Select)
        }
    }
    let mut grants = Member::new(allow);
    grants.column_level = true;
    let p = list(vec![Arc::new(grants), Arc::new(Member::new(deny_views))]);
    p.note_kind(&Securable::View(ident("v")));
    p.note_kind(&table("v"));
    let d = run(p.check(&principal(), &column("v", "a"), Privilege::Select)).unwrap();
    assert!(matches!(d, AccessDecision::Deny { .. }), "{d:?}");
}

#[test]
fn a_member_deciding_relations_only_is_asked_about_a_column_read() {
    // It does not decide `Column(..)` itself, but a column read is a question about its relation.
    let mut grants = Member::new(allow);
    grants.column_level = true;
    let grants = Arc::new(grants);
    let mut tables = Member::new(deny);
    tables.decides = |s| matches!(s, Securable::Table(_) | Securable::View(_));
    let tables = Arc::new(tables);
    let p = list(vec![Arc::clone(&grants), Arc::clone(&tables)]);
    p.note_kind(&table("t"));
    let d = run(p.check(&principal(), &column("t", "a"), Privilege::Select)).unwrap();
    assert_eq!(
        d,
        AccessDecision::Deny {
            reason: "no".to_owned()
        }
    );
    assert_eq!(tables.asked(), vec![(table("t"), Privilege::Select)]);
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
    // Bind asks about the relation before the column pass prefetches its columns.
    p.note_kind(&table("t"));
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

/// A stage with nothing for a member still reaches it, as an empty list: the member replaces
/// its previous stage's state (the `external` provider's prefetched answers) with nothing.
#[test]
fn a_stage_with_nothing_for_a_member_still_reaches_it_empty() {
    let grants = Arc::new(Member::new(allow));
    let mut external = Member::new(allow);
    external.decides = |s| !matches!(s, Securable::System(_));
    let external = Arc::new(external);
    let p = list(vec![Arc::clone(&grants), Arc::clone(&external)]);
    let items = vec![(Securable::System("cluster".to_owned()), Privilege::Manage)];
    run(p.prefetch(&principal(), &items));
    run(p.prefetch(&principal(), &[]));
    assert_eq!(*grants.prefetched.lock().unwrap(), vec![items, Vec::new()]);
    assert_eq!(
        *external.prefetched.lock().unwrap(),
        vec![Vec::<(Securable, Privilege)>::new(), Vec::new()]
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

#[test]
fn a_table_level_member_is_asked_each_relation_once_per_stage() {
    // Review m3: a 3-column read of `t` asks the table-level member `Select` on `t` once.
    let (grants, external, p) = grants_and_external();
    let ana = principal();
    run(p.prefetch(&ana, &[(table("t"), Privilege::Browse)]));
    run(p.check(&ana, &table("t"), Privilege::Browse)).unwrap();
    let columns = [column("t", "a"), column("t", "b"), column("t", "c")];
    for c in &columns {
        assert!(is_allow(&run(p.check(&ana, c, Privilege::Select)).unwrap()));
    }
    let selects = |m: &Member| {
        m.asked()
            .into_iter()
            .filter(|(_, p)| *p == Privilege::Select)
            .count()
    };
    assert_eq!(
        selects(&grants),
        3,
        "the column-level member is asked per column"
    );
    assert_eq!(selects(&external), 1);
    // Another principal is asked on its own.
    let bob = Principal::new("bob", PrincipalKind::User);
    assert!(is_allow(
        &run(p.check(&bob, &columns[0], Privilege::Select)).unwrap()
    ));
    assert_eq!(selects(&external), 2);
    // A new version of the member: the answer is asked again.
    external.bump.store(1, Ordering::Relaxed);
    assert!(is_allow(
        &run(p.check(&ana, &columns[1], Privilege::Select)).unwrap()
    ));
    assert_eq!(selects(&external), 3);
    assert!(is_allow(
        &run(p.check(&ana, &columns[2], Privilege::Select)).unwrap()
    ));
    assert_eq!(selects(&external), 3);
    // A new stage starts with no answers.
    run(p.prefetch(&ana, &[]));
    assert!(is_allow(
        &run(p.check(&ana, &columns[0], Privilege::Select)).unwrap()
    ));
    assert_eq!(selects(&external), 4);
}

#[test]
fn an_answer_given_while_the_version_changed_serves_no_other_column() {
    // CodeRabbit on #1023: the answer is kept under the version it was decided at, and not at
    // all when the member's version moved during the check.
    let mut grants = Member::new(allow);
    grants.column_level = true;
    let grants = Arc::new(grants);
    let mut external = Member::new(allow);
    external.bump_per_check = 1;
    let external = Arc::new(external);
    let p = list(vec![Arc::clone(&grants), Arc::clone(&external)]);
    let ana = principal();
    run(p.prefetch(&ana, &[(table("t"), Privilege::Browse)]));
    run(p.check(&ana, &table("t"), Privilege::Browse)).unwrap();
    for c in [column("t", "a"), column("t", "b")] {
        assert!(is_allow(
            &run(p.check(&ana, &c, Privilege::Select)).unwrap()
        ));
    }
    let selects = external
        .asked()
        .into_iter()
        .filter(|(_, p)| *p == Privilege::Select)
        .count();
    assert_eq!(selects, 2, "each column asks the member again");
}

#[test]
fn a_refused_relation_is_asked_again_for_every_column() {
    // Only an allowed answer serves the stage: a refusal is never remembered as one.
    let mut grants = Member::new(allow);
    grants.column_level = true;
    let grants = Arc::new(grants);
    let external = Arc::new(Member::new(deny));
    let p = list(vec![Arc::clone(&grants), Arc::clone(&external)]);
    run(p.check(&principal(), &table("t"), Privilege::Browse)).unwrap();
    for c in [column("t", "a"), column("t", "b")] {
        let d = run(p.check(&principal(), &c, Privilege::Select)).unwrap();
        assert!(matches!(d, AccessDecision::Deny { .. }), "{d:?}");
    }
    let selects = external
        .asked()
        .into_iter()
        .filter(|(_, p)| *p == Privilege::Select)
        .count();
    assert_eq!(selects, 2);
}
