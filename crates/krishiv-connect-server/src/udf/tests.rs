// SPDX-License-Identifier: Apache-2.0
//! Tests of the sessions' Python UDF state over a recording host (no workers).

use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::error::{
    ErrorClass, KrishivError, KrishivUdfNotAvailable, KrishivUnsupportedFeature,
};
use krishiv_connect::translate::{PythonUdf, UdfHost as _};
use krishiv_exec::python_eval::{PreparedUdf, UdfCallSpec, UdfRuntime};
use krishiv_plan::PythonEvalType;
use krishiv_session::{ANONYMOUS_VERSION_BIT, Session};
use krishiv_types::DataType;

use super::{
    FunctionStore as _, InlineUdfs, RunnerConf, SessionPython, SessionUdfRuntime, UdfHost,
};

/// What the host saw: each runtime's owner, registrations and releases.
#[derive(Default)]
struct Log {
    runtimes: Vec<(String, String)>,
    /// Each runtime's tenant.
    tenants: Vec<Option<String>>,
    /// Each runtime's runner conf source.
    confs: Vec<RunnerConf>,
    registered: Vec<(usize, String, u32)>,
    released: Vec<usize>,
}

#[derive(Default)]
struct Host(Arc<Mutex<Log>>);

impl std::fmt::Debug for Host {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Host")
    }
}

impl Host {
    fn log(&self) -> std::sync::MutexGuard<'_, Log> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

struct Runtime {
    index: usize,
    log: Arc<Mutex<Log>>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Runtime({})", self.index)
    }
}

impl UdfRuntime for Runtime {
    fn prepare(&self, spec: &UdfCallSpec) -> krishiv_exec::Result<Arc<dyn PreparedUdf>> {
        Err(KrishivUdfNotAvailable {
            name: spec.name.clone(),
            version: spec.version.to_string(),
        }
        .into())
    }
}

impl SessionUdfRuntime for Runtime {
    fn register(&self, name: &str, version: u32, _: &PythonUdf) {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .push((self.index, name.to_owned(), version));
    }
    fn release(&self) {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .released
            .push(self.index);
    }
}

impl UdfHost for Host {
    fn session_runtime(
        &self,
        principal: &str,
        tenant: Option<&str>,
        _authenticated: bool,
        session: &str,
        conf: RunnerConf,
    ) -> Arc<dyn SessionUdfRuntime> {
        let mut log = self.log();
        log.runtimes
            .push((principal.to_owned(), session.to_owned()));
        log.tenants.push(tenant.map(str::to_owned));
        log.confs.push(conf);
        Arc::new(Runtime {
            index: log.runtimes.len() - 1,
            log: Arc::clone(&self.0),
        })
    }
}

fn udf(name: &str, command: &[u8]) -> PythonUdf {
    PythonUdf {
        name: name.to_owned(),
        deterministic: true,
        eval_type: 101,
        plan_eval_type: PythonEvalType::ArrowBatched,
        return_type: DataType::Long,
        command: command.to_vec(),
        python_ver: "3.12".to_owned(),
    }
}

fn session() -> Session {
    Session::new().unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn a_server_without_a_pool_refuses_python_udfs() {
    let s = session();
    let python = SessionPython::new(None, "alice", None, true, &s);
    let e = python.register(&s, udf("f", b"x")).expect_err("no pool");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn a_runtime_is_built_for_the_sessions_tenant() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let (a, b, c) = (session(), session(), session());
    let _pa = SessionPython::new(Some(&host), "alice", Some("acme"), true, &a);
    let pb = SessionPython::new(Some(&host), "alice", None, true, &b);
    let _copy = pb.copy_for("alice", Some("globex"), true, &c);
    let log = log.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(
        log.tenants,
        [Some("acme".to_owned()), None, Some("globex".to_owned())]
    );
}

#[test]
fn each_session_gets_a_runtime_of_its_own_owner_and_releases_it() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let a = session();
    let b = session();
    let pa = SessionPython::new(Some(&host), "alice", None, true, &a);
    let pb = SessionPython::new(Some(&host), "bob", None, true, &b);
    {
        let log = log.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(
            log.runtimes,
            [
                ("alice".to_owned(), a.id().to_owned()),
                ("bob".to_owned(), b.id().to_owned())
            ]
        );
    }
    pa.register(&a, udf("Plus", b"p1"))
        .unwrap_or_else(|e| panic!("{e}"));
    pa.register(&a, udf("plus", b"p2"))
        .unwrap_or_else(|e| panic!("{e}"));
    pb.release();
    let log = log.lock().unwrap_or_else(PoisonError::into_inner);
    // Registered under the registry's name and its versions, on alice's runtime only.
    assert_eq!(
        log.registered,
        [(0, "Plus".to_owned(), 1), (0, "plus".to_owned(), 2)]
    );
    assert_eq!(log.released, [1]);
}

#[test]
fn inline_udfs_wait_for_the_grant_and_keep_one_version_per_definition() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let s = session();
    let python = SessionPython::new(Some(&host), "alice", None, true, &s);
    let mut inline = InlineUdfs::execution(&python);
    assert!(!inline.seen());
    let first = inline
        .inline_python_udf(&udf("<lambda>", b"a"))
        .unwrap_or_else(|e| panic!("{e}"));
    let again = inline
        .inline_python_udf(&udf("<lambda>", b"a"))
        .unwrap_or_else(|e| panic!("{e}"));
    let other = inline
        .inline_python_udf(&udf("<lambda>", b"b"))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(first, again);
    assert_ne!(first, other);
    assert_eq!(first.qualifier, ["session"]);
    assert!(inline.seen());
    // Nothing is registered before the grant (the privilege check comes first).
    assert!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .is_empty()
    );
    inline.grant(&s).unwrap_or_else(|e| panic!("{e}"));
    let versions: Vec<u32> = log
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .registered
        .iter()
        .map(|r| r.2)
        .collect();
    assert_eq!(
        versions,
        [ANONYMOUS_VERSION_BIT | 1, ANONYMOUS_VERSION_BIT | 2]
    );
    // A later plan sending the same UDF reuses its version and registers nothing.
    let mut later = InlineUdfs::execution(&python);
    assert_eq!(
        later
            .inline_python_udf(&udf("<lambda>", b"a"))
            .unwrap_or_else(|e| panic!("{e}")),
        first
    );
    later.grant(&s).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .len(),
        2
    );
}

#[test]
fn analysis_names_inline_udfs_and_registers_nothing() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let s = session();
    let python = SessionPython::new(Some(&host), "alice", None, true, &s);
    let mut inline = InlineUdfs::analysis(&python);
    let name = inline
        .inline_python_udf(&udf("f", b"a"))
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(name.name.contains("krishiv.python_udf"));
    inline.grant(&s).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .is_empty()
    );
    // A NUL in the name is refused, never an internal error.
    let e: KrishivError = InlineUdfs::analysis(&python)
        .inline_python_udf(&udf("a\0b", b"a"))
        .expect_err("NUL");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn a_clone_gets_its_own_runtime_with_the_sources_payloads() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let source = session();
    let python = SessionPython::new(Some(&host), "alice", None, true, &source);
    python
        .register(&source, udf("f", b"a"))
        .unwrap_or_else(|e| panic!("{e}"));
    let clone = source.clone_session().unwrap_or_else(|e| panic!("{e}"));
    let copy = python.copy_for("alice", None, true, &clone);
    copy.release();
    let log = log.lock().unwrap_or_else(PoisonError::into_inner);
    // A second runtime, owned by the clone's id, holding the source's registration; releasing the
    // clone's leaves the source's alone.
    assert_eq!(log.runtimes[1], ("alice".to_owned(), clone.id().to_owned()));
    assert_ne!(log.runtimes[0].1, log.runtimes[1].1);
    assert_eq!(
        log.registered,
        [(0, "f".to_owned(), 1), (1, "f".to_owned(), 1)]
    );
    assert_eq!(log.released, [1]);
}

#[test]
fn the_runner_conf_is_read_live_for_every_call() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let s = session();
    let python = SessionPython::new(Some(&host), "alice", None, true, &s);
    python
        .register(&s, udf("f", b"a"))
        .unwrap_or_else(|e| panic!("{e}"));
    let conf = Arc::clone(&log.lock().unwrap_or_else(PoisonError::into_inner).confs[0]);
    let tz = |c: &RunnerConf| c().get("spark.sql.session.timeZone").cloned();
    s.conf()
        .set("spark.sql.session.timeZone", "Asia/Tokyo")
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(tz(&conf).as_deref(), Some("Asia/Tokyo"));
    // A later change, with no new registration or grant, reaches the next call.
    s.conf()
        .set("spark.sql.session.timeZone", "America/Lima")
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(tz(&conf).as_deref(), Some("America/Lima"));
    // Only the runner keys leave the session.
    s.conf()
        .set("my.app.secret", "x")
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(!conf().contains_key("my.app.secret"));
    // A clone's runtime reads the clone's conf, not the source's.
    let clone = s.clone_session().unwrap_or_else(|e| panic!("{e}"));
    let _copy = python.copy_for("alice", None, true, &clone);
    clone
        .conf()
        .set("spark.sql.session.timeZone", "Europe/Oslo")
        .unwrap_or_else(|e| panic!("{e}"));
    let clone_conf = Arc::clone(&log.lock().unwrap_or_else(PoisonError::into_inner).confs[1]);
    assert_eq!(tz(&clone_conf).as_deref(), Some("Europe/Oslo"));
    assert_eq!(tz(&conf).as_deref(), Some("America/Lima"));
}

#[test]
fn a_relation_udf_needs_the_same_grant_as_an_inline_one() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let s = session();
    let python = SessionPython::new(Some(&host), "alice", None, true, &s);
    let map = PythonUdf {
        eval_type: 205,
        plan_eval_type: PythonEvalType::MapPandasIter,
        ..udf("f", b"m")
    };
    let mut inline = InlineUdfs::execution(&python);
    let r = inline
        .relation_python_udf(&map)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (r.name.as_str(), r.version, r.eval_type),
        (
            "f",
            ANONYMOUS_VERSION_BIT | 1,
            PythonEvalType::MapPandasIter
        )
    );
    // The plan calls client code: ExecutePlan checks `Usage` on `System("udf")` before the grant.
    assert!(inline.seen());
    assert!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .is_empty()
    );
    inline.grant(&s).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .len(),
        1
    );
    // An analysis names it and registers nothing.
    let mut analysis = InlineUdfs::analysis(&python);
    let typed = analysis
        .relation_python_udf(&PythonUdf {
            command: b"other".to_vec(),
            ..map
        })
        .unwrap_or_else(|e| panic!("{e}"));
    assert_ne!(typed.version & ANONYMOUS_VERSION_BIT, 0);
    analysis.grant(&s).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .len(),
        1
    );
}

/// A store in memory: tenant -> functions; records who persisted.
#[derive(Default, Debug)]
struct MemStore {
    rows: Mutex<Vec<(Option<String>, String, PythonUdf)>>,
    fail_list: std::sync::atomic::AtomicBool,
}

impl super::FunctionStore for MemStore {
    fn list(&self, tenant: Option<&str>) -> Result<Vec<PythonUdf>, KrishivError> {
        if self.fail_list.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(KrishivUnsupportedFeature {
                feature: "down".to_owned(),
            }
            .into());
        }
        Ok(self
            .rows
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(t, _, _)| t.as_deref() == tenant)
            .map(|(_, _, u)| u.clone())
            .collect())
    }
    fn put(
        &self,
        tenant: Option<&str>,
        owner: &str,
        udf: &PythonUdf,
        override_owner: bool,
    ) -> Result<super::StoreChange, KrishivError> {
        let mut rows = self.rows.lock().unwrap_or_else(PoisonError::into_inner);
        if !override_owner
            && rows
                .iter()
                .any(|(t, o, u)| t.as_deref() == tenant && u.name == udf.name && o != owner)
        {
            return Ok(super::StoreChange::OwnedByOther);
        }
        rows.retain(|(t, _, u)| !(t.as_deref() == tenant && u.name == udf.name));
        rows.push((tenant.map(str::to_owned), owner.to_owned(), udf.clone()));
        Ok(super::StoreChange::Done)
    }
    fn owner(&self, tenant: Option<&str>, name: &str) -> Result<Option<String>, KrishivError> {
        Ok(self
            .rows
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|(t, _, u)| t.as_deref() == tenant && u.name == name)
            .map(|(_, o, _)| o.clone()))
    }
    fn remove(
        &self,
        tenant: Option<&str>,
        name: &str,
        caller: &str,
        override_owner: bool,
    ) -> Result<super::StoreChange, KrishivError> {
        let mut rows = self.rows.lock().unwrap_or_else(PoisonError::into_inner);
        if !override_owner
            && rows
                .iter()
                .any(|(t, o, u)| t.as_deref() == tenant && u.name == name && o != caller)
        {
            return Ok(super::StoreChange::OwnedByOther);
        }
        let before = rows.len();
        rows.retain(|(t, _, u)| !(t.as_deref() == tenant && u.name == name));
        Ok(if rows.len() < before {
            super::StoreChange::Done
        } else {
            super::StoreChange::Absent
        })
    }
}

fn with_store(
    host: &Arc<dyn UdfHost>,
    store: &Arc<MemStore>,
    tenant: Option<&str>,
    s: &Session,
) -> SessionPython {
    let store: Arc<dyn super::FunctionStore> = Arc::clone(store) as _;
    SessionPython::new(Some(host), "alice", tenant, true, s).with_store(Some(&store))
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn a_persisted_udf_reaches_a_later_session_of_the_tenant_only() {
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let store = Arc::new(MemStore::default());
    let (s1, s2, s3) = (session(), session(), session());
    let first = with_store(&host, &store, Some("acme"), &s1);
    first.register(&s1, udf("plus1", b"x")).unwrap();
    first.persist(&s1, "PLUS1", false).await.unwrap();

    let same_tenant = with_store(&host, &store, Some("acme"), &s2);
    assert!(same_tenant.load_pending());
    same_tenant.load_persisted(&s2).await;
    assert!(!same_tenant.load_pending());
    assert_eq!(same_tenant.lock().payloads.len(), 1);
    assert_eq!(same_tenant.lock().payloads[0].0, "plus1");

    let other_tenant = with_store(&host, &store, Some("globex"), &s3);
    other_tenant.load_persisted(&s3).await;
    assert!(other_tenant.lock().payloads.is_empty());
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn persist_needs_a_registered_udf_and_a_store_and_drop_removes() {
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let store = Arc::new(MemStore::default());
    let s = session();
    let python = with_store(&host, &store, None, &s);
    let e = python
        .persist(&s, "nope", false)
        .await
        .expect_err("not registered");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    python.register(&s, udf("f", b"x")).unwrap();
    python.persist(&s, "f", false).await.unwrap();
    assert_eq!(store.rows.lock().unwrap().len(), 1);
    python.persist(&s, "f", true).await.unwrap();
    assert!(store.rows.lock().unwrap().is_empty());

    let bare = SessionPython::new(Some(&host), "alice", None, true, &s);
    assert!(!bare.load_pending());
    let e = bare.persist(&s, "f", false).await.expect_err("no store");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn a_clone_keeps_the_kind_qualified_principal_for_loading_and_ownership() {
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let store = Arc::new(MemStore::default());
    store
        .put(
            Some("p:user:dana"),
            "user:dana",
            &udf("f", b"stored"),
            false,
        )
        .unwrap();
    let (source, clone) = (session(), session());
    let dyn_store: Arc<dyn super::FunctionStore> = Arc::clone(&store) as _;
    let python = SessionPython::new(Some(&host), "user:dana", None, true, &source)
        .with_store(Some(&dyn_store));
    let copy = python.copy_for("user:dana", None, true, &clone);
    // The clone reads dana's key, not another principal's.
    copy.load_persisted(&clone).await;
    assert_eq!(copy.lock().payloads.len(), 1);
    // A persist from the clone replaces dana's row as dana: no `Manage` needed, and the owner stays.
    copy.register(&clone, udf("g", b"clone")).unwrap();
    copy.persist(&clone, "g", false).await.unwrap();
    let rows = store.rows.lock().unwrap();
    let owners: Vec<_> = rows
        .iter()
        .map(|(key, owner, u)| (key.clone(), owner.clone(), u.name.clone()))
        .collect();
    assert!(
        owners.contains(&(
            Some("p:user:dana".to_owned()),
            "user:dana".to_owned(),
            "g".to_owned()
        )),
        "{owners:?}"
    );
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn an_inline_udf_does_not_hold_the_name_of_a_persisted_function() {
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let store = Arc::new(MemStore::default());
    store
        .put(Some("p:alice"), "alice", &udf("f", b"stored"), false)
        .unwrap();
    let s = session();
    let python = with_store(&host, &store, None, &s);
    // An inline UDF named `f` was granted before the load (its first attempt failed, or this is a clone).
    python.lock().payloads.push((
        "f".to_owned(),
        1 | super::ANONYMOUS_VERSION_BIT,
        Arc::new(udf("f", b"inline")),
    ));
    python.load_persisted(&s).await;
    let state = python.lock();
    assert!(
        state.payloads.iter().any(|(n, v, u)| n == "f"
            && v & super::ANONYMOUS_VERSION_BIT == 0
            && u.command == b"stored"),
        "{:?}",
        state
            .payloads
            .iter()
            .map(|(n, v, _)| (n, v))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn persist_never_picks_an_inline_udf_of_the_same_name() {
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let store = Arc::new(MemStore::default());
    let s = session();
    let python = with_store(&host, &store, None, &s);
    python.register(&s, udf("f", b"registered")).unwrap();
    // A later inline UDF named `f` is stored in the same list under a top-bit version.
    python.lock().payloads.push((
        "f".to_owned(),
        1 | super::ANONYMOUS_VERSION_BIT,
        Arc::new(udf("f", b"inline")),
    ));
    python.persist(&s, "f", false).await.unwrap();
    let rows = store.rows.lock().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].2.command, b"registered");
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn persist_picks_the_newest_registration_under_the_registry_name_folding() {
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let store = Arc::new(MemStore::default());
    let s = session();
    let python = with_store(&host, &store, None, &s);
    // `Ä` and `ä` are one name to the registry (`to_lowercase`): the later payload is the function.
    python.register(&s, udf("Ä", b"old")).unwrap();
    python.register(&s, udf("ä", b"new")).unwrap();
    python.persist(&s, "Ä", false).await.unwrap();
    let rows = store.rows.lock().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].2.command, b"new");
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn load_keeps_the_sessions_own_name_retries_after_a_failure_and_skip_stops_it() {
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let store = Arc::new(MemStore::default());
    store
        .put(Some("p:alice"), "bob", &udf("f", b"stored"), false)
        .unwrap();
    store
        .put(Some("p:alice"), "bob", &udf("g", b"stored"), false)
        .unwrap();
    let s = session();
    let python = with_store(&host, &store, None, &s);
    python.register(&s, udf("F", b"mine")).unwrap();

    store
        .fail_list
        .store(true, std::sync::atomic::Ordering::SeqCst);
    python.load_persisted(&s).await;
    assert!(python.load_pending(), "a catalog failure is tried again");
    store
        .fail_list
        .store(false, std::sync::atomic::Ordering::SeqCst);
    python.load_persisted(&s).await;
    let names: Vec<String> = python.lock().payloads.iter().map(|p| p.0.clone()).collect();
    assert_eq!(names, ["F", "g"]);

    let skipped = with_store(&host, &store, None, &session());
    skipped.skip_load();
    assert!(!skipped.load_pending());
}

fn role_session(role: krishiv_session::auth::Role) -> Session {
    use krishiv_session::auth::RolePolicy;
    session().new_session_as(
        krishiv_common::principal::Principal::new(
            "p",
            krishiv_common::principal::PrincipalKind::User,
        ),
        Arc::new(RolePolicy::new(
            Some(role),
            krishiv_session::auth::Role::Admin,
            krishiv_session::PolicyVersion(1),
        )),
    )
}

fn named(
    host: &Arc<dyn UdfHost>,
    store: &Arc<MemStore>,
    who: &str,
    tenant: Option<&str>,
    s: &Session,
) -> SessionPython {
    let store: Arc<dyn super::FunctionStore> = Arc::clone(store) as _;
    SessionPython::new(Some(host), who, tenant, true, s).with_store(Some(&store))
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn another_principal_cannot_replace_or_drop_a_stored_function_without_manage() {
    use krishiv_session::auth::Role;
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let store = Arc::new(MemStore::default());
    let (sa, sb) = (session(), role_session(Role::Operator));
    let alice = named(&host, &store, "alice", Some("acme"), &sa);
    alice.register(&sa, udf("mask", b"alice")).unwrap();
    alice.persist(&sa, "mask", false).await.unwrap();
    // Bob, same tenant, may send code but is not the owner and holds no Manage.
    let bob = named(&host, &store, "bob", Some("acme"), &sb);
    bob.register(&sb, udf("mask", b"bob")).unwrap();
    let e = bob
        .persist(&sb, "mask", false)
        .await
        .expect_err("not the owner");
    assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    let e = bob
        .persist(&sb, "mask", true)
        .await
        .expect_err("not the owner");
    assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    {
        let rows = store.rows.lock().unwrap();
        assert_eq!(
            (rows.len(), rows[0].1.as_str(), rows[0].2.command.as_slice()),
            (1, "alice", b"alice".as_slice())
        );
    }
    // An admin may.
    let sc = role_session(Role::Admin);
    let carol = named(&host, &store, "carol", Some("acme"), &sc);
    carol.register(&sc, udf("mask", b"carol")).unwrap();
    carol.persist(&sc, "mask", false).await.unwrap();
    assert_eq!(store.rows.lock().unwrap()[0].1, "carol");
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn untenanted_authenticated_principals_do_not_share_functions() {
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let store = Arc::new(MemStore::default());
    let (sa, sb) = (session(), session());
    let alice = named(&host, &store, "alice", None, &sa);
    alice.register(&sa, udf("f", b"x")).unwrap();
    alice.persist(&sa, "f", false).await.unwrap();
    let bob = named(&host, &store, "bob", None, &sb);
    bob.load_persisted(&sb).await;
    assert!(bob.lock().payloads.is_empty());
}

/// A store whose `owner` always says "no row", as if another principal's row appeared after the read.
#[derive(Debug, Default)]
struct BlindStore(MemStore);

impl super::FunctionStore for BlindStore {
    fn list(&self, t: Option<&str>) -> Result<Vec<PythonUdf>, KrishivError> {
        self.0.list(t)
    }
    fn put(
        &self,
        t: Option<&str>,
        o: &str,
        u: &PythonUdf,
        ov: bool,
    ) -> Result<super::StoreChange, KrishivError> {
        self.0.put(t, o, u, ov)
    }
    fn owner(&self, _: Option<&str>, _: &str) -> Result<Option<String>, KrishivError> {
        Ok(None)
    }
    fn remove(
        &self,
        t: Option<&str>,
        n: &str,
        c: &str,
        ov: bool,
    ) -> Result<super::StoreChange, KrishivError> {
        self.0.remove(t, n, c, ov)
    }
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn a_row_that_appears_after_the_owner_read_is_still_refused() {
    use krishiv_session::auth::Role;
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let blind = Arc::new(BlindStore::default());
    blind
        .0
        .put(Some("t:acme"), "alice", &udf("mask", b"alice"), false)
        .unwrap();
    let store: Arc<dyn super::FunctionStore> = Arc::clone(&blind) as _;
    let sb = role_session(Role::Operator);
    let bob =
        SessionPython::new(Some(&host), "bob", Some("acme"), true, &sb).with_store(Some(&store));
    bob.register(&sb, udf("mask", b"bob")).unwrap();
    for drop in [false, true] {
        let e = bob.persist(&sb, "mask", drop).await.expect_err("not owner");
        assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    }
    let rows = blind.0.rows.lock().unwrap();
    assert_eq!(
        (rows.len(), rows[0].1.as_str(), rows[0].2.command.as_slice()),
        (1, "alice", b"alice".as_slice())
    );
}

/// A store whose `list` waits until released, to hold the first request mid-load.
#[derive(Debug)]
struct SlowStore {
    inner: MemStore,
    gate: std::sync::Barrier,
    /// Set once `list` is entered (the first request is mid-load).
    entered: std::sync::atomic::AtomicBool,
}

impl super::FunctionStore for SlowStore {
    fn list(&self, tenant: Option<&str>) -> Result<Vec<PythonUdf>, KrishivError> {
        self.entered
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.gate.wait();
        self.inner.list(tenant)
    }
    fn put(
        &self,
        t: Option<&str>,
        o: &str,
        u: &PythonUdf,
        ov: bool,
    ) -> Result<super::StoreChange, KrishivError> {
        self.inner.put(t, o, u, ov)
    }
    fn owner(&self, t: Option<&str>, n: &str) -> Result<Option<String>, KrishivError> {
        self.inner.owner(t, n)
    }
    fn remove(
        &self,
        t: Option<&str>,
        n: &str,
        c: &str,
        ov: bool,
    ) -> Result<super::StoreChange, KrishivError> {
        self.inner.remove(t, n, c, ov)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::disallowed_methods,
    reason = "#[tokio::test] blocks on its own runtime"
)]
async fn a_second_first_request_waits_for_the_load() {
    let host: Arc<dyn UdfHost> = Arc::new(Host::default());
    let slow = Arc::new(SlowStore {
        inner: MemStore::default(),
        gate: std::sync::Barrier::new(2),
        entered: std::sync::atomic::AtomicBool::new(false),
    });
    slow.inner
        .put(Some("p:alice"), "x", &udf("f", b"x"), false)
        .unwrap();
    let s = Arc::new(session());
    let store: Arc<dyn super::FunctionStore> = Arc::clone(&slow) as _;
    let python =
        Arc::new(SessionPython::new(Some(&host), "alice", None, true, &s).with_store(Some(&store)));
    let first = {
        let (python, s) = (Arc::clone(&python), Arc::clone(&s));
        tokio::spawn(async move { python.load_persisted(&s).await })
    };
    // Wait until the first load is inside `list`, then start the second: it must not return early.
    while !slow.entered.load(std::sync::atomic::Ordering::SeqCst) {
        tokio::task::yield_now().await;
    }
    let second = {
        let (python, s) = (Arc::clone(&python), Arc::clone(&s));
        tokio::spawn(async move {
            python.load_persisted(&s).await;
            python.lock().payloads.len()
        })
    };
    // Give the second task every chance to run before the first load ends.
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    assert!(
        !second.is_finished(),
        "the second request waits for the load"
    );
    slow.gate.wait();
    first.await.unwrap();
    assert_eq!(second.await.unwrap(), 1);
    assert!(!python.load_pending());
}
