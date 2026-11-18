// SPDX-License-Identifier: Apache-2.0
//! Tests of the sessions' Python UDF state over a recording host (no workers).

use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::error::{ErrorClass, KrishivError, KrishivUdfNotAvailable};
use krishiv_connect::translate::{PythonUdf, PythonUdtf, UdfHost as _};
use krishiv_exec::python_eval::{PreparedUdf, UdfCallSpec, UdfRuntime};
use krishiv_plan::PythonEvalType;
use krishiv_session::{ANONYMOUS_VERSION_BIT, Session};
use krishiv_types::DataType;

use super::{InlineUdfs, RunnerConf, SessionPython, SessionUdfRuntime, UdfHost};

/// What the host saw: each runtime's owner, registrations and releases.
#[derive(Default)]
struct Log {
    runtimes: Vec<(String, String)>,
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
        session: &str,
        conf: RunnerConf,
    ) -> Arc<dyn SessionUdfRuntime> {
        let mut log = self.log();
        log.runtimes
            .push((principal.to_owned(), session.to_owned()));
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
    let python = SessionPython::new(None, "alice", &s);
    let e = python.register(&s, udf("f", b"x")).expect_err("no pool");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn each_session_gets_a_runtime_of_its_own_owner_and_releases_it() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let a = session();
    let b = session();
    let pa = SessionPython::new(Some(&host), "alice", &a);
    let pb = SessionPython::new(Some(&host), "bob", &b);
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
    let python = SessionPython::new(Some(&host), "alice", &s);
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
    let python = SessionPython::new(Some(&host), "alice", &s);
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
    let python = SessionPython::new(Some(&host), "alice", &source);
    python
        .register(&source, udf("f", b"a"))
        .unwrap_or_else(|e| panic!("{e}"));
    let clone = source.clone_session().unwrap_or_else(|e| panic!("{e}"));
    let copy = python.copy_for("alice", &clone);
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
    let python = SessionPython::new(Some(&host), "alice", &s);
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
    let _copy = python.copy_for("alice", &clone);
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
    let python = SessionPython::new(Some(&host), "alice", &s);
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

fn udtf(name: &str, command: &[u8], return_type: Option<DataType>) -> PythonUdtf {
    PythonUdtf {
        name: name.to_owned(),
        deterministic: false,
        eval_type: 301,
        plan_eval_type: PythonEvalType::ArrowTableUdf,
        return_type,
        command: command.to_vec(),
        python_ver: "3.12".to_owned(),
    }
}

fn row() -> Option<DataType> {
    krishiv_types::parse_data_type("struct<a: bigint>").ok()
}

#[test]
fn register_table_function_registers_the_udtf_and_its_payload_by_its_lower_case_name() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let s = session();
    let python = SessionPython::new(Some(&host), "alice", &s);
    python
        .register_udtf(&s, &udtf("Repeat", b"r", row()))
        .unwrap_or_else(|e| panic!("{e}"));
    python
        .register_udtf(&s, &udtf("Repeat", b"r2", None))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered,
        [(0, "repeat".to_owned(), 1), (0, "repeat".to_owned(), 2)]
    );
    // Without a pool, refused before anything is registered.
    let none = SessionPython::new(None, "alice", &s);
    let e = none
        .register_udtf(&s, &udtf("x", b"x", row()))
        .expect_err("no pool");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn inline_udtfs_get_a_hidden_name_per_definition_and_wait_for_the_grant() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let s = session();
    let python = SessionPython::new(Some(&host), "alice", &s);
    let mut inline = InlineUdfs::execution(&python);
    let first = inline
        .inline_python_udtf(&udtf("Repeat", b"a", row()))
        .unwrap_or_else(|e| panic!("{e}"));
    let again = inline
        .inline_python_udtf(&udtf("Repeat", b"a", row()))
        .unwrap_or_else(|e| panic!("{e}"));
    let other = inline
        .inline_python_udtf(&udtf("Repeat", b"b", row()))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(first, again);
    assert_ne!(first, other);
    assert!(first.starts_with("Repeat#"), "{first}");
    assert!(inline.seen());
    assert!(!inline.analysis_runs_code());
    assert!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .is_empty()
    );
    inline.grant(&s).unwrap_or_else(|e| panic!("{e}"));
    let registered = log
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .registered
        .clone();
    assert_eq!(
        registered,
        [(0, first.to_lowercase(), 1), (0, other.to_lowercase(), 1)]
    );
    // The same definition in a later plan keeps its registration.
    let mut later = InlineUdfs::execution(&python);
    later
        .inline_python_udtf(&udtf("Repeat", b"a", row()))
        .unwrap_or_else(|e| panic!("{e}"));
    later.grant(&s).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .len(),
        2
    );
    // A NUL in the name is refused.
    let e: KrishivError = InlineUdfs::execution(&python)
        .inline_python_udtf(&udtf("a\0b", b"a", row()))
        .expect_err("NUL");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn analysis_types_an_inline_udtf_without_a_payload_until_analyze_needs_one() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let s = session();
    let python = SessionPython::new(Some(&host), "alice", &s);
    // A typed UDTF: in the library for the analyzer, nothing on the workers, no runtime installed.
    let mut typed = InlineUdfs::analysis(&python);
    let name = typed
        .inline_python_udtf(&udtf("T", b"a", row()))
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(!typed.analysis_runs_code());
    typed
        .prepare_analysis(&s, false)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .is_empty()
    );
    // Running it later registers the payload under the version the analysis gave it.
    let mut run = InlineUdfs::execution(&python);
    assert_eq!(
        run.inline_python_udtf(&udtf("T", b"a", row()))
            .unwrap_or_else(|e| panic!("{e}")),
        name
    );
    run.grant(&s).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered,
        [(0, name.to_lowercase(), 1)]
    );
    // A UDTF with `analyze` runs client code at analysis: granted, its payload goes to the workers.
    let mut analyzed = InlineUdfs::analysis(&python);
    analyzed
        .inline_python_udtf(&udtf("A", b"z", None))
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(analyzed.analysis_runs_code());
    analyzed
        .prepare_analysis(&s, true)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .len(),
        2
    );
}

#[test]
fn a_typed_analysis_leaves_the_session_without_a_udf_runtime() {
    let host = Host::default();
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let s = session();
    let python = SessionPython::new(Some(&host), "carol", &s);
    let mut typed = InlineUdfs::analysis(&python);
    typed
        .inline_python_udtf(&udtf("T", b"a", row()))
        .unwrap_or_else(|e| panic!("{e}"));
    typed
        .prepare_analysis(&s, false)
        .unwrap_or_else(|e| panic!("{e}"));
    // No runtime was installed on the session: no plan of it can start a worker.
    assert!(!python.lock().installed);
    assert!(python.lock().payloads.is_empty());
}

#[test]
fn a_registration_under_a_hidden_name_makes_the_inline_udtf_register_again() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let s = session();
    let python = SessionPython::new(Some(&host), "alice", &s);
    let inline_run = || {
        let mut inline = InlineUdfs::execution(&python);
        let name = inline
            .inline_python_udtf(&udtf("Repeat", b"a", row()))
            .unwrap_or_else(|e| panic!("{e}"));
        inline.grant(&s).unwrap_or_else(|e| panic!("{e}"));
        name
    };
    let name = inline_run();
    // The client registers its own UDTF under the hidden name (command 10).
    python
        .register_udtf(&s, &udtf(&name, b"other", row()))
        .unwrap_or_else(|e| panic!("{e}"));
    // The inline call does not resolve to that registration: its definition is registered again,
    // under the library's next version.
    assert_eq!(inline_run(), name);
    let key = name.to_lowercase();
    assert_eq!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered,
        [(0, key.clone(), 1), (0, key.clone(), 2), (0, key, 3)]
    );
}

#[test]
fn a_clone_keeps_the_sources_inline_udtf_without_registering_it_again() {
    let host = Host::default();
    let log = Arc::clone(&host.0);
    let host: Arc<dyn UdfHost> = Arc::new(host);
    let source = session();
    let python = SessionPython::new(Some(&host), "alice", &source);
    let mut inline = InlineUdfs::execution(&python);
    let name = inline
        .inline_python_udtf(&udtf("Repeat", b"a", row()))
        .unwrap_or_else(|e| panic!("{e}"));
    inline.grant(&source).unwrap_or_else(|e| panic!("{e}"));
    let clone = source.clone_session().unwrap_or_else(|e| panic!("{e}"));
    let copy = python.copy_for("alice", &clone);
    // The clone's runtime got the payload at the source's version.
    let key = name.to_lowercase();
    assert_eq!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered,
        [(0, key.clone(), 1), (1, key, 1)]
    );
    // The same relation in the clone resolves to the copied registration: nothing new.
    let mut in_clone = InlineUdfs::execution(&copy);
    assert_eq!(
        in_clone
            .inline_python_udtf(&udtf("Repeat", b"a", row()))
            .unwrap_or_else(|e| panic!("{e}")),
        name
    );
    in_clone.grant(&clone).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        log.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .registered
            .len(),
        2
    );
}
