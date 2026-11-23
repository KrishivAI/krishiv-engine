// SPDX-License-Identifier: Apache-2.0
//! The `external` provider against an in-process `krishiv.access.v1` service.

use std::sync::Mutex;

use krishiv_common::config::Config;
use krishiv_common::secret::OptionValue;
use krishiv_session::PrincipalKind;

use super::*;

/// What the fake service received: each request with its `authorization` header.
type Seen = Arc<Mutex<Vec<(pb::CheckRequest, Option<String>)>>>;

/// How the fake service answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Allow every table whose name does not start with `secret`, deny the rest.
    Rule,
    /// Answer one decision too few.
    Short,
    /// Answer an enum value no build knows.
    Unknown,
    /// Answer after the client's deadline.
    Slow,
    /// Fail the call.
    Fail,
    /// Answer by the rule, each call under a policy version one higher than the last.
    Moving,
}

struct Fake {
    mode: Mode,
    version: u64,
    requests: Seen,
    calls: std::sync::atomic::AtomicU64,
}

fn allows(item: &pb::Item) -> bool {
    match item.securable.as_ref().and_then(|s| s.kind.as_ref()) {
        Some(pb::securable::Kind::Table(t)) => !t.name.starts_with("secret"),
        Some(pb::securable::Kind::Column(c)) => c.column != "ssn",
        Some(pb::securable::Kind::Path(p)) => p.starts_with("s3://lake/"),
        Some(_) => true,
        None => false,
    }
}

#[tonic::async_trait]
impl AccessPolicyService for Fake {
    async fn check(
        &self,
        request: tonic::Request<pb::CheckRequest>,
    ) -> Result<tonic::Response<pb::CheckResponse>, tonic::Status> {
        let auth = request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let message = request.into_inner();
        self.requests.lock().unwrap().push((message.clone(), auth));
        let mut decisions: Vec<i32> = message
            .items
            .iter()
            .map(|i| {
                if allows(i) {
                    pb::Decision::Allow.into()
                } else {
                    pb::Decision::Deny.into()
                }
            })
            .collect();
        match self.mode {
            Mode::Rule | Mode::Moving => {}
            Mode::Short => {
                decisions.pop();
            }
            Mode::Unknown => decisions.iter_mut().for_each(|d| *d = 99),
            Mode::Slow => tokio::time::sleep(Duration::from_secs(2)).await,
            Mode::Fail => return Err(tonic::Status::unavailable("down")),
        }
        let call = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let policy_version = match self.mode {
            Mode::Moving => self.version + call,
            _ => self.version,
        };
        Ok(tonic::Response::new(pb::CheckResponse {
            decisions,
            policy_version,
        }))
    }

    async fn version(
        &self,
        _: tonic::Request<pb::VersionRequest>,
    ) -> Result<tonic::Response<pb::VersionResponse>, tonic::Status> {
        Ok(tonic::Response::new(pb::VersionResponse {
            policy_version: self.version + 1,
        }))
    }
}

struct Service {
    access: Arc<ExternalAccess>,
    requests: Seen,
    _server: tokio::task::JoinHandle<()>,
}

impl Service {
    fn requests(&self) -> Vec<(pb::CheckRequest, Option<String>)> {
        self.requests.lock().unwrap().clone()
    }
}

async fn serve(mode: Mode, token: Option<&str>) -> Service {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let fake = Fake {
        mode,
        version: 41,
        requests: Arc::clone(&requests),
        calls: std::sync::atomic::AtomicU64::new(0),
    };
    let incoming =
        tonic::transport::server::TcpIncoming::bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = incoming.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let _ = tonic::transport::Server::builder()
            .add_service(AccessPolicyServer::new(fake))
            .serve_with_incoming(incoming)
            .await;
    });
    let channel = Endpoint::parse(&format!("http://{addr}"))
        .unwrap()
        .channel()
        .unwrap();
    let token = token.map(|t| format!("Bearer {t}").try_into().unwrap());
    Service {
        access: ExternalAccess::over(
            AccessPolicyClient::new(channel),
            Duration::from_millis(500),
            token,
            false,
        ),
        requests,
        _server: server,
    }
}

fn ident(name: &str) -> krishiv_plan::analyzer::TableIdent {
    krishiv_plan::analyzer::TableIdent {
        catalog: "spark_catalog".to_owned(),
        namespace: vec!["default".to_owned()],
        name: name.to_owned(),
    }
}

fn table(name: &str) -> Securable {
    Securable::Table(ident(name))
}

fn ana() -> Principal {
    let mut p = Principal::new("ana", PrincipalKind::User);
    p.groups = vec!["eng".to_owned()];
    p.tenant = Some("acme".to_owned());
    p.delegated_by = Some("etl".to_owned());
    p
}

fn allowed(d: &AccessDecision) -> bool {
    matches!(d, AccessDecision::Allow)
}

#[tokio::test]
async fn the_service_decides_each_pair() {
    let s = serve(Mode::Rule, Some("t0ken")).await;
    let policy = s.access.policy();
    let p = ana();
    assert!(allowed(
        &policy
            .check(&p, &table("t"), Privilege::Select)
            .await
            .unwrap()
    ));
    assert!(!allowed(
        &policy
            .check(&p, &table("secret_t"), Privilege::Select)
            .await
            .unwrap()
    ));
    let requests = s.requests();
    assert_eq!(requests.len(), 2);
    let (first, auth) = &requests[0];
    assert_eq!(auth.as_deref(), Some("Bearer t0ken"));
    let who = first.principal.as_ref().unwrap();
    assert_eq!(who.name, "ana");
    assert_eq!(who.groups, vec!["eng".to_owned()]);
    assert_eq!(who.tenant, "acme");
    assert_eq!(who.delegated_by, "etl");
    assert_eq!(who.kind, i32::from(pb::PrincipalKind::User));
    assert_eq!(first.items[0].privilege, i32::from(pb::Privilege::Select));
    // The version a response carried is the policy's.
    assert_eq!(policy.version(), PolicyVersion(41));
    assert_eq!(policy.refusal_label(), Some("external"));
}

#[tokio::test]
async fn a_prefetch_overtaken_by_a_later_one_stores_nothing() {
    use futures::FutureExt as _;
    let s = serve(Mode::Rule, None).await;
    let policy = s.access.policy();
    let p = ana();
    let first = [(table("a"), Privilege::Select)];
    let second = [(table("b"), Privilege::Select)];
    // The first prefetch starts its call and waits for the service ...
    let mut early = policy.prefetch(&p, &first);
    assert!(
        (&mut early).now_or_never().is_none(),
        "the call is in flight"
    );
    // ... a later stage's prefetch starts and finishes meanwhile ...
    policy.prefetch(&p, &second).await;
    // ... and the first one's answers, arriving last, are dropped.
    early.await;
    assert_eq!(s.access.calls(), 2);
    let b = policy
        .check(&p, &table("b"), Privilege::Select)
        .await
        .unwrap();
    assert!(matches!(b, AccessDecision::Allow), "{b:?}");
    assert_eq!(s.access.calls(), 2, "the later stage's answer is kept");
    let a = policy
        .check(&p, &table("a"), Privilege::Select)
        .await
        .unwrap();
    assert!(matches!(a, AccessDecision::Allow), "{a:?}");
    assert_eq!(s.access.calls(), 3, "the overtaken answer was not stored");
}

#[tokio::test]
async fn a_prefetch_is_one_call_and_answers_each_pair_once() {
    let s = serve(Mode::Rule, None).await;
    let policy = s.access.policy();
    let p = ana();
    let items = vec![
        (table("a"), Privilege::Select),
        (table("secret_b"), Privilege::Select),
        (Securable::System("cluster".to_owned()), Privilege::Manage),
    ];
    policy.prefetch(&p, &items).await;
    assert_eq!(s.access.calls(), 1);
    // `System(..)` is not the service's by default: never sent.
    assert_eq!(s.requests()[0].0.items.len(), 2);
    assert!(allowed(
        &policy
            .check(&p, &table("a"), Privilege::Select)
            .await
            .unwrap()
    ));
    assert!(!allowed(
        &policy
            .check(&p, &table("secret_b"), Privilege::Select)
            .await
            .unwrap()
    ));
    assert_eq!(s.access.calls(), 1, "both answered from the prefetch");
    // Each prefetched answer serves one check: the next asks again.
    policy
        .check(&p, &table("a"), Privilege::Select)
        .await
        .unwrap();
    assert_eq!(s.access.calls(), 2);
    // Another principal never reads ana's answers.
    policy.prefetch(&p, &items[..1]).await;
    let bob = Principal::new("bob", PrincipalKind::User);
    policy
        .check(&bob, &table("a"), Privilege::Select)
        .await
        .unwrap();
    assert_eq!(s.access.calls(), 4);
    // A securable it does not decide is refused locally.
    let d = policy
        .check(
            &p,
            &Securable::System("cluster".to_owned()),
            Privilege::Manage,
        )
        .await
        .unwrap();
    assert!(!allowed(&d));
    assert_eq!(s.access.calls(), 4);
}

#[tokio::test]
async fn a_large_prefetch_is_split_into_calls_of_the_cap() {
    let s = serve(Mode::Rule, None).await;
    let policy = s.access.policy();
    let items: Vec<_> = (0..MAX_ITEMS_PER_CHECK + 3)
        .map(|i| (table(&format!("t{i}")), Privilege::Select))
        .collect();
    policy.prefetch(&ana(), &items).await;
    let sizes: Vec<usize> = s.requests().iter().map(|(r, _)| r.items.len()).collect();
    assert_eq!(sizes, vec![MAX_ITEMS_PER_CHECK, 3]);
}

#[tokio::test]
async fn every_failure_is_a_denial() {
    for mode in [Mode::Short, Mode::Unknown, Mode::Slow, Mode::Fail] {
        let s = serve(mode, None).await;
        let policy = s.access.policy();
        let p = ana();
        let d = policy
            .check(&p, &table("t"), Privilege::Select)
            .await
            .unwrap();
        assert!(matches!(d, AccessDecision::Deny { .. }), "{mode:?}: {d:?}");
        policy
            .prefetch(
                &p,
                &[
                    (table("t"), Privilege::Select),
                    (table("u"), Privilege::Select),
                ],
            )
            .await;
        for name in ["t", "u"] {
            let d = policy
                .check(&p, &table(name), Privilege::Select)
                .await
                .unwrap();
            assert!(
                matches!(d, AccessDecision::Deny { .. }),
                "{mode:?} prefetched {name}: {d:?}"
            );
        }
    }
}

/// A response with the wrong number of decisions is a failed call: the version it names is not
/// trusted, so no analysis made under it is reused.
#[tokio::test]
async fn a_wrong_length_answer_leaves_the_version_untrusted() {
    let s = serve(Mode::Short, None).await;
    let d = s
        .access
        .policy()
        .check(&ana(), &table("t"), Privilege::Select)
        .await
        .unwrap();
    assert!(matches!(d, AccessDecision::Deny { .. }), "{d:?}");
    let v = s.access.current_version();
    assert_ne!(v.0, 41);
    assert_ne!(v.0 & (1 << 63), 0, "{v:?}");
}

/// A prefetch whose calls report two policy versions keeps none of their answers: each check
/// asks the service again.
#[tokio::test]
async fn a_prefetch_across_a_version_change_keeps_nothing() {
    let s = serve(Mode::Moving, None).await;
    let policy = s.access.policy();
    let items: Vec<_> = (0..MAX_ITEMS_PER_CHECK + 1)
        .map(|i| (table(&format!("t{i}")), Privilege::Select))
        .collect();
    policy.prefetch(&ana(), &items).await;
    assert_eq!(s.requests().len(), 2);
    let d = policy
        .check(&ana(), &table("t0"), Privilege::Select)
        .await
        .unwrap();
    assert!(allowed(&d), "{d:?}");
    assert_eq!(s.requests().len(), 3, "the check asked the service itself");
}

#[tokio::test]
async fn a_service_that_is_down_denies() {
    // Nothing listens on a port just freed.
    let addr = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let channel = Endpoint::parse(&format!("http://{addr}"))
        .unwrap()
        .channel()
        .unwrap();
    let access = ExternalAccess::over(
        AccessPolicyClient::new(channel),
        Duration::from_millis(200),
        None,
        false,
    );
    let d = access
        .policy()
        .check(&ana(), &table("t"), Privilege::Select)
        .await
        .unwrap();
    assert!(!allowed(&d));
}

#[tokio::test]
async fn system_securables_are_sent_only_when_configured() {
    let s = serve(Mode::Rule, None).await;
    let system = ExternalAccess::over(
        s.access.client.clone(),
        Duration::from_millis(500),
        None,
        true,
    );
    let d = system
        .policy()
        .check(
            &ana(),
            &Securable::System("cluster".to_owned()),
            Privilege::Manage,
        )
        .await
        .unwrap();
    assert!(allowed(&d));
    assert_eq!(system.calls(), 1);
}

#[tokio::test]
async fn paths_go_without_credentials() {
    let s = serve(Mode::Rule, None).await;
    let policy = s.access.policy();
    for (path, sent) in [
        // User info leaves every URL.
        ("s3://key:Canary-7f3a9c@lake/data", "s3://lake/data"),
        // An http(s) URL's query and fragment are the request's (a signature), not the object's.
        (
            "https://u:Canary-7f3a9c@store.example.com/lake/x?sig=Canary-2b1d#frag",
            "https://store.example.com/lake/x",
        ),
    ] {
        let path = Securable::Path(OptionValue::from_user("path", path));
        policy
            .check(&ana(), &path, Privilege::Select)
            .await
            .unwrap();
        let last = format!("{:?}", s.requests().last().unwrap().0);
        assert!(!last.contains("Canary"), "{last}");
        assert!(last.contains(sent), "{last}");
    }
    // A local path or an object key keeps `?` and `#`: they are part of the name read.
    for name in ["/data/reports/q3#board.csv", "s3://lake/pub/x?draft"] {
        let path = Securable::Path(OptionValue::from_user("path", name));
        policy
            .check(&ana(), &path, Privilege::Select)
            .await
            .unwrap();
        let last = format!("{:?}", s.requests().last().unwrap().0);
        assert!(last.contains(name), "{last}");
    }
}

#[tokio::test]
async fn an_untrusted_version_never_repeats() {
    // Nothing reported yet: every call is a new value, so no analysis keyed by it is reused.
    let s = serve(Mode::Fail, None).await;
    let policy = s.access.policy();
    let a = policy.version();
    let b = policy.version();
    assert_ne!(a, b);
    assert!(a.0 >= 1 << 63 && b.0 >= 1 << 63);
    // A trusted version holds until a call fails; a lower one replaces it (a restarted service).
    s.access.note_version(5);
    s.access.note_version(7);
    assert_eq!(s.access.current_version(), PolicyVersion(7));
    s.access.note_version(5);
    assert_eq!(s.access.current_version(), PolicyVersion(5));
    s.access.note_version(7);
    policy
        .check(&ana(), &table("t"), Privilege::Select)
        .await
        .unwrap();
    assert_ne!(
        s.access.current_version(),
        PolicyVersion(7),
        "the failed check"
    );
    s.access.note_version(0);
    assert_ne!(s.access.current_version(), s.access.current_version());
    s.access.note_version(1 << 63);
    assert!(s.access.current_version().0 >= 1 << 63);
    assert_ne!(s.access.current_version(), s.access.current_version());
}

#[tokio::test]
async fn an_answer_older_than_the_last_recorded_call_is_ignored() {
    // Review D3: a slow answer of an earlier call never replaces a later call's version.
    let s = serve(Mode::Fail, None).await;
    let early = s.access.begin_call();
    let late = s.access.begin_call();
    s.access.note(late, Some(9));
    s.access.note(early, Some(4));
    assert_eq!(s.access.current_version(), PolicyVersion(9));
    s.access.note(early, None);
    assert_eq!(
        s.access.current_version(),
        PolicyVersion(9),
        "an earlier failure"
    );
}

#[tokio::test]
async fn a_stale_version_is_not_trusted() {
    // Review D4: after an idle spell the first statement gets no reusable version.
    let s = serve(Mode::Fail, None).await;
    s.access.note_version(7);
    s.access
        .noted
        .lock()
        .unwrap()
        .at
        .replace(Instant::now() - VERSION_STALE - Duration::from_millis(1));
    let v = s.access.current_version();
    assert!(v.0 >= 1 << 63, "{v:?}");
}

#[tokio::test]
async fn a_prefetched_answer_is_for_the_principal_it_was_asked_for() {
    let s = serve(Mode::Rule, None).await;
    let policy = s.access.policy();
    let via_etl = ana();
    let mut via_adhoc = ana();
    via_adhoc.delegated_by = Some("adhoc".to_owned());
    policy
        .prefetch(&via_etl, &[(table("t"), Privilege::Select)])
        .await;
    policy
        .check(&via_adhoc, &table("t"), Privilege::Select)
        .await
        .unwrap();
    assert_eq!(s.access.calls(), 2, "another delegate asks again");
    // The next prefetch drops what the last one left.
    policy
        .prefetch(&via_etl, &[(table("u"), Privilege::Select)])
        .await;
    policy
        .check(&via_etl, &table("t"), Privilege::Select)
        .await
        .unwrap();
    assert_eq!(s.access.calls(), 4);
}

#[tokio::test]
async fn the_version_is_refreshed_in_the_background() {
    let s = serve(Mode::Rule, None).await;
    let policy = s.access.policy();
    let done = s.access.version_done.notified();
    tokio::pin!(done);
    done.as_mut().enable();
    // The first read starts the background call and has no answer yet.
    assert_ne!(policy.version(), PolicyVersion(42));
    // Woken when the call has recorded its answer; the bound only stops a hung test.
    tokio::time::timeout(Duration::from_secs(30), done)
        .await
        .expect("the background call finished");
    assert_eq!(policy.version(), PolicyVersion(42));
}

#[test]
fn denials_are_counted() {
    let channel = Endpoint::parse("http://127.0.0.1:1").unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = rt.enter();
    let access = ExternalAccess::over(
        AccessPolicyClient::new(channel.channel().unwrap()),
        Duration::from_millis(100),
        None,
        false,
    );
    access
        .policy()
        .denied(&ana(), &table("t"), Privilege::Select);
    assert_eq!(access.denials(), 1);
}

#[test]
fn endpoints_are_checked() {
    for ok in [
        "https://policy.example.com",
        "https://policy.example.com:8443/",
        "http://127.0.0.1:9000",
        "http://[::1]:9000",
        "unix:/run/krishiv/policy.sock",
    ] {
        Endpoint::parse(ok).unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
    for bad in [
        "http://policy.example.com",
        "http://10.0.0.1:9000",
        // A name is resolved when the channel connects: not provably loopback.
        "http://localhost:9000",
        // No host, a token in the query: the error still echoes nothing.
        "https://?token=Canary-7f3a9c",
        "https:///Canary-7f3a9c",
        "https://u:Canary-7f3a9c@policy.example.com",
        "https://policy.example.com/?token=Canary-7f3a9c",
        "ftp://policy.example.com",
        "unix:relative.sock",
        "policy.example.com",
        "https://",
        "svc:Canary-7f3a9c@policy.example.com:443",
    ] {
        let e = Endpoint::parse(bad).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INVALID_CONFIG_VALUE",
            "{bad}"
        );
        assert!(!e.to_string().contains("Canary"), "{bad}: {e}");
    }
}

#[test]
fn the_config_is_read_and_refused_as_documented() {
    let config = |pairs: &[(&str, &str)]| {
        let mut c = Config::new();
        for (k, v) in pairs {
            c.set(k, v).unwrap();
        }
        c
    };
    let c = ExternalConfig::from_config(&config(&[
        ("krishiv.access.external.endpoint", "http://127.0.0.1:9"),
        ("krishiv.access.external.securables", " Data "),
        ("krishiv.access.external.timeout", "1s"),
    ]))
    .unwrap();
    assert!(!c.system);
    assert_eq!(c.timeout, Duration::from_secs(1));
    assert!(c.token.is_none());
    let c = ExternalConfig::from_config(&config(&[(
        "krishiv.access.external.endpoint",
        "http://127.0.0.1:9",
    )]))
    .unwrap();
    assert!(!c.system);
    assert_eq!(c.timeout, Duration::from_millis(200));
    for pairs in [
        &[][..],
        &[
            ("krishiv.access.external.endpoint", "http://127.0.0.1:9"),
            ("krishiv.access.external.token", "Canary-7f3a9c-literal"),
        ][..],
        &[
            ("krishiv.access.external.endpoint", "http://127.0.0.1:9"),
            ("krishiv.access.external.securables", "data,everything"),
        ][..],
        // Until the admin API and UI guards ask the service too (review M3).
        &[
            ("krishiv.access.external.endpoint", "http://127.0.0.1:9"),
            ("krishiv.access.external.securables", "data,system"),
        ][..],
    ] {
        let e = ExternalConfig::from_config(&config(pairs)).unwrap_err();
        assert!(!e.to_string().contains("Canary"), "{e}");
    }
}

#[test]
fn every_securable_and_privilege_has_a_wire_form() {
    let securables = [
        table("t"),
        Securable::View(ident("v")),
        Securable::Column {
            table: ident("t"),
            column: "a".to_owned(),
        },
        Securable::Function(vec!["f".to_owned()]),
        Securable::Path(OptionValue::from_user("path", "/data")),
        Securable::Namespace(vec!["db".to_owned()]),
        Securable::System("cluster".to_owned()),
        Securable::Secret {
            scope: "s".to_owned(),
        },
        Securable::Connection("c".to_owned()),
    ];
    let privileges = [
        Privilege::Select,
        Privilege::Modify,
        Privilege::Create,
        Privilege::Drop,
        Privilege::Monitor,
        Privilege::Usage,
        Privilege::Browse,
        Privilege::Operate,
        Privilege::Manage,
        Privilege::Impersonate,
    ];
    for s in &securables {
        for p in privileges {
            assert!(convert::known(s, p), "{s} {p:?}");
        }
    }
    // Each privilege goes out as its own number in `access.proto` (`PRIVILEGE_SELECT = 1` ...).
    let wire: Vec<i32> = privileges
        .iter()
        .map(|p| i32::from(convert::privilege(*p)))
        .collect();
    assert_eq!(wire, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "{privileges:?}");
}

#[test]
fn only_a_url_loses_its_user_info() {
    // Review D5: text before `://` that is no scheme is a local path, sent as written.
    assert_eq!(
        convert::credential_free("s3a://key:secret@bucket/t"),
        "s3a://bucket/t"
    );
    assert_eq!(
        convert::credential_free("https://u@h/p?sig=1"),
        "https://h/p"
    );
    assert_eq!(
        convert::credential_free("/data/x://u@pub/f"),
        "/data/x://u@pub/f"
    );
    assert_eq!(convert::credential_free("1a://u@h/f"), "1a://u@h/f");
}

/// A fake answering by the rule, as `serve` builds it.
fn rule_fake() -> (Fake, Seen) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let fake = Fake {
        mode: Mode::Rule,
        version: 41,
        requests: Arc::clone(&requests),
        calls: std::sync::atomic::AtomicU64::new(0),
    };
    (fake, requests)
}

/// Review m9: the `unix:` endpoint's connector reaches a service on a Unix socket.
#[cfg(unix)]
#[tokio::test]
async fn a_unix_socket_endpoint_reaches_the_service() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("policy.sock");
    let listener = tokio::net::UnixListener::bind(&socket).unwrap();
    let incoming = futures::stream::unfold(listener, |l| async move {
        let accepted = l.accept().await.map(|(stream, _)| stream);
        Some((accepted, l))
    });
    let (fake, requests) = rule_fake();
    let _server = tokio::spawn(async move {
        let _ = tonic::transport::Server::builder()
            .add_service(AccessPolicyServer::new(fake))
            .serve_with_incoming(incoming)
            .await;
    });
    let endpoint = Endpoint::parse(&format!("unix:{}", socket.display())).unwrap();
    let access = ExternalAccess::over(
        AccessPolicyClient::new(endpoint.channel().unwrap()),
        Duration::from_secs(5),
        None,
        false,
    );
    let policy = access.policy();
    assert!(allowed(
        &policy
            .check(&ana(), &table("t"), Privilege::Select)
            .await
            .unwrap()
    ));
    assert!(!allowed(
        &policy
            .check(&ana(), &table("secret"), Privilege::Select)
            .await
            .unwrap()
    ));
    assert_eq!(requests.lock().unwrap().len(), 2);
}

/// Review m9: an `https://` endpoint whose TLS handshake fails (here a service that speaks no
/// TLS, as one whose certificate the platform roots refuse) is a denial, not an error.
#[tokio::test]
async fn an_https_endpoint_that_fails_its_handshake_denies() {
    let (fake, requests) = rule_fake();
    let incoming =
        tonic::transport::server::TcpIncoming::bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = incoming.local_addr().unwrap();
    let _server = tokio::spawn(async move {
        let _ = tonic::transport::Server::builder()
            .add_service(AccessPolicyServer::new(fake))
            .serve_with_incoming(incoming)
            .await;
    });
    let endpoint = Endpoint::parse(&format!("https://{addr}")).unwrap();
    let access = ExternalAccess::over(
        AccessPolicyClient::new(endpoint.channel().unwrap()),
        Duration::from_secs(2),
        None,
        false,
    );
    let d = access
        .policy()
        .check(&ana(), &table("t"), Privilege::Select)
        .await
        .unwrap();
    assert!(!allowed(&d));
    assert!(requests.lock().unwrap().is_empty());
}

/// Review m9: the provider's `Debug` form never shows its bearer token.
#[tokio::test]
async fn the_debug_form_hides_the_token() {
    let s = serve(Mode::Rule, Some("Canary-token-91c2")).await;
    let shown = format!("{:?} {:?}", s.access, s.access.policy());
    assert!(!shown.contains("Canary-token-91c2"), "{shown}");
}
