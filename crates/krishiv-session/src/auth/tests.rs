// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the authentication core.

use std::io::Write as _;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::time::{Duration, Instant};

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::principal::PrincipalKind;

use super::config::SPARK_TOKEN_KEY;
use super::throttle::{MAX_PEERS, key, step};
use super::{
    AuthConfig, AuthFailure, Authenticator, Credentials, FailureThrottle, Method, Role, SecretRef,
    ServerAuth, StaticTokens, Surface, ThrottleConfig, bearer_token, is_auth_key,
};

const TOKEN_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const TOKEN_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn peer(last: u8) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, last)), 40_000)
}

/// A secret file with `contents` and `mode`.
pub(super) fn secret_file(dir: &Path, name: &str, contents: &str, mode: u32) -> String {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
    }
    #[cfg(not(unix))]
    let _ = mode;
    format!("file:{}", path.display())
}

fn config(pairs: &[(&str, &str)]) -> Result<AuthConfig, KrishivError> {
    AuthConfig::from_settings(pairs.iter().copied())
}

fn invalid_config(e: &KrishivError) -> bool {
    e.error_class() == ErrorClass::KrishivInvalidConfigValue
}

/// Two tokens, `a` (operator, user) and `prom` (viewer, service), from 0600 files.
fn two_tokens(dir: &Path) -> AuthConfig {
    let a = secret_file(dir, "a", &format!("{TOKEN_A}\n"), 0o600);
    let b = secret_file(dir, "b", TOKEN_B, 0o400);
    config(&[
        ("krishiv.auth.tokens.a.secret", &a),
        ("krishiv.auth.tokens.a.principal", "alice"),
        ("krishiv.auth.tokens.a.role", "operator"),
        ("krishiv.auth.tokens.prom.secret", &b),
        ("krishiv.auth.tokens.prom.principal", "prometheus"),
        ("krishiv.auth.tokens.prom.kind", "service"),
        ("krishiv.auth.tokens.prom.role", "Viewer"),
    ])
    .unwrap()
}

#[test]
fn roles_are_ordered_and_parse_ignoring_case() {
    assert!(Role::Viewer < Role::Operator && Role::Operator < Role::Admin);
    assert_eq!(Role::parse("ADMIN"), Some(Role::Admin));
    assert_eq!(Role::parse("root"), None);
}

#[test]
fn bearer_header_parsing() {
    assert_eq!(bearer_token("Bearer abc"), Some("abc"));
    assert_eq!(bearer_token("bearer  abc"), Some("abc"));
    assert_eq!(bearer_token("Basic abc"), None);
    assert_eq!(bearer_token("Bearer "), None);
    assert_eq!(bearer_token("Bearer"), None);
}

#[test]
fn credentials_debug_hides_the_token() {
    let c = Credentials::new(peer(1), Surface::Connect).with_bearer(TOKEN_A);
    let shown = format!("{c:?}");
    assert!(!shown.contains(TOKEN_A), "{shown}");
}

#[test]
fn auth_keys_are_recognised() {
    assert!(is_auth_key("krishiv.auth.tokens.a.role"));
    assert!(is_auth_key(SPARK_TOKEN_KEY));
    assert!(!is_auth_key("krishiv.connect.port"));
}

#[test]
fn config_reads_named_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let c = two_tokens(dir.path());
    assert_eq!(c.tokens.len(), 2);
    let prom = c.tokens.iter().find(|t| t.name == "prom").unwrap();
    assert_eq!(prom.role, Role::Viewer);
    assert_eq!(prom.kind, PrincipalKind::Service);
    let a = c.tokens.iter().find(|t| t.name == "a").unwrap();
    assert_eq!(a.kind, PrincipalKind::User);
    assert!(c.is_configured());
}

#[test]
fn config_refusals() {
    // Unknown keys never pass silently.
    for key in [
        "krishiv.auth.tokens.a.colour",
        "krishiv.auth.mode",
        "krishiv.auth.tokens.nofield",
    ] {
        let e = config(&[(key, "x")]).unwrap_err();
        assert!(invalid_config(&e), "{key}: {e}");
    }
    // A secret that is not a reference is refused, and its text is not echoed.
    let e = config(&[("krishiv.auth.tokens.a.secret", TOKEN_A)]).unwrap_err();
    assert!(
        invalid_config(&e) && !e.to_string().contains(TOKEN_A),
        "{e}"
    );
    // Every field but kind is required.
    let e = config(&[
        ("krishiv.auth.tokens.a.secret", "env:X"),
        ("krishiv.auth.tokens.a.principal", "alice"),
    ])
    .unwrap_err();
    assert!(e.to_string().contains("krishiv.auth.tokens.a.role"), "{e}");
    let e = config(&[("krishiv.auth.tokens.a.role", "superuser")]).unwrap_err();
    assert!(invalid_config(&e), "{e}");
    let e = config(&[("krishiv.auth.tokens.a.b.secret", "env:X")]).unwrap_err();
    assert!(invalid_config(&e), "{e}");
}

#[test]
fn spark_key_is_one_admin_token() {
    let c = config(&[(SPARK_TOKEN_KEY, TOKEN_A)]).unwrap();
    assert_eq!(c.tokens.len(), 1);
    let t = &c.tokens[0];
    assert_eq!(
        (t.name.as_str(), t.principal.as_str(), t.role),
        ("spark", "spark", Role::Admin)
    );
    assert!(t.secret.is_inline());
    let c = config(&[
        (SPARK_TOKEN_KEY, "env:SPARK_TOKEN"),
        ("krishiv.auth.sparkTokenPrincipal", "ops"),
    ])
    .unwrap();
    assert_eq!(c.tokens[0].principal, "ops");
    assert_eq!(c.tokens[0].secret, SecretRef::Env("SPARK_TOKEN".to_owned()));
    let e = config(&[
        (SPARK_TOKEN_KEY, TOKEN_A),
        ("krishiv.auth.tokens.spark.secret", "env:X"),
        ("krishiv.auth.tokens.spark.principal", "x"),
        ("krishiv.auth.tokens.spark.role", "viewer"),
    ])
    .unwrap_err();
    assert!(invalid_config(&e), "{e}");
}

#[test]
fn throttle_settings_parse() {
    let c = config(&[
        ("krishiv.auth.failureThrottle.maxFailures", "5"),
        ("krishiv.auth.failureThrottle.window", "2m"),
        ("krishiv.auth.failureThrottle.blockFor", "1500ms"),
    ])
    .unwrap();
    assert_eq!(c.throttle.max_failures, 5);
    assert_eq!(c.throttle.window, Duration::from_mins(2));
    assert_eq!(c.throttle.block_for, Duration::from_millis(1500));
    assert!(!c.is_configured());
    for bad in ["soon", "5d", "-1s"] {
        let e = config(&[("krishiv.auth.failureThrottle.window", bad)]).unwrap_err();
        assert!(invalid_config(&e), "{bad}: {e}");
    }
}

#[test]
fn file_secret_drops_one_line_break() {
    let dir = tempfile::tempdir().unwrap();
    let r = secret_file(dir.path(), "s", "abc\r\n", 0o600);
    let s = SecretRef::parse("k", &r, false)
        .unwrap()
        .resolve("k")
        .unwrap();
    assert_eq!(s.expose_secret(), "abc");
}

#[cfg(unix)]
#[test]
fn a_readable_secret_file_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let r = secret_file(dir.path(), "s", TOKEN_A, 0o644);
    let e = SecretRef::parse("k", &r, false)
        .unwrap()
        .resolve("k")
        .unwrap_err();
    assert!(e.to_string().contains("readable by group or others"), "{e}");
}

#[cfg(unix)]
#[test]
fn a_fifo_secret_is_refused_without_blocking() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fifo");
    let made = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .unwrap();
    assert!(made.success());
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    // No writer: opening the FIFO would block; the path check refuses it first.
    let e = SecretRef::File(path).resolve("k").unwrap_err();
    assert!(e.to_string().contains("not a regular file"), "{e}");
}

#[test]
fn an_oversized_secret_file_is_never_cut_to_fit() {
    // A full-length token, CRLF, then more: refused, not read as the first 16 KiB.
    let dir = tempfile::tempdir().unwrap();
    let text = format!("{}\r\nmore", "z".repeat(16 * 1024));
    let r = secret_file(dir.path(), "s", &text, 0o600);
    let e = SecretRef::parse("k", &r, false)
        .unwrap()
        .resolve("k")
        .unwrap_err();
    assert!(e.to_string().contains("longer than"), "{e}");
}

#[test]
fn an_oversized_secret_refuses_startup() {
    let dir = tempfile::tempdir().unwrap();
    let long = "x".repeat(16 * 1024 + 1);
    let r = secret_file(dir.path(), "long", &long, 0o600);
    let c = config(&[
        ("krishiv.auth.tokens.a.secret", &r),
        ("krishiv.auth.tokens.a.principal", "alice"),
        ("krishiv.auth.tokens.a.role", "admin"),
    ])
    .unwrap();
    let e = StaticTokens::build(&c.tokens).unwrap_err();
    assert!(e.to_string().contains("at most 16384 bytes"), "{e}");

    // The longest accepted secret is one a client can present.
    let max = "y".repeat(16 * 1024);
    let r = secret_file(dir.path(), "max", &format!("{max}\n"), 0o600);
    let c = config(&[
        ("krishiv.auth.tokens.a.secret", &r),
        ("krishiv.auth.tokens.a.principal", "alice"),
        ("krishiv.auth.tokens.a.role", "admin"),
    ])
    .unwrap();
    let tokens = StaticTokens::build(&c.tokens).unwrap();
    assert!(tokens.lookup(&max).is_some());
}

#[test]
fn missing_env_secret_is_refused() {
    let e = SecretRef::Env("KRISHIV_TEST_UNSET_AUTH_VARIABLE".to_owned())
        .resolve("k")
        .unwrap_err();
    assert!(invalid_config(&e), "{e}");
}

#[test]
fn tokens_map_to_their_own_principal() {
    let dir = tempfile::tempdir().unwrap();
    let tokens = StaticTokens::build(&two_tokens(dir.path()).tokens).unwrap();
    let (p, role) = tokens.lookup(TOKEN_A).unwrap();
    assert_eq!((p.name.as_str(), role), ("alice", Role::Operator));
    let (p, role) = tokens.lookup(TOKEN_B).unwrap();
    assert_eq!(
        (p.name.as_str(), p.kind, role),
        ("prometheus", PrincipalKind::Service, Role::Viewer)
    );
    assert!(tokens.lookup("wrong").is_none());
    assert!(tokens.lookup(&TOKEN_A[1..]).is_none());
}

#[test]
fn token_compare_is_constant_time() {
    // Structural: a lookup walks every digest whatever matches (timing is not measured).
    let dir = tempfile::tempdir().unwrap();
    let tokens = StaticTokens::build(&two_tokens(dir.path()).tokens).unwrap();
    assert_eq!(tokens.comparisons_per_lookup(), tokens.len());
}

#[test]
fn short_and_shared_secrets_refuse_startup() {
    let dir = tempfile::tempdir().unwrap();
    let short = secret_file(dir.path(), "short", "tooshort", 0o600);
    let c = config(&[
        ("krishiv.auth.tokens.a.secret", &short),
        ("krishiv.auth.tokens.a.principal", "alice"),
        ("krishiv.auth.tokens.a.role", "admin"),
    ])
    .unwrap();
    let e = StaticTokens::build(&c.tokens).unwrap_err();
    assert!(e.to_string().contains("at least 32 bytes"), "{e}");

    let one = secret_file(dir.path(), "one", TOKEN_A, 0o600);
    let two = secret_file(dir.path(), "two", &format!("{TOKEN_A}\n"), 0o600);
    let c = config(&[
        ("krishiv.auth.tokens.a.secret", &one),
        ("krishiv.auth.tokens.a.principal", "alice"),
        ("krishiv.auth.tokens.a.role", "admin"),
        ("krishiv.auth.tokens.b.secret", &two),
        ("krishiv.auth.tokens.b.principal", "bob"),
        ("krishiv.auth.tokens.b.role", "viewer"),
    ])
    .unwrap();
    let e = StaticTokens::build(&c.tokens).unwrap_err();
    assert!(e.to_string().contains("same secret"), "{e}");
    assert!(!e.to_string().contains(TOKEN_A), "{e}");
}

#[test]
fn throttle_blocks_and_releases() {
    let cfg = ThrottleConfig {
        max_failures: 3,
        window: Duration::from_mins(1),
        block_for: Duration::from_secs(30),
    };
    let t = FailureThrottle::new(cfg);
    let ip = peer(1).ip();
    let now = Instant::now();
    assert!(!t.record_failure(ip, now));
    assert!(!t.record_failure(ip, now));
    assert!(
        t.record_failure(ip, now),
        "the third failure starts the block"
    );
    assert!(t.is_blocked(ip, now + Duration::from_secs(29)));
    assert!(
        !t.is_blocked(peer(2).ip(), now),
        "other peers are not affected"
    );
    assert!(!t.is_blocked(ip, now + Duration::from_secs(30)));
    // A new window starts from zero.
    let later = now + step(cfg);
    assert!(!t.record_failure(ip, later));
    assert!(!t.is_blocked(ip, later));
}

#[test]
fn throttle_window_resets_counts() {
    let cfg = ThrottleConfig {
        max_failures: 2,
        window: Duration::from_secs(10),
        block_for: Duration::from_secs(10),
    };
    let t = FailureThrottle::new(cfg);
    let ip = peer(1).ip();
    let now = Instant::now();
    assert!(!t.record_failure(ip, now));
    assert!(!t.record_failure(ip, now + Duration::from_secs(11)));
    assert!(t.record_failure(ip, now + Duration::from_secs(12)));
}

#[test]
fn throttle_off_and_bounded() {
    let off = FailureThrottle::new(ThrottleConfig {
        max_failures: 0,
        ..ThrottleConfig::default()
    });
    let now = Instant::now();
    for _ in 0..100 {
        assert!(!off.record_failure(peer(1).ip(), now));
    }
    assert!(!off.is_blocked(peer(1).ip(), now));

    let t = FailureThrottle::new(ThrottleConfig::default());
    for i in 0..=MAX_PEERS {
        // One /64 per entry: addresses inside one /64 share a key.
        let ip = IpAddr::V6(std::net::Ipv6Addr::from(u128::try_from(i).unwrap() << 64));
        t.record_failure(ip, now);
    }
    assert_eq!(
        t.tracked(),
        MAX_PEERS,
        "a full table of live entries tracks no more"
    );
    // Once the entries are stale, a new address evicts them.
    let later = now + step(ThrottleConfig::default());
    t.record_failure(peer(9).ip(), later);
    assert_eq!(t.tracked(), 1);
}

#[test]
fn server_auth_chain_and_counters() {
    let dir = tempfile::tempdir().unwrap();
    let auth = ServerAuth::new(&two_tokens(dir.path())).unwrap();
    assert!(auth.is_enabled());
    let now = Instant::now();

    let ok = auth
        .authenticate_at(
            &Credentials::new(peer(1), Surface::Metrics).with_bearer(TOKEN_B),
            now,
        )
        .unwrap();
    assert_eq!(ok.principal.name, "prometheus");
    assert_eq!((ok.role, ok.method), (Some(Role::Viewer), Method::Token));

    let missing = auth.authenticate_at(&Credentials::new(peer(1), Surface::Connect), now);
    assert_eq!(missing, Err(AuthFailure::Missing));
    assert_eq!(
        AuthFailure::Missing.message(),
        "No authentication token provided"
    );

    let header = format!("Bearer {}", "x".repeat(40));
    let wrong = auth.authenticate_at(
        &Credentials::new(peer(1), Surface::Connect).with_authorization_header(&header),
        now,
    );
    assert_eq!(wrong, Err(AuthFailure::Invalid));
    assert_eq!(
        AuthFailure::Invalid.message(),
        "Invalid authentication token"
    );

    let count = |s: Surface, m: Method| {
        auth.failure_counts()
            .into_iter()
            .find(|c| c.surface == s && c.method == m)
            .map(|c| c.count)
    };
    assert_eq!(count(Surface::Connect, Method::None), Some(1));
    assert_eq!(count(Surface::Connect, Method::Token), Some(1));
    assert_eq!(count(Surface::Metrics, Method::Token), Some(0));
    assert_eq!(
        auth.failure_counts().len(),
        Surface::ALL.len() * Method::ALL.len()
    );
}

#[test]
fn server_auth_throttles_a_guessing_peer() {
    let dir = tempfile::tempdir().unwrap();
    let auth = ServerAuth::new(&two_tokens(dir.path())).unwrap();
    let now = Instant::now();
    let guess = Credentials::new(peer(7), Surface::Connect).with_bearer("guess");
    for _ in 0..ThrottleConfig::default().max_failures {
        assert_eq!(auth.authenticate_at(&guess, now), Err(AuthFailure::Invalid));
    }
    // Blocked: even the right token is refused, with a retryable failure.
    let right = Credentials::new(peer(7), Surface::Connect).with_bearer(TOKEN_A);
    let r = auth.authenticate_at(&right, now);
    assert_eq!(r, Err(AuthFailure::Throttled));
    assert!(AuthFailure::Throttled.is_retryable() && !AuthFailure::Invalid.is_retryable());
    // Another peer is unaffected; the blocked one recovers.
    let other = Credentials::new(peer(8), Surface::Connect).with_bearer(TOKEN_A);
    assert!(auth.authenticate_at(&other, now).is_ok());
    assert!(
        auth.authenticate_at(&right, now + Duration::from_secs(61))
            .is_ok()
    );
}

#[test]
fn no_tokens_means_disabled() {
    let auth = ServerAuth::new(&AuthConfig::default()).unwrap();
    assert!(!auth.is_enabled());
}

#[test]
fn authenticator_trait_object() {
    let dir = tempfile::tempdir().unwrap();
    let auth: Box<dyn Authenticator> = Box::new(ServerAuth::new(&two_tokens(dir.path())).unwrap());
    let c = Credentials::new(peer(3), Surface::Admin).with_bearer(TOKEN_A);
    let r = futures::executor::block_on(auth.authenticate(&c)).unwrap();
    assert_eq!(r.principal.name, "alice");
}

#[test]
fn one_principal_has_one_role() {
    let dir = tempfile::tempdir().unwrap();
    let one = secret_file(dir.path(), "one", TOKEN_A, 0o600);
    let two = secret_file(dir.path(), "two", TOKEN_B, 0o600);
    let tokens = |role_b: &str| {
        config(&[
            ("krishiv.auth.tokens.a.secret", &one),
            ("krishiv.auth.tokens.a.principal", "alice"),
            ("krishiv.auth.tokens.a.role", "viewer"),
            ("krishiv.auth.tokens.b.secret", &two),
            ("krishiv.auth.tokens.b.principal", "alice"),
            ("krishiv.auth.tokens.b.role", role_b),
        ])
        .unwrap()
    };
    let e = StaticTokens::build(&tokens("admin").tokens).unwrap_err();
    assert!(e.to_string().contains("exactly one role"), "{e}");
    // Two tokens of one principal with one role (rotation) are fine.
    let t = StaticTokens::build(&tokens("viewer").tokens).unwrap();
    assert_eq!(t.len(), 2);
    // Spark's token is admin: naming a viewer principal `spark` collides too.
    let c = config(&[
        (SPARK_TOKEN_KEY, TOKEN_B),
        ("krishiv.auth.tokens.a.secret", &one),
        ("krishiv.auth.tokens.a.principal", "spark"),
        ("krishiv.auth.tokens.a.role", "viewer"),
    ])
    .unwrap();
    assert!(StaticTokens::build(&c.tokens).is_err());
}

#[test]
fn throttle_durations_are_bounded() {
    for bad in ["0s", "25h", "18446744073709551615s"] {
        for k in [
            "krishiv.auth.failureThrottle.window",
            "krishiv.auth.failureThrottle.blockFor",
        ] {
            let e = config(&[(k, bad)]).unwrap_err();
            assert!(invalid_config(&e), "{k}={bad}: {e}");
        }
    }
    // A config built in code with a huge block does not overflow the clock.
    let t = FailureThrottle::new(ThrottleConfig {
        max_failures: 1,
        window: Duration::from_secs(1),
        block_for: Duration::MAX,
    });
    let now = Instant::now();
    assert!(t.record_failure(peer(1).ip(), now));
    assert!(t.is_blocked(peer(1).ip(), now));
}

#[test]
fn throttle_keys_ipv6_by_64_and_maps_ipv4() {
    let a: IpAddr = "2001:db8:1:2:aaaa::1".parse().unwrap();
    let b: IpAddr = "2001:db8:1:2:bbbb::9".parse().unwrap();
    let c: IpAddr = "2001:db8:1:3::1".parse().unwrap();
    assert_eq!(key(a), key(b));
    assert_ne!(key(a), key(c));
    let mapped: IpAddr = "::ffff:10.0.0.1".parse().unwrap();
    assert_eq!(key(mapped), peer(1).ip());
    let cfg = ThrottleConfig {
        max_failures: 2,
        ..ThrottleConfig::default()
    };
    let t = FailureThrottle::new(cfg);
    let now = Instant::now();
    t.record_failure(a, now);
    assert!(
        t.record_failure(b, now),
        "rotating inside the /64 still counts"
    );
    assert!(t.is_blocked(a, now) && !t.is_blocked(c, now));
}

#[test]
fn a_malformed_header_is_an_invalid_token() {
    let dir = tempfile::tempdir().unwrap();
    let auth = ServerAuth::new(&two_tokens(dir.path())).unwrap();
    let now = Instant::now();
    for header in ["Basic dXNlcjpwYXNz", "Bearer ", "Bearer"] {
        let c = Credentials::new(peer(4), Surface::Connect).with_authorization_header(header);
        assert_eq!(
            auth.authenticate_at(&c, now),
            Err(AuthFailure::Invalid),
            "{header}"
        );
    }
    let tokens = auth
        .failure_counts()
        .into_iter()
        .find(|c| c.surface == Surface::Connect && c.method == Method::Token)
        .map(|c| c.count);
    assert_eq!(tokens, Some(3));
}

#[test]
fn throttled_requests_count_as_throttle() {
    let dir = tempfile::tempdir().unwrap();
    let auth = ServerAuth::new(&two_tokens(dir.path())).unwrap();
    let now = Instant::now();
    let guess = Credentials::new(peer(5), Surface::Ui).with_bearer("guess");
    for _ in 0..=ThrottleConfig::default().max_failures {
        let _ = auth.authenticate_at(&guess, now);
    }
    let throttled = auth
        .failure_counts()
        .into_iter()
        .find(|c| c.surface == Surface::Ui && c.method == Method::Throttle)
        .map(|c| c.count);
    assert_eq!(throttled, Some(1));
}

#[test]
fn failure_logs_never_contain_the_token() {
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Sink {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let sink = Sink::default();
    let writer = sink.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_writer(move || writer.clone())
        .finish();
    let dir = tempfile::tempdir().unwrap();
    let near_miss = format!("{}x", &TOKEN_A[..TOKEN_A.len() - 1]);
    tracing::subscriber::with_default(subscriber, || {
        let auth = ServerAuth::new(&two_tokens(dir.path())).unwrap();
        let now = Instant::now();
        for token in [near_miss.as_str(), TOKEN_A, "short"] {
            let c = Credentials::new(peer(6), Surface::Admin).with_bearer(token);
            let _ = auth.authenticate_at(&c, now);
        }
        let header = format!("Basic {TOKEN_B}");
        let c = Credentials::new(peer(6), Surface::Admin).with_authorization_header(&header);
        let _ = auth.authenticate_at(&c, now);
    });
    let logged = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
    assert!(logged.contains("authentication failed"), "{logged}");
    for secret in [TOKEN_A, TOKEN_B, near_miss.as_str()] {
        assert!(!logged.contains(secret), "{logged}");
    }
}

mod roles {
    use krishiv_plan::analyzer::{PolicyVersion, Privilege, Securable, TableIdent};

    use super::super::{Role, RolePolicy};

    fn table(catalog: &str) -> Securable {
        Securable::Table(TableIdent {
            catalog: catalog.to_owned(),
            namespace: vec!["default".to_owned()],
            name: "t".to_owned(),
        })
    }

    fn system_table(name: &str) -> Securable {
        Securable::Table(TableIdent {
            catalog: "SYSTEM".to_owned(),
            namespace: Vec::new(),
            name: name.to_owned(),
        })
    }

    fn sys(name: &str) -> Securable {
        Securable::System(name.to_owned())
    }

    #[test]
    fn required_roles_follow_adr_0046() {
        let p = RolePolicy::new(Some(Role::Viewer), Role::Admin, PolicyVersion(1));
        let cases = [
            (
                table("spark_catalog"),
                Privilege::Browse,
                Some(Role::Viewer),
            ),
            (
                table("spark_catalog"),
                Privilege::Select,
                Some(Role::Operator),
            ),
            (table("system"), Privilege::Select, Some(Role::Operator)),
            (
                system_table("queries"),
                Privilege::Select,
                Some(Role::Viewer),
            ),
            (
                system_table("CAPABILITIES"),
                Privilege::Select,
                Some(Role::Viewer),
            ),
            (
                system_table("metrics"),
                Privilege::Select,
                Some(Role::Viewer),
            ),
            (
                system_table("memory"),
                Privilege::Select,
                Some(Role::Operator),
            ),
            (sys("udf"), Privilege::Usage, Some(Role::Admin)),
            (
                Securable::Secret {
                    scope: "s".to_owned(),
                },
                Privilege::Usage,
                Some(Role::Operator),
            ),
            (sys("metrics"), Privilege::Monitor, Some(Role::Viewer)),
            (sys("system"), Privilege::Monitor, Some(Role::Operator)),
            (sys("queries"), Privilege::Operate, Some(Role::Operator)),
            (sys("sessions"), Privilege::Operate, Some(Role::Operator)),
            (table("spark_catalog"), Privilege::Modify, Some(Role::Admin)),
            (table("spark_catalog"), Privilege::Create, Some(Role::Admin)),
            (table("spark_catalog"), Privilege::Drop, Some(Role::Admin)),
            (sys("cluster"), Privilege::Manage, Some(Role::Admin)),
            (sys("principals"), Privilege::Impersonate, None),
        ];
        for (securable, privilege, need) in cases {
            assert_eq!(
                p.required(&securable, privilege),
                need,
                "{privilege} on {securable}"
            );
        }
    }

    #[test]
    fn udf_min_role_lowers_client_code_only() {
        let p = RolePolicy::new(Some(Role::Operator), Role::Operator, PolicyVersion(1));
        assert!(p.allows(&sys("udf"), Privilege::Usage));
        let strict = RolePolicy::new(Some(Role::Operator), Role::Admin, PolicyVersion(1));
        assert!(!strict.allows(&sys("udf"), Privilege::Usage));
        assert!(!p.allows(&sys("cluster"), Privilege::Manage));
    }

    #[test]
    fn admin_never_impersonates_and_no_role_holds_nothing() {
        let admin = RolePolicy::new(Some(Role::Admin), Role::Admin, PolicyVersion(1));
        assert!(!admin.allows(&sys("principals"), Privilege::Impersonate));
        let none = RolePolicy::new(None, Role::Admin, PolicyVersion(1));
        assert!(!none.allows(&table("spark_catalog"), Privilege::Browse));
        assert_eq!(none.role(), None);
    }
}
