// SPDX-License-Identifier: Apache-2.0
//! `krishiv server` (design P3-T001): start-up refusals, the two listeners,
//! `/healthz` and `/readyz`, and a clean SIGTERM shutdown.
#![cfg(feature = "server")]
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn krishiv() -> Command {
    Command::new(env!("CARGO_BIN_EXE_krishiv"))
}

#[test]
fn a_bad_setting_is_refused_before_anything_starts() {
    let out = krishiv()
        .args([
            "server",
            "--port",
            "0",
            "--connect-port",
            "0",
            "--conf",
            "krishiv.ui.enabled=maybe",
        ])
        .output()
        .expect("run krishiv server");
    assert_eq!(out.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&out.stdout).contains("Spark Connect on"));
}

#[test]
fn an_unsupported_spark_connect_setting_is_refused() {
    let out = krishiv()
        .args([
            "server",
            "--port",
            "0",
            "--connect-port",
            "0",
            "--conf",
            "spark.connect.session.manager.defaultSessionTimeout=-1",
        ])
        .output()
        .expect("run krishiv server");
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("krishiv.connect.session.idleTimeout"), "{err}");
}

#[test]
fn non_loopback_requires_auth() {
    for host in ["0.0.0.0", "::"] {
        let out = krishiv()
            .args([
                "server",
                "--port",
                "0",
                "--connect-port",
                "0",
                "--host",
                host,
            ])
            .output()
            .expect("run krishiv server");
        assert_eq!(out.status.code(), Some(1), "{host}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("needs an authentication method"), "{err}");
        assert!(err.contains("--host"), "{err}");
        assert!(!String::from_utf8_lossy(&out.stdout).contains("Spark Connect on"));
    }
    // The configured address is checked the same way.
    let out = krishiv()
        .args([
            "server",
            "--port",
            "0",
            "--connect-port",
            "0",
            "--conf",
            "spark.connect.grpc.binding.address=0.0.0.0",
        ])
        .output()
        .expect("run krishiv server");
    assert_eq!(out.status.code(), Some(1));
    let out = krishiv()
        .args([
            "server",
            "--port",
            "0",
            "--connect-port",
            "0",
            "--conf",
            "krishiv.connect.address=0.0.0.0",
        ])
        .output()
        .expect("run krishiv server");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("needs an authentication method"));
}

/// A running server and the addresses it printed; killed and reaped on
/// drop, so a failed assertion leaves no server behind.
struct Running {
    child: Child,
    admin: String,
    connect: String,
    /// Kept open: the server may print more lines.
    _stdout: std::io::Lines<BufReader<std::process::ChildStdout>>,
}

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start(spill: &std::path::Path) -> Running {
    start_with(spill, &[])
}

/// [`start`] with more `--conf KEY=VALUE` settings.
fn start_with(spill: &std::path::Path, conf: &[String]) -> Running {
    let mut child = krishiv()
        // `--allow-root`: CI containers may run as root.
        .args([
            "server",
            "--allow-root",
            "--port",
            "0",
            "--connect-port",
            "0",
        ])
        .args(["--conf", "krishiv.metrics.enabled=true", "--conf"])
        .arg(format!("krishiv.spill.dir={}", spill.display()))
        .args(conf.iter().flat_map(|c| ["--conf", c.as_str()]))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn krishiv server");
    let mut lines = BufReader::new(child.stdout.take().expect("stdout")).lines();
    let (mut admin, mut connect) = (None, None);
    while admin.is_none() || connect.is_none() {
        let line = lines
            .next()
            .expect("server printed its addresses")
            .expect("utf-8");
        if let Some(a) = line
            .strip_prefix("krishiv server: admin on http://")
            .or_else(|| line.strip_prefix("krishiv server: admin on https://"))
        {
            admin = Some(a.to_owned());
        }
        if let Some(a) = line.strip_prefix("krishiv server: Spark Connect on sc://") {
            connect = Some(a.trim_end_matches("/;use_ssl=true").to_owned());
        }
    }
    Running {
        child,
        admin: admin.expect("admin"),
        connect: connect.expect("connect"),
        _stdout: lines,
    }
}

fn http_get(addr: &str, path: &str) -> String {
    http_get_as(addr, path, None)
}

/// `GET path` with `Authorization: Bearer token` when `token` is given.
fn http_get_as(addr: &str, path: &str, token: Option<&str>) -> String {
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: x\r\n{auth}Connection: close\r\n\r\n"
    )
    .expect("write");
    let mut out = String::new();
    s.read_to_string(&mut out).expect("read");
    out
}

fn wait_exit(child: &mut Child, within: Duration) -> std::process::ExitStatus {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().expect("wait") {
            return status;
        }
        assert!(
            start.elapsed() < within,
            "server did not exit in {within:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(unix)]
#[test]
fn sigterm_shuts_down_cleanly_and_removes_the_spill_directory() {
    let spill = tempfile::tempdir().expect("tempdir");
    let mut run = start(spill.path());
    let h = http_get(&run.admin, "/healthz");
    assert!(h.starts_with("HTTP/1.1 200"), "{h}");
    let r = http_get(&run.admin, "/readyz");
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    assert!(r.ends_with(r#"{"status":"ready"}"#), "{r}");
    // The Connect port accepts connections.
    TcpStream::connect(&run.connect).expect("connect port is bound");
    // Without `ui` there is no page; with it, the SPA answers.
    let page = http_get(&run.admin, "/");
    if cfg!(feature = "ui") {
        assert!(page.starts_with("HTTP/1.1 200"), "{page}");
    } else {
        assert!(page.starts_with("HTTP/1.1 404"), "{page}");
    }
    // `/metrics` is on (`start` sets it) and needs the admin token.
    let m = http_get(&run.admin, "/metrics");
    assert!(m.starts_with("HTTP/1.1 401"), "{m}");

    let status = Command::new("kill")
        .args(["-TERM", &run.child.id().to_string()])
        .status()
        .expect("kill");
    assert!(status.success());
    let exit = wait_exit(&mut run.child, Duration::from_secs(20));
    assert_eq!(exit.code(), Some(0), "clean exit");
    let left: Vec<_> = std::fs::read_dir(spill.path())
        .expect("spill root")
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .filter(|n| n.to_string_lossy().starts_with("krishiv-spill-"))
        .collect();
    assert!(left.is_empty(), "spill directories left: {left:?}");
}

/// Runs `script` with a PySpark 4.x Connect client against a fresh server,
/// `SPARK_REMOTE` set to its Connect address.
fn pyspark(script: &str) {
    let spill = tempfile::tempdir().expect("tempdir");
    let run = start(spill.path());
    let out = Command::new("python3")
        .args(["-c", script])
        .env("SPARK_REMOTE", format!("sc://{}", run.connect))
        .output()
        .expect("python3");
    drop(run);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
#[ignore = "needs PR-11 (ExecutePlan, Config: P3-T011a)"]
fn sc_localhost_runs_spark_range_collect() {
    pyspark(
        "from pyspark.sql import SparkSession\n\
         s = SparkSession.builder.getOrCreate()\n\
         assert [r.id for r in s.range(3).collect()] == [0, 1, 2]\n\
         s.stop()\n",
    );
}

/// Design P3-T011a §5 `pyspark_minimal_reattach`: the stock client asks
/// for a reattachable execution, gets `ResultComplete`, sends
/// `ReleaseExecute` after the responses, and logs no warning or error.
/// The Connect logger (`pyspark.sql.connect.logging`) is disabled unless
/// `SPARK_CONNECT_LOG_LEVEL` is set when it is imported; a failed release
/// is a warning there (`reattach.py`, "ReleaseExecute failed").
#[test]
#[ignore = "needs pyspark-client 4.3.0 / 4.2.1 on the host (none on CI runners) and the Config RPC (PR-11b)"]
fn pyspark_minimal_reattach() {
    pyspark(
        "import logging, os\n\
         os.environ['SPARK_CONNECT_LOG_LEVEL'] = 'warning'\n\
         from pyspark.sql.connect.logging import logger\n\
         records = []\n\
         class Keep(logging.Handler):\n\
         \x20   def emit(self, record):\n\
         \x20       records.append(record)\n\
         logger.addHandler(Keep())\n\
         from pyspark.sql import SparkSession\n\
         s = SparkSession.builder.getOrCreate()\n\
         assert [r.id for r in s.range(10).collect()] == list(range(10))\n\
         s.stop()\n\
         bad = [r.getMessage() for r in records if r.levelno >= logging.WARNING]\n\
         assert not bad, bad\n",
    );
}

#[test]
#[ignore = "needs PR-11 (ExecutePlan, Config: P3-T011a)"]
fn server_sessions_are_isolated() {
    pyspark(
        "from pyspark.sql import SparkSession\n\
         a = SparkSession.builder.create()\n\
         b = SparkSession.builder.create()\n\
         a.sql('CREATE TEMPORARY VIEW v AS SELECT 1 AS x')\n\
         a.conf.set('spark.sql.ansi.enabled', 'false')\n\
         assert 'v' not in [t.name for t in b.catalog.listTables()]\n\
         assert b.conf.get('spark.sql.ansi.enabled') == 'true'\n\
         a.stop(); b.stop()\n",
    );
}

#[test]
#[ignore = "needs PR-11; nightly: a client idle for 3 minutes keeps its session (keepalive)"]
fn idle_client_survives_keepalive() {
    pyspark(
        "import time\n\
         from pyspark.sql import SparkSession\n\
         s = SparkSession.builder.getOrCreate()\n\
         s.range(1).collect()\n\
         time.sleep(180)\n\
         assert s.range(2).count() == 2\n\
         s.stop()\n",
    );
}

#[test]
fn an_inline_auth_secret_is_refused_and_not_echoed() {
    let out = krishiv()
        .args(["server", "--port", "0", "--connect-port", "0", "--conf"])
        .arg("krishiv.auth.tokens.a.secret=inline-canary-secret-0123456789abcdef")
        .output()
        .expect("run krishiv server");
    assert_eq!(out.status.code(), Some(1));
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!all.contains("inline-canary-secret"), "{all}");
    assert!(!all.contains("Spark Connect on"), "{all}");
}

#[cfg(unix)]
#[test]
fn configured_tokens_guard_the_admin_routes() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let token = "prometheus-scrape-token-0123456789abcdef";
    let path = dir.path().join("prom");
    std::fs::write(&path, token).expect("write secret");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let conf = [
        format!("krishiv.auth.tokens.prom.secret=file:{}", path.display()),
        "krishiv.auth.tokens.prom.principal=prometheus".to_owned(),
        "krishiv.auth.tokens.prom.role=viewer".to_owned(),
    ];
    let server = start_with(dir.path(), &conf);
    let none = http_get(&server.admin, "/metrics");
    assert!(none.starts_with("HTTP/1.1 401"), "{none}");
    let scrape = http_get_as(&server.admin, "/metrics", Some(token));
    assert!(scrape.starts_with("HTTP/1.1 200"), "{scrape}");
    // A viewer sees no other principal's work in the UI's data routes (the route exists only
    // with the `ui` feature).
    let ui = http_get_as(&server.admin, "/api/internal/queries", Some(token));
    let expected = if cfg!(feature = "ui") {
        "HTTP/1.1 403"
    } else {
        "HTTP/1.1 404"
    };
    assert!(ui.starts_with(expected), "{ui}");
    let probe = http_get(&server.admin, "/healthz");
    assert!(probe.starts_with("HTTP/1.1 200"), "{probe}");
    // ADR 0046 §8: the refusals above are on `/metrics` (the counters ride on the engine's).
    let scrape = http_get_as(&server.admin, "/metrics", Some(token));
    assert!(
        scrape.contains(r#"krishiv_auth_failed_total{surface="metrics",method="none"} 1"#),
        "{scrape}"
    );
    if cfg!(feature = "ui") {
        assert!(
            scrape.contains(r#"krishiv_auth_denied_total{surface="ui",privilege="MONITOR"} 1"#),
            "{scrape}"
        );
    }
}

#[cfg(unix)]
#[test]
fn tokens_on_a_public_address_need_tls() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("t");
    std::fs::write(&path, "public-address-token-0123456789abcdef").expect("write secret");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let out = krishiv()
        .args([
            "server",
            "--port",
            "0",
            "--connect-port",
            "0",
            "--host",
            "0.0.0.0",
        ])
        .arg("--conf")
        .arg(format!(
            "krishiv.auth.tokens.a.secret=file:{}",
            path.display()
        ))
        .args(["--conf", "krishiv.auth.tokens.a.principal=a"])
        .args(["--conf", "krishiv.auth.tokens.a.role=viewer"])
        .output()
        .expect("run krishiv server");
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("need TLS"), "{err}");
    assert!(!String::from_utf8_lossy(&out.stdout).contains("Spark Connect on"));
}

#[cfg(unix)]
/// The TLS fixtures of `krishiv-session` (`tests/fixtures/tls/README.md`).
fn tls_fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../krishiv-session/tests/fixtures/tls")
        .join(name)
}

/// A TLS connection to `addr`, trusting the test CA, with its handshake done.
#[cfg(unix)]
fn tls_connect(addr: &str) -> rustls::StreamOwned<rustls::ClientConnection, TcpStream> {
    use rustls::pki_types::pem::PemObject as _;
    use rustls::pki_types::{CertificateDer, ServerName};
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(CertificateDer::from_pem_file(tls_fixture("ca.pem")).expect("ca"))
        .expect("add ca");
    let config = rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(
        rustls::crypto::aws_lc_rs::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("versions")
    .with_root_certificates(roots)
    .with_no_client_auth();
    let name = ServerName::try_from("localhost").expect("name");
    let mut conn =
        rustls::ClientConnection::new(std::sync::Arc::new(config), name).expect("client");
    let mut tcp = TcpStream::connect(addr).expect("connect");
    tcp.set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    while conn.is_handshaking() {
        conn.complete_io(&mut tcp).expect("handshake");
    }
    rustls::StreamOwned::new(conn, tcp)
}

/// The leaf certificate the server presented on `tls`, and the response to `GET /healthz` over
/// that connection.
#[cfg(unix)]
fn tls_healthz_on(
    mut tls: rustls::StreamOwned<rustls::ClientConnection, TcpStream>,
) -> (Vec<u8>, String) {
    tls.write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .expect("write");
    let mut out = Vec::new();
    // The server may close without `close_notify`; keep what came.
    let _ = tls.read_to_end(&mut out);
    let leaf = tls
        .conn
        .peer_certificates()
        .and_then(<[_]>::first)
        .expect("server certificate")
        .to_vec();
    (leaf, String::from_utf8_lossy(&out).into_owned())
}

/// [`tls_healthz_on`] over a new connection to `addr`.
#[cfg(unix)]
fn tls_healthz(addr: &str) -> (Vec<u8>, String) {
    tls_healthz_on(tls_connect(addr))
}

#[cfg(unix)]
#[test]
fn tls_serves_the_admin_listener_and_reloads_on_sighup() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let (cert, key) = (dir.path().join("server.pem"), dir.path().join("server.key"));
    std::fs::copy(tls_fixture("server.pem"), &cert).expect("cert");
    std::fs::copy(tls_fixture("server.key"), &key).expect("key");
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let conf = [
        format!("krishiv.server.tls.certFile={}", cert.display()),
        format!("krishiv.server.tls.keyFile={}", key.display()),
    ];
    let mut server = start_with(dir.path(), &conf);
    let (leaf, answer) = tls_healthz(&server.admin);
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    let first = std::fs::read(tls_fixture("server.der")).expect("der");
    assert_eq!(leaf, first);
    // Plain HTTP on the TLS port gets no HTTP answer.
    let mut plain = TcpStream::connect(&server.admin).expect("connect");
    plain
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    let _ = plain.write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n");
    let mut out = Vec::new();
    let _ = plain.read_to_end(&mut out);
    assert!(
        !out.starts_with(b"HTTP/"),
        "{:?}",
        String::from_utf8_lossy(&out)
    );
    // SIGHUP: the second certificate is served to new connections; one opened before keeps
    // its handshake and is served.
    let before = tls_connect(&server.admin);
    std::fs::copy(tls_fixture("server2.pem"), &cert).expect("cert 2");
    std::fs::copy(tls_fixture("server2.key"), &key).expect("key 2");
    // `copy` gives the file the fixture's mode; a key readable by others is refused.
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let second = std::fs::read(tls_fixture("server2.der")).expect("der 2");
    let hup = Command::new("kill")
        .args(["-HUP", &server.child.id().to_string()])
        .status()
        .expect("kill -HUP");
    assert!(hup.success());
    let start = Instant::now();
    loop {
        let (leaf, answer) = tls_healthz(&server.admin);
        assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
        if leaf == second {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "no reload after SIGHUP"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    let (leaf, answer) = tls_healthz_on(before);
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    assert_eq!(leaf, first, "the old connection keeps its certificate");
    let _ = server.child.kill();
}
