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

/// `krishiv server` with `confs`, expected to refuse startup: its stderr.
fn refused(confs: &[&str]) -> String {
    let mut cmd = krishiv();
    // `--allow-root`: CI containers may run as root, and the root refusal comes before the
    // catalogs are built.
    cmd.args([
        "server",
        "--port",
        "0",
        "--connect-port",
        "0",
        "--allow-root",
    ]);
    for c in confs {
        cmd.args(["--conf", c]);
    }
    let out = cmd.output().expect("run krishiv server");
    assert_eq!(out.status.code(), Some(1), "{confs:?}");
    assert!(!String::from_utf8_lossy(&out.stdout).contains("Spark Connect on"));
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Design P3-T067 §2.3: declarations are parsed and checked at startup, and a refusal stops it.
#[test]
fn catalog_declarations_are_checked_at_startup() {
    // A type no factory of this build builds is refused by name, never skipped.
    let err = refused(&[
        "krishiv.catalog.h.type=hive",
        "krishiv.catalog.h.uri=thrift://metastore:9083",
    ]);
    assert!(err.contains("`hive`") && err.contains("`h`"), "{err}");
    // In any letter case (review 58a2 B1): never dropped, never a silently wider catalog.
    let err = refused(&[
        "KRISHIV.CATALOG.h.type=hive",
        "krishiv.catalog.h.uri=thrift://metastore:9083",
    ]);
    assert!(err.contains("`hive`"), "{err}");
    let err = refused(&[
        "krishiv.catalog.lake.type=iceberg-rest",
        "krishiv.catalog.lake.uri=https://catalog.example.com/api",
        "Krishiv.catalog.lake.tenants=",
    ]);
    if cfg!(feature = "catalog-iceberg-rest") {
        assert!(err.contains("empty tenant list"), "{err}");
    }
    let err = refused(&[
        "Spark.SQL.Catalog.h=org.apache.iceberg.spark.SparkCatalog",
        "spark.sql.catalog.h.type=hive",
        "spark.sql.catalog.h.uri=thrift://metastore:9083",
    ]);
    assert!(err.contains("`hive`"), "{err}");
    // Spark's default-catalog key in any letter case reaches the registry (review 58a2 N3): a
    // default no declaration names stops startup.
    let err = refused(&["SPARK.SQL.DEFAULTCATALOG=nosuch"]);
    assert!(err.contains("names no declared catalog"), "{err}");
    // One field under two spellings is refused, never merged (review 58a2 N4).
    let err = refused(&[
        "krishiv.catalog.lake.type=iceberg-rest",
        "krishiv.catalog.lake.uri=https://catalog.example.com/api",
        "krishiv.catalog.lake.tenants=acme",
        "KRISHIV.CATALOG.lake.tenants=globex",
    ]);
    assert!(err.contains("set twice"), "{err}");
    assert!(!err.contains("globex"), "{err}");
    // A literal secret is refused without being echoed.
    let err = refused(&[
        "krishiv.catalog.lake.type=iceberg-rest",
        "krishiv.catalog.lake.uri=https://catalog.example.com/api",
        "krishiv.catalog.lake.auth=token",
        "krishiv.catalog.lake.token=literal-catalog-secret",
    ]);
    assert!(!err.contains("literal-catalog-secret"), "{err}");
    if cfg!(feature = "catalog-iceberg-rest") {
        assert!(err.contains("krishiv.catalog.lake.token"), "{err}");
        // The egress rule checks a declared literal uri and token endpoint at startup, in any
        // spelling, naming the key, never the address.
        for (key, conf) in [
            (
                "krishiv.catalog.lake.uri",
                "krishiv.catalog.lake.uri=https://2852039166/api",
            ),
            (
                "krishiv.catalog.lake.tokenEndpoint",
                "krishiv.catalog.lake.tokenEndpoint=https://169.254.169.254/token",
            ),
        ] {
            // One value per key: the tested one replaces the accepted one.
            let confs: Vec<&str> = [
                "krishiv.catalog.lake.type=iceberg-rest",
                "krishiv.catalog.lake.uri=https://catalog.example.com/api",
                "krishiv.catalog.lake.auth=client-credentials",
                "krishiv.catalog.lake.clientId=krishiv",
                "krishiv.catalog.lake.clientSecret=env:PATH",
                "krishiv.catalog.lake.tokenEndpoint=https://idp.example.com/token",
            ]
            .into_iter()
            .map(|c| {
                if c.split_once('=').map(|(k, _)| k) == Some(key) {
                    conf
                } else {
                    c
                }
            })
            .collect();
            let err = refused(&confs);
            assert!(err.contains(key), "{err}");
            assert!(!err.contains("set twice"), "{err}");
            assert!(
                !err.contains("169.254") && !err.contains("2852039166"),
                "{err}"
            );
        }
    } else {
        assert!(err.contains("`iceberg-rest`"), "{err}");
    }
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

/// `GET path` with the given `Host` and, when given, a bearer token.
fn http_get_host(addr: &str, path: &str, host: &str, token: Option<&str>) -> String {
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: {host}\r\n{auth}Connection: close\r\n\r\n"
    )
    .expect("write");
    let mut out = String::new();
    s.read_to_string(&mut out).expect("read");
    out
}

/// `GET path` with `Authorization: Bearer token` when `token` is given.
fn http_get_as(addr: &str, path: &str, token: Option<&str>) -> String {
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Connection: close\r\n\r\n"
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
    // Readiness can follow the printed addresses by a moment (503 `not_ready` until then).
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut r = http_get(&run.admin, "/readyz");
    while r.starts_with("HTTP/1.1 503") && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
        r = http_get(&run.admin, "/readyz");
    }
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
/// `SPARK_REMOTE` set to its Connect address. The interpreter is
/// `KRISHIV_TEST_PYSPARK_PYTHON` (a venv with `pyspark-client` installed),
/// else `python3`. The script runs from a file: under `python -c`,
/// `pyspark-client` 4.3.0 takes the run for its own doctests and looks for a
/// `SPARK_HOME` (`pyspark/sql/connect/utils.py`, `check_dependencies`).
fn pyspark(script: &str) {
    pyspark_with(&[], script);
}

/// [`pyspark`] against a server started with `confs`.
fn pyspark_with(confs: &[&str], script: &str) {
    let spill = tempfile::tempdir().expect("tempdir");
    // The spill root must be private (`krishiv-memory`, `check_root_is_private`);
    // a temporary directory follows the umask.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(spill.path(), std::fs::Permissions::from_mode(0o700))
            .expect("private spill root");
    }
    let scripts = tempfile::tempdir().expect("tempdir");
    let file = scripts.path().join("client.py");
    std::fs::write(&file, script).expect("script");
    let python =
        std::env::var("KRISHIV_TEST_PYSPARK_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let confs: Vec<String> = confs.iter().map(|c| (*c).to_owned()).collect();
    let run = start_with(spill.path(), &confs);
    let out = Command::new(python)
        .arg(&file)
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
#[ignore = "needs a Python with pyspark-client 4.3.0 or 4.2.1 (KRISHIV_TEST_PYSPARK_PYTHON); CI hosts have none"]
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
#[ignore = "needs a Python with pyspark-client 4.3.0 or 4.2.1 (KRISHIV_TEST_PYSPARK_PYTHON); CI hosts have none"]
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

/// Design P3-T011a §3.8 `stream_cap_reattach` with the stock client: a
/// 4 KiB `senderMaxStreamSize` ends each stream without `ResultComplete`;
/// the client reattaches after its last response until it has every row,
/// releasing as it goes, and logs no error.
#[test]
#[ignore = "needs a Python with pyspark-client 4.3.0 or 4.2.1 (KRISHIV_TEST_PYSPARK_PYTHON); CI hosts have none"]
fn pyspark_stream_cap_reattach() {
    pyspark_with(
        &[
            "spark.connect.execute.reattachable.senderMaxStreamSize=4096",
            "spark.connect.execute.reattachable.observerRetryBufferSize=8192",
        ],
        "import io, logging\n\
         from pyspark.sql import SparkSession\n\
         log = io.StringIO()\n\
         logging.getLogger('pyspark').addHandler(logging.StreamHandler(log))\n\
         s = SparkSession.builder.getOrCreate()\n\
         s.conf.set('spark.sql.execution.arrow.maxRecordsPerBatch', '100')\n\
         assert [r.id for r in s.range(20000).collect()] == list(range(20000))\n\
         s.stop()\n\
         assert 'ERROR' not in log.getvalue() and 'Traceback' not in log.getvalue(), log.getvalue()\n",
    );
}

/// Design P3-T011a §3.9 with the stock client: `addArtifacts` of a Python
/// file succeeds, the same file again is a no-op, other bytes under the same
/// name are `ARTIFACT_ALREADY_EXISTS`, and a cached blob answers
/// `ArtifactStatus`.
#[test]
#[ignore = "needs a Python with pyspark-client 4.3.0 or 4.2.1 (KRISHIV_TEST_PYSPARK_PYTHON); CI hosts have none"]
fn pyspark_add_artifacts_pyfile() {
    pyspark(
        "import os, tempfile\n\
         from pyspark.sql import SparkSession\n\
         from pyspark.errors import SparkRuntimeException\n\
         s = SparkSession.builder.getOrCreate()\n\
         d = tempfile.mkdtemp()\n\
         p = os.path.join(d, 'helper_mod.py')\n\
         open(p, 'w').write('def f():\\n    return 1\\n')\n\
         s.addArtifacts(p, pyfile=True)\n\
         s.addArtifacts(p, pyfile=True)\n\
         open(p, 'w').write('def f():\\n    return 2\\n')\n\
         try:\n\
         \x20   s.addArtifacts(p, pyfile=True)\n\
         \x20   raise AssertionError('no error')\n\
         except SparkRuntimeException as e:\n\
         \x20   assert e.getCondition() == 'ARTIFACT_ALREADY_EXISTS', e\n\
         h = s._client._artifact_manager.cache_artifact(b'x' * 100000)\n\
         assert s._client._artifact_manager.is_cached_artifact(h)\n\
         assert not s._client._artifact_manager.is_cached_artifact('0' * 64)\n\
         s.stop()\n",
    );
}

/// Design P3-T011a §3.10 with the stock client: a failing query's status
/// carries an `errorId`, the client fetches the details (`FetchErrorDetails`;
/// its server stack trace is then the root error's class line, without
/// frames), and raises the exception with the class and parameters.
#[test]
#[ignore = "needs a Python with pyspark-client 4.3.0 or 4.2.1 (KRISHIV_TEST_PYSPARK_PYTHON); CI hosts have none"]
fn pyspark_fetches_error_details() {
    pyspark(
        "from pyspark.sql import SparkSession\n\
         from pyspark.errors import AnalysisException\n\
         s = SparkSession.builder.getOrCreate()\n\
         try:\n\
         \x20   s.range(3).select('nope').collect()\n\
         \x20   raise AssertionError('no error')\n\
         except AnalysisException as e:\n\
         \x20   assert e.getCondition().startswith('UNRESOLVED_COLUMN'), e.getCondition()\n\
         \x20   assert e.getMessageParameters().get('objectName') == '`nope`', e.getMessageParameters()\n\
         \x20   assert e.getStackTrace() == 'org.apache.spark.sql.AnalysisException', e.getStackTrace()\n\
         s.conf.set('spark.sql.connect.enrichError.enabled', 'false')\n\
         try:\n\
         \x20   s.range(3).select('nope').collect()\n\
         except AnalysisException as e:\n\
         \x20   assert e.getStackTrace() is None, e.getStackTrace()\n\
         s.stop()\n",
    );
}

#[test]
#[ignore = "needs SqlCommand (P3-T012) and a Python with pyspark-client (KRISHIV_TEST_PYSPARK_PYTHON)"]
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
#[ignore = "nightly: a client idle for 3 minutes keeps its session (keepalive); needs a Python with pyspark-client (KRISHIV_TEST_PYSPARK_PYTHON)"]
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

/// `METHOD path` with a JSON `body` and `Authorization: Bearer token`.
#[cfg(feature = "ui")]
fn http_post_as(addr: &str, path: &str, body: &str, token: &str) -> String {
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    write!(
        s,
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .expect("write");
    let mut out = String::new();
    s.read_to_string(&mut out).expect("read");
    out
}

/// The UI editor's sessions are UI sessions (P3-T101 hooks): a statement the caller's role refuses
/// counts under `surface="ui"`, never `connect`.
#[cfg(all(unix, feature = "ui"))]
#[test]
fn the_editors_denials_count_under_the_ui_surface() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let token = "editor-operator-token-0123456789abcdef";
    let path = dir.path().join("op");
    std::fs::write(&path, token).expect("write secret");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let conf = [
        format!("krishiv.auth.tokens.op.secret=file:{}", path.display()),
        "krishiv.auth.tokens.op.principal=editor".to_owned(),
        "krishiv.auth.tokens.op.role=operator".to_owned(),
    ];
    let server = start_with(dir.path(), &conf);
    // An operator may run statements in the editor, but not durable DDL.
    let started = http_post_as(
        &server.admin,
        "/api/internal/sql",
        r#"{"sql": "CREATE TABLE editor_t (a INT) USING parquet"}"#,
        token,
    );
    assert!(started.starts_with("HTTP/1.1 202"), "{started}");
    let run = started
        .split("\"run_id\":\"")
        .nth(1)
        .and_then(|r| r.split('"').next())
        .expect("run id")
        .to_owned();
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        let status = http_get_as(
            &server.admin,
            &format!("/api/internal/sql/{run}"),
            Some(token),
        );
        assert!(status.starts_with("HTTP/1.1 200"), "{status}");
        if !status.contains("\"running\"") && !status.contains("\"queued\"") {
            break status;
        }
        assert!(Instant::now() < deadline, "the run did not end: {status}");
        std::thread::sleep(Duration::from_millis(100));
    };
    // The run ended refused by the role, and that refusal is the one denial counted.
    assert!(status.contains(r#""state":"failed""#), "{status}");
    assert!(
        status.contains("KRISHIV_INSUFFICIENT_PERMISSIONS"),
        "{status}"
    );
    let scrape = http_get_as(&server.admin, "/metrics", Some(token));
    let denied: Vec<&str> = scrape
        .lines()
        .filter(|l| l.starts_with("krishiv_auth_denied_total{"))
        .collect();
    let ui: Vec<&&str> = denied
        .iter()
        .filter(|l| l.contains(r#"surface="ui""#) && !l.ends_with(" 0"))
        .collect();
    assert_eq!(
        ui,
        [&r#"krishiv_auth_denied_total{surface="ui",privilege="CREATE"} 1"#],
        "{denied:?}"
    );
    assert!(
        !denied
            .iter()
            .any(|l| l.contains(r#"surface="connect""#) && !l.ends_with(" 0")),
        "{denied:?}"
    );
}

/// `krishiv.admin.v1` on the admin listener (design P3-T066 PR-52a): any authenticated caller
/// reads the version and the capabilities, which list the surface and its revision; a request
/// without a token is 401, and a `Host` outside the loopback names is 421.
#[test]
fn the_admin_v1_surface_is_served_behind_the_guard() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let token = "admin-v1-viewer-token-0123456789abcdef";
    let path = dir.path().join("viewer");
    std::fs::write(&path, token).expect("write secret");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let conf = [
        format!("krishiv.auth.tokens.v.secret=file:{}", path.display()),
        "krishiv.auth.tokens.v.principal=viewer-service".to_owned(),
        "krishiv.auth.tokens.v.role=viewer".to_owned(),
    ];
    let server = start_with(dir.path(), &conf);
    let none = http_get_host(&server.admin, "/api/admin/v1/version", "localhost", None);
    assert!(none.starts_with("HTTP/1.1 401"), "{none}");
    let version = http_get_host(
        &server.admin,
        "/api/admin/v1/version",
        "localhost",
        Some(token),
    );
    assert!(version.starts_with("HTTP/1.1 200"), "{version}");
    assert!(version.contains("\"adminV1Revision\":\"3\""), "{version}");
    let caps = http_get_host(
        &server.admin,
        "/api/admin/v1/capabilities",
        "localhost",
        Some(token),
    );
    assert!(caps.contains("\"admin.v1\""), "{caps}");
    let rebound = http_get_host(
        &server.admin,
        "/api/admin/v1/version",
        "evil.example",
        Some(token),
    );
    assert!(rebound.starts_with("HTTP/1.1 421"), "{rebound}");
    // The old route stays, marked deprecated, and names its successor.
    let old = http_get_host(&server.admin, "/api/v1/version", "localhost", Some(token));
    assert!(old.to_ascii_lowercase().contains("deprecation: @"), "{old}");
    // With authentication configured the alias needs the credential too (design D6).
    let open = http_get_host(&server.admin, "/api/v1/version", "localhost", None);
    assert!(open.starts_with("HTTP/1.1 401"), "{open}");
    // An unknown path is a 404 of the server's own, not a gRPC status, and the server started
    // with the UI router and the admin routes merged.
    let missing = http_get_host(&server.admin, "/api/nowhere", "localhost", Some(token));
    assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");
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
    tls.write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
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
    let _ = plain.write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n");
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

#[cfg(feature = "ui")]
#[test]
fn the_ui_host_allowlist_follows_krishiv_ui_allowed_hosts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut server = start_with(
        dir.path(),
        &["krishiv.ui.allowedHosts=ui.example.com".to_owned()],
    );
    let line = server
        ._stdout
        .next()
        .expect("the open line")
        .expect("utf-8");
    let token = line
        .split("?token=")
        .nth(1)
        .expect("token")
        .trim()
        .to_owned();
    let named = http_get_host(
        &server.admin,
        "/api/internal/health",
        "ui.example.com",
        Some(&token),
    );
    assert!(named.starts_with("HTTP/1.1 200"), "{named}");
    let local = http_get_host(
        &server.admin,
        "/api/internal/health",
        "localhost",
        Some(&token),
    );
    assert!(local.starts_with("HTTP/1.1 200"), "{local}");
    let other = http_get_host(
        &server.admin,
        "/api/internal/health",
        "evil.example",
        Some(&token),
    );
    assert!(other.starts_with("HTTP/1.1 403"), "{other}");
}

/// The open line's token: the one a server without authentication settings prints at start.
#[cfg(feature = "ui")]
fn startup_token(server: &mut Running) -> String {
    let line = server
        ._stdout
        .next()
        .expect("the open line")
        .expect("utf-8");
    line.split("?token=")
        .nth(1)
        .expect("token")
        .trim()
        .to_owned()
}

/// A server started on a history folder lists what an earlier run stored there; a folder that cannot be read
/// costs the old history, not the server.
#[cfg(feature = "ui")]
#[test]
fn a_restarted_server_lists_the_stored_history() {
    use krishiv_session::{EngineRuntime, Session};
    let history = tempfile::tempdir().expect("tempdir");
    {
        let rt = EngineRuntime::builder()
            .config(
                "krishiv.history.path",
                history.path().to_str().expect("utf-8"),
            )
            .build()
            .expect("runtime");
        let session = Session::builder()
            .runtime(std::sync::Arc::clone(&rt))
            .build()
            .expect("session");
        let df = futures::executor::block_on(
            session.sql("SELECT count(*) AS before_restart FROM range(0, 100)"),
        )
        .expect("sql");
        futures::executor::block_on(df.collect()).expect("collect");
        // Dropping the runtime flushes the history writer.
    }
    let spill = tempfile::tempdir().expect("tempdir");
    let mut server = start_with(
        spill.path(),
        &[format!("krishiv.history.path={}", history.path().display())],
    );
    let token = startup_token(&mut server);
    let listed = http_get_as(&server.admin, "/api/internal/history", Some(&token));
    assert!(listed.starts_with("HTTP/1.1 200"), "{listed}");
    assert!(listed.contains("before_restart"), "{listed}");
}

/// ADR 0049: the `external` access provider's startup refusals (no authentication, no endpoint,
/// any list with `grants`, which `krishiv server` does not wire yet) stop the server before it binds.
#[cfg(unix)]
#[test]
fn the_external_access_provider_is_checked_at_startup() {
    use std::os::unix::fs::PermissionsExt;
    let err = refused(&[
        "krishiv.access.provider=external",
        "krishiv.access.external.endpoint=http://127.0.0.1:9",
    ]);
    assert!(err.contains("needs authentication"), "{err}");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("t");
    std::fs::write(&path, "external-provider-token-0123456789abcdef").expect("write secret");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let secret = format!("krishiv.auth.tokens.a.secret=file:{}", path.display());
    let auth = [
        secret.as_str(),
        "krishiv.auth.tokens.a.principal=a",
        "krishiv.auth.tokens.a.role=viewer",
    ];
    let with = |more: &[&'static str]| {
        let mut confs: Vec<&str> = auth.to_vec();
        confs.extend_from_slice(more);
        refused(&confs)
    };
    let err = with(&["krishiv.access.provider=external"]);
    assert!(err.contains("endpoint"), "{err}");
    let err = with(&[
        "krishiv.access.provider=grants+external",
        "krishiv.access.external.endpoint=http://127.0.0.1:9",
    ]);
    assert!(
        err.contains("not available in `krishiv server` yet"),
        "{err}"
    );
    // Any list naming `grants` is refused, with or without `external`.
    for list in [
        "krishiv.access.provider=grants",
        "krishiv.access.provider=roles+grants",
    ] {
        let err = with(&[list]);
        assert!(
            err.contains("a list naming grants is refused"),
            "{list}: {err}"
        );
    }
    let err = with(&[
        "krishiv.access.provider=external",
        "krishiv.access.external.endpoint=http://policy.example.com",
    ]);
    assert!(err.contains("loopback"), "{err}");
}
